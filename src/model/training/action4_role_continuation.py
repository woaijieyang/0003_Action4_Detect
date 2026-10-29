"""Explicit full-state loader-resource continuation; ordinary resume stays strict.

All IO is project-contained and SHA-bound. This is not a controller/launcher.
The predecessor checkpoint config_id is never rewritten. Only receipt-reviewed
transport/control source deltas are admissible, never mathematical model/data
changes. A verified state does not authorize process ownership or cutover.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path("/data/zy/vln/a-test")
SCHEMA="action4.role_resource_continuation.v1"
CONTROL_PATHS=frozenset({
    "src/model/training/action4_stage2_train.py",
    "src/model/training/action4_stage2_dispatch.py",
    "src/model/training/action4_role_continuation.py",
    "src/model/training/action4_stage2_core.py",  # reviewed bundle metadata only
    "src/eval/action4_role_continuation_eval.py",
    "src/model/training/action4_role_continuation_observer.py",
})


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def contained(value,exists=True):
    require(isinstance(value,(str,Path)),"Path required")
    raw=Path(value);p=raw if raw.is_absolute() else ROOT/raw
    require(".." not in p.parts,"Noncanonical path")
    require(not p.is_symlink() and p.resolve(strict=exists).is_relative_to(ROOT),"Path outside project or symlink")
    # Reject linked ancestors as well, even if they currently resolve inside ROOT.
    current=p
    while current!=ROOT and current.is_relative_to(ROOT):
        require(not current.is_symlink(),"Symlink ancestor")
        current=current.parent
    return p.resolve(strict=exists)


def sha256(path):
    with contained(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def valid_sha(value):
    require(isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value),"SHA256 required")
    return value


class BoundReader:
    def __init__(self):self.seen={}
    def file(self,ref,readonly=False):
        require(isinstance(ref,dict) and set(ref)=={"path","sha256"},"Exact path/SHA reference required")
        p=contained(ref["path"]);sha=valid_sha(ref["sha256"]);st=p.stat()
        stamp=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if readonly:require(not st.st_mode&0o222,"Immutable input must be readonly")
        if str(p) in self.seen:require(self.seen[str(p)]==(sha,stamp),"Bound input changed")
        else:
            require(sha256(p)==sha,"Bound input SHA mismatch")
            self.seen[str(p)]=(sha,stamp)
        after=p.stat()
        require(stamp==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),"Evidence mutated during file verification")
        return p
    def json(self,ref,readonly=False):
        p=self.file(ref,readonly)
        def reject(x):raise ValueError("Nonfinite JSON:"+x)
        value=json.loads(p.read_text(),parse_constant=reject)
        self.unchanged();return value
    def unchanged(self):
        for name,(_,expected) in self.seen.items():
            st=contained(name).stat()
            require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==expected,"Evidence mutated during verification")


def integer(value,name,minimum=0):
    require(type(value) is int and value>=minimum,"Invalid integer:"+name);return value


def relative_name(value):
    require(isinstance(value,str) and value and "\\" not in value,"Relative source name required")
    p=Path(value)
    require(not p.is_absolute() and ".." not in p.parts and str(p)==value and "." not in value.split("/"),"Unsafe source name")
    return p


def snapshot(reader,section):
    spec=reader.json(section["run_spec"],True);cfg=reader.json(section["config"],True)
    require(spec.get("schema")=="action4.immutable_run.v1" and spec.get("project_root")==str(ROOT),"RunSpec identity invalid")
    code=contained(spec["code_root"])
    require(contained(spec["config"])==contained(section["config"]["path"])
            and contained(cfg["run_spec"])==contained(section["run_spec"]["path"]),"Config/RunSpec reference mismatch")
    require(contained(section["config"]["path"])==code/"resolved_config.json","Resolved config must use non-executable snapshot metadata path")
    files=spec["files"];require(isinstance(files,dict) and files,"Full snapshot manifest required")
    for rel,sha in files.items():
        p=code/relative_name(rel)
        reader.file({"path":str(p),"sha256":sha},True)
    require(files.get(str(contained(section["config"]["path"]).relative_to(code)))==section["config"]["sha256"],"Config not bound in manifest")
    # No unlisted importable file may shadow an authenticated import.
    for folder in ("src","runtime/surface_deps"):
        base=code/folder
        if not base.exists():continue
        for p in base.rglob("*"):
            require(not p.is_symlink() and p.resolve().is_relative_to(code),"Import inventory escaped")
            if p.is_file() and (p.suffix in (".py",".pyc",".pyo",".so",".pyd") or ".so." in p.name):
                require(str(p.relative_to(code)) in files,"Unlisted executable import")
    expected={str(code/rel):sha for rel,sha in files.items() if rel.startswith("src/") and rel.endswith(".py")}
    actual={str(contained(name)):sha for name,sha in cfg["source_identity"].items()}
    require(actual==expected and expected,"Consumed source identity differs from manifest")
    relative={str(Path(name).relative_to(code)):sha for name,sha in expected.items()}
    code_id=hashlib.sha256(json.dumps(relative,sort_keys=True).encode()).hexdigest()
    require(cfg["project_code_sha256"]==cfg["tested_project_code_sha256"]==spec["project_code_sha256"]==code_id,
            "Actual source code hash mismatch")
    return cfg,spec,code


def resources(cfg):
    world=cfg["world_size"];gpus=cfg["training_gpus"];evaluators=cfg["eval_gpus"]
    require(isinstance(gpus,list) and len(gpus)==world==1 and isinstance(evaluators,list) and 1<=len(evaluators)<=8,"Resource topology invalid")
    require(all(type(g) is int and 4<=g<=7 for g in gpus+evaluators),"GPU0-3 or invalid GPU forbidden")
    threads=integer(cfg["threads"],"threads",1);workers=integer(cfg["loader_workers"],"loader_workers",1)
    ew=integer(cfg["eval_loader_workers"],"eval_loader_workers")
    affinity=cfg["process_cpu_affinity"]
    require(set(affinity)=={"train"}|{f"eval_{i}" for i in range(len(evaluators))},"CPU role mapping differs")
    union=set()
    for role,ids in affinity.items():
        require(isinstance(ids,list) and ids and all(type(x) is int and 24<=x<=95 for x in ids),"CPU24-95 required")
        require(len(ids)==len(set(ids)) and not union.intersection(ids),"CPU overlaps/duplicates")
        minimum=threads+workers if role=="train" else 2+ew
        require(len(ids)>=minimum,"Worker CPU affinity insufficient")
        union.update(ids)
    need=threads+workers+(2+ew)*len(evaluators)+1
    reserved=integer(cfg["reserved_cpu_budget"],"reserved_cpu_budget",1)
    total=integer(cfg["project_total_reserved_cpu"],"project_total_reserved_cpu",1)
    require(max(need,len(union)+1)<=reserved<=total<=72,"CPU budget exceeds72 or undersized")
    return {"unique_worker_cpus":sorted(union),"minimum_with_controller":need,"project_declared_total":total}


def compare_configs(reader,receipt,old,new,oldcode,newcode):
    purpose=receipt.get("purpose")
    require(purpose in ("validation_probe","training_continuation"),"Explicit continuation purpose required")
    require(old.get("mode")=="full_training","Predecessor must be formal training")
    expected_mode="integration" if purpose=="validation_probe" else "full_training"
    require(new.get("mode")==expected_mode,"Successor mode differs from explicit purpose")
    for cfg in (old,new):
        for key,value in {"role":"A0","batch":48,"micro":2,"world_size":1,
                          "epochs":80,"eval_every":10,"expected_units":10400,"expected_eval_images":650,
                          "require_immutable_source":True}.items():
            require(type(cfg.get(key)) is type(value) and cfg[key]==value,"Mathematical protocol changed:"+key)
    require(old["loader_workers"]==4 and new["loader_workers"]==12,"Only loader4->12 continuation")
    require(set(old)==set(new),"Configuration key set changed")
    paths={"parent_bundle","parent_registry","evaluation_counterpart_bundle","execution_gate","integration_or_admission_receipt"}
    allowed=paths|{"loader_workers","process_cpu_affinity","reserved_cpu_budget","project_total_reserved_cpu",
                   "training_gpus","eval_gpus","eval_loader_workers","output","run_spec","source_identity",
                   "project_code_sha256","tested_project_code_sha256"}
    if purpose=="validation_probe":allowed.add("mode")
    for key in old:
        if key not in allowed:require(canonical(old[key])==canonical(new[key]),"Semantic config field changed:"+key)
    require(contained(old["output"],False)!=contained(new["output"],False),"Separate successor output required")
    require(oldcode!=newcode,"Separate immutable successor code root required")
    artifact=receipt["relocated_artifacts"]
    require(set(artifact)==paths,"Explicit path/hash bindings for all relocated dependencies/gates required")
    for key in paths:
        binding=artifact[key]
        require(set(binding)=={"before","after"},"Relocation binding fields differ")
        for label,cfg in (("before",old),("after",new)):
            require(contained(binding[label]["path"])==contained(cfg[key]),"Relocated path differs:"+key)
            reader.file(binding[label])
        if key in ("parent_bundle","evaluation_counterpart_bundle"):
            hashkey="parent_bundle_sha256" if key=="parent_bundle" else "evaluation_counterpart_sha256"
            require(binding["before"]["sha256"]==binding["after"]["sha256"]==old[hashkey]==new[hashkey],
                    "Parent/counterpart bytes differ")
            require(contained(old[key]).is_relative_to(oldcode/"inputs") and contained(new[key]).is_relative_to(newcode/"inputs"),"Dependency not frozen input")
        if key=="parent_registry":
            require(contained(old[key]).is_relative_to(oldcode/"inputs") and contained(new[key]).is_relative_to(newcode/"inputs"),"Registry must remain frozen non-source input")
            a=reader.json(binding["before"]);b=reader.json(binding["after"])
            # Only exact bound parent path relocation; every other registry field
            # (role, step, architecture and hashes) retains its original value.
            def normalized(value,path):
                if isinstance(value,dict):return {k:normalized(v,path) for k,v in value.items()}
                if isinstance(value,list):return [normalized(v,path) for v in value]
                return "<BOUND_PARENT_PATH>" if value==path else value
            require(canonical(normalized(a,old["parent_bundle"]))==canonical(normalized(b,new["parent_bundle"])),
                    "Parent registry semantics changed")
    return resources(new),resources(old)


def compare_sources(receipt,oldspec,newspec,oldcfg,newcfg,oldcode,newcode):
    old=oldspec["files"];new=newspec["files"];approved=receipt["approved_source_deltas"]
    require(isinstance(approved,list),"Approved source delta list required")
    approvals={}
    for item in approved:
        require(set(item)=={"path","before_sha256","after_sha256","purpose"} and item["purpose"]=="transport_control",
                "Exact reviewed transport/control delta required")
        path=item["path"];require(path in CONTROL_PATHS and path not in approvals,"Unexpected/duplicate approved source delta")
        for name in ("before_sha256","after_sha256"):
            if item[name] is not None:valid_sha(item[name])
        require(item["after_sha256"] is not None,"Deleting transport source is not supported")
        approvals[path]=item
    metadata={str(contained(oldspec["config"]).relative_to(oldcode)),str(contained(newspec["config"]).relative_to(newcode)),
              str(contained(oldcfg["parent_registry"]).relative_to(oldcode)),str(contained(newcfg["parent_registry"]).relative_to(newcode))}
    actual={}
    for path in set(old)|set(new):
        if old.get(path)!=new.get(path) and path not in metadata:
            actual[path]={"path":path,"before_sha256":old.get(path),"after_sha256":new.get(path),"purpose":"transport_control"}
    require(actual==approvals,"Unreviewed/unused source delta; mathematical/data/import changes forbidden")
    return approvals


def check_state(state,old,boundary):
    import torch
    require(isinstance(state,dict) and state.get("resumable") is True and state.get("role")=="A0","Full resumable A0 state required")
    require(state.get("config_id")==canonical(old) and type(state.get("world_size")) is int and state["world_size"]==1,"Predecessor checkpoint/config/topology mismatch")
    fields=("epoch","cursor","step","data_update_attempt","no_supervision_update_count")
    for key in fields:integer(state[key],key)
    require(set(boundary)==set(fields) and all(type(boundary[k]) is int and boundary[k]==state[k] for k in fields),"Captured boundary differs")
    ep,cur,step,attempt,zero=(state[k] for k in fields)
    require(ep<80 and cur<10400 and cur%48==0 and step>0,"Invalid/terminal global update boundary")
    require(attempt==ep*math.ceil(10400/48)+cur//48 and step+zero==attempt,"Traversal/tail/update accounting mismatch")
    require(isinstance(state.get("training_model"),dict) and state["training_model"],"Model state missing")
    opt=state.get("optimizer");sched=state.get("scheduler");rng=state.get("rng_by_rank")
    require(isinstance(opt,dict) and set(opt)=={"state","param_groups"} and opt["state"] and opt["param_groups"],"Optimizer state missing")
    require(isinstance(sched,dict) and sched.get("last_epoch")==step and type(sched.get("last_epoch")) is int
            and sched.get("T_max")==17360,"Cosine scheduler state/horizon differs")
    groups=opt["param_groups"];require(isinstance(groups,list) and all(isinstance(g,dict) and g.get("params") for g in groups),"Optimizer groups missing")
    ids=[p for group in groups for p in group["params"]]
    require(len(ids)==len(set(ids)) and set(opt["state"])<=set(ids),"Optimizer parameter identities invalid")
    require(sched.get("base_lrs")==[g.get("initial_lr") for g in groups]
            and sched.get("_last_lr")==[g.get("lr") for g in groups],"Optimizer/scheduler rates inconsistent")
    require(isinstance(rng,list) and len(rng)==1 and isinstance(rng[0],dict)
            and set(rng[0])=={"torch","cuda","python","numpy_global","roi"},"Full exact per-rank RNG state required")
    r=rng[0]
    require(isinstance(r["torch"],torch.Tensor) and r["torch"].dtype==torch.uint8 and r["torch"].ndim==1
            and isinstance(r["cuda"],list) and len(r["cuda"])==1
            and all(isinstance(t,torch.Tensor) and t.dtype==torch.uint8 and t.ndim==1 for t in r["cuda"]),"CPU/CUDA RNG topology invalid")
    import random
    random.Random().setstate(r["python"])
    require(isinstance(r["numpy_global"],str) and isinstance(r["roi"],dict) and r["roi"],"NumPy/ROI RNG state missing")
    import numpy as np
    n=json.loads(r["numpy_global"],parse_constant=lambda x:(_ for _ in ()).throw(ValueError("Nonfinite NumPy RNG")))
    require(isinstance(n,list) and len(n)==5,"NumPy global RNG structure differs")
    local_global=np.random.RandomState()
    local_global.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    local_roi=np.random.default_rng()
    local_roi.bit_generator.state=copy.deepcopy(r["roi"])
    def finite(value):
        if isinstance(value,torch.Tensor):
            require(value.device.type=="cpu" and bool(torch.isfinite(value).all()),"Nonfinite/non-CPU serialized tensor")
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(tuple,list)):
            for v in value:finite(v)
        elif isinstance(value,float):require(math.isfinite(value),"Nonfinite serialized scalar")
    finite(state)
    return ep*10400+cur


@dataclass(frozen=True)
class VerifiedRoleContinuation:
    state: dict
    lineage: dict
    successor_code_root: str
    successor_config_id: str
    receipt_sha256: str
    runtime_switch_authorized: bool=False


def verify_role_continuation(receipt_ref,successor_config):
    """Read+verify full immutable states. Does not mutate a model or start work."""
    reader=BoundReader();receipt=reader.json(receipt_ref,True)
    require(receipt.get("schema")==SCHEMA,"Continuation receipt schema differs")
    require(receipt.get("purpose") in ("validation_probe","training_continuation"),"Explicit continuation purpose required")
    require(receipt.get("authorization")=={"controlled_checkpoint_switch":True,"scope":"loader_resource_continuation"},
            "Explicit controlled-switch receipt required")
    old,oldspec,oldcode=snapshot(reader,receipt["predecessor"])
    new,newspec,newcode=snapshot(reader,receipt["successor"])
    require(canonical(successor_config)==canonical(new)==receipt["successor"]["config_id"],"Successor config identity mismatch")
    require(canonical(old)==receipt["predecessor"]["config_id"],"Predecessor config identity mismatch")
    cpu_new,cpu_old=compare_configs(reader,receipt,old,new,oldcode,newcode)
    deltas=compare_sources(receipt,oldspec,newspec,old,new,oldcode,newcode)
    dispatch=reader.json(receipt["predecessor"]["dispatch"])
    require(isinstance(dispatch.get("attempt_id"),str) and dispatch["attempt_id"]
            and dispatch["attempt_id"]==receipt["predecessor"]["attempt_id"],"Original attempt identity mismatch")
    if receipt.get("observer_dependency") is not None:
        reader.file(receipt["observer_dependency"],True)
    checkpoint=reader.file(receipt["predecessor"]["checkpoint"],True)
    import torch
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    units=check_state(state,old,receipt["boundary"])
    if "attempt_id" in state:require(state["attempt_id"]==dispatch["attempt_id"],"Checkpoint attempt mismatch")
    reader.unchanged()
    lineage={"schema":SCHEMA,"purpose":receipt["purpose"],"predecessor":copy.deepcopy(receipt["predecessor"]),
             "successor":copy.deepcopy(receipt["successor"]),"boundary":copy.deepcopy(receipt["boundary"]),
             "actual_consumed_units":units,"approved_source_deltas":copy.deepcopy(deltas),
             "CPU_before":cpu_old,"CPU_after":cpu_new,"state_config_id_preserved":state["config_id"],
             "observer_dependency":copy.deepcopy(receipt.get("observer_dependency")),
             "process_control_performed":False,"evaluation_prefix_admitted_here":False}
    return VerifiedRoleContinuation(state,lineage,str(newcode),canonical(new),receipt_ref["sha256"])


def validate_continuation_stop(lineage,loaded_step,stop_after):
    """Called before any update; reuse original integration --stop-after only."""
    integer(loaded_step,"loaded_step",1)
    purpose=lineage.get("purpose")
    require(purpose in ("validation_probe","training_continuation"),"Explicit continuation purpose required")
    if purpose=="validation_probe":
        require(lineage.get("boundary",{}).get("step")==loaded_step and lineage["boundary"]["data_update_attempt"]<=17358,
                "Probe needs at least two remaining global update attempts")
        require(type(stop_after) is int and stop_after==loaded_step+2,
                "Validation probe must stop after exactly two optimizer updates")
    else:
        require(stop_after is None,"Formal continuation cannot use integration stop_after")


def _apply(verified,model,optimizer,scheduler,rng,restore_rng):
    """Internal CPU-testable payload application; production wrapper checks origin."""
    state=copy.deepcopy(verified.state)
    current=scheduler.state_dict()
    require(current.get("T_max")==state["scheduler"]["T_max"] and current.get("eta_min")==state["scheduler"]["eta_min"],
            "Constructed successor scheduler mathematical settings differ")
    model.load_state_dict(state["training_model"],strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng_by_rank"][0],rng)
    return {k:state[k] for k in ("epoch","cursor","step","data_update_attempt","no_supervision_update_count")}


def restore_role_continuation(receipt_ref,successor_config,model,optimizer,scheduler,rng,restore_rng):
    """Verify and restore in successor snapshot; never rewrite predecessor state."""
    verified=verify_role_continuation(receipt_ref,successor_config)
    code=contained(verified.successor_code_root)
    require(Path(__file__).resolve()==code/"src/model/training/action4_role_continuation.py",
            "Continuation loader must run from approved successor immutable source")
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,"__file__",None)
        if filename and name.split(".")[0] in {"data","model","eval"}:
            require(Path(filename).resolve().is_relative_to(code/"src"),"Mutable/non-successor project import")
    _apply(verified,model,optimizer,scheduler,rng,restore_rng)
    return verified.state,copy.deepcopy(verified.lineage)
