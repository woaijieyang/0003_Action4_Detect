import copy
import hashlib
import json
import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from model.training.action4_role_continuation import (
    ROOT,canonical,verify_role_continuation,restore_role_continuation,validate_continuation_stop,_apply,SCHEMA)

OUT=ROOT/"outputs/action4_a0_resource_continuation_v001/state_contract"
def ref(path):
    return {"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2));return ref(path)

class Fixture:
    def __init__(self,root,semantic=None,source_mutation=False,state_mutation=None,eval_count=4,lineage_addition=None,purpose="training_continuation"):
        self.root=root;self.sources=[];self.cfgs=[];self.specs=[];self.artifacts=[]
        for label,workers in [("old",4),("new",12)]:
            base=root/label;code=base/"source";(code/"src/model/training").mkdir(parents=True)
            (code/"src/data").mkdir(parents=True);(code/"inputs").mkdir()
            (code/"src/model/__init__.py").write_text("# frozen fixture package\n")
            (code/"src/model/training/__init__.py").write_text("# frozen fixture training package\n")
            (code/"src/model/training/action4_stage2_train.py").write_text("# reviewed "+label+" transport\n")
            (code/"src/model/training/action4_stage2_dispatch.py").write_text("# stable dispatch\n")
            (code/"src/data/math.py").write_text("# changed math\n" if label=="new" and source_mutation else "# immutable math\n")
            if label=="new":
                helper=ROOT/"src/model/training/action4_role_continuation.py"
                (code/"src/model/training/action4_role_continuation.py").write_bytes(helper.read_bytes())
            if label=="new" and lineage_addition:
                addition=code/lineage_addition;addition.parent.mkdir(parents=True,exist_ok=True)
                addition.write_text("# reviewed lineage transport only\n")
            (code/"inputs/A0.pt").write_bytes(b"identical-parent")
            (code/"inputs/B0.pt").write_bytes(b"identical-counterpart")
            parent_path=str((code/"inputs/A0.pt").relative_to(ROOT))
            write_json(code/"inputs/registry.json",{"entries":{"A0":{"path":parent_path,"sha256":ref(code/"inputs/A0.pt")["sha256"],"selected_step":14000}}})
            count=4 if label=="old" else eval_count
            cpus={"train":list(range(68 if count==8 else 48,(68 if count==8 else 48)+workers+2))}
            cpus.update({f"eval_{i}":list(range(24+4*i,28+4*i)) for i in range(count)})
            codehash={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((code/"src").rglob("*.py"))}
            cid=hashlib.sha256(json.dumps(codehash,sort_keys=True).encode()).hexdigest()
            gate=write_json(base/"GATE.json",{"snapshot":label})
            admission=write_json(base/"ADMISSION.json",{"snapshot":label})
            cfg={"role":"A0","mode":"full_training","batch":48,"micro":2,"world_size":1,"epochs":80,"eval_every":10,
                 "expected_units":10400,"expected_eval_images":650,"require_immutable_source":True,
                 "seed":20261024,"loader_workers":workers,"threads":2,"eval_loader_workers":2,
                 "training_gpus":[7],"eval_gpus":([4,6,7,7] if count==4 else [4,4,6,6,7,7] if count==6 else [4]*4+[6]*4),
                 "process_cpu_affinity":cpus,"reserved_cpu_budget":workers+2+4*count+1,
                 "project_total_reserved_cpu":69,"output":str((base/"run").relative_to(ROOT)),
                 "run_spec":str((base/"RUN_SPEC.json").relative_to(ROOT)),
                 "source_identity":{str((code/p).relative_to(ROOT)):sha for p,sha in codehash.items()},
                 "project_code_sha256":cid,"tested_project_code_sha256":cid,
                 "parent_bundle":parent_path,"parent_bundle_sha256":ref(code/"inputs/A0.pt")["sha256"],
                 "parent_registry":str((code/"inputs/registry.json").relative_to(ROOT)),
                 "evaluation_counterpart_bundle":str((code/"inputs/B0.pt").relative_to(ROOT)),
                 "evaluation_counterpart_sha256":ref(code/"inputs/B0.pt")["sha256"],
                 "execution_gate":gate["path"],"integration_or_admission_receipt":admission["path"],
                 "dataset_root":"data/phrase2","surface_source":{"train_manifest":"a"*64},
                 "preprocess":{"roi_expansion":2.25},"lr":0.0001,"loss_weights":{"xyz":1.0},
                 "parent_best_step":14000,"max_pending":8}
            if label=="new" and purpose=="validation_probe":cfg["mode"]="integration"
            if label=="new" and semantic:cfg.update(semantic)
            cfgref=write_json(code/"resolved_config.json",cfg)
            files={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.rglob("*")) if p.is_file()}
            spec={"schema":"action4.immutable_run.v1","project_root":str(ROOT),"code_root":str(code.relative_to(ROOT)),
                  "config":cfgref["path"],"files":files,"project_code_sha256":cid}
            specref=write_json(base/"RUN_SPEC.json",spec)
            for p in code.rglob("*"):
                if p.is_file():p.chmod(0o444)
            (base/"RUN_SPEC.json").chmod(0o444)
            self.sources.append(code);self.cfgs.append(cfg);self.specs.append(spec)
            self.artifacts.append({"run_spec":specref,"config":cfgref,"config_id":canonical(cfg)})
        model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,17360,eta_min=.000001)
        model(torch.ones(1,2)).sum().backward();optimizer.step();scheduler.step()
        rng={"torch":torch.get_rng_state(),"cuda":[torch.get_rng_state().clone()],"python":random.getstate(),
             "numpy_global":json.dumps(np.random.get_state(),default=lambda x:x.tolist()),"roi":np.random.default_rng(3).bit_generator.state}
        state={"resumable":True,"role":"A0","config_id":canonical(self.cfgs[0]),"world_size":1,
               "training_model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "rng_by_rank":[rng],"epoch":0,"cursor":48,"step":1,"data_update_attempt":1,"no_supervision_update_count":0}
        if state_mutation:state_mutation(state)
        self.state=copy.deepcopy(state);checkpoint=root/"capture.pt";torch.save(state,checkpoint);checkpoint.chmod(0o444)
        dispatch=write_json(root/"DISPATCH.json",{"attempt_id":"original-attempt"});self.artifacts[0].update(checkpoint=ref(checkpoint),dispatch=dispatch,attempt_id="original-attempt")
        approved=[]
        for path in ["src/model/training/action4_stage2_train.py","src/model/training/action4_role_continuation.py"]:
            approved.append({"path":path,"before_sha256":self.specs[0]["files"].get(path),
                "after_sha256":self.specs[1]["files"].get(path),"purpose":"transport_control"})
        if lineage_addition:
            approved.append({"path":lineage_addition,"before_sha256":None,
                "after_sha256":self.specs[1]["files"][lineage_addition],"purpose":"transport_control"})
        paths=["parent_bundle","parent_registry","evaluation_counterpart_bundle","execution_gate","integration_or_admission_receipt"]
        relocated={key:{"before":ref(ROOT/self.cfgs[0][key]),"after":ref(ROOT/self.cfgs[1][key])} for key in paths}
        self.receipt={"schema":SCHEMA,"purpose":purpose,"authorization":{"controlled_checkpoint_switch":True,"scope":"loader_resource_continuation"},
            "predecessor":self.artifacts[0],"successor":self.artifacts[1],"approved_source_deltas":approved,
            "relocated_artifacts":relocated,"boundary":{key:state[key] for key in ["epoch","cursor","step","data_update_attempt","no_supervision_update_count"]}}
        self.receipt_path=root/"RECEIPT.json";self.seal_receipt()
    def seal_receipt(self):
        if self.receipt_path.exists():self.receipt_path.chmod(0o644)
        self.reference=write_json(self.receipt_path,self.receipt);self.receipt_path.chmod(0o444)
    def verify(self):return verify_role_continuation(self.reference,self.cfgs[1])

def tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b

class RoleContinuationTests(unittest.TestCase):
    def setUp(self):
        OUT.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=OUT);self.base=Path(self.temp.name)
        torch.set_num_threads(1)
    def tearDown(self):self.temp.cleanup()
    def fixture(self,**kwargs):return Fixture(self.base/"case",**kwargs)
    def test_verified_exact_state_and_actual_apply_preserve_tensors(self):
        f=self.fixture();verified=f.verify();before=copy.deepcopy(verified.state)
        m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=.0001)
        s=torch.optim.lr_scheduler.CosineAnnealingLR(o,17360,eta_min=.000001);captured={}
        boundary=_apply(verified,m,o,s,captured,lambda state,r:r.update(copy.deepcopy(state)))
        self.assertEqual(boundary["cursor"],48)
        self.assertTrue(tree_equal(before,verified.state))
        self.assertTrue(tree_equal(m.state_dict(),f.state["training_model"]))
        self.assertTrue(tree_equal(o.state_dict(),f.state["optimizer"]))
        self.assertTrue(tree_equal(s.state_dict(),f.state["scheduler"]))
        self.assertTrue(tree_equal(captured,f.state["rng_by_rank"][0]))
        self.assertEqual(verified.state["config_id"],canonical(f.cfgs[0]))
        self.assertNotEqual(verified.state["config_id"],verified.successor_config_id)
        self.assertFalse(verified.runtime_switch_authorized)
        for value in o.state.values():
            for x in value.values():
                if isinstance(x,torch.Tensor):x.add_(1)
        self.assertTrue(tree_equal(before,verified.state))
    def test_public_restore_rejects_mutable_worktree_origin(self):
        f=self.fixture();m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=.0001)
        s=torch.optim.lr_scheduler.CosineAnnealingLR(o,17360,eta_min=.000001)
        with self.assertRaisesRegex(ValueError,"successor immutable"):
            restore_role_continuation(f.reference,f.cfgs[1],m,o,s,{},lambda x,y:None)
    def test_public_restore_from_successor_snapshot_CPU_fixture(self):
        import subprocess,sys
        f=self.fixture()
        script=self.base/"public_restore_probe.py"
        program="""import sys,json,torch
sys.dont_write_bytecode=True
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/"src"))
from model.training.action4_role_continuation import restore_role_continuation,verify_role_continuation
ref=json.loads(Path(sys.argv[2]).read_text())
cfg=json.loads((code/"resolved_config.json").read_text())
m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=.0001)
s=torch.optim.lr_scheduler.CosineAnnealingLR(o,17360,eta_min=.000001)
captured={}
boundary,lineage=restore_role_continuation(ref,cfg,m,o,s,captured,lambda state,r:r.update(state))
v=verify_role_continuation(ref,cfg)
assert all(torch.equal(m.state_dict()[k],t) for k,t in v.state["training_model"].items())
assert boundary["step"]==1 and s.last_epoch==1 and len(captured["cuda"])==1
assert v.state["config_id"]==lineage["state_config_id_preserved"]!=v.successor_config_id
assert all(torch.equal(o.state_dict()["state"][i][k],v.state["optimizer"]["state"][i][k])
 for i in o.state_dict()["state"] for k in o.state_dict()["state"][i])
print("PUBLIC_FROZEN_RESTORE_CPU_FIXTURE_PASS_NO_GPU")
"""
        script.write_text(program);refpath=self.base/"ref.json";refpath.write_text(json.dumps(f.reference))
        result=subprocess.run([sys.executable,"-I",str(script),str(f.sources[1]),str(refpath)],
            cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("PUBLIC_FROZEN_RESTORE_CPU_FIXTURE_PASS_NO_GPU",result.stdout)

    def test_six_eval_workers_resource_only_allowed(self):
        f=self.fixture(eval_count=6);self.assertEqual(f.verify().lineage["CPU_after"]["minimum_with_controller"],39)
    def test_eight_eval_workers_explicit_budget(self):
        f=self.fixture(eval_count=8)
        result=f.verify()
        self.assertEqual(result.lineage["CPU_after"]["minimum_with_controller"],47)
        self.assertEqual(len(result.lineage["CPU_after"]["unique_worker_cpus"]),46)

    def test_explicit_validation_probe_mode_only_and_exact_two_steps(self):
        formal=Fixture(self.base/"formal");probe=Fixture(self.base/"probe",purpose="validation_probe")
        f=formal.verify();p=probe.verify()
        self.assertEqual(p.lineage["purpose"],"validation_probe")
        self.assertEqual({k:v for k,v in formal.specs[1]["files"].items() if k.endswith(".py")},
                         {k:v for k,v in probe.specs[1]["files"].items() if k.endswith(".py")})
        validate_continuation_stop(p.lineage,p.state["step"],3)
        validate_continuation_stop(f.lineage,f.state["step"],None)
        for stop in (None,1,2,4,True):
            with self.assertRaises(ValueError):validate_continuation_stop(p.lineage,1,stop)
        with self.assertRaises(ValueError):validate_continuation_stop(f.lineage,1,3)
        exhausted=copy.deepcopy(p.lineage);exhausted["boundary"].update(step=17359,data_update_attempt=17359)
        with self.assertRaises(ValueError):validate_continuation_stop(exhausted,17359,17361)
        probe.receipt.pop("purpose");probe.seal_receipt()
        with self.assertRaises(ValueError):probe.verify()
        for i,kwargs in enumerate([
            {"purpose":"validation_probe","semantic":{"lr":.1}},
            {"purpose":"validation_probe","semantic":{"mode":"full_training"}},
            {"purpose":"training_continuation","semantic":{"mode":"integration"}},
            {"purpose":"validation_probe","semantic":{"expected_units":48}},
        ]):
            bad=Fixture(self.base/f"probe_bad{i}",**kwargs)
            with self.assertRaises(ValueError):bad.verify()

    def test_changed_semantics_fail(self):
        for i,(key,value) in enumerate([("batch",96),("micro",4),("world_size",2),("lr",.1),("epochs",81),
            ("eval_every",20),("seed",1),("surface_source",{}),("preprocess",{}),("loss_weights",{"xyz":2.})]):
            with self.subTest(key=key):
                f=Fixture(self.base/str(i),semantic={key:value})
                with self.assertRaises(ValueError):f.verify()
    def test_unexpected_source_math_change_and_missing_approval(self):
        f=self.fixture(source_mutation=True)
        with self.assertRaisesRegex(ValueError,"source delta"):f.verify()
        g=Fixture(self.base/"other");g.receipt["approved_source_deltas"].pop();g.seal_receipt()
        with self.assertRaises(ValueError):g.verify()
    def test_exact_lineage_module_additions_only(self):
        for i,name in enumerate(["src/eval/action4_role_continuation_eval.py",
                                 "src/model/training/action4_role_continuation_observer.py",
                                 "src/eval/unapproved.py"]):
            f=Fixture(self.base/str(i),lineage_addition=name)
            if name.endswith("unapproved.py"):
                with self.assertRaises(ValueError):f.verify()
            else:
                self.assertIn(name,f.verify().lineage["approved_source_deltas"])

    def test_missing_hash_old_config_and_parent_bindings(self):
        for i,mutate in enumerate([
            lambda r:r["predecessor"]["checkpoint"].pop("sha256"),
            lambda r:r["predecessor"]["config"].update(sha256="0"*64),
            lambda r:r["predecessor"].update(config_id="0"*64),
            lambda r:r["relocated_artifacts"]["parent_bundle"]["after"].update(sha256="0"*64),
            lambda r:r["relocated_artifacts"]["evaluation_counterpart_bundle"]["after"].update(sha256="0"*64),
            lambda r:r["predecessor"].update(attempt_id="different")]):
            f=Fixture(self.base/str(i));mutate(f.receipt);f.seal_receipt()
            with self.assertRaises(ValueError):f.verify()
    def test_cursor_tail_and_missing_full_state(self):
        for i,mutate in enumerate([
            lambda s:s.update(cursor=47),
            lambda s:s.update(data_update_attempt=2),
            lambda s:s.update(no_supervision_update_count=1),
            lambda s:s.update(training_model={}),
            lambda s:s.update(optimizer={}),
            lambda s:s.update(rng_by_rank=[]),
            lambda s:s["scheduler"].update(T_max=80),
            lambda s:s.update(config_id="0"*64)]):
            f=Fixture(self.base/str(i),state_mutation=mutate)
            with self.assertRaises(ValueError):f.verify()
        def tail(s):
            s.update(epoch=1,cursor=0,step=217,data_update_attempt=217)
            s["scheduler"]["last_epoch"]=217
        f=Fixture(self.base/"tail",state_mutation=tail)
        self.assertEqual(f.verify().lineage["actual_consumed_units"],10400)
    def test_invalid_resource_budgets_and_GPU(self):
        for i,change in enumerate([{"training_gpus":[0]},{"eval_gpus":[4,6,3,7]},
            {"project_total_reserved_cpu":73},{"reserved_cpu_budget":20},{"loader_workers":8}]):
            f=Fixture(self.base/str(i),semantic=change)
            with self.assertRaises(ValueError):f.verify()
    def test_registry_cannot_disguise_source_change(self):
        f=self.fixture()
        new=f.cfgs[1]
        # A source file cannot be declared registry metadata to bypass deltas.
        new["parent_registry"]=str((f.sources[1]/"src/data/math.py").relative_to(ROOT))
        f.receipt["relocated_artifacts"]["parent_registry"]["after"]=ref(f.sources[1]/"src/data/math.py")
        from model.training.action4_role_continuation import BoundReader,compare_configs
        with self.assertRaisesRegex(ValueError,"Registry"):
            compare_configs(BoundReader(),f.receipt,f.cfgs[0],new,*f.sources)

    def test_bound_checkpoint_nonfinite_and_successor_identity(self):
        f=self.fixture(state_mutation=lambda s:s["training_model"]["weight"].fill_(float("nan")))
        with self.assertRaises(ValueError):f.verify()
        g=Fixture(self.base/"other");cfg=copy.deepcopy(g.cfgs[1]);cfg["seed"]=0
        with self.assertRaises(ValueError):verify_role_continuation(g.reference,cfg)

if __name__=="__main__":unittest.main()
