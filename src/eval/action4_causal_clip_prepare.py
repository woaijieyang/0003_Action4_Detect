"""Prepare immutable causal clip jobs without launching any process or GPU.

New caller contract project_code_ref:
{schema: action4.causal_clip_tested_project.v1,
 tests: {passed: positive int, failed: 0}, files_sha256: {relative path: SHA}}.
Its inventory must cover current src/, runtime/surface_deps/ and the existing
frozen-sequence launcher (excluding bytecode). It is a test receipt supplied by
the caller, not a claim inferred by this preparer. Forward never reads current
project files; it inherits the completed discovery snapshot verbatim.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from eval.action4_sam_full_review import ROOT, ReceiptVerifier, checked_path, read_json
from eval.action4_sam_image_eval import validate_source_snapshot, registered_sam_allocation
from eval.action4_sam_b0_sequence import select_seeds, load_instance_seed_points
from model.action4_sam_session_contracts import RegionSpec
from eval.action4_clip_replay import _validated_exit, _terminal_state, _digest

CURRENT_PROJECT_ROOT = ROOT
LAUNCHER = "scripts/action4_run_frozen_sequence.py"


def _sha(path):
    with checked_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _ref(path):
    return {"path":str(checked_path(path).relative_to(ROOT)), "sha256":_sha(path)}


def _relative(value):
    p=Path(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Snapshot entry must be a contained relative path")
    return p


def _json_new(path,value):
    path=checked_path(path)
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
    path.chmod(0o444)


def _inventory(base):
    found={}
    for root,dirs,files in os.walk(checked_path(base),followlinks=False):
        directory=Path(root)
        if any((directory/n).is_symlink() for n in dirs+files):
            raise ValueError("Source inventory refuses symlinks")
        dirs[:]=[d for d in dirs if d!="__pycache__" and d!=".git"]
        for name in files:
            path=directory/name
            if path.suffix in (".pyc",".pyo"):continue
            found[str(path.relative_to(base))]=path
    return found


def _project(ref,verifier):
    receipt=verifier.bound_json(ref)
    if (receipt.get("schema")!="action4.causal_clip_tested_project.v1"
            or type(receipt.get("tests",{}).get("passed")) is not int
            or receipt["tests"]["passed"]<=0 or receipt["tests"].get("failed")!=0):
        raise ValueError("Explicit successful CPU-tested current project inventory required")
    actual={}
    for prefix in ("src","runtime/surface_deps"):
        base=checked_path(CURRENT_PROJECT_ROOT/prefix)
        if not base.is_dir():raise ValueError("Current runtime/project source tree missing: "+prefix)
        actual.update({str(Path(prefix)/rel):p for rel,p in _inventory(base).items()})
    actual[LAUNCHER]=checked_path(CURRENT_PROJECT_ROOT/LAUNCHER)
    expected=receipt["files_sha256"]
    if set(actual)!=set(expected):raise ValueError("Tested inventory does not cover exact current project/runtime files")
    required={"src/eval/action4_sam_b0_sequence.py","src/eval/action4_sam_sequence_backend.py",
              "src/eval/action4_clip_replay.py","src/data/action4_paths.py",LAUNCHER}
    if not required<=set(actual):raise ValueError("Tested source lacks causal producer/consumer modules")
    for name,path in actual.items():verifier.verify(path,expected[name])
    return receipt,actual


def _authority(ref,resources,verifier):
    policy=verifier.bound_json(ref)
    allowed=policy.get("cpu_policy",{}).get("allowed_logical_cpu_ids")
    specific=policy.get("zeroshot_allowed_resources")
    if not isinstance(allowed,list) or not isinstance(specific,dict):
        raise ValueError("Full explicit execution authority required; resource proposal is not authority")
    forbidden=policy.get("forbidden_gpu_ids",[])
    permitted=policy.get("authorized_evaluation_gpu",[])
    if resources["physical_gpu"] in forbidden or resources["physical_gpu"] not in permitted:
        raise ValueError("Registered worker GPU is not authorized by this policy")
    maximum=policy["cpu_policy"].get("max_logical_cpus")
    assigned=[cpu for worker in specific.get("sam_worker_allocations",[]) for cpu in worker["cpu_affinity"]]
    if type(maximum) is not int or not 0<maximum<=72 or len(assigned)>maximum:
        raise ValueError("Registered CPU allocation exceeds explicit policy budget")
    if not registered_sam_allocation(resources,specific,allowed):
        raise ValueError("Resources require an exact registered SAM worker allocation")
    if (set(resources)!={"worker_id","physical_gpu","gpu_uuid","cpu_affinity","allocator_gib","fresh_free_mib","torch_threads"}
            or type(resources["torch_threads"]) is not int or not 1<=resources["torch_threads"]<=len(resources["cpu_affinity"])):
        raise ValueError("Explicit native worker resources/threads required")
    return policy


def _official(ref,verifier):
    manifest=verifier.bound_json(ref)
    source=checked_path(ref["path"]).parent
    if manifest.get("config_file")!="config.json":raise ValueError("Official snapshot has no native template config")
    cfg=read_json(verifier.verify(source/"config.json",manifest["files_sha256"]["config.json"]))
    if validate_source_snapshot(cfg)!=manifest:raise ValueError("Official source snapshot differs")
    native=cfg["native"]
    official=checked_path(native["source_root"])
    if (official!=source/_relative(manifest["official_source_relative"])
            or native["source_commit"]!=manifest["official_commit"]):
        raise ValueError("Official native source root/commit differs")
    receipt=read_json(native["source_receipt"]);asset=read_json(native["asset_receipt"])
    if receipt.get("verified_commit")!=native["source_commit"]:
        raise ValueError("Official acquisition receipt commit mismatch")
    for relative,digest in receipt["inspected_source_sha256"].items():
        verifier.verify(official/_relative(relative),digest)
    weight=verifier.verify(native["weight"],asset["sha256"])
    if asset.get("status")!="ASSET_SHA_AND_WEIGHTS_ONLY_CPU_INSPECTION_PASSED" or weight.stat().st_size!=asset["bytes"]:
        raise ValueError("Verified native asset receipt differs")
    selected={}
    prefix=Path(manifest["official_source_relative"])
    for relative,digest in manifest["files_sha256"].items():
        path=_relative(relative)
        if path.is_relative_to(prefix):
            selected[relative]=verifier.verify(source/path,digest)
    if set(_inventory(official))!={str(Path(x).relative_to(prefix)) for x in selected}:
        raise ValueError("Official snapshot contains unlisted files")
    for key in ("source_receipt","asset_receipt"):
        path=checked_path(native[key])
        if not path.is_relative_to(source):raise ValueError("Native receipt is outside sealed snapshot")
        relative=str(path.relative_to(source))
        selected[relative]=verifier.verify(path,manifest["files_sha256"][relative])
    return manifest,cfg,selected


def _clip(inputs_ref,clip_id,b0_source_ref,verifier):
    inputs=verifier.bound_json(inputs_ref)
    if inputs.get("status")!="CLIP_INPUTS_PREPARED_NOT_TRACKING_RUN":
        raise ValueError("Authenticated prepared clip input receipt required")
    matches=[c for c in inputs["clips"] if c["clip_id"]==clip_id]
    if len(matches)!=1:raise ValueError("Exactly one registered prepared clip required")
    entry=matches[0];doc=verifier.bound_json(entry["region"]);b0=verifier.bound_json(entry["B0"])
    protocol=verifier.bound_json(inputs["protocol"])
    registered=[c for c in protocol["clips"] if c["clip_id"]==clip_id]
    if len(registered)!=1:raise ValueError("Clip absent from registered evaluation protocol")
    c=registered[0];frames=doc["frames"]
    if (doc["source_manifest"]!=protocol["source_manifest"]
            or c.get("reuse_whole_video_track_state") is not False or c.get("core_only_scoring") is not True
            or [f["local_index"] for f in frames]!=list(range(len(frames)))
            or doc["region_id"]!=clip_id or doc.get("session_policy")!="independent_clip_cold_start"
            or doc.get("inherited_session") is not None or doc["core"]!=c["core"]
            or [f["global_frame_id"] for f in frames]!=c["input_frame_ids"]
            or b0["region"]!=entry["region"] or b0["frames"]!=len(frames)
            or b0.get("status")!="B0_REGION_EVIDENCE_COMPLETE"):
        raise ValueError("Prepared clip region/B0 domain differs")
    original=verifier.bound_json(doc["source_manifest"])
    indexed={f["frame_id"]:f for f in original["frames"]}
    for frame in frames:
        native=indexed[frame["global_frame_id"]]
        if (frame["image"],frame["sha256"],frame["pts"])!=(native["image"],native["image_sha256"],native["pts"]):
            raise ValueError("Prepared frame differs from original extraction")
    core=[{"frame_id":f["global_frame_id"],"pts":f["pts"],"RGB_sha256":f["sha256"]}
          for f in frames if f["global_frame_id"] in c["evaluation_frame_ids"]]
    if core!=c["core_frame_bindings"]:raise ValueError("Prepared core binding differs")
    origin=verifier.bound_json(b0_source_ref)
    if _digest({k:v for k,v in origin.items() if k!="identity"})!=origin["identity"] or origin["identity"]!=b0["source_identity"]:
        raise ValueError("Original frozen B0 producer source manifest differs")
    rows=[json.loads(line) for line in verifier.verify(**{"path":b0["pools"]["path"],"sha":b0["pools"]["sha256"]}).read_text().splitlines()]
    if [r["frame"] for r in rows]!=frames:raise ValueError("B0 pool frame identity differs")
    return entry,doc,b0,rows


def _copy_files(files,destination,expected_hashes):
    for relative,path in files.items():
        relative=_relative(relative);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        # Copy even readonly files; never link mutable worktree inodes.
        before=expected_hashes[str(relative)]
        if _sha(path)!=before:raise ValueError("Tested source changed before snapshot copy")
        with checked_path(path).open("rb") as source,target.open("xb") as sink:
            shutil.copyfileobj(source,sink)
        if _sha(target)!=before or _sha(path)!=before:raise ValueError("Source changed during snapshot copy")
        target.chmod(0o444)


def _finish(output,cfg,official_manifest,provenance):
    source=output/"source"
    _json_new(source/"config.json",cfg)
    files={name:_sha(path) for name,path in _inventory(source).items()}
    manifest={"schema":"action4.sam.immutable_source.v1","official_commit":official_manifest["official_commit"],
              "official_source_relative":official_manifest["official_source_relative"],"config_file":"config.json",
              "files_sha256":files,"preparation_provenance":provenance}
    manifest["identity"]=_digest(manifest)
    _json_new(source/"SNAPSHOT_MANIFEST.json",manifest)
    validate_source_snapshot(cfg)
    command=[str(ROOT/".venv/sam31/bin/python"),"-B",str(source/LAUNCHER),
             "--config",str(source/"config.json"),"--mode","sam"]
    receipt={"schema":"action4.causal_clip_prepared.v1","status":"PREPARED_NOT_EXECUTED",
        "config_ref":_ref(source/"config.json"),"source_manifest_ref":_ref(source/"SNAPSHOT_MANIFEST.json"),
        "command":command,"authority_ref":cfg["authority_ref"],"resources":cfg["resources"],
        "launch_authority_checked":False,"requires_live_authority_and_headroom_at_launch":True,
        "native_inference_started":False,"B0_inference_started":False}
    _json_new(output/"PREPARED.json",receipt)
    return receipt


def _destination(output):
    path=checked_path(output)
    if path.exists():raise FileExistsError("Preserve existing preparation/output")
    return path


def _relocate_native(cfg,old_source,new_source):
    for key in ("source_root","source_receipt","asset_receipt"):
        path=checked_path(cfg["native"][key])
        if not path.is_relative_to(old_source):raise ValueError("Native path outside parent source")
        cfg["native"][key]=str((new_source/path.relative_to(old_source)).relative_to(ROOT))


def prepare_discovery(*,clip_inputs_ref,clip_id,b0_source_manifest_ref,official_snapshot_ref,
                      project_code_ref,authority_ref,resources,seed_policy,output):
    """Prepare box/text seed-only prefix discovery, never rerun B0 or dispatch."""
    verifier=ReceiptVerifier();target=_destination(output)
    _authority(authority_ref,resources,verifier)
    entry,doc,b0,rows=_clip(clip_inputs_ref,clip_id,b0_source_manifest_ref,verifier)
    project,files=_project(project_code_ref,verifier)
    official,template,native_files=_official(official_snapshot_ref,verifier)
    cfg=copy.deepcopy(template);old=checked_path(template["source_snapshot"])
    cfg={k:cfg[k] for k in ("schema","native","binding")}
    cfg.update(source_snapshot=str((target/"source").relative_to(ROOT)),output=str((target/"run").relative_to(ROOT)),
        authority_ref=copy.deepcopy(authority_ref),resources=copy.deepcopy(resources),region=copy.deepcopy(entry["region"]),
        b0=copy.deepcopy(b0["b0"]),b0_evidence=copy.deepcopy(entry["B0"]),b0_source_manifest=copy.deepcopy(b0_source_manifest_ref),
        sequence={"frame_count":len(doc["frames"]),"direction":"both","output_prob_thresh":.5,"evict_cached_frame_outputs":False},
        seed_policy=copy.deepcopy(seed_policy),seed_mode="box_text",instance_output_bridge=False,
        discovery_seed_only=True,discovery_decode_prefix=True)
    region=RegionSpec(doc["region_id"],tuple(f["global_frame_id"] for f in doc["frames"]),
        tuple(f["pts"] for f in doc["frames"]),tuple(doc["time_base"]),entry["region"]["sha256"],
        doc["pixel_space_id"],doc["width"],doc["height"])
    select_seeds(rows,region,{"model_sha256":b0["b0"]["sha256"],"source_sha256":b0["source_identity"]},seed_policy)
    _relocate_native(cfg,old,target/"source")
    if set(files)&set(native_files):raise ValueError("Project and official snapshot entries collide")
    (target/"source").mkdir(parents=True);(target/"run").mkdir()
    _copy_files({**files,**native_files},target/"source",
                {**project["files_sha256"],**{k:official["files_sha256"][k] for k in native_files}})
    return _finish(target,cfg,official,{"kind":"current_tested_project_with_verified_official_source",
        "project_code_ref":project_code_ref,"official_snapshot_ref":official_snapshot_ref,"clip_inputs_ref":clip_inputs_ref})


def prepare_forward(*,discovery_config_ref,producer_exit_ref,evidence_refs,authority_ref,resources,output):
    """Inherit only completed discovery's sealed code; never consult worktree."""
    verifier=ReceiptVerifier();target=_destination(output)
    _authority(authority_ref,resources,verifier)
    cfg=copy.deepcopy(verifier.bound_json(discovery_config_ref))
    parent=validate_source_snapshot(cfg)
    if parent.get("preparation_provenance",{}).get("kind")!="current_tested_project_with_verified_official_source":
        raise ValueError("Forward requires this tested causal-capable discovery preparation lineage")
    old=checked_path(cfg["source_snapshot"]);out=checked_path(cfg["output"])
    exit_record=_validated_exit(verifier,producer_exit_ref,discovery_config_ref)
    def bound_path(path):
        key=str(checked_path(path).relative_to(ROOT))
        if key not in evidence_refs:raise ValueError("Missing bound discovery metadata: "+key)
        return verifier.bound_json({"path":key,"sha256":evidence_refs[key]})
    _terminal_state(cfg,exit_record,bound_path)
    done=bound_path(out/"NATIVE_COMPLETE.json");seeds=bound_path(out/"SEEDS.json")["seeds"]
    if (done.get("status")!="DISCOVERY_SEEDS_ONLY_COMPLETE_NOT_TRACKING" or not 1<=len(seeds)<=2
            or [t["track_id"] for t in done["tracks"]]!=[s["track_id"] for s in seeds]):
        raise ValueError("Forward requires actual completed one/two prefix discovery seeds")
    b0=verifier.bound_json(cfg["b0_evidence"]);doc=verifier.bound_json(cfg["region"])
    origin=verifier.bound_json(cfg["b0_source_manifest"])
    if (b0["b0"]!=cfg["b0"] or b0["region"]!=cfg["region"] or b0["frames"]!=len(doc["frames"])
            or b0["source_identity"]!=origin["identity"]
            or _digest({k:v for k,v in origin.items() if k!="identity"})!=origin["identity"]):
        raise ValueError("Discovery frozen B0/model/frame provenance differs")
    pool=verifier.verify(b0["pools"]["path"],b0["pools"]["sha256"])
    rows=[json.loads(line) for line in pool.read_text().splitlines()]
    if [r["frame"] for r in rows]!=doc["frames"]:raise ValueError("Discovery B0 frame domain differs")
    region=RegionSpec(doc["region_id"],tuple(f["global_frame_id"] for f in doc["frames"]),
        tuple(f["pts"] for f in doc["frames"]),tuple(doc["time_base"]),cfg["region"]["sha256"],
        doc["pixel_space_id"],doc["width"],doc["height"])
    selection=select_seeds(rows,region,{"model_sha256":cfg["b0"]["sha256"],
        "source_sha256":b0["source_identity"]},cfg["seed_policy"])
    if bound_path(out/"SEEDS.json")!=selection:raise ValueError("Actual discovery seed selection differs from frozen B0")
    cfg.update(seed_mode="instance_points",instance_output_bridge=True,discovery_seed_only=False,
               discovery_decode_prefix=False,temporal_mode="causal_forward_B0_assist",
               authority_ref=copy.deepcopy(authority_ref),resources=copy.deepcopy(resources))
    cfg["sequence"]["direction"]="forward";cfg.pop("discovery_track_ids",None)
    cfg["instance_seed_sources"]={}
    for seed in seeds:
        tid=seed["track_id"];folder=out/tid
        paths={"seeds":out/"SEEDS.json","native_load":out/"NATIVE_LOAD.json",
               "binding":folder/"SEED_BINDING.json","raw_receipt":folder/"RAW_RECEIPT.json",
               "discovery_resource":out/"DISCOVERY_RESOURCE.json","discovery_complete":out/"NATIVE_COMPLETE.json",
               "discovery_close":folder/"CLOSE.json"}
        refs={"config":copy.deepcopy(discovery_config_ref)}
        for key,path in paths.items():
            bound_path(path);refs[key]={"path":str(path.relative_to(ROOT)),"sha256":evidence_refs[str(path.relative_to(ROOT))]}
        cfg["instance_seed_sources"][tid]=refs
        load_instance_seed_points(cfg,seed,seeds)
    files={}
    for name,digest in parent["files_sha256"].items():
        if name=="config.json":continue
        files[name]=verifier.verify(old/_relative(name),digest)
    if LAUNCHER not in files:raise ValueError("Discovery snapshot lacks frozen sequence launcher")
    cfg["source_snapshot"]=str((target/"source").relative_to(ROOT));cfg["output"]=str((target/"run").relative_to(ROOT))
    _relocate_native(cfg,old,target/"source")
    (target/"source").mkdir(parents=True);(target/"run").mkdir()
    _copy_files(files,target/"source",parent["files_sha256"])
    return _finish(target,cfg,parent,{"kind":"completed_prefix_discovery_immutable_inheritance",
        "discovery_config_ref":discovery_config_ref,"producer_exit_ref":producer_exit_ref})
