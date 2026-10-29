"""Fail-closed baseline video selection and resource admission; no model imports."""
import hashlib
import json
import os
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R, contained_path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_selection(cfg, root=R):
    if cfg.get("temporal_modules") or cfg.get("sam_enabled") or cfg.get("cotracker_enabled"):
        raise ValueError("Framewise baseline renderer has no continuous tracking/session adapter")
    for role in ("a0", "b0"):
        path = contained_path(root, cfg[role+"_bundle"], exists=True)
        if (root/"models/default").resolve() in path.resolve().parents:
            raise ValueError("Video requires immutable selected bundles, not mutable defaults")
        expected = cfg.get(role+"_sha256")
        if not expected or digest(path) != expected:
            raise ValueError("Frozen selected bundle SHA mismatch: " + role)
    if cfg.get("video_mode") == "research_trial":
        return validate_trial_selection(cfg, root)
    if "role_seals" not in cfg:
        raise ValueError("Formal video requires independently sealed selected roles")
    for role in ("A0", "B0"):
        seal = json.loads(contained_path(root, cfg["role_seals"][role], exists=True).read_text())
        if seal.get("schema") != "action4.sealed_independent_role.v2" or seal.get("strict_load") is not True or seal.get("role") != role:
            raise ValueError("Unverified role seal")
        if seal["sha256"] != cfg[role.lower()+"_sha256"] or contained_path(root, seal["bundle"], exists=True) != contained_path(root, cfg[role.lower()+"_bundle"], exists=True):
            raise ValueError("Selection differs from role seal")
    return {k:cfg[k] for k in ("a0_sha256", "b0_sha256")}


def validate_trial_selection(cfg, root=R):
    """Explicit user-requested trial, never a completed/deployment role seal."""
    ref = cfg.get("trial_receipt")
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:
        raise ValueError("Research trial requires an explicit frozen receipt")
    p=contained_path(root,ref["path"],exists=True)
    if p.stat().st_mode & 0o222 or digest(p)!=ref["sha256"]:
        raise ValueError("Trial receipt changed or not frozen")
    trial=json.loads(p.read_text())
    if (trial.get("schema")!="action4.research_video_trial.v1"
            or trial.get("status")!="FROZEN_FOR_RESEARCH_VIDEO"
            or trial.get("authorization_kind") not in ("user_requested_video_trial_during_training","user_requested_completed_training_candidate_comparison")
            or trial.get("deployment_release") is not False
            or trial.get("default_models_modified") is not False):
        raise ValueError("Unsupported trial authority or deployment claim")
    if trial["authorization_kind"]=="user_requested_completed_training_candidate_comparison":
        def frozen_receipt(name):
            ref=trial.get(name)
            if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:
                raise ValueError("Completed comparison requires sealed source bindings")
            file=contained_path(root,ref["path"],exists=True)
            if file.stat().st_mode & 0o222 or digest(file)!=ref["sha256"]:
                raise ValueError("Completed comparison seal changed")
            return json.loads(file.read_text())
        lineage=frozen_receipt("completed_lineage_seal")
        b0=frozen_receipt("fixed_B0_seal")
        if (lineage.get("schema")!="action4.sealed_lineage_training_candidate.v1"
                or lineage.get("role")!="A0" or lineage.get("strict_load") is not True
                or lineage.get("completion",{}).get("formal_training_complete") is not True
                or lineage["completion"].get("epochs",0)<80):
            raise ValueError("Completed eighty-epoch lineage required")
        a=trial["models"]["A0"]
        if not any((x["record"]["step"],x["record"]["config_id"],x["retained_checkpoint"]["sha256"])
                   ==(a["selected_step"],a["config_id"],a["sha256"]) for x in lineage["retained_top_candidates"]):
            raise ValueError("A0 is outside the sealed completed candidate list")
        b=trial["models"]["B0"]
        if (b0.get("schema")!="action4.sealed_independent_role.v2" or b0.get("role")!="B0"
                or b0.get("strict_load") is not True
                or (b0.get("selected_step"),b0.get("config_id"),b0.get("sha256"))!=(b["selected_step"],b["config_id"],b["sha256"])):
            raise ValueError("Fixed B0 seal mismatch")
    for role in ("A0","B0"):
        item=trial["models"][role];key=role.lower()
        if (item["path"]!=cfg[key+"_bundle"] or item["sha256"]!=cfg[key+"_sha256"]
                or item.get("strict_load") is not True or type(item.get("selected_step")) is not int):
            raise ValueError("Trial model selection mismatch")
        if contained_path(root,item["path"],exists=True).stat().st_mode & 0o222:
            raise ValueError("Trial model must be frozen")
    for key in ("intrinsics_context","demo_reference"):
        ref_input=trial["video_inputs"][key]
        if cfg[key]!=ref_input["path"] or digest(contained_path(root,cfg[key],exists=True))!=ref_input["sha256"]:
            raise ValueError("Trial video input changed")
    fixed_inputs={"frames_manifest":"data/action4_real/cache_v001/frames_manifest.json",
                  "canonical_asset":"data/DJI_Action4_construct/r3_v001/asset_manifest.json"}
    for key in ("frames_manifest","canonical_asset"):
        ref_input=trial["video_inputs"][key]
        if contained_path(root,ref_input["path"],exists=True)!=contained_path(root,fixed_inputs[key],exists=True):
            raise ValueError("Trial does not bind actual renderer frame/CAD input")
        if digest(contained_path(root,ref_input["path"],exists=True))!=ref_input["sha256"]:
            raise ValueError("Trial frame/CAD contract changed")
    if trial["b0_scoring_policy"]!=cfg.get("b0_scoring_policy"):
        raise ValueError("Trial detector policy changed")
    if (cfg.get("trial_gpu_uuid")!=trial["resources"]["gpu_uuid"]
            or cfg.get("trial_worker_cpu_groups")!=trial["resources"]["worker_cpu_groups"]):
        raise ValueError("Trial resource allocation changed")
    budget=cfg.get("worker_memory_budget_mib")
    if type(budget) is not int or not 4096<=budget<=14336 or budget!=trial["resources"]["worker_memory_budget_mib"]:
        raise ValueError("Explicit bounded trial allocator required")
    return {"a0_sha256":cfg["a0_sha256"],"b0_sha256":cfg["b0_sha256"],
            "trial_receipt_sha256":ref["sha256"],"evaluation_kind":"research_video_trial"}


def resource_plan(devices, worker_cpus, reserved_cpu, authority):
    allowed = set(authority["authorized_evaluation_gpu"]) - set(authority["forbidden_gpu_ids"])
    if not devices or any(g not in allowed or g not in (4,5,6,7) for g in devices):
        raise ValueError("Only currently authorized physical GPU4-7 may run video")
    ids = list(worker_cpus)
    pool = set(authority["cpu_policy"]["allowed_logical_cpu_ids"])
    if len(ids) != 6*len(devices) or len(set(ids)) != len(ids) or not set(ids) <= pool:
        raise ValueError("Reserve six distinct tail-pool logical CPUs per video worker")
    if reserved_cpu < 0 or len(ids)+reserved_cpu+2 > authority["cpu_budget_limit"]:
        raise ValueError("Total video/control/reserved CPU budget exceeded")
    return [ids[i*6:(i+1)*6] for i in range(len(devices))]


def active_authority():
    pointer=json.loads((R/"configs/execution/action4_active.json").read_text())["active_ref"]
    path=contained_path(R,pointer["path"],exists=True)
    if digest(path)!=pointer["sha256"]: raise ValueError("Active resource authority SHA mismatch")
    return json.loads(path.read_text())


def validate_worker_environment():
    authority=active_authority()
    cpus=set(os.sched_getaffinity(0))
    if len(cpus)>6 or not cpus <= set(authority["cpu_policy"]["allowed_logical_cpu_ids"]):
        raise ValueError("Worker must inherit bounded tail-pool CPU affinity")
    mapping=os.environ.get("CUDA_VISIBLE_DEVICES", "")
    import subprocess
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True).splitlines()
    permitted={row.split(",")[1].strip() for row in rows if int(row.split(",")[0]) in set(authority["authorized_evaluation_gpu"])-set(authority["forbidden_gpu_ids"])}
    if mapping not in permitted: raise ValueError("Explicit single approved physical GPU UUID mapping required")


def load_prediction_cache(manifest_path, frozen_identity, root=R, selection_sha256=None):
    """Only explicitly sealed same-selection caches; legacy JSONL alone is insufficient."""
    manifest=json.loads(contained_path(root,manifest_path,exists=True).read_text())
    if manifest.get("schema")!="action4.baseline_video_prediction_cache.v1" or manifest.get("temporal_mode")!="independent_frames_baseline":
        raise ValueError("Unsupported cache protocol; tracking caches need session-aware consumer")
    if manifest.get("selected_bundles")!=frozen_identity:
        raise ValueError("Prediction cache selected bundle identities differ")
    if selection_sha256 is None or manifest.get("selection_sha256")!=selection_sha256:
        raise ValueError("Cached complete selection SHA mismatch")
    if not manifest.get("producer_sources"):
        raise ValueError("Cache requires frozen producer source identities")
    for entry in manifest["producer_sources"]:
        if digest(contained_path(root,entry["path"],exists=True))!=entry["sha256"]:
            raise ValueError("Frozen cache producer source changed")
    rows_path=contained_path(root,manifest["predictions"],exists=True)
    if digest(rows_path)!=manifest["predictions_sha256"]:raise ValueError("Prediction cache bytes changed")
    rows={}
    for line in rows_path.read_text().splitlines():
        row=json.loads(line);fid=row["frame_id"]
        if fid in rows or row["prediction"]["identity"]!=manifest["prediction_identity"]:
            raise ValueError("Duplicate frame or inconsistent cache model identity")
        if any(row["prediction"]["identity"].get(k)!=v for k,v in frozen_identity.items()):
            raise ValueError("Cached prediction identity omits frozen bundle SHA")
        if not row.get("rgb_sha256"):raise ValueError("Cached RGB identity required")
        rows[fid]=row
    return rows,manifest["prediction_identity"]


def cached_prediction(rows,frame,source_time_base,image_path):
    row=rows.get(frame["frame_id"])
    if row is None or row["pts"]!=frame["pts"] or row["time_base"]!=source_time_base or row["rgb_sha256"]!=digest(image_path):
        raise ValueError("Cached frame timing/RGB mismatch or missing prediction")
    return row["prediction"]


def validate_cpu_affinity(cpus, authority, maximum=6):
    cpus=set(cpus)
    if not cpus or len(cpus)>maximum or not cpus <= set(authority["cpu_policy"]["allowed_logical_cpu_ids"]):
        raise ValueError("Bounded tail-pool CPU affinity required")
    return sorted(cpus)


def gpu_admission(device, workers, free_mib, used_mib, utilization, policy):
    """Explicit allocation policy, not an invented universal utilization cutoff."""
    if policy.get("schema")!="action4.video_resource_admission.v1" or device not in policy.get("allocated_gpu_ids",[]):
        raise ValueError("Explicit allocated video GPU required")
    if not policy.get("allocation_receipt") or type(policy.get("allow_shared_gpu")) is not bool:
        raise ValueError("Allocation receipt and explicit sharing decision required")
    margin=policy.get("uncommitted_margin_mib");budget=policy.get("worker_memory_budget_mib")
    if type(margin) is not int or margin<8192 or type(budget) is not int or budget<4096:
        raise ValueError("Explicit memory budgets must retain at least8GiB margin and4GiB per worker")
    if free_mib < margin+workers*budget:raise ValueError("GPU memory headroom insufficient")
    if not policy["allow_shared_gpu"] and used_mib>policy.get("idle_memory_tolerance_mib",512):
        raise ValueError("Initial idle-only allocation sees existing GPU memory use")
    cap=policy.get("max_instant_utilization_percent")
    if cap is not None:
        if not isinstance(cap,(int,float)) or isinstance(cap,bool) or not 0<=cap<=100:
            raise ValueError("Invalid optional initial utilization limit")
        if utilization>cap:raise ValueError("Configured initial utilization policy not met")
    return {"device":device,"free_mib":free_mib,"used_mib":used_mib,"utilization_percent":utilization,
            "sharing_explicitly_allowed":policy["allow_shared_gpu"],"optional_initial_utilization_limit":cap}
