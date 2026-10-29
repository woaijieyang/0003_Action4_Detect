"""Resumable prediction-only SAM3.1 single-image evaluation.

Importing this module never loads SAM/torch or reads GT. Native execution is
explicit; CPU fixtures exercise the identical job/receipt state machine.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
import numpy as np
from data.action4_paths import PROJECT_ROOT
from model.action4_sam_session_contracts import RegionSpec, SeedBinding, SessionLedger
from model.action4_sam_response_adapter import (
    ResponseEnvelope, MissingOutput, ResponseSchemaError,
    validate_raw_response, parse_target_response, MaskRegistrationReceipt,
)

SCHEMA = "action4.sam.single_image_native.v1"
SCHEMA_V2 = "action4.sam.single_image_native.v2"
SCHEMA_VARIANT = "action4.sam.single_image_rgb_variant.v1"
COMPLETED_ROLE = "frozen_completed_B0_post_soft_nms"



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def contained(value):
    p = Path(value); p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.resolve().is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Project path escape")
    return p


def file_sha(path):
    with contained(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path): return json.loads(contained(path).read_text())


def atomic_json(path, value):
    p = contained(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, p)


def _reject_gt(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().startswith("gt"):
                if not ((key == "GT_instance_id" and item is None) or (key == "GT_used_for_prompt_selection" and item is False)):
                    raise ValueError("GT field forbidden in native prompt inputs")
            _reject_gt(item)
    elif isinstance(value, list):
        for item in value: _reject_gt(item)


def _bound_json(binding):
    if not isinstance(binding,dict) or set(binding)!={"path","sha256"} or file_sha(binding["path"])!=binding["sha256"]:
        raise ValueError("Completed B0 source binding changed")
    return read_json(binding["path"])


def validate_completed_source(config):
    source=config["source_contract"]
    if set(source)!={"sealed_role","run_spec","b0_bundle","selected_step","config_id","predictions"}:
        raise ValueError("Unknown completed B0 source contract")
    seal=_bound_json(source["sealed_role"]);run=_bound_json(source["run_spec"])
    bundle=source["b0_bundle"]
    if not isinstance(bundle,dict) or set(bundle)!={"path","sha256"} or file_sha(bundle["path"])!=bundle["sha256"]:
        raise ValueError("Completed B0 bundle bytes changed")
    if (seal.get("schema")!="action4.sealed_independent_role.v2" or seal.get("role")!="B0"
            or seal.get("strict_load") is not True or seal.get("selected_step")!=source["selected_step"]
            or seal.get("config_id")!=source["config_id"]
            or (seal.get("bundle"),seal.get("sha256"))!=(bundle["path"],bundle["sha256"])
            or (seal.get("run_spec"),seal.get("run_spec_sha256"))!=(source["run_spec"]["path"],source["run_spec"]["sha256"])):
        raise ValueError("Completed B0 release seal identity mismatch")
    completion=seal.get("completion",{})
    if (completion.get("status")!="COMPLETION_METADATA_READY" or completion.get("role")!="B0"
            or completion.get("best_step")!=source["selected_step"] or completion.get("evaluations")!=653
            or completion.get("config_id")!=source["config_id"]):
        raise ValueError("Full completed B0 release evidence required")
    if (run.get("schema")!="action4.immutable_run.v1" or run.get("mutable_worktree_not_used_by_children") is not True
            or run.get("code_root")!=seal.get("actual_code_root")):
        raise ValueError("Completed B0 immutable run mismatch")
    for binding in (source["sealed_role"],source["run_spec"],bundle):
        if contained(binding["path"]).stat().st_mode & 0o222:
            raise ValueError("Completed B0 source must be sealed read-only")
    predictions=_bound_json(source["predictions"])
    _reject_gt(predictions)
    if not isinstance(predictions,list):raise ValueError("Explicit image prediction list required")
    saved={}
    for row in predictions:
        sid=row["sample_id"]
        if not isinstance(sid,str) or not sid or sid in saved or not isinstance(row["instances"],list):
            raise ValueError("Invalid/duplicate prediction image domain")
        saved[sid]=row["instances"]
    return saved


def _completed_candidate(row,candidate,config,saved):
    binding=candidate["identity_binding"]
    if set(binding)!={"sample_id","grid_index","raw_saved_ordinal","prediction_file_sha256"}:
        raise ValueError("Explicit completed B0 candidate binding required")
    grid=binding["grid_index"];ordinal=binding["raw_saved_ordinal"]
    if (type(grid) is not int or grid<0 or type(ordinal) is not int or ordinal<0
            or ordinal>=len(saved[row["image_id"]])
            or binding["prediction_file_sha256"]!=config["source_contract"]["predictions"]["sha256"]):
        raise ValueError("Completed B0 prediction/grid binding mismatch")
    raw=saved[row["image_id"]][ordinal]
    if (candidate.get("raw_saved_prediction")!=raw or raw.get("grid_index")!=grid
            or candidate["candidate_id"]!=f'b0:{row["image_id"]}:grid:{grid}'
            or candidate["bbox_full_native_xyxy"]!=raw["bbox_full_xyxy"]
            or candidate["score"]!=raw["score"]):
        raise ValueError("Completed B0 saved prediction was changed")
    if type(candidate["published"]) is not bool or candidate["published"]!=(candidate["score"]>=.3):
        raise ValueError("Completed B0 published threshold mismatch")
    if not np.isfinite(candidate["score"]):raise ValueError("Nonfinite prediction score")
    full=np.asarray(candidate["bbox_full_native_xyxy"],float)
    if full.shape!=(4,) or not np.isfinite(full).all():raise ValueError("Invalid completed B0 box")
    h,w=row["native_shape_hw"];clipped=np.clip(full,[0,0,0,0],[w,h,w,h])
    expected_eligible=bool(candidate["published"] and (clipped[2:]>clipped[:2]).all())
    if candidate["eligible_for_SAM_diagnostic_prompt"] is not expected_eligible:
        raise ValueError("Completed B0 eligibility must follow frozen score and valid clipped support")



def load_manifest(config):
    if config.get("schema")==SCHEMA_VARIANT:
        from eval.action4_sam_variant_contract import load_native_manifest
        return load_native_manifest(config)
    path = contained(config["manifest"])
    if file_sha(path) != config["manifest_sha256"]:
        raise ValueError("Prompt manifest identity changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != config["expected_images"]: raise ValueError("Incomplete image domain")
    completed=config["schema"]==SCHEMA_V2
    saved=validate_completed_source(config) if completed else None
    if completed and set(saved)!={r["image_id"] for r in rows}:raise ValueError("Completed B0 image domain changed")
    images, candidates, jobs = set(), set(), []
    total = 0
    for ordinal, row in enumerate(rows):
        _reject_gt(row)
        if row["schema"] != ("action4.sam.prediction_prompt_image.v2" if completed else "action4.sam.prediction_prompt_image.v1") or row["image_id"] in images:
            raise ValueError("Duplicate or invalid image identity")
        images.add(row["image_id"])
        if completed:
            if (row.get("split")!=config["split"] or row.get("rgb_variant")!="native_original"
                    or row.get("source_receipt")!=config["source_contract"]
                    or len(row["candidates"])!=len(saved[row["image_id"]])):
                raise ValueError("Completed B0 split/native RGB/source/domain mismatch")
            ordinals=[c["identity_binding"]["raw_saved_ordinal"] for c in row["candidates"]]
            if sorted(ordinals)!=list(range(len(saved[row["image_id"]]))):
                raise ValueError("Completed B0 candidate ordinal coverage mismatch")
        contained(row["image_path"])
        h, w = row["native_shape_hw"]
        if type(h) is not int or type(w) is not int or min(h, w) < 2:
            raise ValueError("Explicit native dimensions required")
        if row["pixel_convention"] != "integer_centers": raise ValueError("Unknown pixel convention")
        if len(row["candidates"]) != row["raw_saved_candidate_count"]:
            raise ValueError("Candidate count mismatch")
        eligible = 0
        for candidate in row["candidates"]:
            cid = candidate["candidate_id"]
            if not isinstance(cid, str) or not cid or cid in candidates:
                raise ValueError("Global stable candidate identity collision")
            candidates.add(cid); total += 1
            if candidate["identity_binding"]["sample_id"] != row["image_id"]:
                raise ValueError("Candidate source image mismatch")
            if completed:_completed_candidate(row,candidate,config,saved)
            if candidate["source"] != ("frozen_completed_B0_saved_prediction" if completed else "frozen_B0_parent_step0_saved_prediction"):
                raise ValueError("Native baseline requires its declared B0 source")
            flag = candidate["eligible_for_SAM_diagnostic_prompt"]
            if type(flag) is not bool: raise ValueError("Explicit prompt eligibility required")
            if not flag: continue
            eligible += 1
            if candidate["published"] is not True or candidate["score"] < .3:
                raise ValueError("Unpublished candidate cannot become a prompt")
            full = np.asarray(candidate["bbox_full_native_xyxy"], float)
            clipped = np.asarray(candidate["bbox_sam_native_xyxy"], float)
            norm = np.asarray(candidate["bbox_sam_normalized_xywh"], float)
            if full.shape != (4,) or clipped.shape != (4,) or norm.shape != (4,) or not np.isfinite(np.r_[full, clipped, norm]).all():
                raise ValueError("Invalid predicted box")
            expected_clip = np.clip(full, [0, 0, 0, 0], [w, h, w, h])
            expected_norm = np.r_[clipped[:2] / [w, h], (clipped[2:] - clipped[:2]) / [w, h]]
            if not np.allclose(clipped, expected_clip, atol=1e-9, rtol=0) or not np.allclose(norm, expected_norm, atol=1e-9, rtol=0) or not (clipped[2:] > clipped[:2]).all():
                raise ValueError("Native/clipped/normalized prompt mismatch")
            jobs.append({"image": row, "candidate": candidate, "image_ordinal": ordinal})
        if eligible != row["SAM_prompt_count"]: raise ValueError("Declared prompt count mismatch")
    if total != config["expected_saved_candidates"] or len(jobs) != config["expected_prompts"]:
        raise ValueError("Frozen candidate/prompt domain changed")
    return rows, jobs


def validate_config(config):
    if config.get("schema")==SCHEMA_VARIANT:
        from eval.action4_sam_variant_contract import validate_native_config
        return validate_native_config(config)
    expected = {"schema", "manifest", "manifest_sha256", "input_contract", "output",
                "expected_images", "expected_saved_candidates", "expected_prompts",
                "native", "resources", "binding", "code_files", "upstream_role",
                "retry_attempt_limit", "K_contract", "authority_ref", "source_snapshot"}
    completed=config.get("schema")==SCHEMA_V2
    if completed:expected |= {"split","rgb_variant","source_contract"}
    if set(config) != expected or config["schema"] not in (SCHEMA,SCHEMA_V2):
        raise ValueError("Unknown or incomplete native config")
    if config["upstream_role"] != (COMPLETED_ROLE if completed else "frozen_B0_parent_step0_post_soft_nms"):
        raise ValueError("B0 baseline cannot be relabeled YOLO-only or another source mode")
    if completed:
        if config["split"] not in ("train","eval") or config["rgb_variant"]!="native_original":
            raise ValueError("Completed B0 cache requires explicit train/eval original native RGB")
        if (type(config["expected_images"]) is not int or config["expected_images"]<1
                or any(type(config[k]) is not int or config[k]<0 for k in ("expected_saved_candidates","expected_prompts"))
                or config["expected_prompts"]>config["expected_saved_candidates"]):
            raise ValueError("Invalid explicit completed B0 domain counts")
    if config["retry_attempt_limit"] != 2: raise ValueError("Bounded explicit retry policy required")
    n = config["native"]
    if set(n) != {"source_root", "source_commit", "source_receipt", "weight", "asset_receipt", "text", "builder_kwargs"}:
        raise ValueError("Unknown native policy fields")
    if n["text"] != "camera" or n["source_commit"] != "7f473cfd983d5eb43040ff608f66bff0d7bdcb53":
        raise ValueError("Native baseline policy changed")
    if n["builder_kwargs"] != {"compile": False, "warm_up": False, "use_fa3": False,
                                "use_rope_real": False, "async_loading_frames": False,
                                "max_num_objects": 16, "multiplex_count": 16}:
        raise ValueError("Unregistered builder policy")
    if config["binding"] != {"bbox_iou_role": "diagnostic_only", "unique_margin": .15,
                             "unique_margin_basis": "target_coverage_minus_best_other_candidate_coverage",
                             "mask_in_box_min": .8, "multi_target_support_min": .7,
                             "multiple_native_policy": "require_single_eligible_mask"}:
        raise ValueError("Unregistered prediction-only binding policy")
    r = config["resources"]
    if set(r) != {"physical_gpu", "gpu_uuid", "fresh_free_mib", "allocator_gib", "cpu_affinity", "torch_threads", "processes", "concurrent_sessions"}:
        raise ValueError("Unknown resource policy; old GPU1 configuration is revoked")
    if completed:
        if (type(r["physical_gpu"]) is not int or r["physical_gpu"] not in (4,5,6,7)
                or not isinstance(r["gpu_uuid"],str) or not r["gpu_uuid"].startswith("GPU-")
                or r["fresh_free_mib"]<24576 or not 0<r["allocator_gib"]<=14):
            raise ValueError("Completed B0 native resources require GPU4-7/headroom/14GiB cap")
        if (r["processes"]!=1 or r["concurrent_sessions"]!=1 or not 1<=r["torch_threads"]<=2
                or not isinstance(r["cpu_affinity"],list) or not 1<=len(r["cpu_affinity"])<=2
                or len(set(r["cpu_affinity"]))!=len(r["cpu_affinity"])
                or any(type(c) is not int or c<0 for c in r["cpu_affinity"])):
            raise ValueError("Completed B0 requires one worker/session and at most two explicit CPUs")
    else:
        if r["physical_gpu"] != 7 or r["gpu_uuid"] != "GPU-a68477c3-1c2a-d557-58f5-405d80e5383e" or r["fresh_free_mib"] < 24576 or not 0 < r["allocator_gib"] <= 14:
            raise ValueError("GPU0-3 forbidden; only registered GPU7/headroom/14GiB allocator allowed")
        if r["processes"] != 1 or r["concurrent_sessions"] != 1 or r["cpu_affinity"] != [47, 48] or not 1 <= r["torch_threads"] <= 2:
            raise ValueError("Single-worker/session and registered CPU47-48 required")
    if config["K_contract"] != {"K": None, "status": "not_supplied_not_consumed_by_native_SAM"}:
        raise ValueError("Do not invent calibration for synthetic images")
    for key in ("manifest", "input_contract", "output"): contained(config[key])
    validate_live_authority(config)
    return config


def registered_sam_allocation(resources,specific,allowed_cpu):
    registrations=specific.get("sam_worker_allocations")
    if registrations is None:return False
    if not isinstance(registrations,list) or not registrations:raise ValueError("Explicit nonempty SAM registrations required")
    ids=[x["worker_id"] for x in registrations]
    cpus=[c for x in registrations for c in x["cpu_affinity"]]
    if len(ids)!=len(set(ids)) or len(cpus)!=len(set(cpus)) or not set(cpus)<=set(allowed_cpu):
        raise ValueError("Duplicate SAM worker/CPU registration")
    if any(type(c) is not int for c in cpus) or set(cpus)&set(specific.get("sam_excluded_cpus",[])):
        raise ValueError("SAM CPU allocation conflicts with reserved work")
    if len(registrations)>specific.get("sam_max_workers",0):raise ValueError("SAM worker count exceeds admission")
    for gpu in (4,5,6,7):
        if sum(x["physical_gpu"]==gpu for x in registrations)>specific.get("sam_per_gpu_worker_limits",{}).get(str(gpu),0):
            raise ValueError("SAM per-GPU worker count exceeds admission")
    quotas=specific.get("sam_allowed_allocator_gib",[8])
    if (not isinstance(quotas,list) or not quotas or len(set(quotas))!=len(quotas)
            or any(type(v) is not int or v not in (8,32) for v in quotas)):
        raise ValueError("Explicit tested SAM allocator tiers required")
    budget=specific.get("sam_per_gpu_allocator_budget_gib",32)
    if type(budget) is not int or not 8<=budget<=32:
        raise ValueError("SAM aggregate GPU budget must preserve headroom")
    for x in registrations:
        if type(x["physical_gpu"]) is not int or x["physical_gpu"] not in (4,5,6,7) or type(x["allocator_gib"]) is not int or x["allocator_gib"] not in quotas or type(x["fresh_free_mib"]) is not int or x["fresh_free_mib"]<(x["allocator_gib"]+8)*1024:
            raise ValueError("Registered SAM quota/headroom invalid")
    for gpu in (4,5,6,7):
        if sum(x["allocator_gib"] for x in registrations if x["physical_gpu"]==gpu)>budget:
            raise ValueError("Registered SAM aggregate GPU memory budget exceeded")
    found=[x for x in registrations if x["worker_id"]==resources.get("worker_id")]
    if len(found)!=1:raise ValueError("Registered SAM worker ID required")
    expected=found[0]
    for key in ("physical_gpu","gpu_uuid","cpu_affinity","allocator_gib","fresh_free_mib"):
        if resources.get(key)!=expected[key]:raise ValueError("SAM allocation differs from registered "+key)
    if expected["physical_gpu"] not in (4,5,6,7) or expected["allocator_gib"] not in quotas or expected["fresh_free_mib"]<(expected["allocator_gib"]+8)*1024:
        raise ValueError("Registered SAM quota/headroom invalid")
    return True

def acquire_sam_worker_lease(worker_id,metadata):
    import re,fcntl
    if not isinstance(worker_id,str) or not re.fullmatch(r"[a-z0-9_-]{1,64}",worker_id):
        raise ValueError("Safe registered worker ID required")
    root=contained("runtime/action4_resource_leases");root.mkdir(exist_ok=True)
    handle=(root/(worker_id+".lock")).open("a+")
    try:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handle.seek(0);handle.truncate();json.dump(metadata,handle);handle.flush()
        return handle
    except BaseException:
        handle.close();raise

def validate_live_authority(config):
    """Read LIVE authority from project root, never a copied authority snapshot."""
    from model.training.action4_resource_policy import validate_device_policy
    r = config["resources"]
    admitted = validate_device_policy({"training_gpus": [], "eval_gpus": [r["physical_gpu"]]}, PROJECT_ROOT)
    if admitted["authority"] != config["authority_ref"]:
        raise ValueError("Latest active authority differs from the pinned resource admission")
    policy = read_json(admitted["authority"]["path"])
    if not {0, 1, 2, 3}.issubset(set(admitted["forbidden_gpu_ids"])):
        raise ValueError("Latest authority must retain GPU0-3 prohibition")
    allowed_cpu = set(policy["cpu_policy"]["allowed_logical_cpu_ids"])
    specific = policy["zeroshot_allowed_resources"]
    if registered_sam_allocation(r,specific,allowed_cpu):return admitted
    if not set(r["cpu_affinity"]) <= allowed_cpu or r["cpu_affinity"] != specific["sam_cpu_affinity"]:
        raise ValueError("CPU affinity differs from live authority")
    if r["physical_gpu"] != specific["sam_candidate_gpu"] or r["allocator_gib"] > specific["sam_candidate_allocator_gib"] or r["fresh_free_mib"] < 1024 * specific["sam_required_free_before_start_gib"] or specific["sam_max_workers"] < 1:
        raise ValueError("SAM resources exceed live authority")
    return admitted


def validate_source_snapshot(config, *, require_execution=False):
    source = contained(config["source_snapshot"]).resolve(strict=True)
    manifest = read_json(source / "SNAPSHOT_MANIFEST.json")
    if manifest["schema"] != "action4.sam.immutable_source.v1":
        raise ValueError("Unknown immutable source manifest")
    payload = {k: v for k, v in manifest.items() if k != "identity"}
    if digest(payload) != manifest["identity"]:
        raise ValueError("Snapshot manifest identity mismatch")
    for relative, expected in manifest["files_sha256"].items():
        p = source / relative
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(source) or file_sha(p) != expected:
            raise ValueError("Snapshot source missing/changed/escaped: " + relative)
        if p.stat().st_mode & 0o222:
            raise ValueError("Snapshot source is writable: " + relative)
    if read_json(source / manifest["config_file"]) != config:
        raise ValueError("Effective config differs from sealed snapshot")
    if require_execution:
        import sys
        if not Path(__file__).resolve().is_relative_to(source / "src"):
            raise ValueError("Native SAM must execute project code from immutable snapshot")
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in {"model", "data", "eval"} and getattr(module, "__file__", None):
                actual = Path(module.__file__).resolve()
                if not actual.is_relative_to(source / "src"):
                    raise ValueError("Live project module leaked into snapshot execution: " + name)
            if name.split(".")[0] == "sam3" and getattr(module, "__file__", None):
                if not Path(module.__file__).resolve().is_relative_to(source / "upstream/sam3/sam3"):
                    raise ValueError("Live official SAM module leaked into execution")
    return manifest


def prepare_run(config):
    validate_config(config)
    snapshot = validate_source_snapshot(config)
    rows, jobs = load_manifest(config)
    out = contained(config["output"]); out.mkdir(parents=True, exist_ok=True)
    code = {p: file_sha(p) for p in config["code_files"]}
    lock = {"schema": config["schema"], "config": config, "config_identity": digest(config),
            "code_sha256": code, "manifest_sha256": config["manifest_sha256"],
            "input_contract_sha256": file_sha(config["input_contract"]),
            "source_snapshot_identity": snapshot["identity"]}
    lock["identity"] = digest(lock)
    lockpath = out / "RUN_LOCK.json"
    if lockpath.exists():
        if read_json(lockpath) != lock: raise ValueError("Frozen config/source/input lock changed")
    else:
        atomic_json(lockpath, lock)
    for job in jobs:
        job["task_key"] = digest({"lock": lock["identity"], "image_id": job["image"]["image_id"],
                                  "image_sha256": job["image"]["image_sha256"],
                                  "candidate": job["candidate"], "text": config["native"]["text"]})
    first = []
    seen = set()
    for job in jobs:
        if job["image"]["image_id"] not in seen:
            first.append(job["task_key"]); seen.add(job["image"]["image_id"])
            if len(first) == 2: break
    cohort = {"schema": config["schema"], "status": "PREPARED_NOT_EXECUTED", "lock_identity": lock["identity"],
              "images": len(rows), "saved_candidates": config["expected_saved_candidates"],
              "eligible_prompt_jobs": len(jobs), "unprompted_candidates": config["expected_saved_candidates"] - len(jobs),
              "upstream_role": config["upstream_role"], "first_batch_task_keys": first,
              "first_batch_policy": "first eligible candidate from each of first two distinct images",
              "all_image_ids": [r["image_id"] for r in rows], "K": config["K_contract"],
              "GT_used_for_prompt_selection": False, "quality_scoring": "not_measured",
              "source_snapshot_identity": snapshot["identity"],
              "execution_module": str(Path(__file__).resolve()),
              "live_authority": validate_live_authority(config)}
    atomic_json(out / "PREPARATION.json", cohort)
    return rows, jobs, lock, first


def _box_iou(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(1e-12, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter))


def bind_native_output(validated, image, candidate, policy):
    """Bind by actual mask coverage and unique support, never full-box IoU/probability.

    Full predicted-box IoU remains a diagnostic. A small correctly located
    visible fragment is not rejected for occupying little of a full-body box.
    Multiple independently eligible native masks remain ambiguous.
    """
    h, w = image["native_shape_hw"]
    target = np.asarray(candidate["bbox_sam_native_xyxy"], float)
    other = [(c["candidate_id"], np.asarray(c["bbox_sam_native_xyxy"], float))
             for c in image["candidates"] if c["eligible_for_SAM_diagnostic_prompt"]]
    rows = []
    for i, obj in enumerate(validated.object_ids):
        x, y, bw, bh = validated.boxes_xywh01[i]
        box = np.array([x*w, y*h, (x+bw)*w, (y+bh)*h], float)
        mask = validated.masks[i]; yy, xx = np.nonzero(mask); area = len(xx)
        overlaps = []
        for cid, b in other:
            inter = int(((xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])).sum())
            overlaps.append({"candidate_id": cid, "mask_in_box": inter / area if area else 0.,
                             "support_overlap_min": inter / min(area, np.prod(b[2:]-b[:2])) if area else 0.})
        local = next(r for r in overlaps if r["candidate_id"] == candidate["candidate_id"])
        competing = [r["mask_in_box"] for r in overlaps if r["candidate_id"] != candidate["candidate_id"]]
        best_other = max(competing, default=0.)
        coverage_gap = local["mask_in_box"] - best_other
        spans = [r["candidate_id"] for r in overlaps
                 if r["mask_in_box"] >= policy["mask_in_box_min"] or r["support_overlap_min"] >= policy["multi_target_support_min"]]
        rejection = []
        removed = obj in validated.removed_object_ids
        if removed: rejection.append("official_removed_object_score_sentinel")
        if not area: rejection.append("empty_native_mask")
        if local["mask_in_box"] < policy["mask_in_box_min"]: rejection.append("insufficient_mask_in_target")
        if coverage_gap < policy["unique_margin"]: rejection.append("insufficient_coverage_margin_over_other_candidate")
        if len(spans) > 1: rejection.append("one_mask_strongly_supports_multiple_candidates")
        eligible = bool(area and not rejection)
        rows.append({"obj_id": obj, "raw_index": i, "probability": None if removed else float(validated.probabilities[i]),
                     "native_score_raw": float(validated.probabilities[i]), "native_object_removed": removed,
                     "probability_is_quality_or_class_confirmation": False,
                     "bbox_native_xyxy": box.tolist(), "mask_pixels": area,
                     "bbox_iou_prompt": _box_iou(box, target), "bbox_iou_role": "diagnostic_only",
                     "support_relations": overlaps, "target_mask_coverage": local["mask_in_box"],
                     "best_other_candidate_mask_coverage": best_other, "coverage_margin": coverage_gap,
                     "spans_candidate_ids": spans, "eligible_binding": eligible,
                     "binding_rejection_reasons": rejection})
    eligible = [r for r in rows if r["eligible_binding"]]
    if len(eligible) == 1:
        return eligible[0]["obj_id"], "bound_unique_prediction_support", rows
    if len(eligible) > 1:
        for row in eligible:
            row["binding_rejection_reasons"].append("multiple_native_masks_compete_for_same_prompt")
            row["eligible_binding"] = False
        return None, "ambiguous_identity", rows
    ambiguous = any(len(r["spans_candidate_ids"]) > 1 or
                    (r["target_mask_coverage"] >= policy["mask_in_box_min"] and r["coverage_margin"] < policy["unique_margin"])
                    for r in rows)
    return None, "ambiguous_identity" if ambiguous else "unbound_no_unique_local_support", rows


def save_raw(raw, directory):
    arrays = {}
    def convert(value, prefix):
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "O": raise ResponseSchemaError("Object arrays forbidden")
            key = prefix.replace("/", "__")
            arrays[key] = value
            return {"array_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {str(k): convert(v, prefix + "/" + str(k)) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [convert(v, prefix + "/" + str(i)) for i, v in enumerate(value)]
        if value is None or isinstance(value, (str, int, float, bool)): return value
        raise ResponseSchemaError("Unsupported native raw metadata type")
    header = convert(raw, "raw")
    rawpath = contained(directory) / "raw.npz"
    np.savez_compressed(rawpath, **arrays)
    atomic_json(contained(directory) / "raw.json", header)
    return {"arrays": str(rawpath.relative_to(PROJECT_ROOT)), "sha256": file_sha(rawpath),
            "header": str((contained(directory) / "raw.json").relative_to(PROJECT_ROOT)),
            "header_sha256": file_sha(contained(directory) / "raw.json")}


class NativeJobFailure(RuntimeError): pass


class ResourceWait(RuntimeError):
    """Controlled resource wait before a new session, never a target outcome."""


def require_free_margin(free_mib, minimum_mib):
    if type(free_mib) is not int or type(minimum_mib) is not int or min(free_mib, minimum_mib) < 0:
        raise ValueError("Integer GPU free/margin measurements required")
    if free_mib < minimum_mib:
        raise ResourceWait("Global free GPU memory below live uncommitted margin")


def completed_result(jobdir, job, lock):
    marker = contained(jobdir) / "COMPLETED.json"
    if not marker.exists(): return None
    entry = read_json(marker)
    if entry["task_key"] != job["task_key"] or entry["lock_identity"] != lock["identity"]:
        raise ValueError("Completion identity mismatch")
    for path, sha in entry["files_sha256"].items():
        p = contained(path)
        if not p.resolve().is_relative_to(contained(jobdir).resolve()) or not p.is_file() or file_sha(p) != sha:
            raise ValueError("Completed raw/result missing or changed; cannot skip")
    result = read_json(entry["result_path"])
    if result["task_key"] != job["task_key"] or result["image_id"] != job["image"]["image_id"] or result["candidate_id"] != job["candidate"]["candidate_id"] or not result["close_success"] or not result["response_valid"]:
        raise ValueError("Invalid completion content")
    return result


def variant_job_fields(row,config):
    variant=config.get("rgb_variant")
    variant_id=row.get("variant_id")
    pixels=row.get("variant_pixels_sha256")
    if (variant not in ("light","blur") or row.get("rgb_variant")!=variant
            or config.get("split")!="train" or row.get("split")!="train"
            or row.get("schema")!="action4.sam.prediction_prompt_image.variant.v1"
            or not isinstance(variant_id,str) or not variant_id.startswith("native_"+variant+":")
            or not isinstance(pixels,str) or len(pixels)!=64 or pixels=="0"*64
            or any(c not in "0123456789abcdef" for c in pixels)):
        raise ValueError("Exact derived variant identity required; no original alias")
    original=row.get("original_source")
    if not isinstance(original,dict) or set(original)!={"path","sha256"}:
        raise ValueError("Separate original RGB provenance required")
    return {"variant_id":variant_id,"variant_pixels_sha256":pixels,"original_source":dict(original)}

def execute_job(job, config, lock, backend, attempt_dir):
    row, candidate = job["image"], job["candidate"]
    variant_fields=variant_job_fields(row,config) if config["schema"]==SCHEMA_VARIANT else {}
    region = RegionSpec("single:" + job["task_key"], (0,), (0,), (1, 1),
                        config["manifest_sha256"], row["pixel_space_id"],
                        row["native_shape_hw"][1], row["native_shape_hw"][0])
    seed = SeedBinding(candidate["candidate_id"], region.identity, candidate["candidate_id"],
                       0, 0, tuple(candidate["bbox_sam_normalized_xywh"]),
                       (row["image_sha256"], row["source_receipt"]["B0_bundle"]["sha256"] if config["schema"]==SCHEMA_VARIANT else row["source_receipt"]["b0_bundle"]["sha256"] if config["schema"]==SCHEMA_V2 else row["source_receipt"]["parent_sha256"]), "frozen_b0")
    ledger = SessionLedger(per_worker_limit=1, global_limit=1)
    record = ledger.plan(region, seed, worker_id="single-native-worker",
                         device_identity=backend.kind, dependency_identity=lock["identity"])
    sid = None
    result = {"schema": config["schema"], "task_key": job["task_key"], "lock_identity": lock["identity"],
              "image_id": row["image_id"], "image_sha256": row["image_sha256"],
              "candidate_id": candidate["candidate_id"], "upstream_role": config["upstream_role"],
              "source_receipt": row["source_receipt"], "pixel_space_id": row["pixel_space_id"],
              "native_shape_hw": row["native_shape_hw"], "K": config["K_contract"],
              "bbox_full_native_xyxy": candidate["bbox_full_native_xyxy"],
              "bbox_prompt_native_xyxy": candidate["bbox_sam_native_xyxy"],
              "bbox_prompt_normalized_xywh": candidate["bbox_sam_normalized_xywh"],
              "single_image_index": 0, "capture_PTS": None, "ledger_time_note": "0/1 transport index only, not capture timing",
              "generation": 0, "backend_kind": backend.kind, "response_valid": False,
              "available": False, "use_as_training_negative": False,
              "class_identity_confirmation": False, "mask_quality_measured": False,
              "close_success": False, "status": "RUNNING", "started_unix": time.time()}
    if config["schema"] in (SCHEMA_V2,SCHEMA_VARIANT):
        result.update(split=config["split"],rgb_variant=config["rgb_variant"],
                      prediction_id=candidate["candidate_id"],identity_binding=candidate["identity_binding"])
    result.update(variant_fields)
    atomic_json(attempt_dir / "RESULT.json", result)
    failure = None
    try:
        ledger.begin_create(record.record_id)
        started = backend.call({"type": "start_session", "resource_path": str(contained(row["image_path"])),
                                "offload_video_to_cpu": True})
        if not isinstance(started, dict) or not isinstance(started.get("session_id"), str) or not started["session_id"]:
            raise ResponseSchemaError("Native start_session acknowledgement invalid")
        sid = started["session_id"]; ledger.created(record.record_id, sid)
        request = ledger.seed_request(record.record_id); request["text"] = config["native"]["text"]
        pending = ledger.get(record.record_id)
        result.update(session_id=sid, prompt_request=request, seed_fingerprint=seed.fingerprint)
        raw = backend.call(request)
        result["raw"] = save_raw(raw, attempt_dir)
        envelope = ResponseEnvelope(sid, region.identity, 0, 0, "seed", job["task_key"], 0)
        output = validate_raw_response(raw, record=pending, region=region, envelope=envelope)
        if isinstance(output, MissingOutput):
            result.update(status="missing_response", reason=output.reason)
            raise NativeJobFailure(output.reason)
        result["response_valid"] = True
        bound, binding, candidates = bind_native_output(output, row, candidate, config["binding"])
        for native_candidate in candidates:
            native_candidate["mask_storage"] = {"npz": result["raw"]["arrays"],
                "array_key": "raw__outputs__out_binary_masks", "row_index": native_candidate["raw_index"],
                "raw_sha256": result["raw"]["sha256"]}
        result.update(binding_status=binding, native_candidates=candidates, raw_object_count=len(output.object_ids), bound_obj_id=bound,
                      binding_reasons=sorted({reason for row in candidates for reason in row["binding_rejection_reasons"]}) if bound is None else [])
        if bound is None:
            ledger.bind_seed(record.record_id, [])
            result.update(status="ambiguous_identity" if binding == "ambiguous_identity" else "target_visibility_unknown",
                          reason="native_public_output_does_not_establish_unique_prompt_target")
        else:
            ledger.bind_seed(record.record_id, [bound])
            def register_mask(request):
                storage = result["raw"]["arrays"] + "#raw__outputs__out_binary_masks[row=" + str(output.object_ids.index(bound)) + ";obj_id=" + str(bound) + "]"
                result["mask_registration"] = {"mask_identity": request.mask_identity,
                    "pixel_sha256": request.pixel_sha256, "binding_identity": request.binding_identity,
                    "binding": json.loads(request.binding_json), "storage_reference": storage}
                return MaskRegistrationReceipt(request.mask_identity, request.pixel_sha256, request.binding_identity, storage)
            parsed = parse_target_response(raw, record=replace(pending, obj_id=bound), region=region,
                                           envelope=envelope, register_mask=register_mask)
            result.update(status=parsed.status, reason=parsed.reason, available=parsed.available,
                          public_mask_nonempty=parsed.public_mask_nonempty,
                          public_output_contains_bound_id=parsed.public_output_contains_bound_id)
    except Exception as exc:
        failure = exc
        if result["status"] == "RUNNING":
            result["status"] = "session_lost" if isinstance(exc, (KeyError, ConnectionError)) else "inference_error"
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        current = ledger.get(record.record_id)
        if current.state != "FAILED":
            ledger.fail(record.record_id, status="missing_response" if result["status"] == "missing_response" else "inference_error",
                        failure_kind="schema" if isinstance(exc, ValueError) else "oom" if "out of memory" in str(exc).lower() else "transient")
    finally:
        if sid is not None:
            try:
                closed = backend.call({"type": "close_session", "session_id": sid})
                success = isinstance(closed, dict) and closed.get("is_success") is True
                ledger.close(record.record_id, released=success)
                result.update(close_response=closed, close_success=success)
                if not success: raise NativeJobFailure("Native close did not acknowledge release")
                backend.assert_idle()
            except Exception as exc:
                result["close_error"] = str(exc)
                failure = failure or exc
                result["close_success"] = False
        try:
            result["resources"] = backend.stats()
        except Exception as stats_error:
            result["resource_stats_error"] = str(stats_error)
            failure = failure or stats_error
        if failure is not None or not result["close_success"]:
            result["native_response_status"] = result["status"]
            result["native_mask_candidate_available"] = result["available"]
            result["status"] = "missing_response" if result["status"] == "missing_response" else "session_lost" if "close_error" in result else "inference_error"
            result["available"] = False
            result["execution_status"] = "FAILED"
        else:
            result["execution_status"] = "COMPLETED"
        result.update(ledger_audit=ledger.audit_log, active_sessions_after_close=ledger.active_count,
                      ended_unix=time.time())
        atomic_json(attempt_dir / "RESULT.json", result)
    if failure is not None or not result["close_success"]:
        atomic_json(attempt_dir / "FAILED.json", {"task_key": job["task_key"], "status": result["status"], "error": str(failure)})
        raise NativeJobFailure("Fail-stop candidate " + candidate["candidate_id"] + ": " + str(failure))
    return result


def run_jobs(config, backend, *, first_batch=False, retry_failed=False):
    rows, jobs, lock, first = prepare_run(config)
    out = contained(config["output"])
    import fcntl
    with (out / "worker.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        verified_images = set()
        progress = {"status": "RUNNING", "lock_identity": lock["identity"], "backend_kind": backend.kind,
                    "expected_images": len(rows), "expected_prompt_jobs": len(jobs), "completed_jobs": 0,
                    "available_masks": 0, "first_batch_only": first_batch, "upstream_role": config["upstream_role"]}
        try:
            # Inventory all verified completions first so a resource wait after
            # a non-prefix first batch never reports completed caches as lost.
            for job in jobs:
                done = completed_result(out / "jobs" / job["task_key"], job, lock)
                if done is not None: results[job["task_key"]] = done
            for job in jobs:
                jobdir = out / "jobs" / job["task_key"]
                if job["task_key"] in results: continue
                if first_batch and job["task_key"] not in first: continue
                previous = sorted(jobdir.glob("attempt_*")) if jobdir.exists() else []
                if previous and not retry_failed:
                    raise NativeJobFailure("Previous incomplete/failed attempt requires explicit retry-failed acknowledgement")
                if len(previous) >= config["retry_attempt_limit"]:
                    raise NativeJobFailure("Explicit retry attempt limit exhausted")
                row = job["image"]
                if row["image_id"] not in verified_images:
                    from PIL import Image
                    imagepath = contained(row["image_path"])
                    if file_sha(imagepath) != row["image_sha256"]: raise ValueError("Source PNG bytes changed")
                    with Image.open(imagepath) as image:
                        if tuple(reversed(image.size)) != tuple(row["native_shape_hw"]): raise ValueError("Native PNG shape mismatch")
                    verified_images.add(row["image_id"])
                # Admission happens before creating an attempt or issuing a
                # session. A resource wait therefore consumes no retry budget.
                if hasattr(backend, "admit_session"): backend.admit_session()
                attempt = jobdir / ("attempt_" + str(len(previous) + 1).zfill(2))
                attempt.mkdir(parents=True, exist_ok=False)
                result = execute_job(job, config, lock, backend, attempt)
                paths = [attempt / "RESULT.json", attempt / "raw.npz", attempt / "raw.json"]
                marker = {"task_key": job["task_key"], "lock_identity": lock["identity"],
                          "result_path": str((attempt / "RESULT.json").relative_to(PROJECT_ROOT)),
                          "files_sha256": {str(p.relative_to(PROJECT_ROOT)): file_sha(p) for p in paths}}
                atomic_json(jobdir / "COMPLETED.json", marker)
                results[job["task_key"]] = result
                progress.update(completed_jobs=len(results), available_masks=sum(r["available"] for r in results.values()),
                                last_candidate=result["candidate_id"], last_status=result["status"])
                atomic_json(out / "PROGRESS.json", progress)
            progress.update(status="NATIVE_TRANSPORT_COMPLETE_MASK_QUALITY_NOT_SCORED" if len(results) == len(jobs) else "FIRST_BATCH_COMPLETE_RESUME_FULL_DOMAIN",
                            completed_jobs=len(results), available_masks=sum(r["available"] for r in results.values()),
                            statuses={status: sum(r["status"] == status for r in results.values()) for status in sorted({r["status"] for r in results.values()})},
                            image_domain_count=len(rows), upstream_saved_candidates=config["expected_saved_candidates"],
                            upstream_unprompted_candidates=config["expected_saved_candidates"]-len(jobs),
                            quality_scoring="not_measured", class_identity_accuracy="not_measured")
            atomic_json(out / "PROGRESS.json", progress)
            return progress
        except ResourceWait as exc:
            progress.update(status="RESOURCE_WAIT_RELEASED_RESTART_SAFE", reason=str(exc),
                            completed_jobs=len(results), available_masks=sum(r["available"] for r in results.values()),
                            pending_jobs=len(jobs)-len(results), new_session_not_started=True,
                            completed_caches_preserved=True, target_outcome=None)
            atomic_json(out / "PROGRESS.json", progress)
            return progress
        except Exception as exc:
            progress.update(status="FAIL_STOP", error_type=type(exc).__name__, error=str(exc),
                            completed_jobs=len(results), available_masks=sum(r["available"] for r in results.values()))
            atomic_json(out / "PROGRESS.json", progress)
            raise


class NativeBackend:
    """Lazy load under the run's exclusive worker lock; no GPU work in __init__."""
    kind = "sam31_official_native"
    def __init__(self, config):
        self.config = config
        self.predictor = None
        self.torch = None
        self.load_receipt = None
        self._session_admitted = False

    def _load(self):
        import sys, subprocess
        # These guards precede torch import, CUDA initialization and source loading.
        authority = validate_live_authority(self.config)
        snapshot = validate_source_snapshot(self.config, require_execution=True)
        resources = self.config["resources"]
        if resources.get("worker_id") is not None:
            self._resource_lease=acquire_sam_worker_lease(resources["worker_id"],
                {"pid":os.getpid(),"output":self.config["output"],"source_identity":snapshot["identity"],"allocation":resources})
        uuid = resources["gpu_uuid"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
            raise RuntimeError("Only the approved configured GPU UUID may be visible; GPU0-3 forbidden")
        os.sched_setaffinity(0, set(resources["cpu_affinity"]))
        out = contained(self.config["output"])
        for key in ("TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
                    "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME",
                    "TMPDIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
            p = out / "runtime_cache" / key; p.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(p)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        n = self.config["native"]; source = contained(n["source_root"])
        head = snapshot["official_commit"]
        if head != n["source_commit"] or source.resolve() != contained(self.config["source_snapshot"]).resolve() / snapshot["official_source_relative"]:
            raise RuntimeError("Official source must be the sealed pinned snapshot copy")
        src = read_json(n["source_receipt"])
        for relative, sha in src["inspected_source_sha256"].items():
            if file_sha(source / relative) != sha: raise RuntimeError("Official source file changed")
        asset = read_json(n["asset_receipt"]); weight = contained(n["weight"])
        if asset["sha256"] != "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6" or weight.stat().st_size != asset["bytes"]:
            raise RuntimeError("Previously validated SAM3.1 asset identity/size mismatch")
        # Query only this assigned GPU, never a different card.
        raw = subprocess.check_output(["nvidia-smi", "--id=" + uuid,
            "--query-gpu=uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
        fields = [x.strip() for x in raw.split(",")]
        if len(fields) != 4 or not fields[3].isdigit() or not 0 <= int(fields[3]) <= 100:
            raise RuntimeError("Fresh utilization measurement unavailable; no admission")
        physical = subprocess.check_output(["nvidia-smi", "--id=" + str(resources["physical_gpu"]),
            "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        if physical != uuid:
            raise RuntimeError("Physical GPU index/UUID changed")
        if fields[0] != uuid:
            raise RuntimeError("Fresh GPU measurement UUID mismatch")
        if int(fields[1]) < resources["fresh_free_mib"]:
            raise ResourceWait("Fresh assigned GPU memory below configured admission; do not displace other jobs")
        import torch
        self.torch = torch
        if np.__version__ != "1.26.4": raise RuntimeError("Use isolated SAM31 NumPy1.26.4 environment")
        torch.set_num_threads(resources["torch_threads"]); torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(resources["allocator_gib"] * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, str(source))
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        import sam3.model_builder as builder
        def no_download(*args, **kwargs): raise RuntimeError("Implicit native downloads forbidden")
        builder.download_ckpt_from_hf = no_download; builder.hf_hub_download = no_download
        calls = []; original = torch.nn.Module.load_state_dict
        def audited(module, state, *args, **kwargs):
            loaded = original(module, state, *args, **kwargs)
            final = type(module).__name__ == "Sam3MultiplexTrackingWithInteractivity"
            calls.append({"class": type(module).__name__, "final_model": final,
                          "missing": list(loaded.missing_keys), "unexpected": list(loaded.unexpected_keys)})
            if final and (loaded.missing_keys or loaded.unexpected_keys):
                raise RuntimeError("Final complete SAM model has missing/unexpected keys")
            return loaded
        torch.nn.Module.load_state_dict = audited
        receipt = {"status": "LOADING", "GPU": raw, "source_commit": head,
                   "fresh_gpu_utilization_percent": int(fields[3]),
                   "utilization_admission": "Measured before construction; root must admit coexistence under live authority. No other process is stopped or resized.",
                   "live_authority": authority, "source_snapshot_identity": snapshot["identity"],
                   "source_snapshot": self.config["source_snapshot"],
                   "weight_sha256_from_prior_validation": asset["sha256"], "weight_rehashed": False,
                   "CPU_affinity": sorted(os.sched_getaffinity(0)), "numpy": np.__version__,
                   "torch": torch.__version__, "precision": "inference_mode/autocast BF16; official complex RoPE",
                   "builder_kwargs": n["builder_kwargs"], "load_calls": calls}
        atomic_json(out / "NATIVE_LOAD.json", receipt)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                self.predictor = build_sam3_multiplex_video_predictor(
                    checkpoint_path=str(weight),
                    bpe_path=str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), **n["builder_kwargs"])
            finals = [r for r in calls if r["final_model"]]
            if len(finals) != 1 or finals[0]["missing"] or finals[0]["unexpected"]:
                raise RuntimeError("Final full-model load not verified")
            from model.action4_sam_native_compat import install_pinned_init_state_compat
            receipt["compat"] = install_pinned_init_state_compat(self.predictor, official_root=source,
                expected_source_sha256=src["inspected_source_sha256"]["sam3/model/sam3_multiplex_tracking.py"],
                snapshot_manifest=contained(self.config["source_snapshot"]) / "SNAPSHOT_MANIFEST.json")
            validate_source_snapshot(self.config, require_execution=True)
            receipt.update(status="LOADED_ZERO_FINAL_MISSING_UNEXPECTED", seconds=time.monotonic()-started)
        except Exception as exc:
            receipt.update(status="FAILED", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            torch.nn.Module.load_state_dict = original
            atomic_json(out / "NATIVE_LOAD.json", receipt)
        self.load_receipt = receipt

    def admit_session(self):
        import subprocess
        self._session_admitted = False
        admitted = validate_live_authority(self.config)
        policy = read_json(admitted["authority"]["path"])
        minimum = int(policy["zeroshot_allowed_resources"]["minimum_uncommitted_memory_margin_gib"] * 1024)
        if minimum < 8192: raise ValueError("Live uncommitted margin cannot be below8GiB")
        measurement = None
        try:
            if self.predictor is None: self._load()
            raw = subprocess.check_output(["nvidia-smi", "--id=" + self.config["resources"]["gpu_uuid"],
                "--query-gpu=uuid,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip()
            fields = [x.strip() for x in raw.split(",")]
            if len(fields) != 4 or fields[0] != self.config["resources"]["gpu_uuid"] or not all(x.isdigit() for x in fields[1:]) or not 0 <= int(fields[3]) <= 100:
                raise RuntimeError("Per-session global resource measurement invalid")
            measurement = {"uuid":fields[0],"free_mib":int(fields[1]),"used_mib":int(fields[2]),
                           "utilization_percent":int(fields[3]),"unix":time.time(),"minimum_free_mib":minimum}
            require_free_margin(measurement["free_mib"], minimum)
        except ResourceWait as exc:
            self.dispose()
            atomic_json(contained(self.config["output"]) / "RESOURCE_WAIT.json",
                {"status":"RESOURCE_WAIT_MODEL_RELEASED","reason":str(exc),"measurement":measurement,
                 "minimum_free_mib":minimum,"new_session_started":False,"model_released":True,
                 "other_processes_modified":False,"target_outcome":None})
            raise
        self._session_admitted = True
        self.last_session_admission = measurement

    def call(self, request):
        if request.get("type") == "start_session":
            validate_live_authority(self.config)
            if not self._session_admitted: self.admit_session()
            self._session_admitted = False
        if self.predictor is None:
            raise RuntimeError("Native backend is not loaded")
        if request.get("type") not in ("start_session", "add_prompt", "close_session"):
            raise RuntimeError("Single-image baseline forbids propagation/corrections")
        torch = self.torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.predictor.handle_request(request)
        torch.cuda.synchronize()
        return output

    def assert_idle(self):
        if self.predictor is not None and self.predictor._all_inference_states:
            raise RuntimeError("Native close left active session state")

    def stats(self):
        if self.torch is None: return {"loaded": False}
        return {"loaded": self.predictor is not None,
                "allocated_mib": self.torch.cuda.memory_allocated()/2**20,
                "reserved_mib": self.torch.cuda.memory_reserved()/2**20,
                "peak_allocated_mib": self.torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_mib": self.torch.cuda.max_memory_reserved()/2**20,
                "last_session_admission": getattr(self,"last_session_admission",None)}

    def dispose(self):
        import gc
        self.predictor = None; self._session_admitted = False; gc.collect()
        if self.torch is not None: self.torch.cuda.empty_cache()
        lease=getattr(self,"_resource_lease",None)
        if lease is not None:lease.close();self._resource_lease=None


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--first-batch", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if args.mode == "prepare":
        rows, jobs, lock, first = prepare_run(config)
        print(json.dumps({"status": "PREPARED_NO_GPU", "images": len(rows), "prompts": len(jobs),
                          "lock_identity": lock["identity"], "first_batch_task_keys": first}))
        return
    if not args.execute_native: raise RuntimeError("Native run requires explicit execute-native flag")
    prepare_run(config)
    backend = NativeBackend(config)
    try:
        progress = run_jobs(config, backend, first_batch=args.first_batch, retry_failed=args.retry_failed)
        print(json.dumps(progress))
        if progress["status"] == "RESOURCE_WAIT_RELEASED_RESTART_SAFE":
            raise SystemExit(75)
    finally:
        backend.dispose()
