"""Independent CPU scoring for frozen SAM single-image outputs; GT never selects masks.

No SAM/torch import, no inference calls, no changes to native binding or raw files.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from data.action4_paths import PROJECT_ROOT

SCHEMA = "action4.sam.independent_mask_metrics.v1"
POLICY = {"upstream_bbox_match_iou": .5, "coverage_min": .5, "target_fraction_min": .8,
          "camera_purity_min": .95, "other_target_leakage_max": .05, "merge_coverage_min": .5}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path(value):
    p = Path(value); p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.resolve().is_relative_to(PROJECT_ROOT): raise ValueError("Scorer path escaped project")
    return p


def sha_file(value):
    with path(value).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(value): return json.loads(path(value).read_text())


def ratio(n, d): return float(n / d) if d else None


def bbox_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a, b]).all() or not np.all(a[2:] > a[:2]) or not np.all(b[2:] > b[:2]):
        raise ValueError("Invalid upstream/GT full box")
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(intersection / (np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))


def match_pairs(target_ids, prediction_ids, edges, quality):
    """Scorer-only max-cardinality then quality matching, stable ID ordering."""
    if not edges.size: return []
    bonus = min(edges.shape) + 1
    rows, cols = linear_sum_assignment(np.where(edges, bonus + quality, -bonus), maximize=True)
    return [(target_ids[i], prediction_ids[j]) for i, j in zip(rows, cols) if edges[i, j]]


def _unique(rows):
    ids = [str(r["id"]) for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)): raise ValueError("Duplicate/empty scorer identity")


def upstream_assignment(candidates, targets):
    """Scorer-only deterministic box association, fixed before inspecting SAM.

    Max cardinality then full-box IoU with stable ID ordering is a benchmark
    correspondence, not proof of semantic uniqueness. Unmatched prompts remain
    unmeasured for conditional target-specific IoU.
    """
    ps = sorted([c for c in candidates if c["eligible_for_SAM_diagnostic_prompt"]], key=lambda r: r["candidate_id"])
    ts = sorted([t for t in targets if t.get("annotation_status") == "valid"], key=lambda r: str(r["id"]))
    q = np.zeros((len(ts), len(ps)))
    for i, t in enumerate(ts):
        for j, p in enumerate(ps): q[i, j] = bbox_iou(t["bbox_full_xyxy"], p["bbox_full_native_xyxy"])
    matched = match_pairs([str(t["id"]) for t in ts], [p["candidate_id"] for p in ps], q >= POLICY["upstream_bbox_match_iou"], q)
    return {pid: gid for gid, pid in matched}


def score_mask_set(predictions, targets):
    """All supplied masks are scored; never returns a revised native selection."""
    ps = sorted(predictions, key=lambda r: str(r["id"])); ts = sorted(targets, key=lambda r: str(r["id"]))
    _unique(ps); _unique(ts)
    unknown = [str(t["id"]) for t in ts if t.get("annotation_status") not in ("valid", "ignore")]
    ignored = [str(t["id"]) for t in ts if t.get("annotation_status") == "ignore"]
    shapes = set()
    for row in ps + ts:
        mask = row.get("mask")
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
                raise ValueError("Scoring requires original native bool masks")
            shapes.add(mask.shape)
    if len(shapes) > 1: raise ValueError("Native pixel shape mismatch")
    visible = [t for t in ts if t.get("annotation_status") == "valid" and t.get("mask") is not None and t["mask"].any()]
    unavailable = [str(t["id"]) for t in ts if t.get("annotation_status") == "valid" and (t.get("mask") is None or not t["mask"].any())]
    known = not unknown and not ignored
    coverage_edges = np.zeros((len(visible), len(ps)), bool)
    strict_edges = coverage_edges.copy(); quality = np.zeros(coverage_edges.shape)
    pairs, diagnostics = [], []
    for j, prediction in enumerate(ps):
        m = prediction.get("mask")
        if m is None: raise ValueError("Raw/selected prediction sets require an actual mask")
        area = int(m.sum())
        hits = [int((m & t["mask"]).sum()) for t in visible]
        merged = [str(t["id"]) for t, hit in zip(visible, hits) if hit / int(t["mask"].sum()) >= POLICY["merge_coverage_min"]]
        for i, t in enumerate(visible):
            gt_area = int(t["mask"].sum()); hit = hits[i]
            own = hit / gt_area
            fraction = ratio(hit, area)
            purity = ratio(hit, sum(hits))
            leak = max([hits[k] / int(other["mask"].sum()) for k, other in enumerate(visible) if k != i], default=0.)
            iou = ratio(hit, gt_area + area - hit)
            ok = own >= POLICY["coverage_min"] and fraction is not None and fraction >= POLICY["target_fraction_min"] and purity is not None and purity >= POLICY["camera_purity_min"] and leak <= POLICY["other_target_leakage_max"]
            coverage_edges[i, j] = own >= POLICY["coverage_min"]
            strict_edges[i, j] = ok
            quality[i, j] = iou or 0.
            pairs.append({"gt_id": str(t["id"]), "prediction_id": str(prediction["id"]), "iou": iou,
                          "gt_coverage": own, "target_fraction": fraction, "camera_purity": purity,
                          "other_target_leakage": leak, "coverage_ok": bool(coverage_edges[i,j]),
                          "strict_support_ok": bool(ok),
                          "gt_occlusion_inframe": t.get("actual_occlusion_fraction"),
                          "gt_groups": t.get("groups", [])})
        if not known: category = "unmeasured_annotation_domain"
        elif len(merged) >= 2: category = "merged_support"
        elif not area: category = "empty_native_support"
        elif not visible or unavailable: category = "support_domain_partly_unmeasured"
        elif not any(hits): category = "background_or_unlabeled"
        elif strict_edges[:,j].any(): category = "supported_target_support"
        elif sum(hit / int(t["mask"].sum()) > POLICY["other_target_leakage_max"] for t, hit in zip(visible, hits)) >= 2:
            # Pair leakage is relative to each possible target. Looking at any
            # pair would misclassify a clean A-mask via its comparison against B.
            category = "cross_target_contamination"
        elif any(p["prediction_id"] == str(prediction["id"]) and p["gt_coverage"] >= POLICY["coverage_min"] and (p["target_fraction"] or 0) < POLICY["target_fraction_min"] for p in pairs):
            category = "target_plus_background_contamination"
        else: category = "insufficient_target_support"
        diagnostics.append({"prediction_id": str(prediction["id"]), "category": category,
                            "mask_pixels": area, "merged_gt_ids": merged,
                            "non_DJI_class_asserted": False})
    gids = [str(t["id"]) for t in visible]; pids = [str(p["id"]) for p in ps]
    cm = match_pairs(gids, pids, coverage_edges, quality)
    sm = match_pairs(gids, pids, strict_edges, quality)
    duplicate_gt = {gids[i]: [pids[j] for j in range(len(ps)) if strict_edges[i,j]]
                    for i in range(len(visible)) if int(strict_edges[i].sum()) > 1}
    seen = {}; duplicate_pixels = []
    for p in ps:
        identity = hashlib.sha256(p["mask"].tobytes()).hexdigest()
        if identity in seen: duplicate_pixels.append([seen[identity],str(p["id"])])
        else: seen[identity] = str(p["id"])
    return {"annotation_domain_eligible": known, "original_gt_count": len(ts),
            "visible_support_gt_count": len(visible) if known else 0,
            "unknown_gt_ids": unknown, "ignored_gt_ids": ignored, "support_unavailable_gt_ids": unavailable,
            "prediction_count": len(ps), "coverage_tp": len(cm) if known else None,
            "strict_support_tp": len(sm) if known else None,
            "coverage_matches": cm if known else None, "strict_support_matches": sm if known else None,
            "unmatched_prediction_ids": [pid for pid in pids if pid not in {p for _,p in sm}] if known else None,
            "pairs": pairs, "diagnostics": diagnostics, "duplicate_gt_support": duplicate_gt if known else {},
            "exact_duplicate_mask_pairs": duplicate_pixels, "class_precision": None,
            "class_precision_reason": "SAM has no independent Action4 classifier; non-target semantic GT not established"}


def score_image(candidates, jobs, targets, *, image_id, groups=(), source_group=None):
    """jobs contain frozen raw masks/bound_id; no GT-based binding or replacement."""
    _unique(targets)
    candidate_map = {c["candidate_id"]: c for c in candidates}
    if len(candidate_map) != len(candidates): raise ValueError("Duplicate upstream candidate IDs")
    assignment = upstream_assignment(candidates, targets)
    raw, bound = [], []
    by_candidate = {}
    for job in jobs:
        cid = job["candidate_id"]
        if cid not in candidate_map or cid in by_candidate: raise ValueError("Unknown/duplicate completed prompt")
        by_candidate[cid] = job
        _unique(job["raw_predictions"])
        raw.extend(job["raw_predictions"])
        selected = job.get("bound_prediction_id")
        if job["available"]:
            matches = [p for p in job["raw_predictions"] if p["id"] == selected]
            if len(matches) != 1 or not matches[0]["mask"].any():
                raise ValueError("Available binding must name exactly one nonempty raw mask")
            bound.append(matches[0])
        elif selected is not None:
            raise ValueError("Unavailable native job cannot publish a selected mask")
    raw_score = score_mask_set(raw, targets); bound_score = score_mask_set(bound, targets)
    pair_lookup = {(p["prediction_id"],p["gt_id"]):p for p in bound_score["pairs"]}
    gt_map = {str(t["id"]):t for t in targets}
    conditional = []
    for cid, gid in sorted(assignment.items()):
        gt = gt_map[gid]
        if gt.get("mask") is None or not gt["mask"].any() or not bound_score["annotation_domain_eligible"]:
            conditional.append({"candidate_id":cid,"gt_id":gid,"eligible":False,"reason":"GT_support_unmeasured"})
            continue
        job = by_candidate.get(cid)
        selected = job.get("bound_prediction_id") if job else None
        pair = pair_lookup.get((selected,gid))
        wrong_instance = [p["gt_id"] for p in bound_score["pairs"] if p["prediction_id"] == selected and p["gt_id"] != gid and p["strict_support_ok"]]
        conditional.append({"candidate_id":cid,"gt_id":gid,"eligible":True,
                            "gt_occlusion_inframe":gt.get("actual_occlusion_fraction"),
                            "wrong_instance_support_gt_ids":wrong_instance,
                            "response_completed":job is not None,"native_status":job["status"] if job else "not_in_completed_score_snapshot",
                            "mask_available":pair is not None,"selected_prediction_id":selected,
                            "iou":pair["iou"] if pair else None,
                            "gt_coverage":pair["gt_coverage"] if pair else None,
                            "target_fraction":pair["target_fraction"] if pair else None,
                            "target_was_fixed_before_SAM_scoring":True})
    expected = [c["candidate_id"] for c in candidates if c["eligible_for_SAM_diagnostic_prompt"]]
    return {"schema":SCHEMA,"image_id":image_id,"groups":list(groups),"source_group":source_group,
            "original_gt_count":len(targets),"expected_prompt_jobs":len(expected),"completed_prompt_jobs":len(jobs),
            "image_transport_complete":set(expected)==set(by_candidate),"upstream_prompt_gt_assignment":assignment,
            "upstream_unmatched_gt_ids":sorted(set(gt_map)-set(assignment.values())),
            "upstream_unassociated_prompt_ids":sorted(set(expected)-set(assignment)),
            "raw_native_diagnostic":raw_score,"frozen_bound_masks":bound_score,"conditional":conditional,
            "GT_used_to_change_native_binding":False}


def load_completed_job(marker_path, job, lock, cohort_row, expected_marker_sha256):
    if sha_file(marker_path) != expected_marker_sha256:
        raise ValueError("Completed marker changed after score snapshot capture")
    marker = read_json(marker_path); directory = path(marker_path).parent
    if marker["task_key"] != job["task_key"] or marker["lock_identity"] != lock["identity"]:
        raise ValueError("Completed task lock mismatch")
    for relative, sha in marker["files_sha256"].items():
        p = path(relative)
        if not p.resolve().is_relative_to(directory.resolve()) or sha_file(p) != sha:
            raise ValueError("Completed raw/result integrity failure")
    result_path = path(marker["result_path"])
    if str(result_path.relative_to(PROJECT_ROOT)) not in marker["files_sha256"]:
        raise ValueError("Unsealed result reference")
    r = read_json(result_path)
    if r["candidate_id"] != job["candidate"]["candidate_id"] or r["image_id"] != job["image"]["image_id"] or r["image_sha256"] != cohort_row["image"]["sha256"] or r["lock_identity"] != lock["identity"]:
        raise ValueError("Scorer image/candidate/RGB identity mismatch")
    if r["backend_kind"] != "sam31_official_native" or not r["close_success"] or not r["response_valid"] or r["execution_status"] != "COMPLETED":
        raise ValueError("Incomplete/system-error/fixture response is not native success")
    raw_path = path(r["raw"]["arrays"])
    if marker["files_sha256"].get(str(raw_path.relative_to(PROJECT_ROOT))) != r["raw"]["sha256"]:
        raise ValueError("Raw reference not sealed by completion")
    with np.load(raw_path,allow_pickle=False) as f:
        ids=f["raw__outputs__out_obj_ids"].copy();masks=f["raw__outputs__out_binary_masks"].copy()
    h,w=cohort_row["native_shape"]
    if ids.dtype.kind not in "iu" or ids.ndim!=1 or len(set(map(int,ids)))!=len(ids) or masks.dtype!=np.bool_ or masks.shape!=(len(ids),h,w):
        raise ValueError("Native IDs/mask shape mismatch")
    metadata=r["native_candidates"]
    if len(metadata)!=len(ids) or any(int(ids[i])!=item["obj_id"] or item["raw_index"]!=i for i,item in enumerate(metadata)):
        raise ValueError("Native raw row/object correspondence changed")
    predictions=[{"id":job["task_key"]+":obj:"+str(int(obj)),"mask":masks[i],
                  "candidate_id":r["candidate_id"],"native_obj_id":int(obj)} for i,obj in enumerate(ids)]
    selected=None
    if r["available"]:
        if r["status"]!="target_present" or r["bound_obj_id"] not in ids:raise ValueError("Illegal available native status")
        index=list(map(int,ids)).index(r["bound_obj_id"]);m=masks[index]
        reg=r["mask_registration"]
        header=json.dumps({"dtype":"bool","shape":list(m.shape)},separators=(",",":")).encode()
        pixel=hashlib.sha256(header+b"\0"+m.tobytes()).hexdigest()
        if pixel!=reg["pixel_sha256"] or digest(reg["binding"])!=reg["binding_identity"] or "sam-mask-sha256:"+digest({"binding":reg["binding_identity"],"pixels":pixel})!=reg["mask_identity"]:
            raise ValueError("Selected mask registration identity mismatch")
        if reg["binding"]["obj_id"]!=r["bound_obj_id"] or reg["binding"]["track_id"]!=r["candidate_id"] or reg["binding"]["session_id"]!=r["session_id"]:
            raise ValueError("Selected mask assignment changed")
        selected=predictions[index]["id"]
    return {"candidate_id":r["candidate_id"],"status":r["status"],"available":r["available"],
            "bound_prediction_id":selected,"raw_predictions":predictions,
            "marker_path":str(path(marker_path).relative_to(PROJECT_ROOT)),"marker_sha256":sha_file(marker_path)}


def summarize(cohort, prompts, records, assignments, expected_jobs, completed_keys):
    by_image={r["image_id"]:r for r in records}
    group_names=sorted({g for image in cohort for g in image["groups"]}|{"all"})
    summaries={}
    for group in group_names:
        images=[i for i in cohort if group=="all" or group in i["groups"]]
        ids={i["sample_id"] for i in images}
        scored=[by_image[i] for i in ids if i in by_image]
        gt=sum(i["original_gt_count"] for i in images)
        eligible=sum(sum(t["annotation_status"]=="valid" and t["support_measurable"] and t["visible_pixels"]>0 for t in i["targets"])
                     for i in images if all(t["annotation_status"]=="valid" for t in i["targets"]))
        jobkeys={j["task_key"] for j in expected_jobs if j["image"]["image_id"] in ids}
        complete=len(jobkeys & completed_keys)==len(jobkeys)
        cov=sum(r["frozen_bound_masks"]["coverage_tp"] or 0 for r in scored)
        strict=sum(r["frozen_bound_masks"]["strict_support_tp"] or 0 for r in scored)
        raw=sum(r["raw_native_diagnostic"]["coverage_tp"] or 0 for r in scored)
        cond_expected=sum(sum(gid in {str(t["id"]) for t in image["targets"] if t["annotation_status"]=="valid" and t["support_measurable"] and t["visible_pixels"]>0}
                              for gid in assignments[image["sample_id"]].values())
                          for image in images if all(t["annotation_status"]=="valid" for t in image["targets"]))
        conditional=[p for r in scored for p in r["conditional"] if p["eligible"]]
        available=[p for p in conditional if p["mask_available"]]
        iou_sum=sum(p["iou"] for p in available)
        upstream_known_gt=sum(len(i["targets"]) for i in images if all(t["annotation_status"]=="valid" for t in i["targets"]))
        upstream_matched=sum(len(assignments[i["sample_id"]]) for i in images if all(t["annotation_status"]=="valid" for t in i["targets"]))
        summaries[group]={"group_level":"image_membership_with_both_original_GT_retained","image_domain":len(images),"original_gt_denominator":gt,
            "upstream_bbox_matched_gt":upstream_matched,
            "upstream_known_gt_without_published_bbox_match":upstream_known_gt-upstream_matched,
            "upstream_assignment_note":"Scorer-only full-box IoU>=.5 matching; unmatched may be localization failure, not proof of non-target class","visible_support_eligible_GT":eligible,
            "scored_images":len(scored),"expected_prompt_jobs":len(jobkeys),"completed_prompt_jobs":len(jobkeys & completed_keys),
            "transport_complete":complete,"bound_coverage_tp_so_far":cov,"bound_strict_support_tp_so_far":strict,
            "bound_coverage_recall":ratio(cov,eligible) if complete else None,
            "bound_strict_support_recall":ratio(strict,eligible) if complete else None,
            "observed_support_recall_lower_bound_full_domain":ratio(cov,eligible) if scored else None,
            "raw_native_coverage_tp_on_observed_images":raw,
            "raw_native_observed_GT":sum(r["raw_native_diagnostic"]["visible_support_gt_count"] for r in scored),
            "conditional_expected_upstream_matched_prompts":cond_expected,
            "conditional_completed_responses":sum(p["response_completed"] for p in conditional),
            "conditional_available_masks":len(available),"conditional_available_coverage":ratio(len(available),cond_expected),
            "conditional_IoU_mean_available_only":ratio(iou_sum,len(available)),
            "availability_weighted_conditional_IoU":ratio(iou_sum,cond_expected) if complete else None,
            "conditional_target_association_or_support_unmeasured":len(jobkeys)-cond_expected,
            "class_specificity_or_non_DJI_precision":None}
    return summaries


def cpu_affinity_receipt():
    """Inherit scheduler affinity; validate it without changing process placement."""
    actual = sorted(os.sched_getaffinity(0))
    if not 1 <= len(actual) <= 2:
        raise ValueError("Scorer requires inherited affinity of at most2 logical CPUs")
    pointer = read_json("configs/execution/action4_active.json")
    ref = pointer["active_ref"]
    if sha_file(ref["path"]) != ref["sha256"]:
        raise ValueError("Live CPU authority digest mismatch")
    authority = read_json(ref["path"])
    allowed = set(authority["cpu_policy"]["allowed_logical_cpu_ids"])
    if not set(actual) <= allowed:
        raise ValueError("Inherited scorer affinity is outside the live allowed CPU pool")
    return {"inherited_logical_CPUs": actual, "affinity_modified": False, "authority": ref}


def validate_native_domain(lock, prompts, jobs):
    cfg=lock["config"];schema=lock["schema"]
    if schema!=cfg.get("schema"):raise ValueError("Schema mismatch")
    if schema=="action4.sam.single_image_native.v1":
        if cfg["upstream_role"]!="frozen_B0_parent_step0_post_soft_nms":raise ValueError("Wrong upstream experiment")
        expected=1298
    elif schema=="action4.sam.single_image_native.v2":
        if cfg.get("split")!="eval" or cfg["upstream_role"]!="frozen_completed_B0_post_soft_nms":raise ValueError("Completed B0 scoring is eval-only")
        if (cfg["expected_images"],cfg["expected_saved_candidates"],cfg["expected_prompts"])!=(650,1300,1300):raise ValueError("Full eval domain required")
        source=cfg["source_contract"]
        if source["selected_step"]!=9500:raise ValueError("Completed B0 step9500 required")
        for key in ("sealed_role","run_spec","b0_bundle","predictions"):
            ref=source[key]
            if sha_file(ref["path"])!=ref["sha256"]:raise ValueError("B0 source changed")
        seal=read_json(source["sealed_role"]["path"])
        if (seal["role"],seal["selected_step"],seal["sha256"],seal["config_id"],seal["strict_load"])!=("B0",9500,source["b0_bundle"]["sha256"],source["config_id"],True):raise ValueError("B0 seal mismatch")
        if seal["run_spec_sha256"]!=source["run_spec"]["sha256"] or path(seal["bundle"]).resolve()!=path(source["b0_bundle"]["path"]).resolve():raise ValueError("Seal references differ")
        expected=1300
    else:raise ValueError("Unsupported native schema")
    if len(prompts)!=650 or len({r["image_id"] for r in prompts})!=650 or len(jobs)!=expected or sum(len(i["candidates"]) for i in prompts)!=1300:raise ValueError("Original prompt denominator changed")
    return {"schema":schema,"upstream_role":cfg["upstream_role"],"expected_prompt_jobs":expected}


def full_conditional_domain(cohort, assignments, records):
    """Pairing metadata only: pending images do not become scored images."""
    by_image={r["image_id"]:r for r in records}
    result=[]
    for image in cohort:
        iid=image["sample_id"]
        if iid in by_image:
            result.append(by_image[iid])
            continue
        targets={str(t["id"]):t for t in image["targets"]}
        known=all(t["annotation_status"]=="valid" for t in image["targets"])
        conditional=[]
        for cid,gid in sorted(assignments[iid].items()):
            target=targets[gid]
            eligible=bool(known and target["annotation_status"]=="valid" and target["support_measurable"] and target["visible_pixels"]>0)
            conditional.append({"candidate_id":cid,"gt_id":gid,"eligible":eligible,
                "response_completed":False,"mask_available":False,"iou":None,
                "native_status":"not_in_completed_score_snapshot",
                "reason":"pending_image" if eligible else "GT_support_unmeasured"})
        result.append({"image_id":iid,"conditional":conditional})
    return result


def paired_conditional(old_records,new_records,domain_keys):
    """Exact image/GT pairing; unavailable outcomes never become zero-IoU successes."""
    domain=set(domain_keys)
    def index(records):
        result={};seen=set()
        for row in records:
            for c in row["conditional"]:
                key=(row["image_id"],str(c["gt_id"]))
                if key not in domain or key in seen:raise ValueError("Unknown/duplicate paired GT identity")
                seen.add(key)
                if c["eligible"]:result[key]=c
        return result
    old=index(old_records);new=index(new_records);pairs=[]
    for key in sorted(old.keys() & new.keys()):
        a,b=old[key],new[key];available=bool(a["mask_available"] and b["mask_available"])
        if available and any(v is None or not np.isfinite(v) or not 0<=v<=1 for v in (a["iou"],b["iou"])):raise ValueError("Invalid paired IoU")
        pairs.append({"image_id":key[0],"gt_id":key[1],"old_available":a["mask_available"],"new_available":b["mask_available"],
            "old_completed":a["response_completed"],"new_completed":b["response_completed"],"old_iou":a["iou"],"new_iou":b["iou"],
            "paired_available":available,"delta_iou":b["iou"]-a["iou"] if available else None})
    measured=[p for p in pairs if p["paired_available"]]
    return {"pair_key":["image_id","gt_id"],"full_GT_domain":len(domain),"old_conditional_eligible":len(old),"new_conditional_eligible":len(new),
        "common_conditional_eligible":len(pairs),"old_only_conditional_keys":[list(k) for k in sorted(old.keys()-new.keys())],
        "new_only_conditional_keys":[list(k) for k in sorted(new.keys()-old.keys())],"both_available_pairs":len(measured),
        "common_not_both_available":len(pairs)-len(measured),"old_mean_paired_IoU":ratio(sum(p["old_iou"] for p in measured),len(measured)),
        "new_mean_paired_IoU":ratio(sum(p["new_iou"] for p in measured),len(measured)),
        "mean_paired_delta_IoU":ratio(sum(p["delta_iou"] for p in measured),len(measured)),"pairs":pairs,
        "interpretation":"Common conditional diagnostic only; changing conditional domains is not mask improvement."}


def run_score(native_root, output, *, first_batch_only=False,
              scorer_source="outputs/action4_sam_full_dev_preparation_v001/SCORER_SOURCE.json",
              cohort_receipt="outputs/action4_full_zeroshot_eval_v001/COHORT_RECEIPT.json", compare_score=None):
    cpu_receipt = cpu_affinity_receipt()
    if path(output).exists(): raise FileExistsError("Score output already exists; preserve prior score snapshots")
    source=read_json(scorer_source);receipt=read_json(cohort_receipt)
    if source["sha256"]!=receipt["manifest"]["sha256"] or sha_file(source["eval_manifest"])!=source["sha256"]:
        raise ValueError("Frozen scorer GT source mismatch")
    if sha_file(receipt["cohort_path"])!=receipt["cohort_sha256"]:raise ValueError("Frozen cohort changed")
    cohort=[json.loads(line) for line in path(receipt["cohort_path"]).read_text().splitlines()]
    if len(cohort)!=650 or sum(len(i["targets"]) for i in cohort)!=1300:raise ValueError("Full650/1300 domain required")
    native_root=path(native_root);lock=read_json(native_root/"RUN_LOCK.json")
    if digest({k:v for k,v in lock.items() if k!="identity"})!=lock["identity"]:raise ValueError("Native run lock changed")
    config=lock["config"]
    if sha_file(config["manifest"])!=config["manifest_sha256"]:raise ValueError("Prompt source changed")
    prompts=[json.loads(line) for line in path(config["manifest"]).read_text().splitlines()]
    if [r["image_id"] for r in prompts]!=[r["sample_id"] for r in cohort]:raise ValueError("Cohort identity/order mismatch")
    by_id={r["image_id"]:r for r in prompts};jobs=[]
    for image in prompts:
        for candidate in image["candidates"]:
            if candidate["eligible_for_SAM_diagnostic_prompt"]:
                key=digest({"lock":lock["identity"],"image_id":image["image_id"],"image_sha256":image["image_sha256"],"candidate":candidate,"text":config["native"]["text"]})
                jobs.append({"task_key":key,"image":image,"candidate":candidate})
    native_domain=validate_native_domain(lock,prompts,jobs)
    allowed=set(read_json(native_root/"PREPARATION.json")["first_batch_task_keys"]) if first_batch_only else {j["task_key"] for j in jobs}
    # Capture a completed index once; in-flight jobs are never read as empty output.
    capture_started=time.time()
    selected={j["task_key"]:native_root/"jobs"/j["task_key"]/"COMPLETED.json" for j in jobs
              if j["task_key"] in allowed and (native_root/"jobs"/j["task_key"]/"COMPLETED.json").exists()}
    captured_marker_sha={k:sha_file(p) for k,p in selected.items()}
    capture_finished=time.time()
    job_by_id={j["task_key"]:j for j in jobs};records=[];assignments={};frozen_index=[]
    for image in cohort:
        iid=image["sample_id"];prompt=by_id[iid]
        if prompt["image_sha256"]!=image["image"]["sha256"] or prompt["native_shape_hw"]!=image["native_shape"]:
            raise ValueError("GT/native image pixel identity mismatch")
        targets=[dict(t) for t in image["targets"]]
        assignments[iid]=upstream_assignment(prompt["candidates"],targets)
        keys=[k for k in selected if job_by_id[k]["image"]["image_id"]==iid]
        if not keys:continue
        from PIL import Image
        for target in targets:
            spec=target.get("visible_mask")
            if spec is None:target["mask"]=None;continue
            if sha_file(spec["path"])!=spec["sha256"]:raise ValueError("GT mask bytes changed")
            with Image.open(path(spec["path"])) as f:mask=np.asarray(f.convert("L"))>0
            if mask.shape!=tuple(image["native_shape"]) or int(mask.sum())!=target["visible_pixels"]:
                raise ValueError("GT native mask/pixel count mismatch")
            target["mask"]=mask
        completed=[load_completed_job(selected[k],job_by_id[k],lock,image,captured_marker_sha[k]) for k in keys]
        frozen_index.extend({"task_key":k,"marker_path":c["marker_path"],"marker_sha256":c["marker_sha256"]}
                            for k,c in zip(keys,completed))
        records.append(score_image(prompt["candidates"],completed,targets,image_id=iid,
                                   groups=image["groups"],source_group=image["source_group"]))
    groups=summarize(cohort,prompts,records,assignments,jobs,set(selected))
    allgroup=groups["all"]
    result={"schema":SCHEMA,"status":"COMPLETE_SUPPORT_SCORING_NOT_CLASS_CONFIRMATION" if len(selected)==len(jobs) else "PARTIAL_NATIVE_CACHE_SNAPSHOT_NO_FULL_RUN_ACCURACY",
            "native_root":str(native_root.relative_to(PROJECT_ROOT)),"native_lock_identity":lock["identity"],
            "CPU_admission":cpu_receipt,
            "upstream_association_semantics":"scorer-only deterministic max-cardinality then full-boxIoU with stable IDs; not semantic disambiguation",
            "source_manifest_sha256":source["sha256"],"cohort_sha256":receipt["cohort_sha256"],
            "scorer_source_sha256":sha_file(__file__),"policy":POLICY,
            "completed_snapshot_capture":{"started_unix":capture_started,"finished_unix":capture_finished,
                "fixed_keys":sorted(selected),"read_running_attempts":False,"new_completions_after_capture_not_included":True},"population":{"images":650,"original_GT":1300,"saved_upstream_candidates":1300,
            "registered_prompt_jobs":len(jobs),"upstream_unprompted_candidates":1300-len(jobs),"completed_jobs_in_score_snapshot":len(selected),
            "pending_or_outside_score_snapshot":len(jobs)-len(selected)},
            "groups":groups,"summary":allgroup,"first_batch_only":first_batch_only,
            "class_identity_precision":None,"non_DJI_specificity":None,
            "semantic_negative_GT_status":"not_measured_no_independent_negative_annotations",
            "GT_changed_native_binding":False,"GPU_used":False,"new_model_calls":0,
            "limitations":["Raw native matches are scorer-only diagnostics, not adopted outputs or GT-selected replacement masks.",
                "Bound metrics use only the exact native runtime binding; no best-mask selection by GT.",
                "Conditional GT association uses frozen upstream full-box IoU>=.5 before looking at SAM; unassociated prompts are separately unmeasured.",
                "Available-only IoU explicitly excludes unavailable masks; availability and full fixed denominators are reported alongside.",
                "Partial lower bounds are not complete-run recall; pending outputs are not fake empty-success.",
                "Unknown/ignore GT domains have no scored precision or recall; background_or_unlabeled is not proof of non-DJI class.",
                "No99% or A0 geometry-accuracy claim; masks alone cannot confirm Action4 category."]}
    result["native_domain"]=native_domain
    result["full1300_end_to_end_support"]={"original_GT_denominator":1300,
        "coverage_tp_so_far":allgroup["bound_coverage_tp_so_far"],"strict_support_tp_so_far":allgroup["bound_strict_support_tp_so_far"],
        "coverage_recall":ratio(allgroup["bound_coverage_tp_so_far"],1300) if allgroup["transport_complete"] else None,
        "strict_support_recall":ratio(allgroup["bound_strict_support_tp_so_far"],1300) if allgroup["transport_complete"] else None,
        "transport_complete":allgroup["transport_complete"],"unmeasured_GT":1300-allgroup["visible_support_eligible_GT"],
        "note":"Fixed original GT support yield; unknowns retained, not class precision."}
    if compare_score is not None:
        if native_domain["schema"]!="action4.sam.single_image_native.v2":raise ValueError("Comparison requires v2 current run")
        oldroot=path(compare_score);old=read_json(oldroot/"RESULT.json")
        if old["status"]!="COMPLETE_SUPPORT_SCORING_NOT_CLASS_CONFIRMATION" or old["population"]["images"]!=650 or old["population"]["original_GT"]!=1300:raise ValueError("Completed full650 historical score required")
        if old["cohort_sha256"]!=result["cohort_sha256"] or old["source_manifest_sha256"]!=result["source_manifest_sha256"] or old["policy"]!=POLICY:raise ValueError("Comparison GT domain/policy mismatch")
        oldlock=read_json(path(old["native_root"])/"RUN_LOCK.json")
        if oldlock["identity"]!=old["native_lock_identity"] or digest({k:v for k,v in oldlock.items() if k!="identity"})!=oldlock["identity"] or oldlock["config"]["upstream_role"]!="frozen_B0_parent_step0_post_soft_nms":raise ValueError("Historical baseline lock mismatch")
        oldrecords=[json.loads(line) for line in (oldroot/"PER_IMAGE.jsonl").read_text().splitlines()]
        if len(oldrecords)!=650 or {r["image_id"] for r in oldrecords}!={i["sample_id"] for i in cohort}:raise ValueError("Historical image domain mismatch")
        paired=paired_conditional(oldrecords,full_conditional_domain(cohort,assignments,records),{(i["sample_id"],str(t["id"])) for i in cohort for t in i["targets"]})
        paired.update(current_run_complete=allgroup["transport_complete"],historical_score={"path":str(oldroot.relative_to(PROJECT_ROOT)),"result_sha256":sha_file(oldroot/"RESULT.json"),"per_image_sha256":sha_file(oldroot/"PER_IMAGE.jsonl")})
        result["paired_historical_conditional"]=paired
    else:result["paired_historical_conditional"]={"status":"NOT_REQUESTED","mean_paired_delta_IoU":None}
    output=path(output);output.mkdir(parents=True,exist_ok=False)
    for filename,value in (("RESULT.json",result),("COMPLETED_SCORE_SNAPSHOT.json",frozen_index)):
        (output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    (output/"PER_IMAGE.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records))
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--native-root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--first-batch-only",action="store_true")
    parser.add_argument("--compare-score")
    args=parser.parse_args()
    result=run_score(args.native_root,args.output,first_batch_only=args.first_batch_only,compare_score=args.compare_score)
    print(json.dumps({"status":result["status"],"population":result["population"],"summary":result["summary"]}))


if __name__=="__main__":main()
