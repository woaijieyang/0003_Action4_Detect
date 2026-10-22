"""Candidate failure events and reviewed synthesis requests, never pseudo-GT."""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

from data.action4_paths import checked_root

import numpy as np

from model.training.action4_runtime import input_path, output_path, sha256_file, write_json_new
from .action4_metrics import bbox_iou, maximum_matching

CATEGORIES = {
    "coordinate_or_timing": {"action": "repair_interface", "diagnostic": "Check native frame PTS, pixel-center transforms, inverse ROI and RGB channel order before training."},
    "missed_or_merged": {"action": "targeted_synthesis", "diagnostic": "Compare full RGB and GT-ROI eval; inspect score, strict duplicate filter and center collisions."},
    "corner_semantics": {"action": "inspect_geometry_then_synthesis", "diagnostic": "Verify fixed object axes/corner order and asset hash; retain same-index corners when adding confirmed side/flip views."},
    "wrist_or_cable_occlusion": {"action": "targeted_synthesis", "diagnostic": "Recreate connected wrist/forearm/cable at the observed side and depth; retain hidden virtual corner labels."},
    "material_or_lighting": {"action": "targeted_synthesis", "diagnostic": "Compare local object/background luminance and material response; correct the baseline before bounded light variation."},
    "motion_or_truncation": {"action": "inspect_inference_then_synthesis", "diagnostic": "Check raw detections, crop truncation and any association/smoothing; add observed wrist turns and bounded motion blur."},
    "domain_gap": {"action": "targeted_synthesis", "diagnostic": "Compare actual wrist installation, body scale, background and appearance with accepted render scenes."},
}

SCENE_FAMILIES = {"wall_rear", "table_fastening", "table_turn", "near_double", "self_forearm_occlusion",
                  "cloth_lift", "wall_transition_truncation", "hard_negative"}


def validate_synthesis_recipe(recipe: dict) -> None:
    """The exact executable JSONL contract shared with DataAdapter plan-s1."""
    required = {"scene_family", "view", "scale", "occluder", "occluder_location", "allowed_perturbations", "reference_frame_ids"}
    if set(recipe) - (required | {"quota", "target_presence"}) or not required.issubset(recipe):
        raise ValueError("Recipe must use the supported explicit fields only")
    if "target_presence" in recipe:
        from data.action4_synthesis import validate_target_presence
        validate_target_presence(recipe["target_presence"], recipe["scene_family"])
    if recipe["scene_family"] not in SCENE_FAMILIES:
        raise ValueError("Unsupported scene family")
    if not isinstance(recipe["view"], dict) or set(recipe["view"]) != {"wrist_delta_xyz_deg"}:
        raise ValueError("view requires wrist_delta_xyz_deg, relative to the evidenced family anchor")

    def finite_between(value, low, high):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and low <= value <= high

    rotation = recipe["view"]["wrist_delta_xyz_deg"]
    if not isinstance(rotation, list) or len(rotation) != 3 or not all(finite_between(x, -8, 8) for x in rotation):
        raise ValueError("Wrist anchor delta must be three values inside +/-8 degrees")
    if not finite_between(recipe["scale"], .9, 1.1):
        raise ValueError("Scale must be a single relative camera-distance factor inside [0.9,1.1]")
    if recipe["occluder"] != "scene_default" or recipe["occluder_location"] != "linked_geometry":
        raise ValueError("Only existing linked scene occluders are implemented; arbitrary 2D coverings are not accepted")
    perturb = recipe["allowed_perturbations"]
    names = {"rotation_jitter_deg", "position_jitter_m", "light_intensity_gain", "exposure_delta_ev"}
    if not isinstance(perturb, dict) or set(perturb) != names:
        raise ValueError("Declare the four supported bounded perturbation parameters")
    if not finite_between(perturb["rotation_jitter_deg"], 0, 8):
        raise ValueError("Rotation jitter must lie inside [0,8] degrees")
    position = perturb["position_jitter_m"]
    if not isinstance(position, list) or len(position) != 3 or not all(finite_between(x, 0, .004) for x in position):
        raise ValueError("Position jitter must be three nonnegative bounds no greater than 0.004 m")
    if not finite_between(perturb["light_intensity_gain"], .9, 1.1) or not finite_between(perturb["exposure_delta_ev"], -.15, .15):
        raise ValueError("Light/exposure adjustment exceeds the evidence-bounded S1 contract")
    references = recipe["reference_frame_ids"]
    if not isinstance(references, list) or not references or any(type(i) is not int or not 0 <= i <= 2849 for i in references):
        raise ValueError("References must be original frame IDs inside this 2850-frame development video")


def read_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or [r["frame_id"] for r in rows] != list(range(len(rows))):
        raise ValueError("Feedback requires every original frame once, in contiguous order")
    if len({r["checkpoint_sha256"] for r in rows}) != 1 or len({r["inference_config_sha256"] for r in rows}) != 1:
        raise ValueError("Predictions mix checkpoint or decoding identities")
    if len({r["time_base"] for r in rows}) != 1 or any(b["pts"] <= a["pts"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Invalid prediction presentation timeline")
    return rows


def candidate_events(rows: list[dict], *, merge_gap_frames: int = 15,
                     center_jump_diagonals: float = .5, low_score: float = .4,
                     zero_detection_frames: int = 30) -> list[dict]:
    """Heuristics prompt human review; true motion/entry/exit may explain each."""
    if (merge_gap_frames < 0 or center_jump_diagonals <= 0 or not 0 <= low_score <= 1
            or type(zero_detection_frames) is not int or zero_detection_frames < 1):
        raise ValueError("Invalid event candidate thresholds")
    # Include every frame of a long empty run so merging retains the entire
    # review interval, including leading/trailing all-empty videos.
    long_empty = set()
    empty_start = None
    for i in range(len(rows) + 1):
        empty = i < len(rows) and not rows[i]["instances"]
        if empty and empty_start is None:
            empty_start = i
        elif not empty and empty_start is not None:
            if i - empty_start >= zero_detection_frames:
                long_empty.update(range(empty_start, i))
            empty_start = None
    triggers = []
    for i, row in enumerate(rows):
        reasons = ["sustained_zero_detections"] if i in long_empty else []
        if any(p["score"] < low_score or "low_corner_peak_heuristic" in p.get("uncertainty_flags", []) for p in row["instances"]):
            reasons.append("low_confidence")
        if i and len(row["instances"]) != len(rows[i - 1]["instances"]):
            reasons.append("instance_count_change")
        if i:
            previous, current = rows[i - 1]["instances"], row["instances"]
            edges = [[j for j, p in enumerate(previous) if bbox_iou(c["bbox_full_xyxy"], p["bbox_full_xyxy"]) > .1]
                     for c in current]
            for ci, pi in maximum_matching(edges, len(previous)):
                now, old = np.asarray(current[ci]["corners_px"]), np.asarray(previous[pi]["corners_px"])
                old_box = np.asarray(previous[pi]["bbox_full_xyxy"])
                diagonal = np.linalg.norm(old_box[2:] - old_box[:2])
                if diagonal > 0 and np.linalg.norm(now - old, axis=1).max() / diagonal > center_jump_diagonals:
                    reasons.append("ordered_corner_motion_jump")
                    break
        if reasons:
            triggers.append((i, reasons))
    grouped = []
    for frame_id, reasons in triggers:
        if grouped and frame_id - grouped[-1]["end_frame_id"] <= merge_gap_frames + 1:
            grouped[-1]["end_frame_id"] = frame_id
            grouped[-1]["trigger_frames"].append(frame_id)
            grouped[-1]["triggers"] = sorted(set(grouped[-1]["triggers"]) | set(reasons))
        else:
            grouped.append({"start_frame_id": frame_id, "end_frame_id": frame_id,
                            "trigger_frames": [frame_id], "triggers": reasons})
    for index, event in enumerate(grouped):
        start, end = rows[event["start_frame_id"]], rows[event["end_frame_id"]]
        event.update({"event_id": f"candidate_{index:05d}", "start_pts": start["pts"], "end_pts": end["pts"],
                      "time_base": start["time_base"], "category": "unclassified", "involved_instances": [],
                      "status": "pending_review", "evidence_strength": "automatic_heuristic",
                      "observation": "Model output changed or remained empty; true scene absence/motion/entry/exit remains an alternative explanation.",
                      "suggested_intervention": "Review original images and raw cuboids before changing data or code.",
                      "screenshots": []})
    return grouped


def reviewed_requests(events: list[dict], rows: list[dict], *, total_quota: int = 2000,
                      per_event_fraction: float = .1) -> dict:
    if total_quota < 1 or not 0 < per_event_fraction <= 1:
        raise ValueError("Invalid synthesis budget")
    requests, interface_repairs = [], []
    used_ids = set()
    cap = max(1, math.floor(total_quota * per_event_fraction))
    for event in events:
        if event.get("status") != "confirmed_failure":
            continue
        identity = event.get("event_id")
        if not isinstance(identity, str) or identity in used_ids:
            raise ValueError("Confirmed review requires unique event_id")
        used_ids.add(identity)
        first, last = event.get("start_frame_id"), event.get("end_frame_id")
        if type(first) is not int or type(last) is not int or not 0 <= first <= last < len(rows):
            raise ValueError("Confirmed failure frame interval is invalid")
        if not event.get("reviewer") or not event.get("evidence") or not event.get("observation"):
            raise ValueError("Confirmed failures require reviewer, evidence and an observed phenomenon")
        category = event.get("category")
        if category not in CATEGORIES:
            raise ValueError("A confirmed event needs a supported causal category")
        recommendation = CATEGORIES[category]
        if recommendation["action"] == "repair_interface":
            interface_repairs.append({"source_event_id": identity, **recommendation, "evidence": event["evidence"]})
            continue
        recipe = event.get("synthesis")
        if not isinstance(recipe, dict):
            raise ValueError("Confirmed data gaps require explicit bounded synthesis parameters")
        required = ("scene_family", "view", "scale", "occluder", "occluder_location", "allowed_perturbations", "reference_frame_ids")
        if any(key not in recipe for key in required):
            raise ValueError(f"Missing synthesis fields; required: {required}")
        validate_synthesis_recipe(recipe)
        if any(type(i) is not int or not first <= i <= last for i in recipe["reference_frame_ids"]):
            raise ValueError("Synthesis reference must be within the reviewed failure interval")
        if not recipe["reference_frame_ids"]:
            raise ValueError("At least one actual reviewed reference frame is required")
        quota = recipe.get("quota", cap)
        if type(quota) is not int or not 1 <= quota <= cap:
            raise ValueError("Each independent event must respect the frozen S1 quota cap")
        requests.append({"request_id": f"s1_{identity}", "source_event_id": identity, "category": category,
                         "status": "reviewed_request_pending_render", "quota": quota, "split": "train",
                         "source_frame_start": first, "source_frame_end": last,
                         "recipe": {key: recipe[key] for key in (*required, "target_presence") if key in recipe},
                         "diagnostic": recommendation["diagnostic"], "ground_truth": "recompute_from_renderer_geometry",
                         "prediction_coordinates_used_as_training_labels": False})
    allocated = sum(r["quota"] for r in requests)
    if allocated > total_quota:
        raise ValueError("Reviewed request quotas exceed the S1 budget; prioritize independently reviewed events")
    return {"requests": requests, "interface_repairs": interface_repairs, "allocated_quota": allocated,
            "unallocated_quota": total_quota - allocated, "per_event_cap": cap,
            "note": "Unallocated quota requires additional reviewed evidence; requests do not manufacture labels."}


def run_feedback(root: str | Path, run_dir: str | Path, output: str | Path, *, reviewed_events: str | Path | None = None,
                 total_quota: int = 2000) -> dict:
    from PIL import Image
    root = checked_root(root)
    run_dir = input_path(root, run_dir)
    target = output_path(root, output)
    if not target.is_relative_to(output_path(root, "outputs")):
        raise ValueError("Feedback candidates and screenshots must be stored under outputs, never data")
    if target.exists():
        raise FileExistsError("Feedback output must be new")
    rows = read_predictions(run_dir / "predictions_raw.jsonl")
    candidates = candidate_events(rows)
    target.mkdir(parents=True)
    screenshots = target / "screenshots"
    screenshots.mkdir()
    for event in candidates:
        indices = sorted({event["start_frame_id"], (event["start_frame_id"] + event["end_frame_id"]) // 2, event["end_frame_id"]})
        for index in indices:
            source = input_path(root, rows[index]["source_image"])
            if sha256_file(source) != rows[index]["source_image_sha256"]:
                raise ValueError("Original source image no longer matches raw predictions")
            destination = screenshots / f"{event['event_id']}_frame_{index:08d}.jpg"
            with Image.open(source) as image:
                image.load()
                image.convert("RGB").save(destination, quality=95)
            event["screenshots"].append(str(destination.relative_to(root)))
    with (target / "failure_events.jsonl").open("x", encoding="utf-8") as handle:
        for event in candidates:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    reviewed = []
    if reviewed_events is not None:
        reviewed_path = input_path(root, reviewed_events)
        reviewed = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    plan = reviewed_requests(reviewed, rows, total_quota=total_quota)
    with (target / "synthesis_requests.jsonl").open("x", encoding="utf-8") as handle:
        for request in plan.pop("requests"):
            handle.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
    report = {"schema_version": "action4.feedback.v1", "source_predictions_sha256": sha256_file(run_dir / "predictions_raw.jsonl"),
              "candidate_count": len(candidates), "reviewed_input_count": len(reviewed), **plan,
              "status": "candidates_pending_review" if not reviewed else "reviewed_requests_generated"}
    write_json_new(target / "feedback_report.json", report)
    return {"output_dir": str(target), **report}
