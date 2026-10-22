"""Independent RGB-only inference, synthetic scoring and frozen test entry."""
from __future__ import annotations

import json
import copy
import itertools
from pathlib import Path

from data.action4_paths import checked_root
from data.action4_split_protocol import binding_identity, check_checkpoint_binding
from typing import Any

import numpy as np

from data import DataAdapter
from model.training.action4_runtime import (canonical_hash, environment_identity, input_path,
                                           jsonable, load_checkpoint, load_config, output_path,
                                           select_device, sha256_file, source_identity, write_json_new, validate_run_id)
from .action4_metrics import is_ignored, score_dataset, selection_key


def inference_contract(config: dict) -> dict:
    return {key: config[key] for key in ("network", "preprocess", "decode")}


def build_model(config: dict, *, checkpoint: str | Path | None = None, weights=None, device="cpu"):
    from model.action4_network import Action4Net
    model = Action4Net(weights=weights, **config["network"])
    if checkpoint is not None:
        state = load_checkpoint(Path(checkpoint))
        if state["config"]["network"] != config["network"]:
            raise ValueError("Checkpoint architecture differs from requested network")
        if state["config"]["preprocess"] != config["preprocess"]:
            raise ValueError("Changing checkpoint preprocessing requires an explicit new experiment")
        from model.training.action4_stage_policy import prepare_checkpoint_topology
        prepare_checkpoint_topology(model, state)
        model.load_state_dict(state["model"], strict=True)
    return model.to(device)


def _map_detection(detection: dict, inverse: np.ndarray) -> dict:
    box = np.asarray(jsonable(detection["bbox_full_xyxy"]), dtype=float)
    transformed = DataAdapter.transform_points(box.reshape(2, 2), inverse)
    center = DataAdapter.transform_points(np.asarray(jsonable(detection["center"])).reshape(1, 2), inverse)[0]
    return {"score": float(detection["score"]), "center_px": center.tolist(),
            "bbox_full_xyxy": transformed.reshape(4).tolist()}


def decode_full_boxes(outputs: dict, orig_to_full: list[np.ndarray], config: dict) -> list[list[dict]]:
    from model.action4_decode import decode_detections
    decode = {k: v for k, v in config["decode"].items() if k != "corner_topk"}
    boxes = decode_detections(outputs, image_size=tuple(config["preprocess"]["full_size"]), **decode)
    return [[_map_detection(d, np.linalg.inv(affine)) for d in row]
            for row, affine in zip(boxes, orig_to_full)]


def heatmap_to_original(affines: list[np.ndarray], roi_size: int, heatmap_size: tuple[int, int]):
    width, height = heatmap_size
    sx, sy = roi_size / width, roi_size / height
    hm_to_roi = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=float)
    return np.asarray([np.linalg.inv(a) @ hm_to_roi for a in affines])


def predict_images(model, images: list, config: dict, device) -> list[list[dict]]:
    """Only raw RGB tensors enter this function. No reference argument exists."""
    import torch
    from data.action4_dataset import prepare_full_images, prepare_rois
    from model.action4_decode import decode_corners
    model.eval()
    with torch.inference_mode():
        full, full_affines = prepare_full_images(images, size=tuple(config["preprocess"]["full_size"]), device=device)
        dense = model(full)
        predictions = decode_full_boxes(dense, full_affines, config)
        image_indices, boxes = [], []
        for image_index, rows in enumerate(predictions):
            image_indices.extend([image_index] * len(rows))
            boxes.extend([row["bbox_full_xyxy"] for row in rows])
        if not boxes:
            return predictions
        roi_size = config["preprocess"]["roi_size"]
        rois, affines = prepare_rois(images, boxes, image_indices, roi_size=roi_size,
                                    expansion=config["preprocess"]["roi_expansion"], device=device)
        logits = model(None, roi_images=rois)["corner_logits"]
        transforms = torch.as_tensor(heatmap_to_original(affines, roi_size, (logits.shape[-1], logits.shape[-2])),
                                     dtype=torch.float32, device=device)
        corners = decode_corners(logits, heatmap_to_image=transforms, topk=config["decode"]["corner_topk"])
        confidence = logits.sigmoid().flatten(2).amax(-1).detach().cpu()
        offset = 0
        for image_index, rows in enumerate(predictions):
            height, width = images[image_index].shape[-2:]
            for instance_index, row in enumerate(rows):
                points = corners[offset].detach().cpu().numpy()
                if not np.isfinite(points).all():
                    raise FloatingPointError("Non-finite predicted corners; frame must not be silently skipped")
                flags = []
                if np.any(points[:, 0] < -.5) or np.any(points[:, 0] > width - .5) or np.any(points[:, 1] < -.5) or np.any(points[:, 1] > height - .5):
                    flags.append("some_corners_out_of_image")
                if float(confidence[offset].min()) < .2:
                    flags.append("low_corner_peak_heuristic")
                row.update({"instance_id": instance_index, "object_id": 1, "corners_px": points.tolist(),
                            "corner_peak_scores": confidence[offset].tolist(), "uncertainty_flags": flags})
                offset += 1
        return predictions


def evaluate_records(model, dataset, config: dict, device, *, batch_size: int = 1) -> tuple[dict, list[dict]]:
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be positive")
    prediction_records = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        prediction_batch = predict_images(model, [item["image"] for item in items], config, device)
        for item, predictions in zip(items, prediction_batch):
            prediction_records.append({"sample_id": item["record"]["sample_id"], "instances": predictions})
    report = score_dataset(prediction_records, dataset.records)
    report.update({"scope": "full_rgb_pipeline", "reference_boxes_enter_inference": False,
                   "sample_count": len(dataset)})
    return report, prediction_records


def evaluate_gt_rois(model, dataset, config: dict, device, *, batch_size: int = 1) -> dict:
    """Explicit diagnostic using GT boxes, never reported as full-system F1."""
    import torch
    from data.action4_dataset import prepare_rois, build_corner_targets
    from model.action4_decode import decode_corners
    model.eval()
    actual_errors, zero_errors, ideal_errors, rows = [], [], [], []
    roi_size = config["preprocess"]["roi_size"]
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            images, records = [i["image"] for i in items], [i["record"] for i in items]
            image_ids, instance_ids, boxes = [], [], []
            for bi, record in enumerate(records):
                for gi, gt in enumerate(record["instances"]):
                    if not is_ignored(gt):
                        image_ids.append(bi)
                        instance_ids.append(gi)
                        boxes.append(gt["bbox_full_xyxy"])
            if not boxes:
                continue
            rois, affines = prepare_rois(images, boxes, image_ids, roi_size=roi_size,
                                        expansion=config["preprocess"]["roi_expansion"], device=device)
            logits = model(None, roi_images=rois)["corner_logits"]
            heatmap_size = (logits.shape[-1], logits.shape[-2])
            targets, valid = build_corner_targets(records, image_ids, instance_ids, affines,
                                                  roi_size=roi_size, heatmap_size=heatmap_size, device=device)
            decoded = decode_corners(logits, topk=config["decode"]["corner_topk"])
            zero_decoded = decode_corners(torch.zeros_like(logits), from_logits=False,
                                          topk=config["decode"]["corner_topk"])
            ideal_decoded = decode_corners(targets, from_logits=False, topk=config["decode"]["corner_topk"])
            hm_to_original = heatmap_to_original(affines, roi_size, heatmap_size)
            for ri, (bi, gi) in enumerate(zip(image_ids, instance_ids)):
                gt_original = np.asarray(records[bi]["instances"][gi]["corners_px"])
                gt_hm = DataAdapter.transform_points(gt_original, np.linalg.inv(hm_to_original[ri]))
                mask = valid[ri].cpu().numpy().astype(bool)
                actual = np.linalg.norm(decoded[ri].cpu().numpy() - gt_hm, axis=1)[mask]
                zero = np.linalg.norm(zero_decoded[ri].cpu().numpy() - gt_hm, axis=1)[mask]
                ideal = np.linalg.norm(ideal_decoded[ri].cpu().numpy() - gt_hm, axis=1)[mask]
                actual_errors.extend(actual.tolist())
                zero_errors.extend(zero.tolist())
                ideal_errors.extend(ideal.tolist())
                rows.append({"sample_id": records[bi]["sample_id"], "instance_id": records[bi]["instances"][gi]["instance_id"],
                             "errors_heatmap_px": actual.tolist(), "valid_corners": int(mask.sum())})
    def p95(value):
        return float(np.percentile(value, 95)) if value else None
    actual_p95, zero_p95, ideal_p95 = p95(actual_errors), p95(zero_errors), p95(ideal_errors)
    return {"schema_version": "action4.roi_diagnostic.v1", "scope": "gt_roi_diagnostic",
            "reference_boxes_enter_inference": True, "images": len(dataset), "rois": len(rows),
            "valid_corners": len(actual_errors), "epe_mean_heatmap_px": float(np.mean(actual_errors)) if actual_errors else None,
            "epe_p95_heatmap_px": actual_p95, "zero_heatmap_p95_px": zero_p95,
            "ideal_heatmap_p95_px": ideal_p95,
            "improvement_over_zero_p95": 1 - actual_p95 / zero_p95 if actual_p95 is not None and zero_p95 else None,
            "per_roi": rows}


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")


def freeze_test_selection(root: Path, config: dict, checkpoint: Path, selection_path: Path) -> dict:
    """Create immutable selection before reading test labels or producing test scores."""
    checkpoint_state = load_checkpoint(checkpoint)
    _require_formal_checkpoint(checkpoint_state)
    test_manifest = input_path(root, config["data"]["test_manifest"])
    _validate_split_lineage(checkpoint_state, test_manifest, "test")
    split_protocol_binding = binding_identity(root, config)
    check_checkpoint_binding(checkpoint_state, split_protocol_binding)
    value = {"split_protocol_binding": split_protocol_binding,
             "schema_version": "action4.final_selection.v1", "checkpoint_sha256": sha256_file(checkpoint),
             "checkpoint": str(checkpoint), "inference_contract": inference_contract(config),
             "inference_contract_sha256": canonical_hash(inference_contract(config)),
             "test_manifest_sha256": sha256_file(test_manifest), "code": source_identity(root),
             "claim_scope": "frozen synthetic test configuration; real-video correctness requires separate reviewed evidence"}
    write_json_new(output_path(root, selection_path), value)
    return value


def _require_formal_checkpoint(checkpoint: dict) -> None:
    if (checkpoint.get("claim_scope") != "accepted_data_model_training"
            or checkpoint.get("stage") not in ("B0", "C1")
            or checkpoint.get("epoch_boundary") is not True
            or type(checkpoint.get("global_step")) is not int
            or checkpoint["global_step"] < 1):
        raise ValueError("Formal test requires a completed B0/C1 checkpoint trained on accepted data")



def _validate_split_lineage(checkpoint: dict, manifest: Path, split: str) -> None:
    """Keep fixed eval/test attached to the dataset frozen by the trainer.

    This also applies to explicit component diagnostics: that flag permits
    diagnostic samples, not replacement of a run's frozen split. Decode-only
    tuning remains possible because no complete config equality is required.
    """
    identity = checkpoint.get("identity")
    digests = identity.get("data_sha256") if isinstance(identity, dict) else None
    expected = digests.get(split) if isinstance(digests, dict) else None
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"Checkpoint lacks a valid frozen {split} manifest identity")
    if sha256_file(manifest) != expected:
        raise ValueError(f"Input differs from checkpoint frozen {split} manifest identity")


def run_evaluation(root: str | Path, config: dict, checkpoint: str | Path, *, split: str, run_id: str,
                   gt_roi: bool = False, selection: str | Path | None = None,
                   allow_diagnostic: bool = False) -> dict:
    import torch
    from data.action4_dataset import Action4Dataset
    root = checked_root(root)
    validate_run_id(run_id)
    if split not in ("eval", "test"):
        raise ValueError("Only fixed eval/test manifests may enter this runner")
    checkpoint_path = input_path(root, checkpoint)
    manifest = input_path(root, config["data"][f"{split}_manifest"])
    checkpoint_state = load_checkpoint(checkpoint_path)
    if split == "test":
        _require_formal_checkpoint(checkpoint_state)
    _validate_split_lineage(checkpoint_state, manifest, split)
    split_protocol_binding = binding_identity(root, config)
    check_checkpoint_binding(checkpoint_state, split_protocol_binding)
    if split == "test":
        if selection is None or gt_roi or allow_diagnostic:
            raise ValueError("Formal test requires a prior frozen selection and full RGB inference on accepted data")
        freeze = json.loads(input_path(root, selection).read_text())
        if (freeze.get("split_protocol_binding") != split_protocol_binding
                or freeze.get("schema_version") != "action4.final_selection.v1"
                or freeze["checkpoint_sha256"] != sha256_file(checkpoint_path)
                or freeze["inference_contract_sha256"] != canonical_hash(inference_contract(config))
                or freeze["test_manifest_sha256"] != sha256_file(manifest)
                or freeze["code"]["sha256"] != source_identity(root)["sha256"]):
            raise ValueError("Test inputs differ from frozen model/config/data/code identity")
    output = output_path(root, Path("outputs/action4") / run_id)
    if output.exists():
        raise FileExistsError("Evaluation requires a new run_id")
    output.mkdir(parents=True)
    device = select_device(config.get("runtime", {}).get("device", "auto"))
    dataset = Action4Dataset(root, manifest, allow_diagnostic=allow_diagnostic, split=split)
    model = build_model(config, checkpoint=checkpoint_path, device=device)
    identity = {"split_protocol_binding": split_protocol_binding, "config": config, "checkpoint_sha256": sha256_file(checkpoint_path),
                "manifest_sha256": sha256_file(manifest), "code": source_identity(root),
                "environment": environment_identity(torch), "split": split, "diagnostic_data_allowed": allow_diagnostic,
                "model_claim_scope": checkpoint_state.get("claim_scope")}
    write_json_new(output / "run_manifest.json", identity)
    if gt_roi:
        report = evaluate_gt_rois(model, dataset, config, device, batch_size=config.get("eval", {}).get("batch_size", 1))
        name = "gt_roi_diagnostic.json"
    else:
        report, predictions = evaluate_records(model, dataset, config, device, batch_size=config.get("eval", {}).get("batch_size", 1))
        write_predictions(output / "predictions_raw.jsonl", predictions)
        name = f"{split}_metrics.json"
    report.update({"checkpoint_sha256": identity["checkpoint_sha256"], "manifest_sha256": identity["manifest_sha256"],
                   "status": "complete", "diagnostic_data_allowed": allow_diagnostic,
                   "model_claim_scope": identity["model_claim_scope"]})
    write_json_new(output / name, report)
    return {"output_dir": str(output), "report": report}


def tune_decode(root: str | Path, config: dict, checkpoint: str | Path, grid: dict, *, run_id: str,
                allow_diagnostic: bool = False) -> dict:
    """Explicit bounded eval-only parameter grid; every trial is retained."""
    from data.action4_dataset import Action4Dataset
    root = checked_root(root)
    validate_run_id(run_id)
    permitted = {"score_threshold", "duplicate_iou", "duplicate_center_ratio"}
    if not grid or set(grid) - permitted or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Grid must explicitly list supported detector thresholds/duplicate rules")
    for key, values in grid.items():
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError("Tuning grid values must be finite numbers")
            if key in ("score_threshold", "duplicate_iou") and not 0 <= value <= 1 or key == "duplicate_center_ratio" and value < 0:
                raise ValueError("Tuning parameter outside supported bounds")
    names = sorted(grid)
    combinations = list(itertools.product(*(grid[name] for name in names)))
    if len(combinations) > 12 or len(set(combinations)) != len(combinations):
        raise ValueError("Declare at most 12 unique trials; increasing the search budget needs an explicit new protocol")
    path = input_path(root, checkpoint)
    manifest = input_path(root, config["data"]["eval_manifest"])
    checkpoint_state = load_checkpoint(path)
    _validate_split_lineage(checkpoint_state, manifest, "eval")
    dataset = Action4Dataset(root, manifest, split="eval", allow_diagnostic=allow_diagnostic)
    output = output_path(root, Path("outputs/action4") / run_id)
    if output.exists():
        raise FileExistsError("Threshold tuning requires a fresh run_id")
    output.mkdir(parents=True)
    device = select_device(config.get("runtime", {}).get("device", "auto"))
    model = build_model(config, checkpoint=path, device=device)
    write_json_new(output / "run_manifest.json", {"scope": "fixed_eval_decode_selection", "split": "eval", "grid": grid,
                   "trial_count": len(combinations), "checkpoint_sha256": sha256_file(path),
                   "manifest_sha256": sha256_file(manifest), "code": source_identity(root),
                   "model_claim_scope": checkpoint_state.get("claim_scope"), "diagnostic_data_allowed": allow_diagnostic})
    trials = []
    for index, combination in enumerate(combinations):
        trial_config = copy.deepcopy(config)
        trial_config["decode"].update(dict(zip(names, combination)))
        report, predictions = evaluate_records(model, dataset, trial_config, device,
                                                 batch_size=config.get("eval", {}).get("batch_size", 1))
        directory = output / f"trial_{index:02d}"
        directory.mkdir()
        write_json_new(directory / "config.json", trial_config)
        write_json_new(directory / "eval_metrics.json", report)
        write_predictions(directory / "predictions_raw.jsonl", predictions)
        key = selection_key(report)
        trials.append({"trial_id": index, "parameters": dict(zip(names, combination)),
                       "selection_key": [value if np.isfinite(value) else -1e30 for value in key],
                       "report": str((directory / "eval_metrics.json").relative_to(root))})
    best = max(trials, key=lambda item: tuple(item["selection_key"]))
    selected = copy.deepcopy(config)
    selected["decode"].update(best["parameters"])
    write_json_new(output / "selected_config.json", selected)
    report = {"status": "complete_eval_selection_pending_video_review", "split": "eval", "selected_trial": best,
              "trials": trials, "test_read_or_scored": False,
              "selected_config": str(output / "selected_config.json"), "output_dir": str(output)}
    write_json_new(output / "tuning_report.json", report)
    return report
