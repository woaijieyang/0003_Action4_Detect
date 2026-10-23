"""Frozen-model diagnostic on explicit candidate training records, never dataset admission."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R, contained_path
from eval.action4_bbox_metrics import score_boxes
from eval.action4_staged_pipeline import load_pipeline
from model.training.action4_runtime import atomic_json

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def load_candidate(ref):
    path = contained_path(R, ref["path"], exists=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("Candidate record changed since benchmark completion")
    row = json.loads(raw)
    if row["split"] != "train" or row["status"] != "diagnostic_render_not_training_accepted":
        raise ValueError("This diagnostic only accepts explicitly non-admitted training previews")
    DataAdapter(R).validate_sample(row)
    cad = np.asarray(row["corners_3d_mm"], dtype=float) / 1000
    frozen = np.asarray(json.loads((R / "data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"]) / 1000
    if not np.array_equal(cad, frozen):
        raise ValueError("Candidate canonical8 identity/order differs from frozen body")
    K = np.asarray(row["camera"]["K"])
    for obj in row["instances"]:
        T = np.asarray(obj["canonical8_geometry"]["object_to_capture_cv_m"])
        camera = cad @ T[:3, :3].T + T[:3, 3]
        h = camera @ K.T
        if (camera[:, 2] <= 0).any() or not np.allclose(h[:, :2] / h[:, 2:], obj["corners_px"], atol=.005, rtol=0):
            raise ValueError("Candidate fixed-ID projection mismatch")
    return row


def overlay(rgb, objects, prediction):
    canvas = rgb.copy()
    draw = ImageDraw.Draw(canvas)
    for obj in objects:
        box = obj["bbox_full_xyxy"]
        draw.rectangle(tuple(box), outline="lime" if prediction else "white", width=5)
        points = obj.get("corners_px")
        if points is None:
            continue
        points = np.asarray(points)
        for a, b in EDGES:
            draw.line([tuple(points[a]), tuple(points[b])], fill="red" if prediction else "white", width=4)
        for i, p in enumerate(points):
            draw.text(tuple(p), str(i), fill="yellow")
    return canvas


def run(config, output):
    cfg = json.loads(contained_path(R, config, exists=True).read_text())
    records = [load_candidate(ref) for ref in cfg["records"]]
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate candidate sample")
    out = contained_path(R, output, exists=False)
    if not out.is_relative_to(R / "outputs"):
        raise ValueError("Diagnostic output must be under outputs")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    pipeline = load_pipeline(cfg["selection"])
    outputs = []
    for row in records:
        with Image.open(contained_path(R, row["image"], exists=True)) as image:
            rgb = image.convert("RGB")
        tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1)
        result = pipeline.predict(tensor, row["camera"]["K"],
                                  np.asarray(row["corners_3d_mm"]) / 1000,
                                  return_box_candidates=True)
        outputs.append({"sample_id": row["sample_id"], **result})
        left = overlay(rgb, row["instances"], False)
        right = overlay(rgb, result["instances"], True)
        panels = Image.new("RGB", (1600, 637), "black")
        for i, panel in enumerate([left, right]):
            panel.thumbnail((800, 607))
            panels.paste(panel, (i * 800, 30))
        draw = ImageDraw.Draw(panels)
        draw.text((10, 8), row["sample_id"] + " | geometric labels (candidate)", fill="white")
        draw.text((810, 8), "Frozen best B0/A0 predictions | no GT inputs", fill="white")
        panels.save(out / (row["sample_id"] + ".jpg"), quality=95)
    boxes = score_boxes([{"sample_id": r["sample_id"], "instances": r["box_candidates"]}
                         for r in outputs], records, .3)
    corner_rows = []
    for gt, predicted, match in zip(records, outputs, boxes["per_image"]):
        if gt["sample_id"] != match["sample_id"]:
            raise ValueError("Metric/record ordering mismatch")
        objects = [g for g in gt["instances"] if not g.get("ignore", False)]
        for pi, gi in match["matches"]:
            g, p = objects[gi], predicted["instances"][pi]
            errors = None
            if p["pose_valid"] and p["corners_px"] is not None:
                diagonal = np.linalg.norm(np.asarray(g["bbox_full_xyxy"])[2:] - g["bbox_full_xyxy"][:2])
                errors = (np.linalg.norm(np.asarray(p["corners_px"]) - g["corners_px"], axis=1) / diagonal).tolist()
            corner_rows.append({"sample_id": gt["sample_id"], "instance_id": g["instance_id"],
                                "errors_normalized": errors, "visibility": g["corner_visible"],
                                "occlusion": g["occlusion_inframe"],
                                "all8": errors is not None and max(errors) <= .05})
    report = {"scope": "Candidate TRAIN previews only; not validation, test, or publication approval",
              "identity": pipeline.identity, "images": len(records), "bbox_only": boxes,
              "GT_targets": sum(not g.get("ignore", False) for r in records for g in r["instances"]),
              "fixedID_all8_success": sum(r["all8"] for r in corner_rows),
              "corner_rows": corner_rows, "candidate_training_allowed": False,
              "GT_input_policy": "RGB, intrinsics, frozen canonical geometry only; target boxes/masks/poses used for scoring/overlay only",
              "record_sources": cfg["records"], "model_weights_changed": False}
    atomic_json(out / "PREDICTIONS.json", outputs)
    atomic_json(out / "REPORT.json", report)
    print(json.dumps({"images": report["images"], "box": boxes["overall"],
                      "all8": report["fixedID_all8_success"], "GT": report["GT_targets"]}))


def gt_roi_control(config, baseline, output):
    """Oracle crop diagnostic only: isolate A0 from B0 crop error, never deployment."""
    from itertools import permutations
    from data.action4_pose_rgb_batch import prepare_pose_rgb

    cfg = json.loads(contained_path(R, config, exists=True).read_text())
    records = [load_candidate(ref) for ref in cfg["records"]]
    reference = json.loads(contained_path(R, Path(baseline) / "REPORT.json", exists=True).read_text())
    by_id = {(r["sample_id"], r["instance_id"]): r for r in reference["corner_rows"]}
    out = contained_path(R, output, exists=False)
    if not out.is_relative_to(R / "outputs"):
        raise ValueError("Diagnostic output must be under outputs")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    pipeline = load_pipeline(cfg["selection"])
    if pipeline.identity != reference["identity"]:
        raise ValueError("Oracle control must use the same frozen models")
    assignments = np.asarray(list(permutations(range(8))))
    rows = []
    for record in records:
        with Image.open(contained_path(R, record["image"], exists=True)) as image:
            rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
        items = [{"image": rgb, "record": record, "instance": j}
                 for j in range(len(record["instances"]))]
        crops, geometry, metadata = prepare_pose_rgb(items, device=pipeline.device)
        # Only crop intrinsics and fixed CAD are inputs; R/t and target points never enter forward.
        with torch.inference_mode():
            predicted = pipeline.a0(pipeline.base.backbone(crops), geometry["K"], geometry["canonical"])
        for j, obj in enumerate(record["instances"]):
            native = DataAdapter.transform_points(predicted["corners"][j].cpu().numpy(),
                                                  np.linalg.inv(metadata[j]["affine_hm"]))
            gt = np.asarray(obj["corners_px"])
            camera = predicted["camera_corners"][j].cpu().numpy()
            if not np.isfinite(native).all() or not np.isfinite(camera).all() or (camera[:, 2] <= 0).any():
                raise ValueError("Oracle diagnostic produced invalid pose; preserve failure instead of reassigning corners")
            diagonal = np.linalg.norm(np.asarray(obj["bbox_full_xyxy"])[2:] - obj["bbox_full_xyxy"][:2])
            errors = np.linalg.norm(native - gt, axis=1) / diagonal
            distances = np.linalg.norm(native[:, None] - gt[None], axis=2) / diagonal
            maxima = distances[np.arange(8)[None], assignments].max(axis=1)
            best = int(np.argmin(maxima))
            prior = by_id.get((record["sample_id"], obj["instance_id"]))
            rows.append({"sample_id": record["sample_id"], "instance_id": obj["instance_id"],
                         "predicted_ROI_errors": prior["errors_normalized"] if prior else None,
                         "oracle_ROI_errors": errors.tolist(), "oracle_all8": bool(errors.max() <= .05),
                         "unordered_best_max_error": float(maxima[best]),
                         "unordered_best_permutation": assignments[best].tolist(),
                         "unordered_is_diagnostic_only": True,
                         "visibility": obj["corner_visible"]})
    atomic_json(out / "REPORT.json", {
        "scope": "Oracle GT ROI diagnostic only; never B0 detection or end-to-end score",
        "identity": pipeline.identity, "rows": rows,
        "predicted_ROI_all8": reference["fixedID_all8_success"],
        "oracle_ROI_all8": sum(r["oracle_all8"] for r in rows),
        "GT_targets": len(rows),
        "unordered_oracle_pass": sum(r["unordered_best_max_error"] <= .05 for r in rows),
        "weights_changed": False})
    print(json.dumps({"oracle_ROI_all8": sum(r["oracle_all8"] for r in rows),
                      "unordered_oracle_pass": sum(r["unordered_best_max_error"] <= .05 for r in rows),
                      "targets": len(rows)}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt-roi-baseline", help="Explicit oracle control using a prior frozen baseline directory")
    args = parser.parse_args()
    if args.gt_roi_baseline:
        gt_roi_control(args.config, args.gt_roi_baseline, args.output)
    else:
        run(args.config, args.output)
