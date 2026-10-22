"""Eight ordered cuboid-corner metrics; labels never enter inference.

An instance passes only when all eight same-index corners pass a threshold
relative to the complete ground-truth cuboid box diagonal. Matching is maximum
cardinality, not greedy. Undefined aggregate ratios are JSON null.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from data.action4_dataset import is_ignored


def bbox_iou(first, second) -> float:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a, b]).all():
        raise ValueError("IoU requires finite xyxy boxes")
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def maximum_matching(edges: Sequence[Sequence[int]], right_count: int) -> list[tuple[int, int]]:
    """Deterministic augmenting-path maximum bipartite matching."""
    right_to_left: dict[int, int] = {}

    def augment(left: int, visited: set[int]) -> bool:
        for right in sorted(set(edges[left])):
            if not 0 <= right < right_count:
                raise ValueError("Matching endpoint outside graph")
            if right in visited:
                continue
            visited.add(right)
            if right not in right_to_left or augment(right_to_left[right], visited):
                right_to_left[right] = left
                return True
        return False

    for left in range(len(edges)):
        augment(left, set())
    return sorted((left, right) for right, left in right_to_left.items())


def _corners(instance: dict[str, Any]) -> np.ndarray:
    value = np.asarray(instance["corners_px"], dtype=float)
    if value.shape != (8, 2) or not np.isfinite(value).all():
        raise ValueError("Every predicted and reference instance needs eight finite ordered corners")
    return value


def score_image(predictions: list[dict], references: list[dict], threshold: float = .05,
                ignore_iou: float = .5) -> dict[str, Any]:
    if not 0 < threshold < 1 or not 0 <= ignore_iou <= 1:
        raise ValueError("Invalid scoring threshold")
    pred_corners = [_corners(p) for p in predictions]
    ref_corners = [_corners(g) for g in references]
    valid = [i for i, g in enumerate(references) if not is_ignored(g)]
    ignored = [i for i, g in enumerate(references) if is_ignored(g)]
    diagonals = []
    for i in valid:
        box = np.asarray(references[i]["bbox_full_xyxy"], dtype=float)
        diagonal = float(np.linalg.norm(box[2:] - box[:2]))
        if box.shape != (4,) or not math.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("Valid reference needs a nondegenerate complete box")
        diagonals.append(diagonal)
    edges = [[j for j, i in enumerate(valid)
              if np.all(np.linalg.norm(pc - ref_corners[i], axis=1) <= threshold * diagonals[j])]
             for pc in pred_corners]
    matches = [(p, valid[g]) for p, g in maximum_matching(edges, len(valid))]
    matched_predictions = {p for p, _ in matches}
    matched_references = {g for _, g in matches}
    remaining = [p for p in range(len(predictions)) if p not in matched_predictions]
    ignore_edges = [[j for j, g in enumerate(ignored)
                     if bbox_iou(predictions[p]["bbox_full_xyxy"], references[g]["bbox_full_xyxy"]) >= ignore_iou]
                    for p in remaining]
    ignore_matches = [(remaining[p], ignored[g])
                      for p, g in maximum_matching(ignore_edges, len(ignored))]
    ignored_predictions = {p for p, _ in ignore_matches}
    fp = [p for p in remaining if p not in ignored_predictions]
    fn = [g for g in valid if g not in matched_references]
    # Bad corners remain both FP and FN in the primary metric. This additional
    # diagnostic explains which failed instances were spatially localized.
    bad_edges = [[j for j, g in enumerate(fn)
                  if bbox_iou(predictions[p]["bbox_full_xyxy"], references[g]["bbox_full_xyxy"]) >= .5]
                 for p in fp]
    bad_matches = [(fp[p], fn[g]) for p, g in maximum_matching(bad_edges, len(fn))]
    errors = [float(e) for p, g in matches for e in np.linalg.norm(pred_corners[p] - ref_corners[g], axis=1)]
    return {
        "threshold": threshold, "tp": len(matches), "fp": len(fp), "fn": len(fn),
        "valid_gt": len(valid), "ignored_gt": len(ignored),
        "ignored_predictions": len(ignore_matches), "predictions": len(predictions),
        "matches": matches, "ignore_matches": ignore_matches,
        "false_positive_indices": fp, "false_negative_indices": fn,
        "bad_corner_matches": bad_matches, "bad_corner_instances": len(bad_matches),
        "missed_localization_instances": len(fn) - len(bad_matches),
        "epe_success_only_px": errors,
        "empty_valid_and_prediction": not valid and not predictions,
    }


def _summary(records: list[dict]) -> dict[str, Any]:
    counts = {k: sum(r[k] for r in records) for k in
              ("tp", "fp", "fn", "valid_gt", "ignored_gt", "ignored_predictions", "predictions",
               "bad_corner_instances", "missed_localization_instances")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    errors = [error for record in records for error in record["epe_success_only_px"]]
    return {**counts, "images": len(records),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "epe_success_only_mean_px": float(np.mean(errors)) if errors else None,
            "epe_success_only_p95_px": float(np.percentile(errors, 95)) if errors else None,
            "epe_success_only_corner_count": len(errors),
            "empty_valid_and_prediction_images": sum(r["empty_valid_and_prediction"] for r in records)}


def score_dataset(prediction_records: list[dict], reference_samples: list[dict]) -> dict[str, Any]:
    refs = {s["sample_id"]: s for s in reference_samples}
    preds = {p["sample_id"]: p for p in prediction_records}
    if len(refs) != len(reference_samples) or len(preds) != len(prediction_records):
        raise ValueError("Duplicate sample IDs in scoring inputs")
    if set(preds) != set(refs):
        raise ValueError("Predictions must cover every reference exactly once")
    result: dict[str, Any] = {"schema_version": "action4.metrics.v1", "thresholds": {},
                              "epe_scope": "successfully matched corners only; misses are FP/FN, never zero error"}
    for threshold in (.02, .05):
        rows, groups, strata = [], defaultdict(list), defaultdict(lambda: [0, 0])
        for identity in sorted(refs):
            sample = refs[identity]
            row = score_image(preds[identity]["instances"], sample["instances"], threshold)
            row["sample_id"] = identity
            rows.append(row)
            groups[sample["scene_family"]].append(row)
            hit = {g for _, g in row["matches"]}
            for gi, gt in enumerate(sample["instances"]):
                if is_ignored(gt):
                    continue
                box = gt["bbox_full_xyxy"]
                ratio = (box[2] - box[0]) / sample["width"]
                bins = ["scale/lt0.06" if ratio < .06 else "scale/0.06to0.12" if ratio < .12 else "scale/ge0.12"]
                occ = gt.get("occlusion_inframe")
                bins.append("occlusion/undefined" if occ is None else "occlusion/lt0.1" if occ < .1
                            else "occlusion/0.1to0.4" if occ < .4 else "occlusion/ge0.4")
                trunc = gt.get("truncation")
                bins.append("truncation/undefined" if trunc is None else "truncation/none" if trunc <= .001
                            else "truncation/partial")
                for name in bins:
                    strata[name][0] += 1
                    strata[name][1] += gi in hit
        result["thresholds"][f"{threshold:.2f}"] = {
            **_summary(rows), "per_scene_family": {name: _summary(values) for name, values in sorted(groups.items())},
            "per_gt_stratum": {name: {"valid_gt": n, "tp": h, "recall": h / n if n else None}
                                for name, (n, h) in sorted(strata.items())}, "per_image": rows}
    return result


def selection_key(report: dict[str, Any]) -> tuple[float, float, float]:
    loose, strict = report["thresholds"]["0.05"], report["thresholds"]["0.02"]
    return (loose["f1"] if loose["f1"] is not None else -1.,
            strict["f1"] if strict["f1"] is not None else -1.,
            -loose["epe_success_only_mean_px"] if loose["epe_success_only_mean_px"] is not None else -math.inf)
