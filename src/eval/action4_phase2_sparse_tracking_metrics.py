"""Sparse, query-aware tracking metrics; annotation arrays are scorer-only inputs."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean {shape}")
    return a


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _summary(values):
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "p95": None}
    return {"count": len(values), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95))}


@dataclass(frozen=True)
class VisibleBoxScale:
    """Only independently annotated visible-box diagonals, never inferred envelopes."""
    diagonal_px: object
    annotation_identity: str
    source: str = "independent_visible_bbox_annotation"

    def __post_init__(self):
        if self.source != "independent_visible_bbox_annotation" or not self.annotation_identity:
            raise ValueError("independent visible-box annotation provenance required")


def score_sparse(gt_uv, prediction_uv, annotation_valid, gt_visible, is_query,
                 prediction_available, accepted, predicted_visible, visibility_response_available, *, image_width,
                 image_height, visible_box_scale=None, domain_override=None):
    """All arrays share arbitrary point/frame shape S, UV has shape S+(2,).

    P includes finite coordinates. Accepted O must be a subset of P. Missing raw
    coordinates remain failures in G. TAP FP includes claimed-visible missing UV.
    domain_override may further exclude registered queries for a common report.
    """
    gt, pred = np.asarray(gt_uv, float), np.asarray(prediction_uv, float)
    if gt.ndim < 2 or gt.shape[-1] != 2 or pred.shape != gt.shape:
        raise ValueError("aligned GT/prediction XY required")
    shape = gt.shape[:-1]
    A = _mask(annotation_valid, shape, "annotation_valid")
    V = _mask(gt_visible, shape, "gt_visible")
    Q = _mask(is_query, shape, "is_query")
    P = _mask(prediction_available, shape, "prediction_available") & np.isfinite(pred).all(-1)
    O = _mask(accepted, shape, "accepted")
    PV = _mask(predicted_visible, shape, "predicted_visible")
    R = _mask(visibility_response_available, shape, "visibility_response_available")
    if (O & (~P | ~R)).any():
        raise ValueError("accepted rows require finite coordinates and a valid visibility response")
    if not np.isfinite(gt[A & V]).all():
        raise ValueError("annotated visible rows require finite GT")
    if any(type(x) is not int or x < 1 for x in (image_width, image_height)):
        raise ValueError("positive integer full native image dimensions required")
    M = A & ~Q
    if domain_override is not None:
        domain = _mask(domain_override, shape, "domain_override")
        if (domain & ~M).any():
            raise ValueError("common domain must be subset of own nonquery annotation domain")
        M = domain
    G = M & V
    count = int(G.sum())
    error = np.full(shape, np.nan)
    error[G & P] = np.linalg.norm(pred[G & P] - gt[G & P], axis=-1)
    accepted_visible = G & O
    invisible = M & ~V
    native = {str(d): _ratio(int((G & P & (error < d)).sum()), count) for d in (2, 5, 10)}
    tap_error = np.full(shape, np.nan)
    tap_error[G & P] = np.linalg.norm((pred[G & P] - gt[G & P]) * [256 / image_width, 256 / image_height], axis=-1)
    taps = {}
    for threshold in (1, 2, 4, 8, 16):
        within = P & (tap_error < threshold)
        tp = int((G & R & PV & within).sum())
        fp = int((M & R & PV & (~V | ~within)).sum())
        taps[str(threshold)] = {"pck": _ratio(int((G & within).sum()), count),
                                "tp": tp, "fp": fp, "jaccard": _ratio(tp, count + fp)}
    jaccards = [v["jaccard"] for v in taps.values()]
    normalized = {"status": "not_measured", "reason": "independent_visible_bbox_absent"}
    if visible_box_scale is not None:
        if not isinstance(visible_box_scale, VisibleBoxScale):
            raise ValueError("VisibleBoxScale required")
        diagonal = np.asarray(visible_box_scale.diagonal_px, float)
        if diagonal.shape != shape:
            raise ValueError("visible box diagonal must align to annotation rows")
        valid = np.isfinite(diagonal) & (diagonal > 0)
        mask = G & P & valid
        norm = error[mask] / diagonal[mask]
        am = accepted_visible & valid
        normalized = {"status": "measured" if mask.any() else "not_measured",
                      "annotation_identity": visible_box_scale.annotation_identity,
                      "scale_coverage": _ratio(int((G & valid).sum()), count),
                      "epe_over_visible_diagonal": _summary(norm),
                      "accepted_r50": _summary(error[am] / np.maximum(2., .005 * diagonal[am])),
                      "accepted_r95": _summary(error[am] / np.maximum(5., .02 * diagonal[am]))}
    return {"status": "measured" if M.any() and count else "insufficient_evidence",
            "domain_count": int(M.sum()), "visible_count": count, "known_invisible_count": int(invisible.sum()),
            "query_excluded_count": int((A & Q).sum()), "unknown_count": int((~A).sum()),
            "full_annotation_visible_count": int((A & V).sum()),
            "full_annotation_nonquery_accepted_coverage": _ratio(int((A & V & ~Q & O).sum()), int((A & V).sum())),
            "raw": {"epe_px": _summary(error[G & P]), "coverage": _ratio(int((G & P).sum()), count),
                    "missing_count": int((G & ~P).sum()), "pck": native},
            "accepted": {"epe_px": _summary(error[accepted_visible]),
                         "coverage": _ratio(int(accepted_visible.sum()), count),
                         "false_visible_count": int((invisible & O).sum()),
                         "false_visible_rate": _ratio(int((invisible & O).sum()), int(invisible.sum())),
                         "visible_precision": _ratio(int(accepted_visible.sum()), int((M & O).sum())),
                         "visible_recall": _ratio(int(accepted_visible.sum()), count)},
            "normalized": normalized,
            "tap_sparse": {"thresholds": taps,
                           "average_jaccard": float(np.mean(jaccards)) if all(x is not None for x in jaccards) else None,
                           "occlusion_accuracy": _ratio(int((M & R & (PV == V)).sum()), int((M & R).sum())),
                           "visibility_response_coverage": _ratio(int((M & R).sum()), int(M.sum())),
                           "missing_visibility_response_count": int((M & ~R).sum()),
                           "definition": "this_sequence_sparse_TAP_style_not_public_benchmark"}}


def compare_common_domain(annotation_ids, annotation_valid, gt_visible, gt_uv, groups,
                          runs, *, image_width, image_height, required_groups=(),
                          visible_box_scale=None, common_annotation_manifest_id,
                          expected_run_names=("C0", "C1", "C2")):
    """Score own and shared domains for aligned frozen (material identity, frame) IDs.

    Each run provides exactly ordered annotation_ids, registered_query_ids (requests,
    not corrected outputs), prediction_uv/available/accepted/predicted_visible.
    Query ID sets may contain requests outside the sparse annotation list.
    """
    if not isinstance(common_annotation_manifest_id, str) or not common_annotation_manifest_id:
        raise ValueError("frozen annotation manifest identity required")
    expected = tuple(expected_run_names)
    if not {"C0", "C1", "C2"}.issubset(expected) or len(set(expected)) != len(expected) or set(runs) != set(expected):
        raise ValueError("all expected runs required; represent failed runs with aligned missing outputs")
    ids = tuple(annotation_ids)
    n = len(ids)
    try:
        unique = set(ids)
    except TypeError as exc:
        raise ValueError("hashable full material/frame identities required") from exc
    if len(unique) != n or any(not isinstance(k, tuple) or len(k) != 4 or
                                not isinstance(k[0], str) or not k[0] or not isinstance(k[1], str) or not k[1] or
                                type(k[2]) is not int or k[2] < 0 or type(k[3]) is not int for k in ids):
        raise ValueError("IDs must be unique (track_id,anchor_id,generation,frame_id)")
    A = _mask(annotation_valid, (n,), "annotation_valid")
    V = _mask(gt_visible, (n,), "gt_visible")
    if len(groups) != n or not runs:
        raise ValueError("aligned groups and at least one run required")
    queries, requests = {}, {}
    for name, run in runs.items():
        if tuple(run["annotation_ids"]) != ids:
            raise ValueError(f"silent material/frame misalignment in {name}")
        requested = set(run["registered_query_ids"])
        if any(not isinstance(k, tuple) or len(k) != 4 or not isinstance(k[0], str) or not k[0] or
               not isinstance(k[1], str) or not k[1] or type(k[2]) is not int or k[2] < 0 or type(k[3]) is not int for k in requested):
            raise ValueError("invalid registered query identity")
        queries[name] = np.array([k in requested for k in ids], dtype=bool)
        requests[name] = sorted(requested)
    union = np.logical_or.reduce(list(queries.values()))
    common = A & ~union
    group_counts = {group: int(sum(m and g == group for m, g in zip(common, groups))) for group in sorted(set(groups))}
    visible_group_counts = {group: int(sum(m and v and g == group for m, v, g in zip(common, V, groups))) for group in sorted(set(groups))}
    empty_groups = [g for g in required_groups if visible_group_counts.get(g, 0) == 0]
    query_union_manifest_id = hashlib.sha256(json.dumps(requests, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity_payload = {"annotation_manifest": common_annotation_manifest_id, "query_manifest": query_union_manifest_id,
                        "ids": ids, "annotation_valid": A.tolist(), "query_union": union.tolist(), "mask": common.tolist()}
    mask_identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reports = {}
    for name, run in runs.items():
        kwargs = dict(gt_uv=gt_uv, prediction_uv=run["prediction_uv"], annotation_valid=A,
                      gt_visible=gt_visible, is_query=queries[name], prediction_available=run["prediction_available"],
                      accepted=run["accepted"], predicted_visible=run["predicted_visible"],
                      visibility_response_available=run["visibility_response_available"],
                      image_width=image_width, image_height=image_height, visible_box_scale=visible_box_scale)
        failures = run.get("system_seed_failure_counts")
        if failures is not None and (not isinstance(failures, dict) or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in failures.items())):
            raise ValueError("system/seed failure counts must be named nonnegative integer counts")
        reports[name] = {"own": score_sparse(**kwargs), "common": score_sparse(**kwargs, domain_override=common),
                         "system_seed_failure_counts": {"status": "provided", "counts": failures} if failures is not None else {"status": "not_measured", "counts": None}}
    return {"status": "insufficient_evidence" if not (common & V).any() or empty_groups else "measured",
            "expected_run_names": list(expected), "common_annotation_manifest_id": common_annotation_manifest_id,
            "query_union_manifest_id": query_union_manifest_id,
            "common_mask_identity": mask_identity, "common_mask": common.tolist(),
            "query_union": union.tolist(), "query_excluded_positions": [ids[i] for i in np.flatnonzero(A & union)],
            "query_excluded_fraction": _ratio(int((A & union).sum()), int(A.sum())),
            "common_count": int(common.sum()), "common_visible_count": int((common & V).sum()),
            "group_counts": group_counts, "common_visible_group_counts": visible_group_counts, "empty_required_groups": empty_groups,
            "per_run_common_missing": {k: v["common"]["raw"]["missing_count"] for k, v in reports.items()},
            "runs": reports}
