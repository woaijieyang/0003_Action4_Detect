"""Diagnostic mask motion evidence; never a tracking update or physical-motion proof.

All thresholds are caller supplied, uncalibrated diagnostic settings. Pixel axes
are (x, y). Alignment uses rounded centroid displacement without wraparound;
its IoU retains off-image pixels in the union denominator to penalize clipping.
External instance identity, frame correspondence and amodal extent cannot be
established from two visible masks, including copied/shared masks.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class MotionThresholds:
    max_area_fraction_change: float
    min_aligned_iou: float
    min_subset_fraction: float
    max_b0_delta_disagreement_px: float

    def __post_init__(self):
        for key in ("max_area_fraction_change", "min_aligned_iou", "min_subset_fraction"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite fraction")
        value = self.max_b0_delta_disagreement_px
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative B0 disagreement threshold required")


@dataclass(frozen=True)
class B0MotionEvidence:
    """Caller-reported complete-body box-center delta over the SAME frame pair.

    These flags are provenance assertions, not independently checked identities.
    No mask bounding box may be substituted for either complete-body B0 box.
    """
    delta_xy: tuple
    same_instance_verified: bool
    same_frame_pair_verified: bool
    complete_body_boxes_verified: bool

    def __post_init__(self):
        delta = np.asarray(self.delta_xy, dtype=float)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError("B0 delta must be finite (x, y)")
        object.__setattr__(self, "delta_xy", tuple(float(v) for v in delta))
        for key in ("same_instance_verified", "same_frame_pair_verified", "complete_body_boxes_verified"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("B0 provenance flags must be bool")


def _intersection_after_shift(previous, current, dx, dy):
    h, w = previous.shape
    x0, x1 = max(0, -dx), min(w, w - dx)
    y0, y1 = max(0, -dy), min(h, h - dy)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(
        previous[y0:y1, x0:x1] & current[y0+dy:y1+dy, x0+dx:x1+dx]))


def diagnose_mask_motion(previous, current, *, thresholds, b0_motion=None):
    """Return JSON-compatible evidence, without changing masks or any tracker.

    translation_shape_compatible is a shape diagnostic, NOT an admission gate.
    direct_centroid_translation_sufficient stays False: this uncalibrated module
    cannot authenticate identity or infer rigid physical motion from visibility.
    """
    if not isinstance(thresholds, MotionThresholds):
        raise TypeError("Explicit MotionThresholds required")
    if b0_motion is not None and not isinstance(b0_motion, B0MotionEvidence):
        raise TypeError("B0MotionEvidence or None required")
    for mask in (previous, current):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("Original-image two-dimensional boolean masks required")
    if previous.shape != current.shape:
        raise ValueError("Masks must share the same original-image pixel dimensions")
    a, b = int(np.count_nonzero(previous)), int(np.count_nonzero(current))
    result = {
        "schema": "action4.mask_motion_evidence.v1",
        "thresholds": asdict(thresholds), "thresholds_calibrated": False,
        "previous_area_px": a, "current_area_px": b,
        "area_ratio_current_over_previous": b / a if a else None,
        "area_fraction_change": abs(b-a) / max(a, b) if max(a, b) else None,
        "centroid_delta_xy_px": None, "rounded_alignment_delta_xy_px": None,
        "unshifted_iou": None, "aligned_iou": None,
        "current_subset_of_previous_fraction": None,
        "previous_subset_of_current_fraction": None,
        "subset_change_suspected": None, "visible_support_change_suspected": None,
        "translation_shape_compatible": False,
        "b0_motion": asdict(b0_motion) if b0_motion is not None else None,
        "b0_delta_disagreement_px": None, "b0_corroborates_delta": None,
        "direct_centroid_translation_sufficient": False,
        "physical_motion": "unknown", "external_identity_verified_by_module": False,
        "unknown_reasons": ["uncalibrated_diagnostic_not_admission",
                            "mask_pair_cannot_authenticate_instance_or_frame_identity",
                            "visible_support_does_not_determine_rigid_body_motion"],
    }
    if b0_motion is None:
        result["unknown_reasons"].append("no_current_complete_body_b0_motion_evidence")
    if not a or not b:
        result["unknown_reasons"].append("empty_previous_or_current_mask")
        return result
    py, px = np.nonzero(previous)
    cy, cx = np.nonzero(current)
    delta = np.array([cx.mean()-px.mean(), cy.mean()-py.mean()])
    dx, dy = (int(v) for v in np.rint(delta))
    overlap = int(np.count_nonzero(previous & current))
    aligned = _intersection_after_shift(previous, current, dx, dy)
    aligned_iou = aligned / (a+b-aligned)
    area_ok = result["area_fraction_change"] <= thresholds.max_area_fraction_change
    subset = a != b and overlap / min(a, b) >= thresholds.min_subset_fraction
    shape_ok = area_ok and aligned_iou >= thresholds.min_aligned_iou and not subset
    result.update(
        centroid_delta_xy_px=delta.tolist(),
        rounded_alignment_delta_xy_px=[dx, dy],
        unshifted_iou=overlap/(a+b-overlap), aligned_iou=aligned_iou,
        current_subset_of_previous_fraction=overlap/b,
        previous_subset_of_current_fraction=overlap/a,
        subset_change_suspected=bool(subset),
        visible_support_change_suspected=bool(not area_ok or aligned_iou < thresholds.min_aligned_iou or subset),
        translation_shape_compatible=bool(shape_ok))
    if not shape_ok:
        result["unknown_reasons"].append("visibility_or_shape_change_confounds_centroid_motion")
    if np.any(delta != np.rint(delta)):
        result["unknown_reasons"].append("alignment_quantized_to_integer_pixels")
    if b0_motion is not None:
        distance = float(np.linalg.norm(delta - np.asarray(b0_motion.delta_xy)))
        provenance = (b0_motion.same_instance_verified and b0_motion.same_frame_pair_verified
                      and b0_motion.complete_body_boxes_verified)
        result["b0_delta_disagreement_px"] = distance
        result["b0_corroborates_delta"] = bool(provenance and distance <= thresholds.max_b0_delta_disagreement_px)
        if not provenance:
            result["unknown_reasons"].append("b0_provenance_not_confirmed_by_caller")
        elif distance > thresholds.max_b0_delta_disagreement_px:
            result["unknown_reasons"].append("b0_complete_body_motion_conflicts_with_mask_centroid")
    return result
