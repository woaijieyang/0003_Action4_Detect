"""Stable focal and project-specific image-normalized corner losses.

Focal/gather mathematical reference: CenterNet @
4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c, src/lib/models/losses.py
(_neg_loss, _reg_loss) and utils.py (_transpose_and_gather_feat).
Copyright (c) 2019 Xingyi Zhou, MIT. CenterNet also attributes its focal
formulation to CornerNet, copyright (c) 2018 University of Michigan, BSD-3.
This independently written implementation takes logits, uses FP32 logsigmoid,
averages complete images and supports ignore regions/zero-instance images.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _check_unit_targets(target: Tensor, name: str) -> None:
    if not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError(f"{name} must contain finite values in [0,1]")


def focal_center_loss(logits: Tensor, target: Tensor, negative_mask: Tensor | None = None) -> Tensor:
    """CornerNet/CenterNet focal, alpha=2 beta=4, positive priority over ignore.

    ``negative_mask=True`` enables a negative location. Positive target==1 is
    always supervised. Each complete image divides by max(num_positive,1),
    then images are averaged; there is no global ROI-count normalization.
    """
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("center logits and targets must share shape [B,1,H,W]")
    if logits.shape[0] == 0:
        raise ValueError("Complete-image batch must be nonempty")
    _check_unit_targets(target, "center targets")
    target = target.float()
    logits = logits.float()
    positive = target.eq(1)
    negative = target.lt(1)
    if negative_mask is not None:
        if negative_mask.shape != target.shape or negative_mask.dtype != torch.bool:
            raise ValueError("negative_mask must be boolean with the center target shape")
        negative = negative & negative_mask
    probability = torch.sigmoid(logits)
    positive_term = -F.logsigmoid(logits) * (1 - probability).square() * positive
    negative_term = -F.logsigmoid(-logits) * probability.square() * (1 - target).pow(4) * negative
    denominator = positive.flatten(1).sum(1).clamp_min(1)
    return ((positive_term + negative_term).flatten(1).sum(1) / denominator).mean()


def gather_at_indices(feature: Tensor, indices: Tensor) -> Tensor:
    """[B,C,H,W] plus valid integer [B,M] flat indices -> [B,M,C]."""
    if feature.ndim != 4 or indices.ndim != 2 or feature.shape[0] != indices.shape[0]:
        raise ValueError("Expected a BCHW feature and BM index tensor")
    if indices.dtype != torch.long:
        raise ValueError("indices must use torch.long")
    if bool(((indices < 0) | (indices >= feature.shape[-2] * feature.shape[-1])).any()):
        raise ValueError("A flat index is outside the feature grid")
    flattened = feature.flatten(2).transpose(1, 2)
    return flattened.gather(1, indices[..., None].expand(-1, -1, feature.shape[1]))


def _masked_regression(feature: Tensor, target: Tensor, indices: Tensor, mask: Tensor) -> Tensor:
    if target.shape != (*indices.shape, feature.shape[1]):
        raise ValueError("Regression target shape must be [B,M,C]")
    if not bool(torch.isfinite(target[mask]).all()):
        raise ValueError("Valid regression targets must be finite")
    safe_indices = torch.where(mask, indices, torch.zeros_like(indices))
    predicted = gather_at_indices(feature, safe_indices).float()
    safe_target = torch.where(mask[..., None], target, torch.zeros_like(target)).float()
    per_instance = F.smooth_l1_loss(predicted, safe_target, beta=1.0, reduction="none").mean(-1)
    per_image = (per_instance * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_image.mean()


def detection_losses(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return scalar ``center``, ``offset``, ``box`` losses.

    Targets: center[B,1,H,W], indices[B,M] int64, instance_mask[B,M] bool,
    offset[B,M,2], ltrb[B,M,4] normalized by full-image W/H, optional boolean
    negative_mask[B,1,H,W]. Padded indices may be -1; valid indices may not.
    Same-cell valid targets are rejected rather than silently overwritten.
    """
    center = outputs["center_logits"]
    mask, indices = targets["instance_mask"], targets["indices"]
    if mask.dtype != torch.bool or indices.dtype != torch.long or mask.ndim != 2 or mask.shape != indices.shape:
        raise ValueError("instance_mask and int64 indices must share shape [B,M]")
    if mask.shape[0] != center.shape[0]:
        raise ValueError("Target batch size does not match outputs")
    if outputs["offset"].shape != (center.shape[0], 2, *center.shape[-2:]):
        raise ValueError("offset must have shape [B,2,H,W]")
    if outputs["ltrb"].shape != (center.shape[0], 4, *center.shape[-2:]):
        raise ValueError("ltrb must have shape [B,4,H,W]")
    for image_indices in (indices[i][mask[i]] for i in range(mask.shape[0])):
        if image_indices.unique().numel() != image_indices.numel():
            raise ValueError("Two supervised instances collide on one detector grid cell")
    if bool((targets["ltrb"][mask] < 0).any()):
        raise ValueError("Target ltrb distances must be nonnegative")
    safe_indices = torch.where(mask, indices, torch.zeros_like(indices))
    peaks = gather_at_indices(targets["center"], safe_indices)[..., 0]
    if bool((peaks[mask] != 1).any()):
        raise ValueError("Each supervised integer anchor must have center target exactly one")
    if not torch.equal(targets["center"].eq(1).flatten(1).sum(1), mask.sum(1)):
        raise ValueError("Center positive peaks and valid instance anchors disagree")
    return {
        "center": focal_center_loss(center, targets["center"], targets.get("negative_mask")),
        "offset": _masked_regression(outputs["offset"], targets["offset"], indices, mask),
        "box": _masked_regression(outputs["ltrb"], targets["ltrb"], indices, mask),
    }


def corner_heatmap_loss(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    roi_batch_index: Tensor,
    batch_size: int,
    *,
    peak_threshold: float = 0.1,
) -> Tensor:
    """Balanced SmoothL1: pixels -> valid corners -> valid ROIs -> images.

    Occluded cuboid corners are valid when their projection lies on the ROI
    canvas. ``valid`` denotes supervised projection/domain validity, not the
    physically visible metadata. Empty foreground/background regions have no
    weight; the remaining region receives weight one. Images without any
    valid ROI contribute a graph-connected zero and remain in the final mean.
    """
    if logits.ndim != 4 or logits.shape[1] != 8 or target.shape != logits.shape:
        raise ValueError("corner logits and targets must share shape [N,8,H,W]")
    if valid.shape != logits.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [N,8]")
    if (roi_batch_index.shape != (logits.shape[0],) or roi_batch_index.dtype != torch.long
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("roi_batch_index must be int64 [N] and batch_size positive")
    if not 0 < peak_threshold < 1:
        raise ValueError("peak_threshold must be inside (0,1)")
    if bool(((roi_batch_index < 0) | (roi_batch_index >= batch_size)).any()):
        raise ValueError("An ROI refers to an image outside this batch")
    if logits.shape[0] == 0:
        return logits.float().sum() * 0
    _check_unit_targets(target[valid], "valid corner heatmaps")
    safe_target = torch.where(valid[..., None, None], target, torch.zeros_like(target)).float()
    error = F.smooth_l1_loss(logits.float().sigmoid(), safe_target, beta=1.0, reduction="none")
    foreground = safe_target >= peak_threshold
    background = ~foreground
    foreground_count = foreground.flatten(2).sum(-1)
    background_count = background.flatten(2).sum(-1)
    foreground_mean = (error * foreground).flatten(2).sum(-1) / foreground_count.clamp_min(1)
    background_mean = (error * background).flatten(2).sum(-1) / background_count.clamp_min(1)
    region_count = (foreground_count > 0).to(error.dtype) + (background_count > 0).to(error.dtype)
    per_corner = (foreground_mean + background_mean) / region_count.clamp_min(1)
    valid_count = valid.sum(1)
    per_roi = (per_corner * valid).sum(1) / valid_count.clamp_min(1)
    image_sum = error.new_zeros(batch_size).scatter_add(0, roi_batch_index, per_roi)
    image_count = error.new_zeros(batch_size).scatter_add(0, roi_batch_index, (valid_count > 0).to(error.dtype))
    return (image_sum / image_count.clamp_min(1)).mean()


def joint_loss(
    outputs: Mapping[str, Tensor],
    detection_targets: Mapping[str, Tensor],
    corner_targets: Tensor,
    corner_valid: Tensor,
    roi_batch_index: Tensor,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Default four unit-weight terms; the training runner owns their schedule."""
    losses = detection_losses(outputs, detection_targets)
    losses["heatmap"] = corner_heatmap_loss(
        outputs["corner_logits"], corner_targets, corner_valid, roi_batch_index,
        outputs["center_logits"].shape[0],
    )
    coefficients = {"center": 1.0, "offset": 1.0, "box": 1.0, "heatmap": 1.0}
    if weights is not None:
        if set(weights) - set(coefficients):
            raise ValueError("Unknown joint-loss weight")
        coefficients.update(weights)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in coefficients.values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    losses["total"] = sum(losses[key] * coefficients[key] for key in coefficients)
    return losses
