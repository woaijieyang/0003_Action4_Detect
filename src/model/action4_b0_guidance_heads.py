"""Shared-appearance B0 guidance heads and separate explicit supervision sums.

Inference has no GT arguments. The caller owns BoxRefinement; this module never
registers or copies its appearance parameters. No training loop or model loading.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .action4_box_refinement import BoxRefinement, paired_iou_giou
from .action4_local_evidence import LocalEvidenceBatch

LOSS_KEYS = ("identity_positive", "identity_negative", "full_box_regression", "full_box_giou", "full_box_quality")
POSITIVE, BACKGROUND, UNKNOWN, IGNORE, MIXED = 1, 0, -1, -2, -3


def _ids(values):
    values = tuple(values)
    if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError("Unique explicit candidate IDs required")
    return values


def _tensor(value, shape, reference, label, *, boolean=False):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.device != reference.device:
        raise ValueError(label + " shape/device mismatch")
    if boolean:
        if value.dtype != torch.bool: raise ValueError(label + " must be bool")
    elif not value.is_floating_point() or value.dtype != reference.dtype or not torch.isfinite(value).all():
        raise ValueError(label + " finite floating dtype mismatch")


def shared_appearance(refiner, pooled, geometry):
    """Use the exact existing 128 appearance +16 geometry modules, with live gradients."""
    if not isinstance(refiner, BoxRefinement): raise ValueError("Existing BoxRefinement owner required")
    _tensor(pooled, (len(pooled), 2304), pooled, "9point pooled features")
    _tensor(geometry, (len(pooled), 4), pooled, "proposal geometry")
    visual = torch.cat((refiner.appearance.appearance(pooled),
                        refiner.appearance.geometry(geometry)), -1)
    _tensor(visual, (len(pooled), 144), pooled, "shared appearance")
    return visual


@dataclass(frozen=True)
class GuidanceOutput:
    candidate_ids: tuple
    identity_logits: torch.Tensor
    full_boxes: torch.Tensor
    quality_logits: torch.Tensor
    deltas: torch.Tensor
    original_boxes: torch.Tensor
    yolo_only: torch.Tensor

    @property
    def p_action4(self): return self.identity_logits.sigmoid()

    @property
    def q_full(self): return self.quality_logits.sigmoid()


class B0GuidanceHeads(nn.Module):
    """Only new heads: identity150->1 and prior-free YOLO144->128->(4,1)."""
    def __init__(self):
        super().__init__()
        self.identity_head = nn.Linear(150, 1)
        self.yolo_fusion = nn.Sequential(nn.Linear(144, 128), nn.ReLU())
        self.yolo_delta = nn.Linear(128, 4)
        self.yolo_quality = nn.Linear(128, 1)
        nn.init.zeros_(self.yolo_delta.weight)
        nn.init.zeros_(self.yolo_delta.bias)

    def forward(self, refiner, pooled, geometry, boxes, *, candidate_ids, evidence,
                yolo_only, b0_priors=None, b0_prior_ids=()):
        ids = _ids(candidate_ids); n = len(ids)
        _tensor(pooled, (n, 2304), self.identity_head.weight, "proposal features")
        _tensor(geometry, (n, 4), pooled, "proposal geometry")
        _tensor(boxes, (n, 4), pooled, "native candidate boxes")
        _tensor(yolo_only, (n,), pooled, "source route", boolean=True)
        if not ((boxes[:, 2:] - boxes[:, :2]) > 0).all():
            raise ValueError("Finite ordered native candidate boxes required")
        if not isinstance(evidence, LocalEvidenceBatch) or evidence.candidate_ids != ids:
            raise ValueError("Local evidence candidate IDs/order mismatch")
        ext = torch.as_tensor(evidence.values, device=pooled.device, dtype=pooled.dtype)
        _tensor(ext, (n, 6), pooled, "six local evidence features")
        if ((ext < 0) | (ext > 1)).any() or not ((ext[:, 3:] == 0) | (ext[:, 3:] == 1)).all():
            raise ValueError("Invalid local scores/availability")
        if (ext[:, :3][ext[:, 3:] == 0] != 0).any():
            raise ValueError("Unavailable scores must be zero; absence is not a negative label")
        visual = shared_appearance(refiner, pooled, geometry)
        logits = self.identity_head(torch.cat((visual, ext.detach()), -1)).squeeze(-1)
        boxes = boxes.detach()
        full = boxes.new_empty((n, 4)); deltas = boxes.new_empty((n, 4))
        quality = boxes.new_empty((n,))
        bi = torch.where(~yolo_only)[0]; yi = torch.where(yolo_only)[0]
        expected_prior_ids = tuple(ids[i] for i in bi.tolist())
        if _ids(b0_prior_ids) != expected_prior_ids:
            raise ValueError("B0 priors must align to B0-only stable IDs")
        if len(bi):
            _tensor(b0_priors, (len(bi), 3), pooled, "actual B0 priors")
            if ((b0_priors < 0) | (b0_priors > 1)).any(): raise ValueError("Invalid B0 priors")
            # Preserve the original B0 refiner exactly, including its real priors.
            rb, rq, rd = refiner(pooled[bi], geometry[bi], b0_priors, boxes[bi])
            full = full.index_copy(0, bi, rb)
            quality = quality.index_copy(0, bi, rq)
            deltas = deltas.index_copy(0, bi, rd)
        elif b0_priors is not None:
            raise ValueError("YOLO-only routes must not carry fabricated detector priors")
        if len(yi):
            hidden = self.yolo_fusion(visual[yi])
            delta = self.yolo_delta(hidden)
            wh = boxes[yi, 2:] - boxes[yi, :2]
            center = (boxes[yi, :2] + boxes[yi, 2:]) * .5
            center = center + delta[:, :2] * wh
            refined_wh = wh * delta[:, 2:].clamp(-2, 2).exp()
            refined = torch.cat((center - refined_wh * .5, center + refined_wh * .5), -1)
            full = full.index_copy(0, yi, refined)
            quality = quality.index_copy(0, yi, self.yolo_quality(hidden).squeeze(-1))
            deltas = deltas.index_copy(0, yi, delta)
        # Keep every new head in the graph even when this rank has no YOLO rows.
        # This is numerically zero; clear inactive independent heads only AFTER
        # update-wide/rank-wide counts have been reduced by the orchestrator.
        connected_zero = sum(parameter.sum() * 0 for parameter in self.parameters())
        logits = logits + connected_zero
        full = full + connected_zero
        quality = quality + connected_zero
        deltas = deltas + connected_zero
        for x in (logits, full, quality, deltas):
            if not torch.isfinite(x).all(): raise ValueError("Nonfinite guidance output")
        return GuidanceOutput(ids, logits, full, quality, deltas, boxes, yolo_only)


@dataclass(frozen=True)
class GuidanceTargets:
    """Loss-only aligned labels. Unknown/ignore/mixed may carry NaN box sentinels."""
    candidate_ids: tuple
    identity_state: torch.Tensor
    unique_match: torch.Tensor
    full_box_valid: torch.Tensor
    full_boxes: torch.Tensor


@dataclass(frozen=True)
class GuidanceLossSums:
    sums: dict
    counts: dict
    quality_targets: torch.Tensor
    quality_eligible: torch.Tensor
    route_counts: dict


def guidance_head_dependencies():
    """Use with existing clear_unsupervised_heads and GLOBAL UPDATE counts.

    Never pass one rank/microbatch's local counts to this optimizer gate.
    Counts below exclusively describe these new losses, not original B0 losses.
    """
    return (
        {"parameter_prefixes": ["identity_head"], "losses": ["identity_positive", "identity_negative"]},
        {"parameter_prefixes": ["yolo_fusion"], "losses": ["full_box_regression", "full_box_giou", "full_box_quality"]},
        {"parameter_prefixes": ["yolo_delta"], "losses": ["full_box_regression", "full_box_giou"]},
        {"parameter_prefixes": ["yolo_quality"], "losses": ["full_box_quality"]},
    )


def guidance_loss_sums(output, targets, *, loss_weights=None):
    """Local sums/counts only; caller allreduces counts before global normalization.

    Use existing weighted_objective / step_if_supervised at orchestration.
    Identity positive/negative sums are separate for global class balancing.
    New regression/GIoU/quality supervise YOLO-only rows; existing B0 full-box
    losses remain owned by the original training path, with no duplicate weight.
    Regression is per-instance mean over 4 deltas, GIoU per instance.
    Quality is VFL-form sum, count = YOLO eligible positives+known background;
    target is current detached full-box IoU (background0), not an identity score.
    Disabled losses return zero sums/counts and do not inspect their GT boxes.
    ALL counts here are LOCAL. Accumulate all microbatches and allreduce across
    ranks before existing clear_unsupervised_heads/step_if_supervised decisions.
    """
    if not isinstance(output, GuidanceOutput) or not isinstance(targets, GuidanceTargets):
        raise ValueError("Typed prediction and loss-only supervision required")
    weights = {key: 1. for key in LOSS_KEYS}
    if loss_weights is not None:
        if not isinstance(loss_weights, dict) or set(loss_weights) - set(LOSS_KEYS):
            raise ValueError("Unknown guidance loss weights")
        for key, value in loss_weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError("Finite nonnegative loss weights required")
        weights.update(loss_weights)
    enabled = {key: weights[key] > 0 for key in LOSS_KEYS}
    ids = _ids(output.candidate_ids); n = len(ids)
    if _ids(targets.candidate_ids) != ids: raise ValueError("Supervision candidate ID/order mismatch")
    ref = output.identity_logits
    for label, value, shape in (("identity logits", ref, (n,)), ("full boxes", output.full_boxes, (n, 4)),
                                ("quality logits", output.quality_logits, (n,)), ("deltas", output.deltas, (n, 4)),
                                ("original boxes", output.original_boxes, (n, 4))):
        _tensor(value, shape, ref, label)
    state = targets.identity_state
    if not isinstance(state, torch.Tensor) or state.shape != (n,) or state.device != ref.device or state.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Explicit integer identity state per candidate required")
    if not torch.isin(state, state.new_tensor([POSITIVE, BACKGROUND, UNKNOWN, IGNORE, MIXED])).all():
        raise ValueError("Unknown supervision state")
    _tensor(targets.unique_match, (n,), ref, "unique match", boolean=True)
    _tensor(targets.full_box_valid, (n,), ref, "full box validity", boolean=True)
    gt = targets.full_boxes
    if not isinstance(gt, torch.Tensor) or gt.shape != (n, 4) or gt.device != ref.device or gt.dtype != ref.dtype:
        raise ValueError("GT box shape/device/dtype mismatch")
    positive = state == POSITIVE
    negative = state == BACKGROUND
    _tensor(output.yolo_only, (n,), ref, "output source route", boolean=True)
    box_eligible = positive & targets.unique_match & targets.full_box_valid & output.yolo_only
    any_box_loss = any(enabled[k] for k in ("full_box_regression", "full_box_giou", "full_box_quality"))
    if not any_box_loss: box_eligible = torch.zeros_like(box_eligible)
    if ((~positive) & (targets.unique_match | targets.full_box_valid)).any():
        raise ValueError("Unknown/mixed/background cannot claim matched full-box supervision")
    zero = ref.sum() * 0 + output.full_boxes.sum() * 0 + output.quality_logits.sum() * 0 + output.deltas.sum() * 0
    sums = {key: zero for key in LOSS_KEYS}
    counts = {key: 0 for key in LOSS_KEYS}
    for key, mask, value in (("identity_positive", positive, 1.), ("identity_negative", negative, 0.)):
        if enabled[key] and mask.any():
            sums[key] = F.binary_cross_entropy_with_logits(ref[mask], torch.full_like(ref[mask], value), reduction="sum")
            counts[key] = int(mask.sum())
    quality_target = torch.full_like(ref, float("nan"))
    quality_background = negative & output.yolo_only
    quality_eligible = (box_eligible | quality_background) if enabled["full_box_quality"] else torch.zeros_like(negative)
    quality_target[quality_background & quality_eligible] = 0
    if box_eligible.any():
        # Select first: never run box arithmetic on inactive NaN/invalid GT.
        selected_gt = gt[box_eligible].detach()
        if not torch.isfinite(selected_gt).all() or not ((selected_gt[:, 2:] - selected_gt[:, :2]) > 0).all():
            raise ValueError("Illegal eligible GT before regression/IoU arithmetic")
        predicted = output.full_boxes[box_eligible]
        original = output.original_boxes[box_eligible]
        if not ((original[:, 2:] - original[:, :2]) > 0).all():
            raise ValueError("Illegal eligible original box")
        iou, giou = paired_iou_giou(predicted, selected_gt)
        wh = original[:, 2:] - original[:, :2]
        target_wh = selected_gt[:, 2:] - selected_gt[:, :2]
        target_delta = torch.cat((((selected_gt[:, :2] + selected_gt[:, 2:]) -
                                    (original[:, :2] + original[:, 2:])) * .5 / wh,
                                   (target_wh / wh).log()), -1)
        if enabled["full_box_regression"]:
            sums["full_box_regression"] = F.smooth_l1_loss(output.deltas[box_eligible], target_delta, beta=1/9, reduction="none").mean(-1).sum()
            counts["full_box_regression"] = int(box_eligible.sum())
        if enabled["full_box_giou"]:
            sums["full_box_giou"] = (1 - giou).sum()
            counts["full_box_giou"] = int(box_eligible.sum())
        if enabled["full_box_quality"]:
            quality_target[box_eligible] = iou.detach()
    if quality_eligible.any():
        q = output.quality_logits[quality_eligible]
        target = quality_target[quality_eligible].detach()
        weight = torch.where(target > 0, target, .75 * (q.sigmoid() - target).abs().square())
        sums["full_box_quality"] = (F.binary_cross_entropy_with_logits(q, target, reduction="none") * weight).sum()
        counts["full_box_quality"] = int(quality_eligible.sum())
    _tensor(output.yolo_only, (n,), ref, "output source route", boolean=True)
    route_counts = {}
    for label, route in (("b0", ~output.yolo_only), ("yolo_only", output.yolo_only)):
        route_counts[label] = {
            "identity_positive": int((positive & route).sum()) if enabled["identity_positive"] else 0,
            "identity_negative": int((negative & route).sum()) if enabled["identity_negative"] else 0,
            "full_box_regression": int((box_eligible & route).sum()) if enabled["full_box_regression"] else 0,
            "full_box_giou": int((box_eligible & route).sum()) if enabled["full_box_giou"] else 0,
            "full_box_quality": int((quality_eligible & route).sum()),
        }
    return GuidanceLossSums(sums, counts, quality_target.detach(), quality_eligible, route_counts)
