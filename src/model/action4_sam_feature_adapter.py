"""Independent SAM C5 residual interface; no backbone, pose head or runtime loading.

Input masks are predicted and already sampled/perturbed on the RGB ROI grid.
Ground truth is accepted only by mask_quality_loss, never by forward.
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROTOCOL_ID = "action4.sam_feature.c5.v001"
QUALITY_WEIGHT = 0.1


@dataclass(frozen=True)
class FeatureIdentity:
    image_id: str
    instance_id: str
    roi_id: str
    pixel_space_id: str


def _identities(items: Sequence[FeatureIdentity], batch: int):
    items = tuple(items)
    if len(items) != batch or any(
        not isinstance(x, FeatureIdentity) or any(
            not isinstance(v, str) or not v.strip()
            for v in (x.image_id, x.instance_id, x.roi_id, x.pixel_space_id)
        ) for x in items
    ):
        raise ValueError("identity requires one nonempty image/instance/ROI/pixel-space record per sample")
    if len(set(items)) != batch:
        raise ValueError("duplicate identity")
    return items


def _tensor(name, x, shape, *, device=None, dtype=None, bounds=None, binary=False, check_finite=True):
    if not isinstance(x, Tensor) or tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {shape}")
    if check_finite and not torch.isfinite(x).all():
        raise ValueError(f"{name}: non-finite value")
    if device is not None and x.device != device:
        raise ValueError(f"{name}: device mismatch")
    if dtype is not None and x.dtype != dtype:
        raise ValueError(f"{name}: dtype mismatch")
    if bounds is not None and ((x < bounds[0]).any() or (x > bounds[1]).any()):
        raise ValueError(f"{name}: out of range")
    if binary and ((x != 0) & (x != 1)).any():
        raise ValueError(f"{name}: must be binary")


@dataclass(frozen=True)
class MaskAux:
    visible: Tensor
    distance: Tensor
    available: Tensor
    candidate_overlap: Tensor
    dropout: Tensor
    identities: tuple[FeatureIdentity, ...]
    protocol_id: str = PROTOCOL_ID


@dataclass(frozen=True)
class SamFeatureOutput:
    c5: Tensor
    residual: Tensor
    q_mask: Tensor
    quality_logits: Tensor
    visible: Tensor
    available: Tensor
    dropout: Tensor
    identities: tuple[FeatureIdentity, ...]


@dataclass(frozen=True)
class QualityLoss:
    # Unweighted local SUM; caller applies 0.1 * world_size / GLOBAL count.
    loss_sum: Tensor
    count: Tensor
    target_iou: Tensor
    eligible: Tensor


def prepare_mask_aux(visible: Tensor, available: Tensor, *,
                     identities: Sequence[FeatureIdentity],
                     candidate_overlap: Tensor, dropout: Tensor | None = None) -> MaskAux:
    """Compute exact CPU EDT after caller's ROI sampling AND mask perturbation.

    This preprocessing helper detaches predicted evidence. Empty, missing and
    dropout inputs are canonicalized to M=D=0, available=0. No GT is accepted.
    """
    from scipy.ndimage import distance_transform_edt
    if visible.ndim != 4:
        raise ValueError("visible: expected BCHW")
    batch = visible.shape[0]
    if not visible.is_floating_point():
        raise ValueError("visible: floating binary tensor required")
    _tensor("visible", visible, (batch, 1, 448, 448), binary=True)
    _tensor("available", available, (batch, 1, 1, 1),
            device=visible.device, dtype=visible.dtype, binary=True)
    _tensor("candidate_overlap", candidate_overlap, (batch, 1),
            device=visible.device, dtype=visible.dtype, bounds=(0, 1))
    ids = _identities(identities, batch)
    dropout = torch.zeros(batch, dtype=torch.bool, device=visible.device) if dropout is None else dropout
    _tensor("dropout", dropout, (batch,), device=visible.device, dtype=torch.bool)
    effective = available.detach().bool().flatten() & visible.detach().bool().flatten(1).any(1) & ~dropout
    mask = visible.detach().clone() * effective[:, None, None, None]
    distance = torch.zeros_like(mask)
    for index in range(batch):
        if bool(effective[index]):
            binary = mask[index, 0].cpu().numpy().astype(bool)
            edt = (distance_transform_edt(binary) - distance_transform_edt(~binary)) / 16.0
            distance[index, 0] = torch.as_tensor(edt.clip(-1, 1), device=mask.device, dtype=mask.dtype)
    return MaskAux(mask, distance, effective.to(mask.dtype).reshape(batch, 1, 1, 1),
                   candidate_overlap.detach().clone(), dropout.detach().clone(), ids)


class SamFeatureAdapter(nn.Module):
    """Maps an existing RGB C5 to SAM-enhanced C5 for current A0 or future FPN."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
        )
        self.projection = nn.Conv2d(64, 2048, 1)
        self.quality_head = nn.Sequential(nn.Linear(2048 + 8, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.zeros_(self.quality_head[-1].bias)

    def get_extra_state(self):
        return {"protocol_id": PROTOCOL_ID, "c5_shape": [2048, 14, 14],
                "quality_statistics": ["area", "bbox_width", "bbox_height", "fill",
                                       "perimeter", "candidate_overlap", "missing", "dropout"]}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("SAM feature state protocol identity mismatch")

    def forward(self, c5: Tensor, *, rgb: Tensor, identities: Sequence[FeatureIdentity],
                mask_aux: MaskAux | None = None) -> SamFeatureOutput:
        if not isinstance(c5, Tensor) or c5.ndim != 4 or not c5.is_floating_point():
            raise ValueError("C5 must be floating BCHW")
        batch = c5.shape[0]
        if batch == 0:
            raise ValueError("empty batch is unsupported; use an explicit dummy sample")
        _tensor("c5", c5, (batch, 2048, 14, 14))
        _tensor("rgb", rgb, (batch, 3, 448, 448), device=c5.device, dtype=c5.dtype)
        ids = _identities(identities, batch)
        if mask_aux is None:
            mask_aux = MaskAux(c5.new_zeros(batch, 1, 448, 448), c5.new_zeros(batch, 1, 448, 448),
                               c5.new_zeros(batch, 1, 1, 1), c5.new_zeros(batch, 1),
                               torch.zeros(batch, dtype=torch.bool, device=c5.device), ids)
        if mask_aux.protocol_id != PROTOCOL_ID or _identities(mask_aux.identities, batch) != ids:
            raise ValueError("mask/RGB identity or protocol mismatch")
        for name, shape, bounds, binary in (
            ("visible", (batch, 1, 448, 448), (0, 1), True),
            ("distance", (batch, 1, 448, 448), (-1, 1), False),
            ("available", (batch, 1, 1, 1), (0, 1), True),
            ("candidate_overlap", (batch, 1), (0, 1), False),
        ):
            _tensor(name, getattr(mask_aux, name), shape, device=c5.device, dtype=c5.dtype,
                    bounds=bounds, binary=binary)
        _tensor("dropout", mask_aux.dropout, (batch,), device=c5.device, dtype=torch.bool)
        m, d, a = (x.detach() for x in (mask_aux.visible, mask_aux.distance, mask_aux.available))
        missing = ~a.bool().flatten()
        if (m[missing] != 0).any() or (d[missing] != 0).any():
            raise ValueError("unavailable mask must have M=D=0")
        if (mask_aux.dropout & ~missing).any() or ((~missing) & ~m.bool().flatten(1).any(1)).any():
            raise ValueError("empty/dropout mask must be canonicalized unavailable")
        # All predicted statistics are detached: quality cannot train RGB or adapter.
        with torch.no_grad():
            mask = m[:, 0]
            area = mask.mean((1, 2))
            coordinate = torch.arange(448, device=c5.device)
            def extent(support):
                lo = torch.where(support, coordinate, 448).amin(1)
                hi = torch.where(support, coordinate, -1).amax(1)
                return (hi - lo + 1).clamp_min(0).to(c5.dtype) / 448
            width = extent(mask.bool().any(1))
            height = extent(mask.bool().any(2))
            fill = area / (width * height).clamp_min(1 / (448 * 448))
            perimeter = ((mask[:, 1:, :] - mask[:, :-1, :]).abs().sum((1, 2)) +
                         (mask[:, :, 1:] - mask[:, :, :-1]).abs().sum((1, 2))) / (4 * 448 * 448)
            stats = torch.stack((area, width, height, fill, perimeter,
                                 mask_aux.candidate_overlap[:, 0], missing.to(c5.dtype),
                                 mask_aux.dropout.to(c5.dtype)), 1)
        logits = self.quality_head(torch.cat((c5.detach().mean((2, 3)), stats.detach()), 1))
        q = logits.sigmoid()
        encoded = self.encoder(torch.cat((m, d), 1))
        residual = self.projection(F.interpolate(encoded, size=(14, 14), mode="bilinear", align_corners=False))
        enhanced = c5 + a * q.detach().reshape(batch, 1, 1, 1) * residual
        if not all(torch.isfinite(x).all() for x in (q, residual, enhanced)):
            raise ValueError("non-finite SAM feature output")
        return SamFeatureOutput(enhanced, residual, q, logits, m, a,
                                mask_aux.dropout.detach(), ids)


def mask_quality_loss(output: SamFeatureOutput, gt_visible: Tensor, *,
                      gt_identities: Sequence[FeatureIdentity], supervised: Tensor | None = None) -> QualityLoss:
    """SmoothL1 quality supervision from the ACTUAL input mask, never gate by GT IoU.

    supervised=False excludes unknown/ignored/dummy labels; empty GT is a known
    negative when the predicted mask is nonempty. All-zero loss stays connected
    to quality_head only. Do not call optimizer.step for globally zero supervision.
    """
    batch = output.q_mask.shape[0]
    if _identities(gt_identities, batch) != output.identities:
        raise ValueError("GT/prediction identity mismatch")
    _tensor("q_mask", output.q_mask, (batch, 1), bounds=(0, 1))
    # Unknown/ignored/dummy labels may contain NaN sentinels. Validate metadata
    # first; only known labels have a finite binary-value contract.
    _tensor("gt_visible", gt_visible, (batch, 1, 448, 448), device=output.visible.device,
            dtype=output.visible.dtype, check_finite=False)
    if supervised is None:
        supervised = torch.ones(batch, dtype=torch.bool, device=output.visible.device)
    _tensor("supervised", supervised, (batch,), device=output.visible.device, dtype=torch.bool)
    known_gt = gt_visible[supervised]
    _tensor("supervised gt_visible", known_gt, (int(supervised.sum()), 1, 448, 448), binary=True)
    with torch.no_grad():
        safe_gt = torch.where(supervised[:, None, None, None], gt_visible.detach(), 0)
        pred, target = output.visible.bool().flatten(1), safe_gt.bool().flatten(1)
        intersection = (pred & target).sum(1).to(output.q_mask.dtype)
        union = (pred | target).sum(1).to(output.q_mask.dtype)
        target_iou = intersection / union.clamp_min(1)
        eligible = output.available.bool().flatten() & pred.any(1) & ~output.dropout & supervised
    per_sample = F.smooth_l1_loss(output.q_mask.flatten(), target_iou, reduction="none")
    return QualityLoss((per_sample * eligible.to(per_sample.dtype)).sum(), eligible.sum(),
                       target_iou, eligible)
