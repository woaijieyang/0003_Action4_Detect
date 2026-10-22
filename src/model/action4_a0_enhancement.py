"""Additive A0 direction-field module; original best A0 remains immutable.

PVNet-inspired visible-pixel supervision is an auxiliary task. This is not a
full PVNet reproduction. No automatic replacement/fusion of base corners occurs.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class DirectionEvidence(nn.Module):
    """Small local branch on native decoder features, RGB, and image coordinates."""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels + 5, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU())
        self.direction = nn.Conv2d(64, 16, 1)
        self.foreground = nn.Conv2d(64, 2, 1)

    def forward(self, features, rgb):
        batch, _, height, width = features.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=features.device),
            torch.linspace(-1, 1, width, device=features.device), indexing="ij")
        coordinates = torch.stack((x, y))[None].expand(batch, -1, -1, -1)
        low_rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
        hidden = self.body(torch.cat((features, low_rgb, coordinates), dim=1))
        directions = self.direction(hidden).reshape(batch, 8, 2, height, width)
        return {"direction_vectors": directions,
                "foreground_logits": self.foreground(hidden)}


class A0Enhancement(nn.Module):
    """Frozen mainline plus trainable evidence. Disable gives original heatmaps."""
    def __init__(self, base):
        super().__init__()
        if base.role != "A0":
            raise ValueError("Expected the original A0 component")
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.evidence = DirectionEvidence(base.network["decoder_channels"])

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, roi_images, *, enhancement_enabled=True):
        with torch.no_grad():
            features = self.base.decoder(self.base.backbone(roi_images))
            logits = self.base.corner_head(features)
        result = {"corner_logits": logits, "base_preserved": True}
        if enhancement_enabled:
            result.update(self.evidence(features.detach(), roi_images.detach()))
        return result

    def trainable_parameters(self):
        return self.evidence.parameters()


def direction_supervision(points_hm, visible_mask):
    """Visible surface pixels vote for every fixed-ID corner, including hidden ones.

    Mask is an object-pixel mask, not corner visibility. No hidden corner is
    removed. Coordinates must already share the decoder pixel-centre grid.
    """
    if points_hm.ndim != 3 or points_hm.shape[1:] != (8, 2):
        raise ValueError("Eight ordered 2D corners required")
    if visible_mask.ndim != 3 or visible_mask.shape[0] != points_hm.shape[0]:
        raise ValueError("Object mask must be BHW")
    if not bool(torch.isfinite(points_hm).all()):
        raise ValueError("Nonfinite corner target")
    height, width = visible_mask.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=points_hm.device),
                         torch.arange(width, device=points_hm.device), indexing="ij")
    pixels = torch.stack((x, y)).to(points_hm.dtype)
    difference = points_hm[..., None, None] - pixels[None, None]
    norm = torch.linalg.vector_norm(difference, dim=2, keepdim=True)
    target = difference / norm.clamp_min(1e-6)
    support = visible_mask[:, None].bool() & (norm.squeeze(2) > 1e-6)
    return target, support


def evidence_loss(output, points_hm, visible_mask):
    targets, support = direction_supervision(points_hm, visible_mask)
    vectors = output["direction_vectors"]
    if vectors.shape != targets.shape:
        raise ValueError("Prediction/target direction grids differ")
    # PVNet SmoothL1 direction supervision, normalized over visible support.
    error = F.smooth_l1_loss(vectors, targets, reduction="none").mean(2)
    direction = (error * support).sum() / support.sum().clamp_min(1)
    # Exact binary softmax-CE identity; avoids CUDA nll_loss2d nondeterminism.
    foreground_logits = output["foreground_logits"]
    foreground = F.binary_cross_entropy_with_logits(
        foreground_logits[:, 1] - foreground_logits[:, 0], visible_mask.float())
    return {"direction": direction, "foreground": foreground}


@torch.no_grad()
def auxiliary_vote_evidence(output):
    """Predict auxiliary points from predicted masks only; never replaces base.

    Numerical validity is kept separate from accuracy/calibrated reliability.
    Consumers must explicitly evaluate any fusion policy before deployment.
    """
    from model.action4_decode import decode_corners
    from model.ops.pvnet_voting import ransac_voting_layer_v2
    from model.ops.action4_vote_refinement import refine_votes
    vectors=output["direction_vectors"].permute(0,3,4,1,2).contiguous()
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("Nonfinite auxiliary directions")
    mask=output["foreground_logits"].argmax(1)
    initial=ransac_voting_layer_v2(mask,vectors,2,64)[:,0]
    refined=refine_votes(mask,vectors,initial)
    baseline=decode_corners(output["corner_logits"])
    evidence=torch.where(refined["valid"][...,None],refined["points"],baseline)
    return {"base_corners_hm":baseline,"auxiliary_corners_hm":evidence,
            "numerically_valid":refined["valid"],
            "support_fraction":refined["support_fraction"],
            "production_replacement_enabled":False}
