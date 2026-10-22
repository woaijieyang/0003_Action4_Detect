"""RGB-derived detector and fixed-index corner decoding.

Independent implementation of CenterNet max-pool NMS/top-k/gather mathematics
(Xingyi Zhou, MIT; commit 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c), and the
per-corner top-20 position-mean baseline in BoxDreamer recover_bb8_corners
(ZJU3DV, PRL v1.0; commit 106f3c16c746770cbad80720ce4a557eb31f30f5).
No BoxDreamer network, reference RGB, reference keypoints or checkpoint is used.
All coordinates here are pixel centres; no duplicated [-1,1] conversions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .action4_losses import gather_at_indices


def transform_points(points: Tensor, matrix: Tensor) -> Tensor:
    """Apply one affine 3x3 matrix per leading sample to [...,2] points."""
    if points.ndim < 2 or points.shape[-1] != 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected points [...,2] and a 3x3 affine matrix")
    matrix = torch.as_tensor(matrix, device=points.device, dtype=points.dtype)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Coordinate matrix must be finite")
    expected = matrix.new_tensor([0, 0, 1]).expand_as(matrix[..., 2, :])
    if not torch.allclose(matrix[..., 2, :], expected, atol=1e-6, rtol=0):
        raise ValueError("Only affine matrices are accepted")
    if bool((torch.linalg.det(matrix[..., :2, :2]).abs() < 1e-12).any()):
        raise ValueError("Coordinate matrix is singular")
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.matmul(homogeneous, matrix.transpose(-1, -2))[..., :2]


def decode_corners(
    heatmaps: Tensor,
    *,
    heatmap_to_image: Tensor | None = None,
    topk: int = 20,
    from_logits: bool = True,
) -> Tensor:
    """Decode [N,8,H,W] to [N,8,2], optionally mapping to original pixels.

    The baseline averages the positions of the k highest pixels, preserving
    fixed corner channel order. Constant heatmaps have no meaningful location;
    this function does not turn them into confidence claims. Evaluation should
    retain scores separately. Near a heatmap border this decoder has inward
    bias; formal ROI construction leaves a margin around the full cuboid box.
    """
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 8:
        raise ValueError("Corner heatmaps must be [N,8,H,W]")
    if type(topk) is not int or topk < 1:
        raise ValueError("topk must be a positive integer")
    if not bool(torch.isfinite(heatmaps).all()):
        raise ValueError("Corner heatmaps contain nonfinite predictions")
    height, width = heatmaps.shape[-2:]
    if height < 1 or width < 1:
        raise ValueError("Heatmap spatial dimensions must be positive")
    values = heatmaps.float().sigmoid() if from_logits else heatmaps.float()
    if not from_logits and bool(((values < 0) | (values > 1)).any()):
        raise ValueError("Probability heatmaps must be in [0,1]")
    indices = values.flatten(2).topk(min(topk, height * width), dim=-1, sorted=False).indices
    points = torch.stack([(indices % width).float().mean(-1),
                          torch.div(indices, width, rounding_mode="floor").float().mean(-1)], dim=-1)
    if heatmap_to_image is not None:
        matrix = torch.as_tensor(heatmap_to_image, dtype=points.dtype, device=points.device)
        if matrix.shape not in ((3, 3), (points.shape[0], 3, 3)):
            raise ValueError("heatmap_to_image must be 3x3 or Nx3x3")
        points = transform_points(points, matrix)
    return points


def box_iou(first: Tensor, second: Tensor) -> Tensor:
    """Pairwise IoU for ordinary continuous xyxy envelopes (no +1 pixels)."""
    if first.shape[-1] != 4 or second.shape[-1] != 4:
        raise ValueError("Boxes must end in four xyxy coordinates")
    left_top = torch.maximum(first[..., :2], second[..., :2])
    right_bottom = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0).prod(-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0).prod(-1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-12)


def decode_detections(
    outputs: Mapping[str, Tensor],
    *,
    image_size: Sequence[int],
    heatmap_to_image: Tensor | None = None,
    score_threshold: float = 0.3,
    topk: int = 20,
    max_detections: int = 2,
    duplicate_iou: float = 0.95,
    duplicate_center_ratio: float = 0.05,
    nms_kernel: int = 3,
) -> list[list[dict[str, Any]]]:
    """Return zero, one or two RGB-only candidate instances per full image.

    ``image_size=(W,H)`` is the detector input canvas. Distances are normalized
    in that canvas. Optional ``heatmap_to_image`` is its heatmap -> original
    pixel transform, with shape [B,3,3] or [3,3]. It transforms both centre and
    complete box; boxes are never clipped to the source image. Candidate
    records contain a Python score and tensor center/bbox_full_xyxy.
    """
    center, offset, ltrb = outputs["center_logits"], outputs["offset"], outputs["ltrb"]
    if center.ndim != 4 or center.shape[1] != 1:
        raise ValueError("center_logits must have shape [B,1,H,W]")
    batch, _, height, width = center.shape
    if offset.shape != (batch, 2, height, width) or ltrb.shape != (batch, 4, height, width):
        raise ValueError("Detector map shapes disagree")
    if (len(image_size) != 2 or any(type(x) is not int or x < 1 for x in image_size)
            or type(topk) is not int or topk < 1 or max_detections not in (1, 2)):
        raise ValueError("Positive image_size/topk and max_detections in {1,2} are required")
    if not 0 <= score_threshold <= 1 or not 0 <= duplicate_iou <= 1 or duplicate_center_ratio < 0:
        raise ValueError("Invalid score or duplicate threshold")
    if type(nms_kernel) is not int or nms_kernel < 1 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer")
    if not all(bool(torch.isfinite(value).all()) for value in (center, offset, ltrb)):
        raise ValueError("Detector output contains nonfinite predictions")
    if bool((ltrb < 0).any()):
        raise ValueError("ltrb must be nonnegative normalized distances")
    canvas_width, canvas_height = image_size
    scale_x, scale_y = canvas_width / width, canvas_height / height
    hm_to_canvas = center.new_tensor([
        [scale_x, 0, (scale_x - 1) / 2],
        [0, scale_y, (scale_y - 1) / 2],
        [0, 0, 1],
    ], dtype=torch.float32)
    if heatmap_to_image is None:
        hm_to_image = hm_to_canvas.expand(batch, -1, -1)
    else:
        hm_to_image = torch.as_tensor(heatmap_to_image, dtype=torch.float32, device=center.device)
        if hm_to_image.shape == (3, 3):
            hm_to_image = hm_to_image.expand(batch, -1, -1)
        if hm_to_image.shape != (batch, 3, 3):
            raise ValueError("heatmap_to_image must be 3x3 or Bx3x3")
    canvas_to_image = hm_to_image @ torch.linalg.inv(hm_to_canvas)
    probabilities = center.float().sigmoid()
    maxima = F.max_pool2d(probabilities, nms_kernel, stride=1, padding=nms_kernel // 2)
    filtered = probabilities * probabilities.eq(maxima)
    scores, indices = filtered.flatten(1).topk(min(topk, height * width), dim=1, sorted=True)
    shifts = gather_at_indices(offset.float(), indices)
    distances = gather_at_indices(ltrb.float(), indices) * center.new_tensor(
        [canvas_width, canvas_height, canvas_width, canvas_height], dtype=torch.float32)
    anchors = torch.stack([(indices % width).float(),
                           torch.div(indices, width, rounding_mode="floor").float()], dim=-1)
    centers_hm = anchors + shifts
    centers_canvas = transform_points(centers_hm, hm_to_canvas)
    centers_output = transform_points(centers_hm, hm_to_image)
    minimum = centers_canvas - distances[..., :2]
    maximum = centers_canvas + distances[..., 2:]
    bbox_canvas_corners = torch.stack([
        minimum, torch.stack([maximum[..., 0], minimum[..., 1]], -1),
        maximum, torch.stack([minimum[..., 0], maximum[..., 1]], -1),
    ], dim=-2)
    output_corners = transform_points(bbox_canvas_corners.flatten(1, 2), canvas_to_image).reshape(
        batch, indices.shape[1], 4, 2)
    boxes = torch.cat([output_corners.amin(-2), output_corners.amax(-2)], dim=-1)
    result: list[list[dict[str, Any]]] = []
    for image_index in range(batch):
        selected: list[dict[str, Any]] = []
        for candidate_index in range(indices.shape[1]):
            score = float(scores[image_index, candidate_index].detach())
            if score < score_threshold or score <= 0:
                continue
            box = boxes[image_index, candidate_index]
            point = centers_output[image_index, candidate_index]
            if bool((box[2:] <= box[:2]).any()):
                continue
            duplicate = False
            for kept in selected:
                diagonal = torch.minimum(torch.linalg.vector_norm(box[2:] - box[:2]),
                                         torch.linalg.vector_norm(kept["bbox_full_xyxy"][2:] - kept["bbox_full_xyxy"][:2]))
                center_distance = torch.linalg.vector_norm(point - kept["center"])
                if (float(box_iou(box, kept["bbox_full_xyxy"])) >= duplicate_iou
                        and float(center_distance) <= duplicate_center_ratio * float(diagonal)):
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append({"score": score, "center": point,
                             "bbox_full_xyxy": box,
                             "grid_index": int(indices[image_index, candidate_index])})
            if len(selected) == max_detections:
                break
        result.append(selected)
    return result
