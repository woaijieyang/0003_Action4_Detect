"""Action 4 manifest loading, shared RGB sampling, and geometric supervision.

No model/eval imports and no torch import at module load. Geometry is delegated
to DataAdapter. Both training and inference use the same pixel-centre matrices
and torch bilinear sampler (align_corners=False, black RGB padding).
Gaussian radius equations follow CenterNet, commit
4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c, src/lib/utils/image.py.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from .action4_adapter import DataAdapter


def _torch():
    import torch
    return torch


def is_ignored(instance: dict) -> bool:
    return (bool(instance.get('ignore', False)) or bool(instance.get('unobservable', False))
            or instance.get('status') == 'unobservable')


class Action4Dataset:
    """Map-style Dataset. Formal training rejects unaccepted synthetic samples."""

    def __init__(self, root, manifest, *, allow_diagnostic=False, split=None,
                 validation_workers=1, decoded_cache_size=0):
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        if type(validation_workers) is not int or not 1 <= validation_workers <= 72:
            raise ValueError('validation_workers must be 1..72')
        if type(decoded_cache_size) is not int or not 0 <= decoded_cache_size <= 128:
            raise ValueError('decoded_cache_size must be 0..128')
        self.root = Path(root).resolve(strict=True)
        self.adapter = DataAdapter(self.root)
        self.records = self.adapter.read_samples(manifest, check_images=False)
        self.decoded_cache_size = decoded_cache_size
        self._decoded_cache = OrderedDict()
        self._validated_stats = {}
        if split is not None:
            if split not in ('train', 'eval', 'test'):
                raise ValueError('Unknown split')
            if any(row.get('split') != split for row in self.records):
                raise ValueError('Manifest split identity differs from requested split')
        self.diagnostic = any(row.get('status') != 'training_accepted' for row in self.records)
        if self.diagnostic and not allow_diagnostic:
            raise ValueError('Dataset contains unaccepted diagnostic/candidate samples; formal training is blocked')
        def validate_image(row):
            path = self.adapter._source(row['image'])
            before = self._file_identity(path)
            # Preserve full PNG decoding, dimensions, label checks and RGB SHA.
            self.adapter.validate_sample(row, check_image=True)
            expected = row.get('rgb_sha256')
            if row.get('status') == 'training_accepted' and expected is None:
                raise ValueError('Accepted sample requires rgb_sha256 for immutable RGB identity')
            if expected is not None:
                if (not isinstance(expected, str) or len(expected) != 64
                        or any(c not in '0123456789abcdef' for c in expected)):
                    raise ValueError('rgb_sha256 must be a lowercase SHA256 hex string')
                digest = hashlib.sha256()
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f"RGB hash mismatch for {row['sample_id']}; data identity has changed")
            for instance in row['instances']:
                if not is_ignored(instance) and instance.get('bbox_visible_xyxy') is None:
                    raise ValueError('Non-ignored instance needs an actual visible mask bounding box')
            if self._file_identity(path) != before:
                raise ValueError('RGB changed during validation')
            return row['image'], before
        if validation_workers == 1:
            validated = map(validate_image, self.records)
            self._validated_stats.update(validated)
        else:
            with ThreadPoolExecutor(max_workers=validation_workers) as pool:
                self._validated_stats.update(pool.map(validate_image, self.records))

    @staticmethod
    def _file_identity(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        row = self.records[index]
        path = self.adapter._source(row['image'])
        if self.decoded_cache_size:
            if self._file_identity(path) != self._validated_stats[row['image']]:
                raise ValueError('RGB identity changed after validation; cached input refused')
            if index in self._decoded_cache:
                image = self._decoded_cache.pop(index)
                self._decoded_cache[index] = image
                return {'image': image.clone(), 'record': row}
        with Image.open(path) as im:
            array = np.array(im.convert('RGB'), dtype=np.uint8, copy=True)
        tensor = _torch().from_numpy(array).permute(2, 0, 1).contiguous()
        if self.decoded_cache_size:
            if self._file_identity(path) != self._validated_stats[row['image']]:
                raise ValueError('RGB changed while filling decoded cache')
            self._decoded_cache[index] = tensor
            while len(self._decoded_cache) > self.decoded_cache_size:
                self._decoded_cache.popitem(last=False)
            tensor = tensor.clone()
        return {'image': tensor, 'record': row}

    @staticmethod
    def collate_fn(items):
        return items


def _rgb_tensor(image, device):
    torch = _torch()
    if not isinstance(image, torch.Tensor):
        import numpy as np
        image = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) <= 0:
        raise ValueError('Expected RGB CHW image')
    if image.dtype == torch.uint8:
        return image.to(device=device, dtype=torch.float32).div(255)
    if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Floating RGB input must be finite and in [0,1]')
    return image.to(device=device, dtype=torch.float32)


def _normalize(image):
    torch = _torch()
    mean = torch.as_tensor([.485, .456, .406], dtype=image.dtype, device=image.device)[:, None, None]
    std = torch.as_tensor([.229, .224, .225], dtype=image.dtype, device=image.device)[:, None, None]
    return (image - mean) / std


def prepare_full_images(images, size=(1024, 768), device='cpu'):
    """Return normalized full-image batch and original->full 3x3 matrices."""
    torch = _torch()
    import torch.nn.functional as F
    width, height = map(int, size)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError('Full dimensions must be positive multiples of32')
    output, matrices = [], []
    for image in images:
        x = _rgb_tensor(image, device)
        matrix, info = DataAdapter.letterbox_affine((int(x.shape[2]), int(x.shape[1])), (width, height))
        rw, rh = info['resize_size']
        left, top, right, bottom = info['padding_ltrb']
        # Same continuous centre map as DataAdapter; padding is black in RGB.
        resized = F.interpolate(x[None], size=(rh, rw), mode='bilinear', align_corners=False)[0]
        output.append(_normalize(F.pad(resized, (left, right, top, bottom), value=0)))
        matrices.append(matrix)
    if not output:
        return torch.empty((0, 3, height, width), device=device), matrices
    return torch.stack(output), matrices


def virtual_roi(box, expansion=1.25):
    import numpy as np
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Expected finite ordered full bbox')
    if not math.isfinite(expansion) or expansion < 1:
        raise ValueError('ROI expansion must be >=1')
    centre = (box[:2] + box[2:]) / 2
    side = float(max(box[2:] - box[:2])) * expansion
    return [centre[0] - side / 2, centre[1] - side / 2, centre[0] + side / 2, centre[1] + side / 2]


def prepare_rois(images, boxes, image_indices, roi_size=256, expansion=1.25, device='cpu'):
    """Sample virtual square ROIs without clipping, return RGB and original->ROI."""
    torch = _torch()
    import torch.nn.functional as F
    import numpy as np
    if len(boxes) != len(image_indices) or type(roi_size) is not int or roi_size <= 0 or roi_size % 32:
        raise ValueError('Bad ROI inputs or size')
    if len(boxes) == 0:
        return torch.empty((0, 3, roi_size, roi_size), device=device), []
    outputs, matrices, cache = [], [], {}
    yy, xx = torch.meshgrid(torch.arange(roi_size, device=device, dtype=torch.float32),
                            torch.arange(roi_size, device=device, dtype=torch.float32), indexing='ij')
    destination = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    for box, index in zip(boxes, image_indices):
        index = int(index)
        if not 0 <= index < len(images):
            raise ValueError('ROI image index outside batch')
        if index not in cache:
            cache[index] = _rgb_tensor(images[index], device)
        source = cache[index]
        matrix = DataAdapter.crop_affine(virtual_roi(box, expansion), (roi_size, roi_size))
        inverse = torch.as_tensor(np.linalg.inv(matrix), dtype=torch.float32, device=device)
        # Pixel coordinates must not be rounded by the model's outer AMP scope.
        # Large native-image translations lose subpixel accuracy in FP16 and
        # otherwise make training RGB disagree with FP32 evaluation and labels.
        with torch.autocast(device_type=source.device.type, enabled=False):
            xy = destination.float() @ inverse.float().T
            grid = torch.stack((2 * (xy[..., 0] + .5) / source.shape[2] - 1,
                                2 * (xy[..., 1] + .5) / source.shape[1] - 1), dim=-1)
            sampled = F.grid_sample(source.float()[None], grid[None], mode='bilinear', padding_mode='zeros', align_corners=False)[0]
        outputs.append(_normalize(sampled))
        matrices.append(matrix)
    return torch.stack(outputs), matrices


def _gaussian_radius(height, width, overlap=.7):
    # The three overlap-bound equations in CenterNet's documented implementation.
    candidates = []
    for a, b, c in [(1, height + width, width * height * (1 - overlap) / (1 + overlap)),
                    (4, 2 * (height + width), (1 - overlap) * width * height),
                    (4 * overlap, -2 * overlap * (height + width), (overlap - 1) * width * height)]:
        candidates.append((b + math.sqrt(max(0, b * b - 4 * a * c))) / 2)
    return max(0, int(min(candidates)))


def build_detection_targets(records, orig_to_full, full_size=(1024, 768), stride=4, device='cpu'):
    """Build visible-centre/full-cuboid targets; refuse same-cell ambiguities."""
    import numpy as np
    torch = _torch()
    width, height = map(int, full_size)
    if stride <= 0 or width % stride or height % stride or len(records) != len(orig_to_full):
        raise ValueError('Invalid heatmap dimensions or affine batch')
    hw, hh, batch = width // stride, height // stride, len(records)
    center = torch.zeros((batch, 1, hh, hw), device=device)
    negative = torch.ones_like(center, dtype=torch.bool)
    indices = torch.zeros((batch, 2), dtype=torch.long, device=device)
    mask = torch.zeros((batch, 2), dtype=torch.bool, device=device)
    offset = torch.zeros((batch, 2, 2), device=device)
    ltrb = torch.zeros((batch, 2, 4), device=device)
    down = np.array([[1 / stride, 0, (1 / stride - 1) / 2],
                     [0, 1 / stride, (1 / stride - 1) / 2], [0, 0, 1.]])
    for b, (row, matrix) in enumerate(zip(records, orig_to_full)):
        occupied = set()
        for instance in row['instances']:
            if not is_ignored(instance):
                continue
            box = instance['bbox_full_xyxy']
            edges = DataAdapter.transform_points([[box[0], box[1]], [box[2], box[3]]], down @ matrix)
            x0, y0 = np.ceil(edges[0]).astype(int)
            x1, y1 = np.floor(edges[1]).astype(int) + 1
            x0, x1 = max(0, min(hw, x0)), max(0, min(hw, x1))
            y0, y1 = max(0, min(hh, y0)), max(0, min(hh, y1))
            if x1 > x0 and y1 > y0:
                negative[b, 0, y0:y1, x0:x1] = False
        for j, instance in enumerate(row['instances']):
            if is_ignored(instance):
                continue
            visible = instance.get('bbox_visible_xyxy')
            if visible is None:
                raise ValueError('Visible bbox needed for a valid detection anchor')
            p = (np.asarray(visible[:2]) + visible[2:]) / 2
            point_full = DataAdapter.transform_points([p], matrix)[0]
            point_hm = DataAdapter.transform_points([p], down @ matrix)[0]
            anchor = np.clip(np.floor(point_hm).astype(int), [0, 0], [hw - 1, hh - 1])
            ix, iy = map(int, anchor)
            if (ix, iy) in occupied:
                raise ValueError(f"Same heatmap cell for two valid instances in {row['sample_id']}; resolve detector resolution/capacity")
            occupied.add((ix, iy))
            corners = DataAdapter.transform_points(instance['corners_px'], matrix)
            low, high = corners.min(axis=0), corners.max(axis=0)
            distances = np.r_[point_full - low, high - point_full]
            if (distances < -1e-4).any():
                raise ValueError('Visible centre outside full cuboid envelope')
            indices[b, j] = iy * hw + ix
            mask[b, j] = True
            offset[b, j] = torch.as_tensor(point_hm - anchor, device=device)
            ltrb[b, j] = torch.as_tensor(np.maximum(distances, 0) / [width, height, width, height], device=device)
            radius = min(_gaussian_radius((high[1] - low[1]) / stride, (high[0] - low[0]) / stride), max(hw, hh))
            x0, x1, y0, y1 = max(0, ix - radius), min(hw, ix + radius + 1), max(0, iy - radius), min(hh, iy + radius + 1)
            yy, xx = torch.meshgrid(torch.arange(y0, y1, device=device), torch.arange(x0, x1, device=device), indexing='ij')
            gaussian = torch.exp(-((xx - ix).float().square() + (yy - iy).float().square()) / (2 * ((2 * radius + 1) / 6) ** 2))
            center[b, 0, y0:y1, x0:x1] = torch.maximum(center[b, 0, y0:y1, x0:x1], gaussian)
            center[b, 0, iy, ix] = 1
            negative[b, 0, iy, ix] = True
    return {'center': center, 'indices': indices, 'instance_mask': mask,
            'offset': offset, 'ltrb': ltrb, 'negative_mask': negative}


def make_corner_heatmaps(corners_hm, heatmap_size=(64, 64)):
    """Distance-exponential supervision in the heatmap-centre domain [0,W-1].

    A corner outside that centre domain is unsupported by this discrete head;
    it is not clamped to the border. Its geometric label remains unchanged.
    Physical visibility never changes a virtual cuboid corner's supervision.
    """
    torch = _torch()
    p = torch.as_tensor(corners_hm, dtype=torch.float32)
    if p.ndim != 3 or p.shape[1:] != (8, 2) or not torch.isfinite(p).all():
        raise ValueError('Expected finite [N,8,2] heatmap-domain corners')
    width, height = map(int, heatmap_size)
    if width <= 0 or height <= 0:
        raise ValueError('Positive heatmap size required')
    valid = (p[..., 0] >= 0) & (p[..., 0] <= width - 1) & (p[..., 1] >= 0) & (p[..., 1] <= height - 1)
    if p.shape[0] == 0:
        return torch.zeros((0, 8, height, width), device=p.device), valid
    yy, xx = torch.meshgrid(torch.arange(height, device=p.device), torch.arange(width, device=p.device), indexing='ij')
    xy = torch.stack((xx, yy), dim=-1).float()
    radius = torch.linalg.vector_norm(p - p.mean(dim=1, keepdim=True), dim=-1)
    scale = (radius / 10).square().clamp_min(1)
    distance = torch.linalg.vector_norm(xy[None, None] - p[:, :, None, None], dim=-1)
    maps = torch.exp(-distance / scale[..., None, None])
    maps = maps / (maps.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return maps * valid[..., None, None], valid


def build_corner_targets(records, image_indices, instance_indices, orig_to_roi,
                         roi_size=256, heatmap_size=64, device='cpu'):
    import numpy as np
    torch = _torch()
    if not len(image_indices) == len(instance_indices) == len(orig_to_roi):
        raise ValueError('Inconsistent ROI metadata')
    hm = (heatmap_size, heatmap_size) if isinstance(heatmap_size, int) else tuple(heatmap_size)
    down, _ = DataAdapter.letterbox_affine((roi_size, roi_size), hm)
    points = []
    for image_idx, instance_idx, matrix in zip(image_indices, instance_indices, orig_to_roi):
        instance = records[int(image_idx)]['instances'][int(instance_idx)]
        if is_ignored(instance):
            raise ValueError('Ignored instance cannot become a corner-loss ROI')
        points.append(DataAdapter.transform_points(instance['corners_px'], down @ matrix))
    tensor = torch.as_tensor(np.asarray(points).reshape(-1, 8, 2), dtype=torch.float32, device=device)
    return make_corner_heatmaps(tensor, hm)
