"""Verify actual RGB sampling, formal-data gates and detector/corner targets."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from data.action4_adapter import DataAdapter
from data.action4_dataset import (
    Action4Dataset, prepare_full_images, prepare_rois,
    build_detection_targets, build_corner_targets, make_corner_heatmaps,
)
from model.action4_decode import decode_corners, decode_detections


def unnormalize(tensor):
    return tensor * tensor.new_tensor([.229, .224, .225])[None, :, None, None] + tensor.new_tensor([.485, .456, .406])[None, :, None, None]


def instance(box=(10., 8., 45., 40.), visible=None, ignore=False):
    x0, y0, x1, y1 = box
    return {'instance_id': 0, 'object_id': 1,
            'corners_px': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] * 2,
            'bbox_full_xyxy': list(box), 'bbox_visible_xyxy': list(visible or box),
            'corner_visible': [False] * 8, 'ignore': ignore,
            'occlusion_inframe': .2, 'truncation': 0.}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        base = Path('.cache/action4_dataset_tests')
        base.mkdir(parents=True, exist_ok=True)
        cls.root = Path(tempfile.mkdtemp(prefix='run-', dir=base)).resolve()

    def test_full_rgb_interpolation_agrees_with_affine_and_padding(self):
        yy, xx = torch.meshgrid(torch.arange(47), torch.arange(63), indexing='ij')
        rgb = torch.stack([xx / 63, yy / 47, torch.ones_like(xx) * .25])
        out, matrices = prepare_full_images([rgb], size=(96, 64))
        pixels = unnormalize(out)[0]
        matrix, meta = DataAdapter.letterbox_affine((63, 47), (96, 64))
        np.testing.assert_allclose(matrix, matrices[0])
        left, top, right, bottom = meta['padding_ltrb']
        self.assertGreater(left, 0)
        self.assertLess(float(pixels[:, :, :left].abs().max()), 1e-6)
        destinations = np.array([[left + 10, top + 12], [left + 60, top + 42]], float)
        source = DataAdapter.transform_points(destinations, matrix, inverse=True)
        for (dx, dy), (sx, sy) in zip(destinations.astype(int), source):
            np.testing.assert_allclose(pixels[:2, dy, dx].numpy(), [sx / 63, sy / 47], atol=3e-6)

    def test_outer_autocast_cannot_change_native_roi_coordinates_or_pixels(self):
        yy, xx = torch.meshgrid(torch.arange(768), torch.arange(1024), indexing='ij')
        rgb = torch.stack([(xx % 17).float()/16, (yy % 19).float()/18, ((xx+yy)%23).float()/22])
        box = [[811.25, 602.75, 873.65, 664.15]]
        expected, affine = prepare_rois([rgb], box, [0], roi_size=64, device='cpu')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, amp_affine = prepare_rois([rgb], box, [0], roi_size=64, device='cpu')
        self.assertTrue(torch.equal(expected, actual))
        np.testing.assert_array_equal(affine, amp_affine)

    def test_virtual_roi_rgb_matches_pixel_centre_matrix_and_black_outside(self):
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        rgb = torch.stack([xx / 64, yy / 64, torch.ones_like(xx) * .5])
        out, matrices = prepare_rois([rgb], [[12.5, 9.5, 44.5, 41.5], [-32.5, -32.5, -.5, -.5]],
                                    [0, 0], roi_size=32, expansion=1)
        out = unnormalize(out)
        np.testing.assert_allclose(out[0, :2, 0, 0].numpy(), [13 / 64, 10 / 64], atol=1e-6)
        np.testing.assert_allclose(out[0, :2, 31, 31].numpy(), [44 / 64, 41 / 64], atol=1e-6)
        np.testing.assert_allclose(DataAdapter.transform_points([[13, 10]], matrices[0]), [[0, 0]], atol=1e-6)
        self.assertLess(float(out[1].abs().max()), 1e-6)
        empty, transforms = prepare_rois([rgb], [], [], roi_size=32)
        self.assertEqual(tuple(empty.shape), (0, 3, 32, 32))
        self.assertEqual(transforms, [])

    def test_detection_targets_decode_to_original_complete_bbox(self):
        first = instance((-7, 10, 80, 70), visible=(10, 20, 54, 64))
        second = instance((100, 55, 175, 130), visible=(105, 70, 166, 120))
        second['instance_id'] = 1
        row = {'sample_id': 'two', 'instances': [first, second]}
        matrix, _ = DataAdapter.letterbox_affine((200, 150), (128, 96))
        targets = build_detection_targets([row], [matrix], full_size=(128, 96))
        logits = torch.full_like(targets['center'], -30.)
        offset = torch.zeros((1, 2, 24, 32))
        ltrb = torch.zeros((1, 4, 24, 32))
        for j in range(2):
            idx = int(targets['indices'][0, j]); y, x = divmod(idx, 32)
            logits[0, 0, y, x] = 30
            offset[0, :, y, x] = targets['offset'][0, j]
            ltrb[0, :, y, x] = targets['ltrb'][0, j]
        down, _ = DataAdapter.letterbox_affine((128, 96), (32, 24))
        predictions = decode_detections({'center_logits': logits, 'offset': offset, 'ltrb': ltrb},
                                       image_size=(128, 96), heatmap_to_image=torch.tensor(np.linalg.inv(down @ matrix)))
        self.assertEqual(len(predictions[0]), 2)
        boxes = sorted([p['bbox_full_xyxy'].numpy() for p in predictions[0]], key=lambda x: x[0])
        np.testing.assert_allclose(boxes, [first['bbox_full_xyxy'], second['bbox_full_xyxy']], atol=3e-5)

    def test_ignore_bounds_positive_priority_and_same_cell_are_explicit(self):
        outside = instance((-120, -60, -20, -10), ignore=True)
        target = build_detection_targets([{'sample_id': 'outside', 'instances': [outside]}], [np.eye(3)], full_size=(64, 64))
        self.assertTrue(bool(target['negative_mask'].all()))
        cover = instance((0, 0, 63, 63), ignore=True)
        target = build_detection_targets([{'sample_id': 'overlap', 'instances': [cover, instance()]}], [np.eye(3)], full_size=(64, 64))
        self.assertTrue(bool((~target['negative_mask']).any()))
        i = int(target['indices'][0, 1])
        self.assertTrue(bool(target['negative_mask'].flatten()[i]))
        with self.assertRaisesRegex(ValueError, 'Same heatmap cell'):
            build_detection_targets([{'sample_id': 'collision', 'instances': [instance(), instance()]}], [np.eye(3)], full_size=(64, 64))

    def test_hidden_virtual_corners_keep_supervision_without_border_clamp(self):
        row = {'sample_id': 'hidden', 'instances': [instance()]}
        images = [torch.zeros((3, 64, 64), dtype=torch.uint8)]
        _, matrices = prepare_rois(images, [row['instances'][0]['bbox_full_xyxy']], [0])
        maps, valid = build_corner_targets([row], [0], [0], matrices)
        self.assertTrue(bool(valid.all()))
        predicted = decode_corners(maps, from_logits=False)
        down, _ = DataAdapter.letterbox_affine((256, 256), (64, 64))
        expected = DataAdapter.transform_points(row['instances'][0]['corners_px'], down @ matrices[0])
        self.assertLess(float(torch.linalg.vector_norm(predicted[0] - torch.tensor(expected), dim=-1).max()), 1.)
        border = torch.full((1, 8, 2), 10.)
        border[0, 0, 0] = -.1
        maps, valid = make_corner_heatmaps(border)
        self.assertFalse(bool(valid[0, 0]))
        self.assertEqual(float(maps[0, 0].sum()), 0.)
        self.assertEqual(float(border[0, 0, 0]), float(torch.tensor(-.1)))

    def test_formal_loader_rejects_unaccepted_and_checks_split(self):
        case = Path(tempfile.mkdtemp(prefix='manifest-', dir=self.root))
        Image.new('RGB', (64, 64), (10, 20, 30)).save(case / 'sample.png')
        row = {'schema_version': 'action4.v1', 'sample_id': 'test-fixture', 'image': 'sample.png',
               'width': 64, 'height': 64, 'group_id': 'fixture-group', 'background_source_id': 'fixture-background',
               'scene_family': 'test_fixture', 'instances': [instance()], 'split': 'train',
               'status': 'diagnostic_pilot_not_training_accepted'}
        manifest = case / 'manifest.jsonl'
        manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unaccepted'):
            Action4Dataset(case, manifest)
        dataset = Action4Dataset(case, manifest, allow_diagnostic=True, split='train')
        self.assertTrue(dataset.diagnostic)
        self.assertEqual(tuple(dataset[0]['image'].shape), (3, 64, 64))
        with self.assertRaisesRegex(ValueError, 'split identity'):
            Action4Dataset(case, manifest, allow_diagnostic=True, split='test')
        row['status'] = 'training_accepted'  # An isolated engineering fixture; never a production acceptance.
        manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'requires rgb_sha256'):
            Action4Dataset(case, manifest, split='train')
        row['rgb_sha256'] = hashlib.sha256((case / 'sample.png').read_bytes()).hexdigest()
        manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')
        self.assertFalse(Action4Dataset(case, manifest, split='train').diagnostic)
        Image.new('RGB', (64, 64), (11, 20, 30)).save(case / 'sample.png')
        with self.assertRaisesRegex(ValueError, 'RGB hash mismatch'):
            Action4Dataset(case, manifest, split='train')


if __name__ == '__main__':
    unittest.main()
