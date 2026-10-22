"""CPU functional checks, not dataset evaluation or a D0 convergence claim."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from model.action4_decode import decode_corners, decode_detections
from model.action4_losses import corner_heatmap_loss, detection_losses, focal_center_loss
from model.action4_network import Action4Net


class Action4ModelTests(unittest.TestCase):
    def test_corner_bias_variant_changes_only_declared_initialization(self):
        from model.action4_network import Action4Net
        torch.manual_seed(77)
        default=Action4Net(weights=None,decoder_channels=16,head_channels=8)
        torch.manual_seed(77)
        candidate=Action4Net(weights=None,decoder_channels=16,head_channels=8,corner_probability_bias=.1)
        differences=[k for k,v in default.state_dict().items() if not torch.equal(v,candidate.state_dict()[k])]
        self.assertEqual(differences,["corner_head.3.bias"])
        torch.testing.assert_close(candidate.corner_head[-1].bias.sigmoid(),torch.full((8,),.1))
        torch.testing.assert_close(default.corner_head[-1].bias,torch.zeros(8))
        for invalid in (0,1,-.1,float('nan'),True):
            with self.assertRaises(ValueError):Action4Net(weights=None,corner_probability_bias=invalid)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261018)

    def test_shared_backbone_gradient_update_and_bn_freeze(self):
        # ResNet-50 is unchanged. Decoder width 64 and 64x64 inputs make this an
        # explicitly small functional test, not the formal 256-wide training.
        network = Action4Net(weights=None, decoder_channels=64, head_channels=32).train()
        observations = []
        hook = network.backbone.register_forward_hook(lambda _m, _a, out: observations.append(out.shape))
        first = next(network.backbone.parameters())
        before = first.detach().clone()
        running_before = network.backbone.bn1.running_mean.clone()
        rgb = torch.randn(1, 3, 64, 64)
        rois = torch.randn(1, 3, 64, 64)

        def roi_builder(dense):
            self.assertTrue(all(not value.requires_grad for value in dense.values()))
            return rois, {"image_indices": [0], "source": "test"}

        outputs = network(rgb, roi_builder=roi_builder)
        hook.remove()
        self.assertEqual(len(observations), 2)
        self.assertEqual(outputs["center_logits"].shape, (1, 1, 16, 16))
        self.assertEqual(outputs["corner_logits"].shape, (1, 8, 16, 16))
        self.assertTrue(bool((outputs["ltrb"] >= 0).all()))
        self.assertEqual(outputs["roi_metadata"]["source"], "test")
        # This loss uses the ROI branch only: it must still update the backbone.
        loss = outputs["corner_logits"].square().mean()
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertTrue(bool(torch.isfinite(first.grad).all()))
        self.assertGreater(float(first.grad.abs().sum()), 0)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        optimizer.step()
        self.assertFalse(torch.equal(before, first.detach()))
        self.assertTrue(torch.equal(running_before, network.backbone.bn1.running_mean))
        self.assertFalse(network.backbone.bn1.training)
        self.assertTrue(network.backbone.bn1.weight.requires_grad)
        network.set_backbone_trainable(False)
        self.assertTrue(all(not parameter.requires_grad for parameter in network.backbone.parameters()))
        network.set_backbone_trainable(True)
        self.assertTrue(all(parameter.requires_grad for parameter in network.backbone.parameters()))
        self.assertAlmostEqual(float(network.center_head[-1].bias.detach()), math.log(0.01 / 0.99), places=6)

    def test_empty_roi_branch_keeps_connected_zero(self):
        network = Action4Net(weights=None, decoder_channels=64, head_channels=32).train()
        outputs = network(torch.randn(1, 3, 32, 32))
        self.assertEqual(outputs["corner_logits"].shape, (0, 8, 64, 64))
        loss = corner_heatmap_loss(outputs["corner_logits"], torch.empty(0, 8, 64, 64),
                                   torch.empty(0, 8, dtype=torch.bool), torch.empty(0, dtype=torch.long), 1)
        self.assertEqual(float(loss.detach()), 0)
        (loss + outputs["center_logits"].square().mean()).backward()
        for parameter in network.corner_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertEqual(float(parameter.grad.abs().sum()), 0)

    def test_focal_extreme_logits_and_ignore_positive_priority(self):
        logits = torch.tensor([[[[100., -100.]]]], requires_grad=True)
        target = torch.tensor([[[[0., 1.]]]])
        loss = focal_center_loss(logits, target)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertAlmostEqual(float(loss.detach()), 200., places=4)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad, torch.tensor([[[[1., -1.]]]])))
        ignored = focal_center_loss(logits.detach(), target, torch.zeros_like(target, dtype=torch.bool))
        self.assertAlmostEqual(float(ignored), 100., places=4)
        empty = focal_center_loss(torch.zeros(2, 1, 1, 1), torch.zeros(2, 1, 1, 1))
        self.assertAlmostEqual(float(empty), -math.log(0.5) * 0.25, places=7)

    def test_detection_losses_count_complete_images_and_reject_collision(self):
        center = torch.zeros(2, 1, 4, 4)
        center[0, 0, 1, 1] = 1
        mask = torch.tensor([[True, False], [False, False]])
        targets = {"center": center, "indices": torch.tensor([[5, -1], [-1, -1]]),
                   "instance_mask": mask,
                   "offset": torch.full((2, 2, 2), float("nan")),
                   "ltrb": torch.full((2, 2, 4), float("nan"))}
        targets["offset"][0, 0] = 1
        targets["ltrb"][0, 0] = 1
        outputs = {"center_logits": torch.full((2, 1, 4, 4), -4., requires_grad=True),
                   "offset": torch.zeros(2, 2, 4, 4, requires_grad=True),
                   "ltrb": torch.zeros(2, 4, 4, 4, requires_grad=True)}
        losses = detection_losses(outputs, targets)
        self.assertAlmostEqual(float(losses["offset"].detach()), 0.25)
        self.assertAlmostEqual(float(losses["box"].detach()), 0.25)
        sum(losses.values()).backward()
        self.assertTrue(all(bool(torch.isfinite(value.grad).all()) for value in outputs.values()))
        targets["indices"][0, 1] = 5
        targets["instance_mask"][0, 1] = True
        with self.assertRaisesRegex(ValueError, "collide"):
            detection_losses(outputs, targets)

    def test_corner_loss_region_and_image_normalization(self):
        targets = torch.zeros(3, 8, 4, 4)
        targets[:, :, 1, 1] = 1
        logits = torch.full_like(targets, math.log(0.2 / 0.8))
        logits[2] = 0
        logits.requires_grad_(True)
        valid = torch.ones(3, 8, dtype=torch.bool)
        result = corner_heatmap_loss(logits, targets, valid, torch.tensor([0, 0, 1]), 2)
        # Image 0 has two equivalent ROIs: equal FG/BG means (0.32+0.02)/2.
        # Image 1 has one ROI: both regions 0.125. Images still get equal weight.
        self.assertAlmostEqual(float(result.detach()), (0.17 + 0.125) / 2, places=6)
        result.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        zero = corner_heatmap_loss(logits.detach().requires_grad_(), targets,
                                   torch.zeros_like(valid), torch.tensor([0, 0, 1]), 3)
        self.assertEqual(float(zero.detach()), 0)

    def test_detector_pixel_centres_empty_and_strict_duplicate_handling(self):
        center = torch.full((1, 1, 16, 16), -20.)
        center[0, 0, 5, 5] = 3
        center[0, 0, 5, 7] = 2
        outputs = {"center_logits": center, "offset": torch.zeros(1, 2, 16, 16),
                   "ltrb": torch.full((1, 4, 16, 16), 0.1)}
        outputs["offset"][0, :, 5, 5] = torch.tensor([0.25, -0.25])
        outputs["offset"][0, :, 5, 7] = torch.tensor([0.25, -0.25])
        decoded = decode_detections(outputs, image_size=(64, 64))
        self.assertEqual(len(decoded[0]), 2)
        self.assertTrue(torch.allclose(decoded[0][0]["center"], torch.tensor([22.5, 20.5])))
        self.assertTrue(torch.allclose(decoded[0][0]["bbox_full_xyxy"],
                                       torch.tensor([16.1, 14.1, 28.9, 26.9]), atol=1e-5))
        matrix = torch.tensor([[8., 0, 3.5], [0, 8., 3.5], [0, 0, 1.]])
        transformed = decode_detections(outputs, image_size=(64, 64), heatmap_to_image=matrix)
        self.assertTrue(torch.allclose(transformed[0][0]["center"], torch.tensor([45.5, 41.5])))
        self.assertEqual(decode_detections(outputs, image_size=(64, 64), score_threshold=1), [[]])
        outputs["offset"][0, 0, 5, 7] = -1.75  # Same continuous centre, distinct grid peak.
        self.assertEqual(len(decode_detections(outputs, image_size=(64, 64))[0]), 1)

    def test_ideal_corner_labels_preserve_channel_order_and_affine(self):
        points = torch.tensor([[[8.15, 9.5], [20.2, 8.3], [51.6, 10.7], [53.2, 47.8],
                                [18.1, 51.3], [31.25, 29.75], [45.3, 33.45], [8.9, 45.9]]])
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).float()
        scale = (torch.linalg.vector_norm(points - points.mean(1, keepdim=True), dim=-1) / 10).square().clamp_min(1)
        maps = torch.exp(-torch.linalg.vector_norm(grid[None, None] - points[:, :, None, None], dim=-1)
                         / scale[:, :, None, None])
        maps = maps / (maps.amax((-2, -1), keepdim=True) + 1e-8)
        result = decode_corners(maps, from_logits=False)
        error = torch.linalg.vector_norm(result - points, dim=-1)
        self.assertLess(float(torch.quantile(error, 0.95)), 1)
        matrix = torch.tensor([[4., 0, 1.5], [0, 4., 1.5], [0, 0, 1.]])
        transformed = decode_corners(maps, from_logits=False, heatmap_to_image=matrix)
        self.assertTrue(torch.allclose(transformed, result * 4 + 1.5))

    def test_local_checkpoint_rejects_hash_and_partial_state(self):
        network = Action4Net(weights=None, decoder_channels=64, head_channels=32)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "partial.pt"
            torch.save({"conv1.weight": network.backbone.conv1.weight.detach()}, source)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                network.load_backbone_weights(source, "0" * 64)
            with self.assertRaisesRegex(ValueError, "complete Torchvision"):
                network.load_backbone_weights(source)


if __name__ == "__main__":
    unittest.main()
