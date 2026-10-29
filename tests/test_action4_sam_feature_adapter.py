"""CPU-only engineering evidence. The synthetic readout is NOT pose evaluation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

import torch
from model.action4_sam_feature_adapter import (
    FeatureIdentity, MaskAux, SamFeatureAdapter, prepare_mask_aux, mask_quality_loss, PROTOCOL_ID,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/action4_sam_feature_preparation_v001"
REPORT = {"scope": "CPU engineering gradient/update tests; no actual pose metric or formal training",
          "protocol_id": PROTOCOL_ID}
IDS = (FeatureIdentity("synthetic-image", "instance-a", "roi-a", "roi448"),)


def fixture():
    generator = torch.Generator().manual_seed(1024)
    rgb = torch.randn(1, 3, 448, 448, generator=generator)
    c5 = torch.randn(1, 2048, 14, 14, generator=generator)
    correct = torch.zeros(1, 1, 448, 448)
    correct[:, :, 80:260, 40:210] = 1
    wrong = torch.zeros_like(correct)
    wrong[:, :, 160:340, 250:420] = 1
    return rgb, c5, correct, wrong


def auxiliary(mask, available=1.0, dropout=False):
    return prepare_mask_aux(mask, torch.full((1, 1, 1, 1), available, dtype=mask.dtype, device=mask.device),
                            identities=IDS, candidate_overlap=torch.tensor([[0.6]]),
                            dropout=torch.tensor([dropout]))


def grad_norm(module):
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)


def readout(features):
    # Synthetic spatial readout exercises the future C5->FPN gradient connection.
    return features[:, :8, 3:8, 2:7].mean()


class SamFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)

    def test_two_real_updates_and_strict_reload(self):
        rgb, c5, correct, wrong = fixture()
        model = SamFeatureAdapter()
        inputs = {"correct": auxiliary(correct), "wrong": auxiliary(wrong), "missing": None}
        rgb_before = rgb.clone()
        with torch.no_grad():
            initial = {key: model(c5, rgb=rgb, identities=IDS, mask_aux=aux)
                       for key, aux in inputs.items()}
        for result in initial.values():
            self.assertTrue(torch.equal(result.c5, c5))
        self.assertGreater(float(model.encoder[0].weight.detach().abs().sum()), 0)
        self.assertGreater(float(model.encoder[3].weight.detach().abs().sum()), 0)
        self.assertGreater(float(initial["correct"].q_mask), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        updates = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(c5, rgb=rgb, identities=IDS, mask_aux=inputs["correct"])
            location_loss = (readout(output.c5) - 0.75).square()
            location_loss.backward()
            row = {"step": step + 1, "synthetic_location_loss": float(location_loss.detach()),
                   "projection_grad_l1": grad_norm(model.projection),
                   "encoder_grad_l1": grad_norm(model.encoder),
                   "quality_grad_l1": grad_norm(model.quality_head)}
            self.assertGreater(row["projection_grad_l1"], 0)
            self.assertEqual(row["quality_grad_l1"], 0)
            if step == 0:
                self.assertEqual(row["encoder_grad_l1"], 0)
            else:
                self.assertGreater(row["encoder_grad_l1"], 0)
            optimizer.step()
            updates.append(row)
        with torch.no_grad():
            final = {key: model(c5, rgb=rgb, identities=IDS, mask_aux=aux)
                     for key, aux in inputs.items()}
        changed = float((final["correct"].c5 - c5).abs().max())
        mask_delta = float((final["correct"].c5 - final["wrong"].c5).abs().max())
        readout_delta = float((readout(final["correct"].c5) - readout(final["wrong"].c5)).abs())
        self.assertGreater(changed, 1e-6)
        self.assertGreater(mask_delta, 1e-6)
        self.assertGreater(readout_delta, 1e-7)
        self.assertTrue(torch.equal(final["missing"].c5, c5))
        self.assertTrue(torch.equal(rgb, rgb_before))
        OUT.mkdir(parents=True, exist_ok=True)
        saved = OUT / "engineering_adapter_state.pt"
        torch.save(model.state_dict(), saved)
        restored = SamFeatureAdapter()
        restored.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            restored_output = restored(c5, rgb=rgb, identities=IDS, mask_aux=inputs["correct"])
        self.assertTrue(torch.equal(restored_output.c5, final["correct"].c5))
        self.assertTrue(torch.equal(restored_output.quality_logits, final["correct"].quality_logits))
        broken = copy.deepcopy(model.state_dict())
        del broken["projection.bias"]
        with self.assertRaises(RuntimeError):
            SamFeatureAdapter().load_state_dict(broken, strict=True)
        bad_identity = copy.deepcopy(model.state_dict())
        bad_identity["_extra_state"]["protocol_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "protocol identity"):
            SamFeatureAdapter().load_state_dict(bad_identity, strict=True)
        REPORT["two_optimizer_updates"] = updates
        REPORT["fixed_input"] = {
            "initial_correct_wrong_missing_equal_rgb_c5": True,
            "after_updates_max_c5_change": changed,
            "correct_vs_wrong_max_c5_delta": mask_delta,
            "correct_vs_wrong_synthetic_readout_delta": readout_delta,
            "missing_exact_rgb_c5_fallback": True, "rgb_unchanged": True,
            "strict_reload_bitwise_equal": True,
            "checkpoint_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        }

    def test_quality_gradient_isolation_and_actual_mask_iou(self):
        rgb, c5, correct, wrong = fixture()
        rgb.requires_grad_()
        c5.requires_grad_()
        model = SamFeatureAdapter()
        # A perturbed prediction overlapping only half the original GT.
        perturbed = correct.clone()
        perturbed[:, :, :, :125] = 0
        aux = auxiliary(perturbed)
        out = model(c5, rgb=rgb, identities=IDS, mask_aux=aux)
        loss = mask_quality_loss(out, correct, gt_identities=IDS)
        self.assertEqual(int(loss.count), 1)
        self.assertAlmostEqual(float(loss.target_iou), 0.5, places=7)
        optimizer = torch.optim.SGD(model.quality_head.parameters(), lr=0.1)
        # Use empty GT for a known wrong nonempty prediction: target must be 0.
        negative = mask_quality_loss(out, torch.zeros_like(correct), gt_identities=IDS)
        self.assertEqual(float(negative.target_iou), 0)
        self.assertEqual(int(negative.count), 1)
        before = out.quality_logits.detach().clone()
        (0.1 * negative.loss_sum / negative.count).backward()
        self.assertGreater(grad_norm(model.quality_head), 0)
        self.assertEqual(grad_norm(model.encoder), 0)
        self.assertEqual(grad_norm(model.projection), 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertIsNone(c5.grad)
        self.assertIsNone(rgb.grad)
        optimizer.step()
        with torch.no_grad():
            after = model(c5, rgb=rgb, identities=IDS, mask_aux=aux).quality_logits
        self.assertFalse(torch.equal(before, after))
        REPORT["quality"] = {"perturbed_input_iou": float(loss.target_iou),
                             "nonempty_prediction_empty_gt_target": float(negative.target_iou),
                             "eligible_count": int(negative.count),
                             "quality_logits_change": float((after - before).abs().max()),
                             "quality_only_gradient": True}

    def test_missing_empty_dropout_and_zero_count_stay_connected(self):
        rgb, c5, correct, _ = fixture()
        model = SamFeatureAdapter()
        cases = {"missing": auxiliary(correct, available=0),
                 "empty": auxiliary(torch.zeros_like(correct)),
                 "dropout": auxiliary(correct, dropout=True)}
        counts = {}
        for name, aux in cases.items():
            model.zero_grad(set_to_none=True)
            self.assertEqual(float(aux.available.sum()), 0)
            self.assertEqual(float(aux.visible.abs().sum() + aux.distance.abs().sum()), 0)
            output = model(c5, rgb=rgb, identities=IDS, mask_aux=aux)
            loss = mask_quality_loss(output, correct, gt_identities=IDS)
            self.assertEqual(int(loss.count), 0)
            self.assertEqual(float(loss.loss_sum.detach()), 0)
            self.assertTrue(loss.loss_sum.requires_grad)
            loss.loss_sum.backward()
            self.assertTrue(all(p.grad is not None for p in model.quality_head.parameters()))
            self.assertEqual(grad_norm(model.quality_head), 0)
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            self.assertTrue(torch.equal(output.c5, c5))
            counts[name] = int(loss.count)
        output = model(c5, rgb=rgb, identities=IDS, mask_aux=auxiliary(correct))
        unknown = mask_quality_loss(output, correct, gt_identities=IDS, supervised=torch.tensor([False]))
        self.assertEqual(int(unknown.count), 0)
        REPORT["zero_count"] = {**counts, "unknown_or_dummy": int(unknown.count),
                                "connected_zero": True, "optimizer_step_not_called": True}

    def test_edt_sign_scale_and_strict_missing_fallback_gradients(self):
        rgb, c5, correct, _ = fixture()
        aux = auxiliary(correct)
        self.assertEqual(float(aux.distance[0, 0, 100, 100]), 1)
        self.assertEqual(float(aux.distance[0, 0, 80, 100]), 1 / 16)
        self.assertEqual(float(aux.distance[0, 0, 79, 100]), -1 / 16)
        self.assertEqual(float(aux.distance[0, 0, 0, 0]), -1)
        model = SamFeatureAdapter()
        with torch.no_grad():
            model.projection.weight.fill_(0.01)
            model.projection.bias.fill_(0.3)
        c5.requires_grad_()
        output = model(c5, rgb=rgb, identities=IDS)
        self.assertTrue(torch.equal(output.c5, c5))
        output.c5.sum().backward()
        self.assertTrue(torch.equal(c5.grad, torch.ones_like(c5)))
        self.assertEqual(grad_norm(model.encoder), 0)
        self.assertEqual(grad_norm(model.projection), 0)
        REPORT["edt"] = {"inside_boundary": 1/16, "outside_boundary": -1/16,
                          "clipped_range": [-1, 1], "missing_c5_gradient_identity": True}

    def test_shapes_finite_ranges_and_identity_fail_closed(self):
        from dataclasses import replace
        rgb, c5, correct, _ = fixture()
        aux = auxiliary(correct)
        model = SamFeatureAdapter()
        for bad in (c5[:, :, :-1], c5.fill_(float("nan"))):
            with self.assertRaises(ValueError):
                model(bad, rgb=rgb, identities=IDS, mask_aux=aux)
        _, c5, _, _ = fixture()
        for bad_aux in (
            replace(aux, identities=(FeatureIdentity("other", "instance-a", "roi-a", "roi448"),)),
            replace(aux, protocol_id="other"),
            replace(aux, distance=torch.full_like(aux.distance, float("inf"))),
            replace(aux, distance=aux.distance + 2),
            replace(aux, available=torch.zeros_like(aux.available)),
            replace(aux, candidate_overlap=torch.tensor([[2.0]])),
            replace(aux, visible=torch.full_like(aux.visible, 0.5)),
            replace(aux, dropout=torch.tensor([True])),
        ):
            with self.assertRaises(ValueError):
                model(c5, rgb=rgb, identities=IDS, mask_aux=bad_aux)
        with self.assertRaises(ValueError):
            model(c5, rgb=rgb[:, :, :-1], identities=IDS)
        with self.assertRaises(ValueError):
            model(c5, rgb=rgb, identities=())
        output = model(c5, rgb=rgb, identities=IDS, mask_aux=aux)
        with self.assertRaises(ValueError):
            mask_quality_loss(output, correct, gt_identities=(FeatureIdentity("x", "x", "x", "x"),))
        with self.assertRaises(ValueError):
            mask_quality_loss(output, torch.full_like(correct, float("nan")), gt_identities=IDS)
        with torch.no_grad():
            model.projection.bias.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "non-finite SAM feature"):
            model(c5, rgb=rgb, identities=IDS, mask_aux=aux)
        REPORT["validation"] = {"shape_nonfinite_range_identity_mismatch_rejected": True}

    def test_sum_count_partition_and_sparse_shape(self):
        from dataclasses import replace
        rgb, c5, correct, _ = fixture()
        ids = (IDS[0], FeatureIdentity("synthetic-image", "instance-b", "roi-b", "roi448"))
        sparse = torch.zeros_like(correct)
        sparse[:, :, 100, 20] = 1
        sparse[:, :, 100, 320] = 1
        aux = prepare_mask_aux(torch.cat((sparse, correct)), torch.ones(2, 1, 1, 1),
                               identities=ids, candidate_overlap=torch.tensor([[0.6], [0.2]]))
        model = SamFeatureAdapter()
        captured = []
        handle = model.quality_head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        out = model(c5.expand(2, -1, -1, -1), rgb=rgb.expand(2, -1, -1, -1),
                    identities=ids, mask_aux=aux)
        handle.remove()
        self.assertAlmostEqual(float(captured[0][0, 2049]), 301 / 448, places=7)
        self.assertAlmostEqual(float(captured[0][0, 2050]), 1 / 448, places=7)
        gt = torch.cat((sparse, torch.zeros_like(correct)))
        full = mask_quality_loss(out, gt, gt_identities=ids)
        partitions = []
        for i in range(2):
            one = replace(out, c5=out.c5[i:i+1], residual=out.residual[i:i+1],
                          q_mask=out.q_mask[i:i+1], quality_logits=out.quality_logits[i:i+1],
                          visible=out.visible[i:i+1], available=out.available[i:i+1],
                          dropout=out.dropout[i:i+1], identities=(ids[i],))
            partitions.append(mask_quality_loss(one, gt[i:i+1], gt_identities=(ids[i],)))
        self.assertEqual(int(full.count), 2)
        self.assertTrue(torch.equal(full.loss_sum, sum(x.loss_sum for x in partitions)))
        mixed = mask_quality_loss(out, gt, gt_identities=ids, supervised=torch.tensor([True, False]))
        self.assertEqual(int(mixed.count), 1)
        self.assertTrue(torch.equal(mixed.loss_sum, partitions[0].loss_sum))
        gt_nan = gt.clone()
        gt_nan[1] = float("nan")
        unknown_nan = mask_quality_loss(out, gt_nan, gt_identities=ids, supervised=torch.tensor([True, False]))
        self.assertEqual(int(unknown_nan.count), 1)
        self.assertTrue(torch.equal(unknown_nan.loss_sum, partitions[0].loss_sum))
        self.assertTrue(torch.isfinite(unknown_nan.target_iou).all())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mask_quality_loss(out, gt_nan, gt_identities=ids, supervised=torch.tensor([True, True]))
        unknown_all = mask_quality_loss(out, torch.full_like(gt, float("nan")), gt_identities=ids,
                                        supervised=torch.tensor([False, False]))
        self.assertEqual(int(unknown_all.count), 0)
        self.assertEqual(float(unknown_all.loss_sum.detach()), 0)
        self.assertTrue(unknown_all.loss_sum.requires_grad)
        bad_output = replace(out, q_mask=torch.full_like(out.q_mask, float("nan")))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mask_quality_loss(bad_output, gt_nan, gt_identities=ids, supervised=torch.tensor([False, False]))
        REPORT["unknown_gt"] = {"mixed_nan_count": 1, "equals_known_only": True,
                                 "all_unknown_nan_count": 0, "known_nan_rejected": True,
                                 "nonfinite_model_quality_rejected_even_unknown": True}
        REPORT["normalization"] = {"full_count": 2, "mixed_known_dummy_count": 1,
                                    "partition_sum_equal": True,
                                    "sparse_mask_bbox_width_pixels": 301}

    def test_two_instance_order_binding(self):
        rgb, c5, correct, wrong = fixture()
        ids = (IDS[0], FeatureIdentity("synthetic-image", "instance-b", "roi-b", "roi448"))
        masks = torch.cat((correct, wrong))
        aux = prepare_mask_aux(masks, torch.ones(2, 1, 1, 1), identities=ids,
                               candidate_overlap=torch.tensor([[0.6], [0.2]]))
        model = SamFeatureAdapter()
        with torch.no_grad():
            model.projection.weight.fill_(0.001)
            model.projection.bias.fill_(0.02)
            output = model(c5.expand(2, -1, -1, -1), rgb=rgb.expand(2, -1, -1, -1),
                           identities=ids, mask_aux=aux)
            permutation = torch.tensor([1, 0])
            swapped = MaskAux(aux.visible[permutation], aux.distance[permutation],
                              aux.available[permutation], aux.candidate_overlap[permutation],
                              aux.dropout[permutation], ids[::-1])
            reversed_output = model(c5.expand(2, -1, -1, -1), rgb=rgb.expand(2, -1, -1, -1),
                                    identities=ids[::-1], mask_aux=swapped)
            self.assertTrue(torch.equal(output.c5[permutation], reversed_output.c5))
            with self.assertRaisesRegex(ValueError, "identity"):
                model(c5.expand(2, -1, -1, -1), rgb=rgb.expand(2, -1, -1, -1),
                      identities=ids, mask_aux=swapped)
        REPORT["two_instance_permutation"] = "equal after joint permutation; mismatched ordering rejected"


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run this engineering test with CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamFeatureTests))
    REPORT.update({"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                   "passed": result.wasSuccessful(), "torch_version": torch.__version__,
                   "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                   "device": "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    (OUT / "verification.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
