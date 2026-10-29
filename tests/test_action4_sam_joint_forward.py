"""Actual A0 surface/GDR forward with synthetic CPU ROI/mask engineering inputs.

This is not SAM inference, real-image evaluation, pose accuracy or formal training.
Only the new opt-in adapter is optimized; actual existing head is held fixed for
attribution. Parent checkpoint/default files are read-only.
"""
import hashlib
import json
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from model.action4_joint_surface_rgb import JointSurfaceRGB
from model.action4_geometry_forward import instancewise_geometry
from model.action4_sam_feature_adapter import FeatureIdentity, MaskAux, prepare_mask_aux, mask_quality_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/action4_sam_joint_forward_preparation_v001"
PARENT = ROOT / "models/default/A0.pt"
REPORT = {"scope": "ACTUAL_A0_FORWARD_SYNTHETIC_ENGINEERING_FIXTURE_NOT_PERFORMANCE",
          "formal_training": False, "sam_prediction_cache_used": False,
          "parent": str(PARENT), "device": "cpu"}
IDS = (FeatureIdentity("engineering-roi-a", "instance-a", "crop-a", "roi448"),
       FeatureIdentity("engineering-roi-b", "instance-b", "crop-b", "roi448"))


def digest_file(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grad_l1(module):
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)


def synthetic_fixture():
    # Deterministic normalized RGB448, known painted supports and explicit
    # synthetic fixed-order cuboids; no GT/SAM cache or real-pose claim.
    axis = torch.linspace(0, 1, 448)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    first = torch.stack((0.12 + 0.22 * xx, 0.18 + 0.22 * yy, 0.20 + 0.15 * xx * yy))
    second = torch.stack((0.30 - 0.12 * yy, 0.10 + 0.25 * xx, 0.25 + 0.15 * yy))
    masks = torch.zeros(2, 1, 448, 448)
    masks[0, :, 95:320, 75:230] = 1
    masks[1, :, 125:345, 220:390] = 1
    images = torch.stack((first, second))
    colors = torch.tensor([[0.72, 0.56, 0.22], [0.18, 0.63, 0.77]])[:, :, None, None]
    images = images * (1 - masks) + colors * masks
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = (images - mean) / std
    K = torch.tensor([[[170., 0., 55.5], [0., 172., 55.5], [0., 0., 1.]],
                      [[180., 0., 53.5], [0., 178., 57.5], [0., 0., 1.]]])
    canonical = torch.tensor([[-.035, -.022, -.016], [.035, -.022, -.016],
                              [.035, .022, -.016], [-.035, .022, -.016],
                              [-.035, -.022, .016], [.035, -.022, .016],
                              [.035, .022, .016], [-.035, .022, .016]])
    canonical = canonical[None].repeat(2, 1, 1)
    return rgb, K, canonical, masks


def aux_for(mask):
    return prepare_mask_aux(mask, torch.ones(2, 1, 1, 1), identities=IDS,
                            candidate_overlap=torch.tensor([[0.7], [0.6]]))


def output_equal(test, actual, expected):
    for key in expected:
        if isinstance(expected[key], dict):
            output_equal(test, actual[key], expected[key])
        else:
            test.assertTrue(torch.equal(actual[key], expected[key]), key)


def pose_record(output):
    return {name: output[name].detach().tolist() for name in ("rotation", "translation", "corners", "depth")}


class SamJointForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU-only engineering test requires empty CUDA_VISIBLE_DEVICES")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(481)
        OUT.mkdir(parents=True, exist_ok=True)
        cls.parent_sha = digest_file(PARENT)
        registry = json.loads((ROOT / "configs/model/action4_best_registry_v003.json").read_text())
        if cls.parent_sha != registry["entries"]["A0"]["sha256"]:
            raise ValueError("Actual A0 parent differs from registry")
        cls.disabled = JointSurfaceRGB(str(PARENT), device="cpu")
        cls.enabled = JointSurfaceRGB(str(PARENT), device="cpu", enable_sam=True)
        cls.disabled.eval()
        cls.enabled.train()
        # Engineering attribution: real pretrained head stays fixed; derivatives
        # still flow THROUGH its surface/GDR computations into the new adapter.
        cls.enabled.head.requires_grad_(False)
        cls.rgb, cls.K, cls.cad, cls.mask = synthetic_fixture()
        cls.correct = aux_for(cls.mask)
        cls.wrong = aux_for(cls.mask.flip(-1))
        cls.base_hash = module_digest(cls.enabled.base)
        cls.head_hash = module_digest(cls.enabled.head)
        REPORT["parent_sha256"] = cls.parent_sha
        REPORT["parent_selected_step"] = cls.enabled.parent["selected_step"]
        REPORT["fixed_head_for_adapter_attribution"] = True
        REPORT["fixture"] = {"batch": 2, "rgb_grid": 448, "surface_grid": 112,
                             "kind": "deterministic painted rectangles with synthetic cuboid/K",
                             "mask_source": "explicit synthetic engineering fixture; not SAM predictions"}

    @classmethod
    def tearDownClass(cls):
        REPORT["parent_sha_unchanged"] = digest_file(PARENT) == cls.parent_sha
        REPORT["torch_threads"] = torch.get_num_threads()
        REPORT["cuda_initialized"] = torch.cuda.is_initialized()
        (OUT / "verification.json").write_text(json.dumps(REPORT, indent=2))

    def forward_checked(self, mask=None, rgb=None, identities=IDS, K=None, cad=None):
        model = self.enabled
        calls = []
        head_inputs = []
        backbone_hook = model.base.backbone.register_forward_hook(lambda *args: calls.append(1))
        surface_hook = model.head.surface.register_forward_pre_hook(
            lambda module, args: head_inputs.append(args[0].detach().clone()))
        try:
            output, aux = model(self.rgb if rgb is None else rgb, self.K if K is None else K,
                                self.cad if cad is None else cad, mask_aux=mask,
                                identities=identities, return_aux=True)
        finally:
            backbone_hook.remove()
            surface_hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(head_inputs), 2)
        self.assertTrue(torch.equal(torch.cat(head_inputs), aux.c5))
        for value in output.values():
            for tensor in value.values() if isinstance(value, dict) else (value,):
                self.assertTrue(torch.isfinite(tensor).all())
        return output, aux

    def test_actual_surface_pose_updates_and_quality_isolation(self):
        model = self.enabled
        with torch.no_grad():
            original_route = instancewise_geometry(self.disabled.head,
                                                    self.disabled.base.backbone(self.rgb),
                                                    self.K, self.cad)
            default = self.disabled(self.rgb, self.K, self.cad)
            output_equal(self, default, original_route)
            initial, initial_aux = self.forward_checked(self.correct)
            wrong0, _ = self.forward_checked(self.wrong)
            missing0, _ = self.forward_checked()
            for value in (initial, wrong0, missing0):
                output_equal(self, value, default)
        expected_old_keys = {"head." + name for name in self.disabled.head.state_dict()}
        expected_old_keys |= {"base." + name for name in self.disabled.base.state_dict()}
        self.assertEqual(set(self.disabled.state_dict()), expected_old_keys)
        sam_parameters = [(name, p) for name, p in model.named_parameters() if name.startswith("sam_adapter.")]
        self.assertGreater(len(sam_parameters), 0)
        self.assertTrue(all(p.requires_grad for _, p in sam_parameters))
        self.assertFalse(any(p.requires_grad for p in model.base.parameters()))
        optimizer = torch.optim.AdamW(model.sam_adapter.parameters(), lr=0.001, weight_decay=0)
        # Synthetic detached target derived from initial ACTUAL projected corners;
        # it is an engineering perturbation, not a claimed ground-truth pose.
        target = initial["corners"].detach() + torch.tensor([2.0, -1.5])
        updates = []
        for step in range(2):
            model.zero_grad(set_to_none=True)
            output, aux = self.forward_checked(self.correct)
            localization = F.smooth_l1_loss(output["corners"] / 112, target / 112, beta=0.05)
            localization.backward()
            row = {"step": step + 1, "actual_corner_loss": float(localization.detach()),
                   "projection_grad_l1": grad_l1(model.sam_adapter.projection),
                   "encoder_grad_l1": grad_l1(model.sam_adapter.encoder),
                   "quality_grad_l1": grad_l1(model.sam_adapter.quality_head)}
            self.assertGreater(row["projection_grad_l1"], 0)
            self.assertEqual(row["quality_grad_l1"], 0)
            if step == 0:
                self.assertEqual(row["encoder_grad_l1"], 0)
            else:
                self.assertGreater(row["encoder_grad_l1"], 0)
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            torch.nn.utils.clip_grad_norm_(model.sam_adapter.parameters(), 1.0)
            optimizer.step()
            updates.append(row)
        with torch.no_grad():
            correct, correct_aux = self.forward_checked(self.correct)
            wrong, _ = self.forward_checked(self.wrong)
            missing, _ = self.forward_checked()
        corner_change = float((correct["corners"] - default["corners"]).abs().max())
        wrong_corner_delta = float((correct["corners"] - wrong["corners"]).abs().max())
        pose_delta = float((correct["translation"] - wrong["translation"]).abs().max())
        rotation_delta = float((correct["rotation"] - wrong["rotation"]).abs().max())
        surface_delta = float((correct["surface"]["masks"] - wrong["surface"]["masks"]).abs().max())
        self.assertGreater(corner_change, 1e-6)
        self.assertGreater(wrong_corner_delta, 1e-6)
        self.assertGreater(max(pose_delta, rotation_delta), 1e-8)
        self.assertGreater(surface_delta, 1e-6)
        output_equal(self, missing, default)
        self.assertEqual(module_digest(model.base), self.base_hash)
        self.assertEqual(module_digest(model.head), self.head_hash)
        self.assertTrue(all(not m.training for m in model.base.modules()))
        self.assertTrue(torch.equal(self.rgb, synthetic_fixture()[0]))
        REPORT["two_actual_corner_updates"] = updates
        REPORT["actual_output_effect"] = {
            "max_corner_change112_px": corner_change,
            "correct_wrong_corner_delta112_px": wrong_corner_delta,
            "correct_wrong_translation_delta": pose_delta,
            "correct_wrong_rotation_delta": rotation_delta,
            "correct_wrong_surface_mask_logits_delta": surface_delta,
            "missing_exact_original_actual_output": True,
            "one_backbone_per_forward_verified": True, "surface_received_exact_enhanced_c5": True,
            "backbone_parameters_and_buffers_unchanged": True, "actual_head_weights_unchanged": True}
        REPORT["actual_outputs"] = {"initial": pose_record(default), "correct_mask_after": pose_record(correct),
                                    "wrong_mask_after": pose_record(wrong), "missing_mask_after": pose_record(missing)}
        REPORT["new_parameter_names"] = [name for name, _ in sam_parameters]
        # Prove quality isolation through the real backbone/geometry wrapper.
        model.zero_grad(set_to_none=True)
        grad_rgb = self.rgb.detach().clone().requires_grad_()
        _, quality_aux = self.forward_checked(self.correct, rgb=grad_rgb)
        quality = mask_quality_loss(quality_aux, torch.zeros_like(self.mask), gt_identities=IDS)
        (0.1 * quality.loss_sum / quality.count).backward()
        self.assertGreater(grad_l1(model.sam_adapter.quality_head), 0)
        self.assertTrue(all(p.grad is None for p in model.sam_adapter.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.sam_adapter.projection.parameters()))
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertIsNone(grad_rgb.grad)
        REPORT["quality_actual_wrapper"] = {"eligible_count": int(quality.count),
                                           "only_quality_head_gradient": True, "rgb_gradient": None}
        self.assertEqual(digest_file(PARENT), self.parent_sha)

    def test_enabled_instance_binding_and_joint_permutation(self):
        p = torch.tensor([1, 0])
        aux = MaskAux(self.correct.visible[p], self.correct.distance[p], self.correct.available[p],
                      self.correct.candidate_overlap[p], self.correct.dropout[p], IDS[::-1])
        with torch.no_grad():
            baseline, _ = self.forward_checked(self.correct)
            swapped, _ = self.forward_checked(aux, rgb=self.rgb[p], K=self.K[p], cad=self.cad[p], identities=IDS[::-1])
        for name in ("rotation", "translation", "corners", "depth"):
            self.assertTrue(torch.equal(baseline[name][p], swapped[name]), name)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.enabled(self.rgb, self.K, self.cad, mask_aux=aux, identities=IDS)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.enabled(self.rgb, self.K, self.cad, mask_aux=self.correct)
        REPORT["instance_alignment"] = {"joint_rgb_K_CAD_mask_id_permutation_equal": True,
                                        "mask_only_identity_swap_rejected": True}

    def test_disabled_rejects_mask_and_keeps_original_return_contract(self):
        self.assertIsNone(self.disabled.sam_adapter)
        self.assertFalse(self.disabled.enable_sam)
        self.assertFalse(any(name.startswith("sam_adapter.") for name, _ in self.disabled.named_parameters()))
        with self.assertRaisesRegex(ValueError, "enable_sam"):
            self.disabled(self.rgb, self.K, self.cad, mask_aux=self.correct)
        with torch.no_grad():
            output, auxiliary = self.disabled(self.rgb, self.K, self.cad, return_aux=True)
        self.assertIsNone(auxiliary)
        self.assertIn("corners", output)
        REPORT["disabled_contract"] = {"default_disabled": True, "no_extra_state_keys": True,
                                       "mask_without_opt_in_rejected": True}


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamJointForwardTests))
    REPORT.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  passed=result.wasSuccessful())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
