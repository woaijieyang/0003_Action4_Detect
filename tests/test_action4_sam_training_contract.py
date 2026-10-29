"""Real CPU optimizer/loss/bundle contract with explicit synthetic engineering items.

No formal train/eval, SAM prediction cache, GPU, default/model/data write.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from model.training.action4_stage2_core import construct, optimizer_for, loss_sums_on_batch, losses_on_batch, bundle_snapshot
from model.training.action4_supervision import local_counts, weighted_objective, step_if_supervised, A0_KEYS
from model.action4_release_bundle import load_bundle, export, warm_start, SCHEMA, SAM_SCHEMA
from model.action4_sam_joint import QUALITY_KEY, QUALITY_PREFIX, SAM_VARIANT, SAM_ARCHITECTURE, BUNDLE_PROTOCOL
from model.action4_sam_feature_adapter import PROTOCOL_ID, MaskAux
from model.action4_sparse_surface_losses import sample_points
from test_action4_sam_joint_forward import synthetic_fixture, aux_for, IDS, module_digest, grad_l1, output_equal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/action4_sam_training_contract_preparation_v001"
PARENT = ROOT / "models/default/A0.pt"
REPORT = {"scope": "CPU_SYNTHETIC_TRAINING_CONTRACT_AND_REAL_BUNDLE_ROUNDTRIP_NOT_FORMAL_TRAINING",
          "data_eval_pipeline_integrated": False, "sam_prediction_cache_used": False,
          "formal_training": False, "device": "cpu"}


def file_sha(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixture_config(sam=True):
    cfg = {"role": "A0", "mode": "integration", "train_backbone": False,
           "parent_bundle": str(PARENT.relative_to(ROOT)), "parent_best_step": 14000,
           "parent_bundle_sha256": file_sha(PARENT), "surface_lr": 1e-5, "lr": 5e-5,
           "weight_decay": 1e-4, "loss_weights": {key: 1.0 for key in A0_KEYS},
           "source_scope": "synthetic engineering fixture, no formal data source"}
    if sam:
        cfg.update(a0_variant=SAM_VARIANT, sam_enabled=True, sam_protocol_id=PROTOCOL_ID,
                   sam_adapter_lr=1e-4, sam_quality_lr=1e-4,
                   independent_head_dependencies=[{"parameter_prefixes": [QUALITY_PREFIX], "losses": [QUALITY_KEY]}])
        cfg["loss_weights"][QUALITY_KEY] = 0.1
    return cfg


def slice_aux(aux, i):
    return MaskAux(aux.visible[i:i+1], aux.distance[i:i+1], aux.available[i:i+1],
                   aux.candidate_overlap[i:i+1], aux.dropout[i:i+1], (aux.identities[i],))


def items_from_output(rgb, K, cad, masks, aux, output):
    result = []
    xy = torch.tensor([[25., 35.], [55., 60.], [85., 80.]])
    for i in range(2):
        result.append({
            "active": True, "roi": rgb[i].clone(),
            "geometry": {"K": K[i], "canonical": cad[i], "rotation": output["rotation"][i].detach(),
                         "translation": output["translation"][i].detach() + torch.tensor([0., 0., .002]),
                         "corners": output["corners"][i].detach() + torch.tensor([1., -.75]),
                         "diagonal": torch.tensor(100.),
                         "visible": torch.tensor([True, True, True, True, False, False, False, False])},
            "surface_masks": F.interpolate(masks[i:i+1], size=(112, 112), mode="nearest")[0].repeat(2, 1, 1),
            "surface_sparse": {"xy_hm": xy,
                               "xyz": sample_points(output["surface"]["xyz"][i].detach(), xy) + .001,
                               "regions": torch.tensor([1, 2, 3])},
            "metadata": {"target_visible_input_support_count": int(masks[i].sum()),
                         "target_identity_valid": True, "gt_geometry_valid": True},
            "sam_enabled": True, "mask_aux": slice_aux(aux, i), "identity": IDS[i],
            "gt_identity": IDS[i], "gt_visible448": masks[i].clone(), "sam_quality_supervised": True,
        })
    return result


def tensor_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensor_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(tensor_state_equal(x, y) for x, y in zip(a, b))
    return a == b


class SamTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Expected CPU-only environment")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(612)
        OUT.mkdir(parents=True, exist_ok=True)
        cls.cfg = fixture_config()
        cls.model, cls.parent = construct(cls.cfg, "cpu")
        cls.model.train()
        cls.rgb, cls.K, cls.cad, cls.masks = synthetic_fixture()
        cls.aux = aux_for(cls.masks)
        with torch.no_grad():
            original = cls.model(cls.rgb, cls.K, cls.cad, mask_aux=cls.aux, identities=IDS)
        cls.items = items_from_output(cls.rgb, cls.K, cls.cad, cls.masks, cls.aux, original)
        cls.base_digest = module_digest(cls.model.base)
        cls.parent_sha = file_sha(PARENT)

    def test_actual_optimizer_counts_updates_and_bundle_roundtrip(self):
        model, cfg, items = self.model, self.cfg, self.items
        optimizer = optimizer_for(model, cfg)
        expected_lr = {"surface": 1e-5, "pose": 5e-5, "sam_adapter": 1e-4, "sam_quality": 1e-4}
        groups = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(groups, expected_lr)
        self.assertTrue(all(group["weight_decay"] == 1e-4 for group in optimizer.param_groups))
        owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned), {id(p) for p in model.parameters() if p.requires_grad})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-7)
        counts = local_counts(items, sam_enabled=True)
        self.assertEqual(set(counts), set(A0_KEYS) | {QUALITY_KEY})
        self.assertEqual(counts[QUALITY_KEY], 2)
        before = {name: next(module.parameters()).detach().clone() for name, module in
                  (("surface", model.head.surface), ("pose", model.head.pose),
                   ("adapter", model.sam_adapter.projection), ("quality", model.sam_adapter.quality_head))}
        updates = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            sums = loss_sums_on_batch(model, items, cfg, "cpu")
            self.assertEqual(set(sums), set(counts))
            objective = weighted_objective(sums, counts, cfg["loss_weights"], 1)
            self.assertTrue(torch.isfinite(objective))
            objective.backward()
            row = {"step": step + 1, "objective": float(objective.detach()),
                   "projection_grad_l1": grad_l1(model.sam_adapter.projection),
                   "encoder_grad_l1": grad_l1(model.sam_adapter.encoder),
                   "quality_grad_l1": grad_l1(model.sam_adapter.quality_head)}
            self.assertGreater(row["projection_grad_l1"], 0)
            self.assertGreater(row["quality_grad_l1"], 0)
            if step == 1:
                self.assertGreater(row["encoder_grad_l1"], 0)
            did_update, norm = step_if_supervised(model, optimizer, scheduler, counts, cfg["loss_weights"],
                                                  cfg["independent_head_dependencies"], clip=1.)
            self.assertTrue(did_update)
            row["preclip_norm"] = norm
            updates.append(row)
        for name, module in (("surface", model.head.surface), ("pose", model.head.pose),
                             ("adapter", model.sam_adapter.projection), ("quality", model.sam_adapter.quality_head)):
            self.assertFalse(torch.equal(before[name], next(module.parameters())), name)
        self.assertEqual(module_digest(model.base), self.base_digest)
        REPORT["optimizer"] = {"initial_lrs": groups, "weight_decay": 1e-4,
                               "coverage_exact_unique": True, "all_four_groups_changed": True}
        REPORT["two_updates"] = updates

        # Quality eligibility counts stay present for all-mask-missing and dummy.
        no_quality = [dict(item, sam_quality_supervised=False, gt_visible448=torch.full_like(item["gt_visible448"], float("nan")))
                      for item in items]
        cq = local_counts(no_quality, sam_enabled=True)
        self.assertEqual(cq[QUALITY_KEY], 0)
        self.assertEqual(set(cq), set(counts))
        quality_params = list(model.sam_adapter.quality_head.parameters())
        old_quality = [p.detach().clone() for p in quality_params]
        old_states = [copy.deepcopy(optimizer.state[p]) for p in quality_params]
        optimizer.zero_grad(set_to_none=True)
        sums = loss_sums_on_batch(model, no_quality, cfg, "cpu")
        self.assertEqual(float(sums[QUALITY_KEY].detach()), 0.)
        weighted_objective(sums, cq, cfg["loss_weights"], 1).backward()
        did_update, _ = step_if_supervised(model, optimizer, scheduler, cq, cfg["loss_weights"],
                                           cfg["independent_head_dependencies"])
        self.assertTrue(did_update)
        for p, old, state in zip(quality_params, old_quality, old_states):
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, old))
            self.assertTrue(tensor_state_equal(optimizer.state[p], state))
        REPORT["quality_no_supervision"] = {"global_quality_count": 0, "shared_update_happened": True,
                                            "quality_params_moments_and_step_unchanged": True, "grad_none": True}

        dummy = [dict(item, active=False) for item in items]
        cd = local_counts(dummy, sam_enabled=True)
        self.assertEqual(set(cd), set(counts))
        self.assertTrue(all(v == 0 for v in cd.values()))
        old_step = scheduler.last_epoch
        did_update, _ = step_if_supervised(model, optimizer, scheduler, cd, cfg["loss_weights"],
                                           cfg["independent_head_dependencies"])
        self.assertFalse(did_update)
        self.assertEqual(scheduler.last_epoch, old_step)
        REPORT["all_dummy"] = {"keys_same": True, "global_zero_optimizer_scheduler_skipped": True}

        # Actual final geometry and quality must survive complete bundle save/load.
        with torch.no_grad():
            reference, reference_aux = model(self.rgb, self.K, self.cad, mask_aux=self.aux, identities=IDS, return_aux=True)
        state = bundle_snapshot(model, self.parent, cfg, 3)
        self.assertEqual(state["architecture"], SAM_ARCHITECTURE)
        self.assertEqual(state["schema"], SAM_SCHEMA)
        self.assertEqual(state["sam_protocol"], BUNDLE_PROTOCOL)
        self.assertIn("_extra_state", state["sam_adapter"])
        bundle = OUT / "engineering_a0_sam_step3_sam_schema_v1.pt"
        torch.save(state, bundle)
        loaded, base, metadata = load_bundle(str(bundle), "A0", "cpu")
        with torch.no_grad():
            actual, actual_aux = loaded(base.backbone(self.rgb), self.K, self.cad,
                                        rgb=self.rgb, mask_aux=self.aux, identities=IDS, return_aux=True)
        output_equal(self, actual, reference)
        self.assertTrue(torch.equal(actual_aux.quality_logits, reference_aux.quality_logits))
        self.assertTrue(torch.equal(actual_aux.c5, reference_aux.c5))
        resumed_cfg = dict(cfg, parent_bundle=str(bundle.relative_to(ROOT)), parent_best_step=3,
                           parent_bundle_sha256=file_sha(bundle))
        resumed, actual_parent = construct(resumed_cfg, "cpu")
        with torch.no_grad():
            resumed_output = resumed(self.rgb, self.K, self.cad, mask_aux=self.aux, identities=IDS)
        output_equal(self, resumed_output, reference)
        self.assertEqual(actual_parent["selected_step"], 3)
        REPORT["roundtrip"] = {"path": str(bundle.relative_to(ROOT)), "sha256": file_sha(bundle),
                               "strict_geometry_adapter_load": True, "actual_surface_pose_corners_bitwise_equal": True,
                               "quality_logits_and_enhanced_c5_bitwise_equal": True,
                               "enhanced_parent_restored_without_dropping_adapter": True,
                               "parent_selected_step": state["parent_selected_step"],
                               "scope": "engineering artifact; not selected/formal checkpoint"}
        # A real old immutable loader must reject before silently discarding SAM.
        import subprocess, sys
        code = ROOT / "runtime/action4_runs/p2_b0_formal80_v001/source"
        program = "\n".join([
            "import sys, pathlib, os, json",
            "sys.dont_write_bytecode=True",
            "root=pathlib.Path(" + repr(str(ROOT)) + ")",
            "code=pathlib.Path(" + repr(str(code)) + ")",
            "os.environ['ACTION4_PROJECT_ROOT']=str(root)",
            "sys.path[:0]=[str(code/'runtime/surface_deps'),str(code/'src')]",
            "import torch;torch.set_num_threads(2);torch.set_num_interop_threads(1)",
            "import model.action4_release_bundle as old",
            "try:",
            " old.load_bundle(" + repr(str(bundle)) + ", 'A0', 'cpu')",
            "except ValueError as error:",
            " assert 'schema' in str(error), str(error)",
            " print(json.dumps({'rejected':True,'loader':old.__file__,'reason':str(error)}))",
            "else: raise RuntimeError('Old loader silently accepted enhanced bundle')",
        ])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        torch.set_num_threads(2)
        try:
            check = subprocess.run([sys.executable, "-I", "-c", program], cwd=ROOT, env=env,
                                   capture_output=True, text=True)
        finally:
            torch.set_num_threads(4)
        (OUT / "legacy_loader_schema_rejection.stdout.log").write_text(check.stdout)
        (OUT / "legacy_loader_schema_rejection.stderr.log").write_text(check.stderr)
        self.assertEqual(check.returncode, 0, check.stderr)
        REPORT["old_immutable_loader"] = json.loads(check.stdout)
        # Strict missing-key and protocol rejection, without constructing a fake real checkpoint.
        broken = dict(state)
        broken["sam_adapter"] = dict(state["sam_adapter"])
        broken["sam_adapter"].pop("projection.bias")
        with mock.patch("model.action4_release_bundle.torch.load", return_value=broken):
            with self.assertRaises(RuntimeError):
                load_bundle(str(bundle), "A0", "cpu")
        old_schema = dict(state, schema=SCHEMA)
        with mock.patch("model.action4_release_bundle.torch.load", return_value=old_schema):
            with self.assertRaisesRegex(ValueError, "schema/architecture"):
                load_bundle(str(bundle), "A0", "cpu")
        bad_protocol = dict(state, sam_protocol={"protocol_id": "wrong"})
        with mock.patch("model.action4_release_bundle.torch.load", return_value=bad_protocol):
            with self.assertRaisesRegex(ValueError, "protocol"):
                load_bundle(str(bundle), "A0", "cpu")
        from model.action4_joint_surface_rgb import JointSurfaceRGB
        with self.assertRaisesRegex(ValueError, "discard"):
            JointSurfaceRGB(str(bundle), enable_sam=False)
        with self.assertRaisesRegex(ValueError, "parameter loss"):
            bundle_snapshot(model, self.parent, fixture_config(False), 3)
        with mock.patch("model.action4_release_bundle.load_bundle", return_value=(loaded, base, state)):
            with self.assertRaisesRegex(ValueError, "Enhanced warm start"):
                warm_start(str(bundle), "A0", device="cpu")
        cfgpath = OUT / "engineering_config.json"
        cfgpath.write_text(json.dumps(cfg))
        selection = OUT / "legacy_export_attempt.json"
        selection.write_text(json.dumps({"a0_config": str(cfgpath.relative_to(ROOT)),
                                          "a0_checkpoint": str(bundle.relative_to(ROOT))}))
        destination = OUT / "legacy_must_not_publish"
        with self.assertRaisesRegex(ValueError, "Legacy export"):
            export(str(selection.relative_to(ROOT)), str(destination.relative_to(ROOT)))
        self.assertFalse(destination.exists())
        REPORT["fail_closed"] = {"missing_adapter_key": True, "protocol_mismatch": True,
                                 "enhanced_parent_as_baseline": True, "serializer_model_cfg_mismatch": True,
                                 "legacy_warm_start": True, "legacy_export_no_output": True}
        self.assertEqual(file_sha(PARENT), self.parent_sha)

    def test_count_normalization_gradient_separation_and_baseline(self):
        model, cfg, items = self.model, self.cfg, self.items
        model.zero_grad(set_to_none=True)
        rgb_items = [dict(item, roi=item["roi"].detach().clone().requires_grad_()) for item in items]
        sums = loss_sums_on_batch(model, rgb_items, cfg, "cpu")
        sums[QUALITY_KEY].backward()
        self.assertGreater(grad_l1(model.sam_adapter.quality_head), 0)
        for module in (model.head, model.sam_adapter.encoder, model.sam_adapter.projection, model.base):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(all(item["roi"].grad is None for item in rgb_items))
        model.zero_grad(set_to_none=True)
        sums = loss_sums_on_batch(model, items, cfg, "cpu")
        (sums["visible_corners"] + sums["hidden_corners"]).backward()
        self.assertGreater(grad_l1(model.sam_adapter.projection), 0)
        self.assertTrue(all(p.grad is None for p in model.sam_adapter.quality_head.parameters()))
        with torch.no_grad():
            full = loss_sums_on_batch(model, items, cfg, "cpu")
            split = [loss_sums_on_batch(model, [item], cfg, "cpu") for item in items]
        for key in full:
            self.assertTrue(torch.allclose(full[key], split[0][key] + split[1][key], rtol=2e-5, atol=2e-5), key)
        cf = local_counts(items, sam_enabled=True)
        cs = [local_counts([item], sam_enabled=True) for item in items]
        self.assertEqual(cf, {k: cs[0][k] + cs[1][k] for k in cf})
        REPORT["normalization"] = {"sum_count_partition_consistent": True, "all_terms_keep_global_units": True,
                                   "quality_gradient_isolated": True, "corner_gradient_reaches_adapter": True}
        missing = []
        for item in items:
            aux = item["mask_aux"]
            no_mask = MaskAux(torch.zeros_like(aux.visible), torch.zeros_like(aux.distance),
                              torch.zeros_like(aux.available), aux.candidate_overlap, aux.dropout, aux.identities)
            missing.append(dict(item, mask_aux=no_mask))
        cm = local_counts(missing, sam_enabled=True)
        self.assertEqual(set(cm), set(cf))
        self.assertEqual(cm[QUALITY_KEY], 0)
        baseline_cfg = fixture_config(False)
        baseline, baseline_parent = construct(baseline_cfg, "cpu")
        base_items = [{k: v for k, v in item.items() if k not in
                       ("sam_enabled", "mask_aux", "identity", "gt_identity", "gt_visible448", "sam_quality_supervised")}
                      for item in items]
        self.assertEqual(set(local_counts(base_items)), set(A0_KEYS))
        with self.assertRaisesRegex(ValueError, "explicit enhanced"):
            loss_sums_on_batch(baseline, items, baseline_cfg, "cpu")
        baseline_optim = optimizer_for(baseline, baseline_cfg)
        self.assertEqual(len(baseline_optim.param_groups), 2)
        with torch.no_grad():
            result = baseline(self.rgb, self.K, self.cad)
            from model.action4_geometry_forward import instancewise_geometry
            original = instancewise_geometry(baseline.head, baseline.base.backbone(self.rgb), self.K, self.cad)
        output_equal(self, result, original)
        snapshot = bundle_snapshot(baseline, baseline_parent, baseline_cfg, 0)
        self.assertEqual(snapshot["architecture"], "joint_pose")
        self.assertNotIn("sam_adapter", snapshot)
        self.assertNotIn("sam_protocol", snapshot)
        self.assertFalse(any(k.startswith("sam_adapter.") for k in baseline.state_dict()))
        with self.assertRaisesRegex(ValueError, "globally reduced"):
            losses_on_batch(model, items, cfg, "cpu")
        REPORT["baseline"] = {"counts_keys_unchanged": True, "optimizer_two_groups": True,
                              "state_keys_no_sam": True, "actual_output_original_route_equal": True}
        self.assertEqual(module_digest(model.base), self.base_digest)

    def test_invalid_contracts_rejected_before_training(self):
        from model.action4_sam_joint import validate_sam_training_config
        bad = copy.deepcopy(self.cfg)
        bad["sam_enabled"] = False
        with self.assertRaisesRegex(ValueError, "agree"):
            validate_sam_training_config(bad)
        for prefix in ("sam_adapter", "head"):
            bad = copy.deepcopy(self.cfg)
            bad["independent_head_dependencies"][0]["parameter_prefixes"] = [prefix]
            with self.assertRaisesRegex(ValueError, "ownership"):
                validate_sam_training_config(bad)
        bad = copy.deepcopy(self.cfg)
        bad["sam_adapter_lr"] = 1e-3
        with self.assertRaisesRegex(ValueError, "learning-rate"):
            validate_sam_training_config(bad)
        with self.assertRaisesRegex(ValueError, "marked items"):
            local_counts([self.items[0], dict(self.items[1], sam_enabled=False)], sam_enabled=True)
        with self.assertRaisesRegex(ValueError, "GT identity"):
            local_counts([dict(self.items[0], gt_identity=IDS[1])], sam_enabled=True)
        selection = OUT / "legacy_declared_sam_kind_attempt.json"
        selection.write_text(json.dumps({"a0_kind": SAM_ARCHITECTURE}))
        with self.assertRaisesRegex(ValueError, "Legacy export"):
            export(str(selection.relative_to(ROOT)), str((OUT / "undeclared_legacy_output").relative_to(ROOT)))
        self.assertFalse((OUT / "undeclared_legacy_output").exists())
        REPORT["invalid_contracts_rejected"] = True


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamTrainingContractTests))
    REPORT.update(tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                  passed=result.wasSuccessful(), torch_threads=torch.get_num_threads(),
                  cuda_initialized=torch.cuda.is_initialized(), parent_unchanged=file_sha(PARENT)==SamTrainingContractTests.parent_sha)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
