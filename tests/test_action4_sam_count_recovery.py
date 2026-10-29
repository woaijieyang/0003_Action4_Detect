"""Configured-rank schema and real CPU same-run recovery for the SAM enhancement.

These are synthetic engineering batches, not formal training or multi-rank DDP.
"""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import unittest
import numpy as np
import torch

from model.training.action4_stage2_core import construct, optimizer_for, loss_sums_on_batch
from model.training.action4_supervision import local_counts, weighted_objective, has_supervision, step_if_supervised, A0_KEYS
from model.training.action4_runtime import save_checkpoint_atomic, canonical_hash
from model.training.action4_component_train import rng_state, restore_rng
from model.action4_sam_joint import sam_variant, QUALITY_KEY
from model.action4_sam_feature_adapter import MaskAux
from test_action4_sam_training_contract import fixture_config, items_from_output, tensor_state_equal, file_sha
from test_action4_sam_joint_forward import synthetic_fixture, aux_for, IDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/action4_sam_training_contract_preparation_v001/exact_recovery_v001"
REPORT = {"scope": "SAM_ENHANCEMENT_CPU_CONFIGURED_SCHEMA_AND_SAME_RUN_RECOVERY",
          "formal_training": False, "distributed_collective_executed": False, "world_size": 1}


def metadata_item(index):
    _, K, cad, masks = synthetic_fixture()
    aux = aux_for(masks)
    one = MaskAux(aux.visible[index:index+1], aux.distance[index:index+1], aux.available[index:index+1],
                  aux.candidate_overlap[index:index+1], aux.dropout[index:index+1], (IDS[index],))
    return {"active": True, "geometry": {"visible": torch.tensor([True]*4 + [False]*4)},
            "surface_masks": torch.ones(2,112,112), "surface_sparse": {"xy_hm": torch.empty(0,2)},
            "metadata": {"target_visible_input_support_count": 1, "target_identity_valid": True, "gt_geometry_valid": True},
            "sam_enabled": True, "mask_aux": one, "identity": IDS[index], "gt_identity": IDS[index],
            "gt_visible448": masks[index], "sam_quality_supervised": True}


def materialize_attempt(items, kind, rng):
    # Explicit deterministic engineering data order and augmentation. All four RNG
    # streams are consumed even for zero-supervision attempts, matching cursor/RNG
    # advancement semantics rather than replaying until an update succeeds.
    order = rng.permutation(len(items)).tolist()
    scale = 0.99 + 0.02 * random.random()
    shift = float(np.random.uniform(-0.002, 0.002))
    batch = []
    for index in order:
        item = dict(items[index])
        item["roi"] = item["roi"] * scale + shift + 0.0005 * torch.randn_like(item["roi"])
        if kind in ("quality_off", "global_zero"):
            item["sam_quality_supervised"] = False
            item["gt_visible448"] = torch.full_like(item["gt_visible448"], float("nan"))
        if kind == "global_zero":
            # Real data units remain active, but all their supervision is invalid.
            # This differs from an empty/dummy rank and advances data cursor.
            item["metadata"] = dict(item["metadata"], target_identity_valid=False,
                                     gt_geometry_valid=False, target_visible_input_support_count=0)
        batch.append(item)
    signature = hashlib.sha256(torch.stack([x["roi"] for x in batch]).numpy().tobytes()).hexdigest()
    return batch, {"order": order, "rgb_sha256": signature, "scale": scale, "shift": shift}


def attempt(model, optimizer, scheduler, cfg, items, kind, rng, counters):
    batch, signature = materialize_attempt(items, kind, rng)
    counts = local_counts(batch, sam_enabled=sam_variant(cfg))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = None
    if has_supervision(counts, cfg["loss_weights"]):
        sums = loss_sums_on_batch(model, batch, cfg, "cpu")
        objective = weighted_objective(sums, counts, cfg["loss_weights"], 1)
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite engineering objective")
        objective.backward()
        loss_value = float(objective.detach())
    did_update, norm = step_if_supervised(model, optimizer, scheduler, counts, cfg["loss_weights"],
                                         cfg["independent_head_dependencies"], clip=1.)
    counters = dict(counters)
    counters["data_update_attempt"] += 1
    counters["cursor"] += len(items)
    counters["step"] += int(did_update)
    counters["no_supervision_update_count"] += int(not did_update)
    qparameter = next(model.sam_adapter.quality_head.parameters())
    return counters, {"kind": kind, "data": signature, "counts": counts,
                      "did_update": did_update, "objective": loss_value, "preclip_norm": norm,
                      "scheduler_epoch": scheduler.last_epoch,
                      "quality_parameter_step": float(optimizer.state[qparameter].get("step", 0)),
                      "quality_grad_none": qparameter.grad is None,
                      "counters": counters}


def restore_same_run(checkpoint, model, optimizer, scheduler, cfg, rng):
    # Same strict identity and state restoration used by the existing trainer.
    # This helper is TEST-ONLY; it is not a second training/resume implementation.
    if (checkpoint.get("resumable") is not True or checkpoint["config_id"] != canonical_hash(cfg)
            or checkpoint["world_size"] != 1 or checkpoint["role"] != "A0"):
        raise ValueError("Exact recovery requires same run/config/topology, not a best-weight warm start")
    model.load_state_dict(checkpoint["training_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng(checkpoint["rng_by_rank"][0], rng)
    return {key: checkpoint[key] for key in ("epoch", "cursor", "step", "data_update_attempt", "no_supervision_update_count")}


class SamConfiguredSchemaRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU-only environment required")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        OUT.mkdir(parents=True, exist_ok=True)

    def test_two_rank_schema_is_configured_and_missing_marker_rejected(self):
        cfg = fixture_config()
        r0 = metadata_item(0)
        r1 = metadata_item(1)
        old = r1["mask_aux"]
        r1["mask_aux"] = MaskAux(torch.zeros_like(old.visible), torch.zeros_like(old.distance),
                                 torch.zeros_like(old.available), old.candidate_overlap, old.dropout, old.identities)
        counts0 = local_counts([r0], sam_enabled=sam_variant(cfg))
        missing = local_counts([r1], sam_enabled=sam_variant(cfg))
        dummy = local_counts([dict(r1, active=False)], sam_enabled=sam_variant(cfg))
        self.assertEqual(sorted(counts0), sorted(missing))
        self.assertEqual(sorted(counts0), sorted(dummy))
        self.assertEqual(counts0[QUALITY_KEY], 1)
        self.assertEqual(missing[QUALITY_KEY], 0)
        self.assertEqual(dummy[QUALITY_KEY], 0)
        for variant in (r1, dict(r1, active=False)):
            invalid = dict(variant)
            invalid.pop("sam_enabled")
            with self.subTest(active=variant["active"]), self.assertRaisesRegex(ValueError, "marked items"):
                local_counts([invalid], sam_enabled=sam_variant(cfg))
        with self.assertRaisesRegex(ValueError, "configured baseline"):
            local_counts([r0])
        baseline = {k:v for k,v in r0.items() if k != "sam_enabled"}
        self.assertEqual(set(local_counts([baseline])), set(A0_KEYS))
        REPORT["two_rank_schema_contract"] = {
            "scope": "rank0/rank1 inputs evaluated separately; no distributed collective",
            "ordered_keys": sorted(counts0), "rank0_quality_count": 1,
            "rank1_all_missing_quality_count": 0, "rank1_dummy_quality_count": 0,
            "missing_marker_rejected_before_count_return": True, "baseline_default_keys_unchanged": True}

    def test_actual_same_run_recovery_preserves_enhanced_state(self):
        torch.manual_seed(2718)
        random.seed(3141)
        np.random.seed(1618)
        rng = np.random.default_rng(5772)
        cfg = fixture_config()
        parent_sha = cfg["parent_bundle_sha256"]
        model, parent = construct(cfg, "cpu")
        model.train()
        rgb, K, cad, masks = synthetic_fixture()
        aux = aux_for(masks)
        with torch.no_grad():
            output = model(rgb, K, cad, mask_aux=aux, identities=IDS)
        items = items_from_output(rgb, K, cad, masks, aux, output)
        del output
        optimizer = optimizer_for(model, cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-7)
        counters = {"epoch":0, "cursor":0, "step":0, "data_update_attempt":0, "no_supervision_update_count":0}
        counters, prefix = attempt(model, optimizer, scheduler, cfg, items, "normal", rng, counters)
        self.assertEqual(counters["step"], 1)
        self.assertIn("sam_adapter._extra_state", model.state_dict())
        state = {"training_model":model.state_dict(), "optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict(), "rng_by_rank":[rng_state(rng)],
                 "world_size":1, "role":"A0", "config_id":canonical_hash(cfg), "resumable":True,
                 "engineering_scope":"same-run recovery fixture; not selected best",
                 **counters}
        checkpoint_path = OUT / "same_run_midpoint.pt"
        save_checkpoint_atomic(checkpoint_path, state)
        midpoint_sha = file_sha(checkpoint_path)
        del state
        suffix_kinds = ("quality_off", "global_zero", "normal")
        uninterrupted = []
        for kind in suffix_kinds:
            counters, row = attempt(model, optimizer, scheduler, cfg, items, kind, rng, counters)
            uninterrupted.append(row)
        final_model = copy.deepcopy(model.state_dict())
        final_optimizer = copy.deepcopy(optimizer.state_dict())
        final_scheduler = copy.deepcopy(scheduler.state_dict())
        final_rng = copy.deepcopy(rng_state(rng))
        final_counters = dict(counters)
        del model, optimizer, scheduler, parent
        gc.collect()

        # Deliberately disturb initialization/RNG before restoring the SAME run.
        torch.manual_seed(99991)
        random.seed(99992)
        np.random.seed(99993)
        resumed_rng = np.random.default_rng(99994)
        resumed_model, resumed_parent = construct(cfg, "cpu")
        resumed_optimizer = optimizer_for(resumed_model, cfg)
        resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=8, eta_min=1e-7)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        counters = restore_same_run(checkpoint, resumed_model, resumed_optimizer, resumed_scheduler, cfg, resumed_rng)
        self.assertTrue(tensor_state_equal(resumed_model.state_dict(), checkpoint["training_model"]))
        self.assertGreater(float(resumed_model.sam_adapter.projection.weight.detach().abs().sum()), 0)
        self.assertEqual(resumed_scheduler.last_epoch, 1)
        restored_extra = copy.deepcopy(resumed_model.state_dict()["sam_adapter._extra_state"])
        resumed = []
        for kind in suffix_kinds:
            counters, row = attempt(resumed_model, resumed_optimizer, resumed_scheduler, cfg, items, kind, resumed_rng, counters)
            resumed.append(row)
        self.assertEqual(uninterrupted, resumed)
        self.assertEqual(counters, final_counters)
        self.assertTrue(tensor_state_equal(resumed_model.state_dict(), final_model))
        self.assertTrue(tensor_state_equal(resumed_optimizer.state_dict(), final_optimizer))
        self.assertTrue(tensor_state_equal(resumed_scheduler.state_dict(), final_scheduler))
        self.assertTrue(tensor_state_equal(rng_state(resumed_rng), final_rng))
        self.assertEqual(restored_extra, final_model["sam_adapter._extra_state"])
        self.assertEqual(final_counters["step"], 3)
        self.assertEqual(final_counters["data_update_attempt"], 4)
        self.assertEqual(final_counters["no_supervision_update_count"], 1)
        self.assertEqual([row["did_update"] for row in uninterrupted], [True, False, True])
        self.assertTrue(uninterrupted[0]["quality_grad_none"])
        self.assertEqual(uninterrupted[0]["quality_parameter_step"], 1)
        self.assertEqual(uninterrupted[1]["quality_parameter_step"], 1)
        self.assertEqual(uninterrupted[2]["quality_parameter_step"], 2)
        self.assertEqual(file_sha(checkpoint_path), midpoint_sha)
        self.assertEqual(file_sha(ROOT / cfg["parent_bundle"]), parent_sha)
        # Best-weight initialization is a different operation and is not accepted.
        with self.assertRaisesRegex(ValueError, "best-weight"):
            restore_same_run(dict(checkpoint, resumable=False), resumed_model, resumed_optimizer, resumed_scheduler, cfg, resumed_rng)
        with self.assertRaisesRegex(ValueError, "same run"):
            restore_same_run(dict(checkpoint, config_id="different-run"), resumed_model, resumed_optimizer, resumed_scheduler, cfg, resumed_rng)
        with self.assertRaises(FileNotFoundError):
            torch.load(OUT / "missing_recovery_no_default_fallback.pt", map_location="cpu", weights_only=True)
        REPORT["exact_recovery"] = {
            "checkpoint":str(checkpoint_path.relative_to(ROOT)), "sha256":midpoint_sha,
            "checkpoint_source":"existing save_checkpoint_atomic; full training_model/optimizer/scheduler/rng_by_rank",
            "extra_state_restored":restored_extra, "prefix":prefix,
            "uninterrupted_suffix":uninterrupted, "resumed_suffix":resumed,
            "model_parameters_buffers_extra_state_bitwise_equal":True,
            "optimizer_full_state_bitwise_equal":True, "scheduler_full_state_equal":True,
            "all_rng_streams_equal":True, "data_order_rgb_and_cursor_equal":True,
            "final_counters":final_counters, "best_or_wrong_run_rejected":True,
            "missing_resume_has_no_default_fallback":True, "parent_unchanged":True}


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamConfiguredSchemaRecoveryTests))
    REPORT.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  passed=result.wasSuccessful(), torch_threads=torch.get_num_threads(),
                  cuda_initialized=torch.cuda.is_initialized())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
