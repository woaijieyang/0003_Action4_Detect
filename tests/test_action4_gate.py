"""CPU contracts only; fabricated test measures never represent a formal run."""
import copy
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from model.training.action4_gate import (SOURCE_FILES, build_gate_binding,
                                         make_d0_gate, validate_d0_gate)
from model.training.action4_runtime import sha256_file


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (*SOURCE_FILES, "configs/eval/action4_protocol.json"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source " + name)
        (self.root / "image.png").write_bytes(b"test identity only, never loaded as an image")
        (self.root / "weight.pt").write_bytes(b"test identity only, never loaded as a model")
        self.records = [{"sample_id": str(i), "image": "image.png", "instances": [
            {"instance_id": "camera", "corners_px": [[i, i]] * 8, "ignore": False}
        ]} for i in range(32)]
        self.manifest = self.root / "train.jsonl"
        self.write_records()
        self.config = {
            "server_extension": {"schema_version": "action4.server_extension.v1", "version": "fixture"},
            "network": {"width": 1}, "preprocess": {"roi_size": 256}, "decode": {"corner_topk": 20},
            "initialization": {"backbone_weights": "weight.pt", "sha256": sha256_file(self.root / "weight.pt")},
            "stages": {"D0": {"roi_count": 32, "epochs": 500}},
            "runtime": {"seed": 1, "amp": False, "deterministic": True},
        }

    def write_records(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def binding(self):
        return build_gate_binding(self.root, self.config, self.manifest, self.records)

    def gate(self):
        return make_d0_gate(self.binding(), {
            "gate_pass": True, "geometry_checks": {"gate_pass": True},
            "epe_report": {"rois": 32, "ideal_heatmap_p95_px": .2,
                           "epe_p95_heatmap_px": 1.8, "improvement_over_zero_p95": .9},
            "finite_nonzero_backbone_gradient": True, "backbone_parameter_changed": True,
        }, component=False, global_step=500)

    def test_current_binding_is_valid(self):
        validate_d0_gate(self.gate(), self.binding())

    def test_every_relevant_source_change_invalidates_gate(self):
        gate = self.gate()
        for name in SOURCE_FILES:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    validate_d0_gate(gate, self.binding())
                path.write_bytes(original)

    def test_all_method_configuration_changes_invalidate(self):
        gate, original = self.gate(), copy.deepcopy(self.config)
        changes = (("network", {"width": 2}), ("preprocess", {"roi_size": 128}),
                   ("decode", {"corner_topk": 4}), ("heatmap", {"sigma": 2}),
                   ("loss", {"alpha": 4}), ("protocol", {"version": 2}),
                   ("server_extension", {"schema_version": "action4.server_extension.v1", "version": "changed"}),
                   ("runtime", {"seed": 2}), ("stages", {"D0": {"roi_count": 32, "epochs": 400}}))
        for key, value in changes:
            with self.subTest(key=key):
                self.config = {**original, key: value}
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    validate_d0_gate(gate, self.binding())
        self.config = original

    def test_manifest_fixed_roi_and_image_changes_invalidate(self):
        gate = self.gate()
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_d0_gate(gate, self.binding())
        self.write_records()
        self.records.reverse()
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_d0_gate(gate, self.binding())
        self.records.reverse()
        (self.root / "image.png").write_bytes(b"modified actual RGB")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_d0_gate(gate, self.binding())

    def test_weight_change_rejected_even_if_declared_hash_is_refreshed(self):
        gate = self.gate()
        (self.root / "weight.pt").write_bytes(b"changed weight")
        with self.assertRaisesRegex(ValueError, "Initialization weight SHA"):
            self.binding()
        self.config["initialization"]["sha256"] = sha256_file(self.root / "weight.pt")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_d0_gate(gate, self.binding())

    def test_boolean_cannot_promote_legacy_component_or_failed_measurements(self):
        gate, binding = self.gate(), self.binding()
        legacy = {"gate_pass": True, "network": self.config["network"]}
        with self.assertRaisesRegex(ValueError, "historical"):
            validate_d0_gate(legacy, binding)
        for change in ({"component_diagnostic": True}, {"global_step": 501},
                       {"finite_nonzero_backbone_gradient": False},
                       {"backbone_parameter_changed": False}, {"geometry_checks": {"gate_pass": False}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_d0_gate({**gate, **change}, binding)
        for key, value in (("epe_p95_heatmap_px", 2.01), ("ideal_heatmap_p95_px", 1.01),
                           ("improvement_over_zero_p95", .79), ("rois", 31),
                           ("epe_p95_heatmap_px", float("nan"))):
            bad = copy.deepcopy(gate)
            bad["epe_report"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_d0_gate(bad, binding)

    def test_protocol_file_and_in_place_config_mutation_cannot_rewrite_binding(self):
        gate = self.gate()
        self.config["network"]["width"] = 2
        self.assertEqual(gate["binding"]["method"]["network"]["width"], 1)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_d0_gate(gate, self.binding())
        self.config["network"]["width"] = 1
        (self.root / "configs/eval/action4_protocol.json").write_text("changed protocol")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_d0_gate(gate, self.binding())

    def test_formal_requires_explicit_server_protocol(self):
        for extension in (None, {}, {"schema_version": "unknown"}):
            self.config["server_extension"] = extension
            with self.subTest(extension=extension), self.assertRaisesRegex(ValueError, "declared"):
                self.binding()

    def test_component_writer_cannot_emit_pass(self):
        result = make_d0_gate(self.binding(), {"gate_pass": True}, component=True, global_step=1)
        self.assertFalse(result["gate_pass"])

    def test_nonignored_selection_and_execution_only_changes(self):
        gate = self.gate()
        self.config["runtime"]["workers"] = 8
        self.config["runtime"]["device"] = "cuda"
        self.config["video"] = {"output": "different"}
        validate_d0_gate(gate, self.binding())
        self.records[0]["instances"][0]["ignore"] = True
        with self.assertRaisesRegex(ValueError, "Not enough"):
            self.binding()


class TrainerGateIntegrationTests(unittest.TestCase):
    def test_component_trainer_emits_v2_but_cannot_pass(self):
        from test_action4_training_eval import fixture, TinyControlNet
        from model.training.action4_trainer import train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            with mock.patch("model.action4_network.Action4Net", TinyControlNet):
                train(root, config, stage="D0", run_id="gate_fixture", allow_diagnostic=True,
                      component_smoke=True, max_updates=1)
            gate = json.loads((root / "outputs/action4/gate_fixture/d0_gate.json").read_text())
            self.assertEqual(gate["schema_version"], "action4.d0_gate.v2")
            self.assertTrue(gate["component_diagnostic"])
            self.assertFalse(gate["gate_pass"])
            self.assertEqual(len(gate["binding"]["fixed_rois"]), 2)
            with self.assertRaises(ValueError):
                validate_d0_gate(gate, gate["binding"])

    def test_all_followup_stages_reject_legacy_gate_before_model_creation(self):
        from test_action4_training_eval import fixture
        from model.training.action4_trainer import train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture(root)
            config["runtime"]["require_cuda"] = False
            (root / "frames.json").write_text(json.dumps({
                "complete": True, "source": {"frame_count": 1}, "frames": [{"frame_id": 0}]}))
            config["video"]["frame_manifest"] = "frames.json"
            s1 = json.loads((root / "train.jsonl").read_text().splitlines()[0])
            for field in ("sample_id", "group_id", "background_source_id"):
                s1[field] = "s1_" + s1[field]
            (root / "s1.jsonl").write_text(json.dumps(s1) + "\n")
            config["data"]["s1_manifest"] = "s1.jsonl"
            (root / "legacy.json").write_text(json.dumps({
                "gate_pass": True, "geometry_checks": {"gate_pass": True},
                "network": config["network"], "preprocess": config["preprocess"]}))
            for stage in ("A0", "B0", "C1"):
                with self.subTest(stage=stage), mock.patch("model.action4_network.Action4Net") as constructor:
                    with self.assertRaisesRegex(ValueError, "historical"):
                        train(root, config, stage=stage, run_id="rejected_" + stage,
                              d0_gate="legacy.json", allow_diagnostic=True)
                    constructor.assert_not_called()
                    self.assertFalse((root / "models/action4" / ("rejected_" + stage)).exists())


if __name__ == "__main__":
    unittest.main()
