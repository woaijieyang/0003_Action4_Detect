"""Metadata-only pairing tests; dummy bytes never certify a real checkpoint."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_finalize", PROJECT / "src/eval/action4_stage2_finalize.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class RolePairMetadataTests(unittest.TestCase):
    def setUp(self):
        evidence = PROJECT / "outputs/action4_role_sealing_review_v001"
        evidence.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="METADATA_PAIR_ONLY_", dir=evidence)
        self.root = Path(self.temp.name)
        self.before = pair.R
        pair.R = self.root
        self.verified = {}
        self.paths = {}
        for role in ("A0", "B0"):
            folder = self.root / "models/candidates" / (role + "_METADATA_ONLY")
            folder.mkdir(parents=True)
            bundle = folder / (role + ".pt")
            bundle.write_bytes(("NOT A MODEL: " + role).encode())
            cfg = {"role": role, "mode": "full_training", "validation_receipt": "validation.json",
                   "validation_receipt_sha256": "fixture_dataset_sha",
                   "score_schema": "action4.pose_projection_all8_002.v1",
                   "b0_scoring_policy": "center", "evaluation_scoring_policy": "center",
                   "intrinsics_context": "fixture_context.json"}
            seal = {"schema": "action4.sealed_independent_role.v2", "role": role, "strict_load": True,
                    "bundle": str(bundle.relative_to(self.root)), "sha256": pair._digest(bundle),
                    "selected_step": 40, "config_id": role + "_fixture_config",
                    "actual_code_root": "snapshot_" + role,
                    "validation_receipt": cfg["validation_receipt"],
                    "validation_receipt_sha256": cfg["validation_receipt_sha256"],
                    "score_schema": cfg["score_schema"], "fixture_only": True}
            self.paths[role] = folder / "SEALED_ROLE.json"
            self.verified[role] = {"seal": seal, "config": cfg}
        b = self.verified["B0"]["seal"]
        a = self.verified["A0"]
        a["seal"].update(evaluation_counterpart_sha256=b["sha256"], counterpart_selected_step=b["selected_step"],
                         counterpart_config_id=b["config_id"])
        a["config"]["evaluation_counterpart_sha256"] = b["sha256"]
        self.persist()

    def persist(self):
        for role, value in self.verified.items():
            self.paths[role].write_text(json.dumps(value["seal"]))
            value["seal_sha256"] = pair._digest(self.paths[role])

    def tearDown(self):
        pair.R = self.before
        self.temp.cleanup()

    def verifier(self, seal_path):
        role = next(role for role, p in self.paths.items() if p == seal_path)
        return copy.deepcopy(self.verified[role])

    def inspect(self):
        return pair.inspect_pair(self.paths["A0"], self.paths["B0"], verifier=self.verifier)

    def test_different_code_roots_metadata_pair_supported(self):
        result = self.inspect()
        self.assertNotEqual(result[0]["seal"]["actual_code_root"], result[1]["seal"]["actual_code_root"])

    def test_counterpart_sha_step_and_config_must_match_pair(self):
        original = copy.deepcopy(self.verified["A0"])
        for key, value in (("evaluation_counterpart_sha256", "wrong"), ("counterpart_selected_step", 20),
                           ("counterpart_config_id", "other")):
            self.verified["A0"] = copy.deepcopy(original)
            self.verified["A0"]["seal"][key] = value
            self.persist()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "counterpart"):
                self.inspect()

    def test_data_and_scoring_policy_mismatch_rejected(self):
        self.verified["A0"]["config"]["b0_scoring_policy"] = "different"
        with self.assertRaisesRegex(ValueError, "scoring policy"):
            self.inspect()
        self.verified["A0"]["config"]["b0_scoring_policy"] = "center"
        self.verified["A0"]["seal"]["validation_receipt_sha256"] = "other"
        self.verified["A0"]["config"]["validation_receipt_sha256"] = "other"
        self.persist()
        with self.assertRaisesRegex(ValueError, "dataset"):
            self.inspect()

    def test_old_unvalidated_seal_rejected(self):
        self.verified["A0"]["seal"]["schema"] = "action4.sealed_independent_role.v1"
        self.persist()
        with self.assertRaisesRegex(ValueError, "v2"):
            self.inspect()

    def test_bundle_mutation_after_verifier_rejected(self):
        (self.root / self.verified["B0"]["seal"]["bundle"]).write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.inspect()

    def test_pair_profile_references_only_sealed_bundles(self):
        with mock.patch.object(pair, "_verify_role", side_effect=self.verifier):
            profile = pair.seal_pair(self.paths["A0"], self.paths["B0"], "models/candidates/METADATA_PAIR")
        for role in ("A0", "B0"):
            self.assertEqual(profile[role.lower() + "_bundle"], self.verified[role]["seal"]["bundle"])
            self.assertNotIn("default", profile[role.lower() + "_bundle"])
            self.assertNotIn("output/best.pt", profile[role.lower() + "_bundle"])
        record = json.loads((self.root / "models/candidates/METADATA_PAIR/SEALED.json").read_text())
        self.assertTrue(record["separate_immutable_loaders"])
        self.assertFalse(record["accuracy_accepted"])

    def test_legacy_immutable_config_entry_refuses_before_loading(self):
        for role in ("A0", "B0"):
            (self.root / (role + ".json")).write_text(json.dumps(self.verified[role]["config"]))
        with self.assertRaisesRegex(ValueError, "a0-seal"):
            pair.seal("A0.json", "B0.json", "unused")

    def test_each_role_verifier_uses_separate_cpu_subprocess(self):
        responses = [mock.Mock(stdout=json.dumps({"fixture_role": role})) for role in ("A0", "B0")]
        with mock.patch.object(pair.subprocess, "run", side_effect=responses) as process:
            pair._verify_role(self.paths["A0"])
            pair._verify_role(self.paths["B0"])
        self.assertEqual(process.call_count, 2)
        for call, role in zip(process.call_args_list, ("A0", "B0")):
            command = call.args[0]
            self.assertIn("-I", command)
            self.assertEqual(command[-1], str(self.paths[role]))
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "2")


if __name__ == "__main__":
    unittest.main()
