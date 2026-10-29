"""Metadata-only fixtures. Dummy bytes are NOT models or formal completion proof.

Original seven tests only checked terminal fields/file presence; these fixtures
now bind requests, reports, config, attempts, datasets, parents and selection.
Actual immutable model loading is intentionally deferred to completed real runs.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("role_sealer", PROJECT / "scripts/action4_seal_completed_role.py")
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


def fixture_rank(result):
    # Explicit metadata fixture ranking; production imports immutable rank_result.
    r = result["report"]
    return (r["mAP50_95"], r["overall"]["f1"], -result["step"])


class RoleSealMetadataTests(unittest.TestCase):
    def setUp(self):
        evidence = PROJECT / "outputs/action4_role_sealing_review_v001"
        evidence.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="METADATA_ONLY_", dir=evidence)
        self.root = Path(self.temp.name)
        self.before = seal.ROOT
        seal.ROOT = self.root
        self.out = self.root / "output"
        (self.out / "eval").mkdir(parents=True)
        (self.out / "controller.lock").touch()
        (self.out / "selection.lock").touch()
        (self.out / "best.pt").write_bytes(b"NOT A MODEL: metadata fixture")
        self.code = self.root / "snapshot"
        (self.code / "inputs").mkdir(parents=True)
        parent = self.code / "inputs/B0_parent.pt"
        parent.write_bytes(b"NOT A MODEL: frozen parent metadata fixture")
        (self.root / "validation.json").write_text('{"scope":"metadata fixture"}')
        self.cfg = {"mode": "full_training", "epochs": 80, "role": "B0", "output": "output",
                    "expected_units": 1, "batch": 1, "eval_every": 20, "expected_eval_images": 650,
                    "require_immutable_source": True, "run_spec": "spec.json",
                    "evaluation_scoring_policy": "center", "parent_bundle": "snapshot/inputs/B0_parent.pt",
                    "parent_bundle_sha256": seal.digest(parent), "parent_best_step": 3220,
                    "validation_receipt": "validation.json",
                    "validation_receipt_sha256": seal.digest(self.root / "validation.json"),
                    "test_scope": "METADATA_ONLY_NOT_FORMAL_TRAINING"}
        self.write("snapshot/resolved.json", self.cfg)
        self.spec = {"schema": "action4.immutable_run.v1", "project_root": str(self.root),
                     "code_root": "snapshot", "config": "snapshot/resolved.json",
                     "files": {"inputs/B0_parent.pt": seal.digest(parent),
                               "resolved.json": seal.digest(self.code / "resolved.json")}}
        self.write("spec.json", self.spec)
        self.cid = seal.canonical(self.cfg)
        self.train = {"config_id": self.cid, "attempt_id": "fixture", "mode": "full_training",
                      "epochs": 80, "role": "B0", "units_per_epoch": 1, "step": 80,
                      "formal_training_complete": True, "data_update_attempt": 80, "planned_update_attempts": 80}
        self.evaluation = {"config_id": self.cid, "attempt_id": "fixture", "all_steps": [0, 20, 40, 60, 80],
                           "evaluations": 5, "best_step": 40}
        self.controller = {"config_id": self.cid, "attempt_id": "fixture", "role": "B0",
                           "mode": "full_training", "best_step": 40}
        for step in self.evaluation["all_steps"]:
            request = {"step": step, "config_id": self.cid, "role": "B0",
                       "snapshot": f"output/eval_step_{step:08d}.pt"}
            self.write(f"output/eval/step_{step:08d}.json", request)
            raw = {"schema": "action4.bbox_metrics_candidate.v1", "scoring_policy": "center",
                   "mAP50_95": 0.9 if step == 40 else 0.1, "overall": {"images": 650, "f1": 0.5},
                   "per_image": [{"sample_id": f"fixture_{i}"} for i in range(650)],
                   "fixture_only": True}
            raw_path = f"output/eval_runs/step_{step:08d}_fixture_0/REPORT.json"
            self.write(raw_path, raw)
            done = {"step": step, "config_id": self.cid, "role": "B0", "training_mode": "full_training",
                    "report": dict(raw, evaluation_images=650, evaluation_wall_seconds=0.01),
                    "raw_report": raw_path, "checkpoint": "output/best.pt"}
            self.write(f"output/eval/step_{step:08d}.done.json", done)
            if step == 40:
                self.best = done
        self.terminals()

    def tearDown(self):
        seal.ROOT = self.before
        self.temp.cleanup()

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def terminals(self):
        for name, value in (("TRAIN_COMPLETE", self.train), ("EVAL_COMPLETE", self.evaluation),
                            ("CONTROLLER_COMPLETE", self.controller), ("BEST", self.best)):
            self.write("output/" + name + ".json", value)

    def inspect(self):
        return seal.inspect("spec.json", rank_fn=fixture_rank)

    def test_complete_metadata_ready(self):
        self.assertEqual(self.inspect()[3]["status"], "COMPLETION_METADATA_READY")

    def test_missing_terminal_waits(self):
        (self.out / "TRAIN_COMPLETE.json").unlink()
        self.assertEqual(self.inspect()[3]["status"], "WAITING_FOR_COMPLETION")

    def test_wrong_attempt_rejected(self):
        self.controller["attempt_id"] = "other"
        self.terminals()
        with self.assertRaisesRegex(ValueError, "attempt"):
            self.inspect()

    def test_unevaluated_step_rejected(self):
        (self.out / "eval/step_00000060.done.json").unlink()
        with self.assertRaisesRegex(ValueError, "Queue"):
            self.inspect()

    def test_selected_report_mismatch_rejected(self):
        self.best = dict(self.best, report={})
        self.terminals()
        with self.assertRaisesRegex(ValueError, "Selected eval"):
            self.inspect()

    def test_short_traversal_rejected(self):
        self.train["step"] = 79
        self.terminals()
        with self.assertRaises(ValueError):
            self.inspect()

    def test_failure_receipt_rejected(self):
        self.write("output/eval/step_00000020.failed.json", {"fixture_only": True})
        with self.assertRaisesRegex(ValueError, "Failed evaluations"):
            self.inspect()

    def test_every_request_and_done_identity_checked(self):
        for suffix in ("json", "done.json"):
            name = "output/eval/step_00000060." + suffix
            original = self.read(name)
            for key, value in (("step", 20), ("config_id", "foreign"), ("role", "A0"), ("attempt_id", "foreign")):
                with self.subTest(suffix=suffix, key=key):
                    self.write(name, dict(original, **{key: value}))
                    with self.assertRaises(ValueError):
                        self.inspect()
            self.write(name, original)

    def test_done_mode_schema_and_coverage(self):
        name = "output/eval/step_00000060.done.json"
        original = self.read(name)
        for key, value in (("schema", "wrong"), ("evaluation_images", 649), ("scoring_policy", "wrong")):
            changed = copy.deepcopy(original)
            changed["report"][key] = value
            self.write(name, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.inspect()
        changed = dict(original, training_mode="integration")
        self.write(name, changed)
        with self.assertRaisesRegex(ValueError, "mode"):
            self.inspect()

    def test_raw_report_attempt_and_content(self):
        name = "output/eval/step_00000060.done.json"
        original = self.read(name)
        changed = dict(original, raw_report="output/eval_runs/step_00000060_foreign_0/REPORT.json")
        self.write(changed["raw_report"], self.read(original["raw_report"]))
        self.write(name, changed)
        with self.assertRaisesRegex(ValueError, "attempt/path"):
            self.inspect()
        self.write(name, original)
        self.write(original["raw_report"], {"different": True})
        with self.assertRaisesRegex(ValueError, "Raw/report"):
            self.inspect()

    def test_real_rank_winner_not_terminal_agreement(self):
        name = "output/eval/step_00000060.done.json"
        changed = self.read(name)
        changed["report"]["mAP50_95"] = 0.95
        self.write(name, changed)
        self.write(changed["raw_report"], {k: v for k, v in changed["report"].items()
                                          if k not in ("evaluation_images", "evaluation_wall_seconds")})
        with self.assertRaisesRegex(ValueError, "rank winner"):
            self.inspect()

    def test_zero_negative_boolean_and_fractional_counts(self):
        original = copy.deepcopy(self.train)
        for key, value in (("step", 0), ("step", -1), ("step", True), ("step", 80.0),
                           ("no_supervision_update_count", -1), ("data_update_attempt", 79)):
            self.train = dict(original, **{key: value})
            self.terminals()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.inspect()

    def test_duplicate_steps_and_nonfinite_reports(self):
        self.evaluation["all_steps"].append(40)
        self.terminals()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.inspect()
        self.evaluation["all_steps"].pop()
        self.terminals()
        name = "output/eval/step_00000060.done.json"
        changed = self.read(name)
        changed["report"]["mAP50_95"] = float("nan")
        self.write(name, changed)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.inspect()

    def test_current_data_and_frozen_parent_sha_checked(self):
        seal.verify_dependencies(self.spec, self.cfg)
        (self.root / "validation.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Validation"):
            seal.verify_dependencies(self.spec, self.cfg)
        (self.root / "validation.json").write_text('{"scope":"metadata fixture"}')
        (self.code / "inputs/B0_parent.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA"):
            seal.verify_dependencies(self.spec, self.cfg)

    def test_parent_and_counterpart_cannot_point_at_live_or_default(self):
        for name in ("models/default/B0.pt", "output/best.pt"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"unfrozen")
            cfg = dict(self.cfg, parent_bundle=name, parent_bundle_sha256=seal.digest(p))
            with self.assertRaisesRegex(ValueError, "frozen run input"):
                seal.verify_dependencies(self.spec, cfg)
        cfg = dict(self.cfg, role="A0", evaluation_counterpart_bundle="output/best.pt",
                   evaluation_counterpart_sha256=seal.digest(self.out / "best.pt"))
        with self.assertRaisesRegex(ValueError, "frozen run input"):
            seal.verify_dependencies(self.spec, cfg)

    def test_frozen_counterpart_uses_existing_config_sha_key(self):
        counterpart = self.code / "inputs/B0_counterpart.pt"
        counterpart.write_bytes(b"NOT A MODEL: counterpart metadata fixture")
        sha = seal.digest(counterpart)
        cfg = dict(self.cfg, role="A0", evaluation_counterpart_bundle="snapshot/inputs/B0_counterpart.pt",
                   evaluation_counterpart_sha256=sha)
        spec = copy.deepcopy(self.spec)
        spec["files"]["inputs/B0_counterpart.pt"] = sha
        dependencies = seal.verify_dependencies(spec, cfg)
        self.assertEqual(dependencies["evaluation_counterpart_sha256"], sha)
        self.assertNotIn("evaluation_counterpart_bundle_sha256", dependencies)
        cfg["evaluation_counterpart_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "SHA"):
            seal.verify_dependencies(spec, cfg)

    def test_per_image_duplicates_rejected_even_when_summary_claims650(self):
        name = "output/eval/step_00000060.done.json"
        changed = self.read(name)
        changed["report"]["per_image"][-1] = changed["report"]["per_image"][0]
        self.write(name, changed)
        with self.assertRaisesRegex(ValueError, "Per-image"):
            self.inspect()

    def test_a0_score_schema_and_eligibility_required(self):
        self.cfg["role"] = "A0"
        self.cfg["score_schema"] = "action4.pose_projection_all8_002.v1"
        self.write("snapshot/resolved.json", self.cfg)
        cid = seal.canonical(self.cfg)
        for record in (self.train, self.evaluation, self.controller):
            record["config_id"] = cid
            if "role" in record:
                record["role"] = "A0"
        for step in self.evaluation["all_steps"]:
            name = f"output/eval/step_{step:08d}"
            req = self.read(name + ".json")
            req.update(role="A0", config_id=cid)
            self.write(name + ".json", req)
            done = self.read(name + ".done.json")
            done.update(role="A0", config_id=cid, deployment_eligible=True)
            done["report"]["score_schema"] = self.cfg["score_schema"]
            done["report"]["bbox_only"] = {"per_image": done["report"]["per_image"]}
            self.write(name + ".done.json", done)
            self.write(done["raw_report"], {k: v for k, v in done["report"].items()
                                            if k not in ("evaluation_images", "evaluation_wall_seconds")})
            if step == 40:
                self.best = done
        self.terminals()
        self.assertEqual(self.inspect()[3]["role"], "A0")
        name = "output/eval/step_00000060.done.json"
        original = self.read(name)
        for kind in ("schema", "eligible"):
            changed = copy.deepcopy(original)
            if kind == "schema":
                changed["report"]["score_schema"] = "old"
            else:
                changed.pop("deployment_eligible")
            self.write(name, changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.inspect()

    def test_actual_bundle_parent_fields_checked(self):
        state = {"role": "B0", "config_id": self.cid, "config": self.cfg, "selected_step": 40,
                 "parent_selected_step": 3220, "parent_bundle": self.cfg["parent_bundle"]}
        seal.verify_state(state, self.cfg, 40)
        for key, value in (("parent_selected_step", 1), ("parent_bundle", "output/best.pt"), ("selected_step", 20)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                seal.verify_state(dict(state, **{key: value}), self.cfg, 40)

    def fake_worker(self, run_spec, copied, sha):
        self.assertNotEqual(Path(copied), self.out / "best.pt")
        self.assertIn(".pending_", str(copied))
        self.assertEqual(seal.digest(Path(copied)), sha)
        status = self.inspect()[3]
        return {"status": status, "dependencies": seal.verify_dependencies(self.spec, self.cfg),
                "run_spec_sha256": seal.digest(self.root / "spec.json"),
                "strict_load": True, "bundle_sha256": sha}

    def test_copy_is_verified_in_pending_not_live_alias(self):
        with mock.patch.object(seal, "run_worker", side_effect=self.fake_worker) as verifier:
            result = seal.seal_role("spec.json", "models/candidates/METADATA_ONLY")
        self.assertEqual(result["sha256"], seal.digest(self.out / "best.pt"))
        self.assertEqual(verifier.call_count, 1)

    def test_copy_race_and_receipt_race_fail(self):
        for which in ("source", "receipt"):
            def race(run_spec, copied, sha):
                result = self.fake_worker(run_spec, copied, sha)
                if which == "source":
                    (self.out / "best.pt").write_bytes(b"replaced source")
                else:
                    self.controller["fixture_mutation"] = True
                    self.terminals()
                return result
            destination = "models/candidates/METADATA_RACE_" + which
            with mock.patch.object(seal, "run_worker", side_effect=race), self.assertRaises(ValueError):
                seal.seal_role("spec.json", destination)
            self.assertFalse((self.root / destination).exists())

    def test_active_controller_lock_rejected(self):
        import fcntl
        with (self.out / "controller.lock").open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active controller"):
                with seal.completion_locks(self.out):
                    self.fail("active controller accepted")

    def test_import_path_preserves_controlled_venv_and_rejects_other_project_roots(self):
        code = self.root / "selected/source"
        site = self.root / ".venv/lib/python3.12/site-packages"
        other = self.root / "other/source/src"
        current = [str(site), str(self.root / "src"), str(other), "/usr/lib/python3.12"]
        with mock.patch.object(seal.sysconfig, "get_path", return_value=str(site)):
            selected = seal.snapshot_pythonpath(code, current)
        self.assertEqual(selected[:2], [str(code / "runtime/surface_deps"), str(code / "src")])
        self.assertIn(str(site), selected)
        self.assertNotIn(str(other), selected)
        self.assertNotIn(str(self.root / "src"), selected)

    def test_verified_readonly_hash_reuse_and_mutation_rejection(self):
        p = self.code / "inputs/B0_parent.pt"
        p.chmod(0o444)
        original = seal.digest(p)
        verified = {str(p): (original, seal.stat_fingerprint(p))}
        with mock.patch.object(seal, "digest", side_effect=AssertionError("unexpected repeat hash")):
            self.assertEqual(seal.verified_digest(p, verified), original)
        p.chmod(0o644)
        p.write_bytes(b"same-name changed content")
        with self.assertRaisesRegex(ValueError, "Previously verified"):
            seal.verified_digest(p, verified)
        writable = {str(p): (seal.digest(p), seal.stat_fingerprint(p))}
        with mock.patch.object(seal, "digest", wraps=seal.digest) as real_digest:
            seal.verified_digest(p, writable)
            real_digest.assert_called_once_with(p)

    def test_real_isolated_snapshot_dependency_import_only(self):
        import os
        import subprocess
        import sys
        evidence = PROJECT / "outputs/action4_role_sealing_review_v001"
        command = [sys.executable, "-I", str(evidence / "import_only_probe.py")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
                   TMPDIR=str(evidence / "tmp"), XDG_CACHE_HOME=str(evidence / "cache"),
                   TORCH_HOME=str(evidence / "cache"))
        result = subprocess.run(command, cwd=PROJECT, env=env, text=True, capture_output=True)
        (evidence / "real_import_probe.stdout.log").write_text(result.stdout)
        (evidence / "real_import_probe.stderr.log").write_text(result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["isolated"], 1)
        self.assertTrue(report["passed"])
        self.assertFalse(report["checkpoint_loaded"])
        self.assertFalse(report["cuda_initialized"])
        (evidence / "REAL_IMPORT_VERIFICATION.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
