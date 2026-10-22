"""Queued handoff CPU/mock contracts. No test is a real model/visual review."""
import copy
import csv
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model.training.action4_video_queue import (VideoQueue, execution_mode, video_due,
                                                 epoch_pause_required, require_dispatch_owner, validate_queue_config)
from model.training.action4_trainer import amp_initial_scale
from model.training.action4_runtime import sha256_file
from model.training.action4_dispatch import Dispatcher, ALLOWED_GPU_UUIDS


class FakeDispatcher:
    """Persist exact normalized specs but never launch a process or reserve GPUs."""
    pools = {}

    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "outputs/action4_dispatch"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs = self.pools.setdefault(str(root), {})

    def state(self):
        return {"jobs": copy.deepcopy(self.jobs)}

    def status(self):
        return self.state()

    def enqueue(self, spec, *, profile=False):
        if spec["job_id"] in self.jobs:
            raise AssertionError("Duplicate enqueue is not idempotent")
        # Exercise the real normalization/hash/resource/config contract, in a
        # nested isolated fixture. Validation does not enqueue or launch jobs.
        with mock.patch("model.training.action4_dispatch.PROJECT_ROOT", self.root):
            checked = Dispatcher(self.root)._validate(spec, profile=profile)
        self.jobs[spec["job_id"]] = {"spec": checked, "status": "queued"}
        return self.jobs[spec["job_id"]]


def _queued_final_epoch_worker(rank, root_text, port):
    from test_action4_training_eval import TinyControlNet
    from model.training.action4_trainer import train
    root = Path(root_text)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    config = json.loads((root / "ddp_config.json").read_text())
    with mock.patch("model.action4_network.Action4Net", TinyControlNet), \
         mock.patch("model.training.action4_trainer.require_dispatch_owner"), \
         mock.patch("model.training.action4_trainer.validate_d0_gate"), \
         mock.patch("model.training.action4_video_queue.Dispatcher", FakeDispatcher):
        result = train(root, config, stage="B0", run_id="ddp_final_fixture", init_weights=root / "ddp_a0.pt",
                       d0_gate=root / "ddp_gate.json", allow_diagnostic=True)
    (root / f"rank_{rank}_queued_result.json").write_text(json.dumps(result))


class VideoQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = mock.patch("model.training.action4_video_queue.Dispatcher", FakeDispatcher)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        FakeDispatcher.pools.pop(str(self.root), None)
        for name in ("src/example.py", "scripts/action4_video.py", "scripts/action4_dispatch.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Explicit isolated code-identity fixture\n")
        frames = [{"frame_id": i, "pts": 400 * i, "duration_pts": 400} for i in range(2)]
        (self.root / "frames.json").write_text(json.dumps({
            "complete": True, "source": {"frame_count": 2, "time_base": "1/11988"}, "frames": frames}))
        self.config = {
            "runtime": {"device": "auto"}, "stages": {"B0": {"epochs": 60}, "C1": {"epochs": 20}},
            "video": {"execution": "queued", "every_epochs": 5, "frame_manifest": "frames.json",
                      "dispatch": {"allowed_gpu_uuids": [ALLOWED_GPU_UUIDS[0]], "reservation_mib": 12000,
                                   "cpu_threads": 9, "profile_key": "unit-test-not-real-profile",
                                   "profile_first": True}}}
        self.queue = VideoQueue(self.root, "unitrun", self.config)

    def checkpoint(self, epoch=5, stage="B0"):
        path = self.root / "models/action4/unitrun" / f"epoch_{epoch:03d}_step_{epoch:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": "action4.checkpoint.v1", "stage": stage,
                    "run_id": "unitrun", "next_epoch": epoch, "epoch_boundary": True, "config": self.config,
                    "global_step": epoch, "claim_scope": "accepted_data_model_training"}, path)
        return path

    def event(self, epoch=5, stage="B0"):
        return self.queue.ensure_event(stage, epoch, self.checkpoint(epoch, stage))

    def mock_completed_review(self, event, stage="B0"):
        """Artificial fixture explicitly supplies all evidence; production never calls this."""
        job = self.queue.dispatcher.jobs[event["job_id"]]
        job["status"] = "succeeded"
        out = self.root / event["result_dir"]
        out.mkdir(parents=True)
        (out / "run_manifest.json").write_text(json.dumps({
            "checkpoint_sha256": event["identity"]["checkpoint_sha256"],
            "frame_manifest_sha256": event["identity"]["frame_manifest_sha256"], "frame_count": 2}))
        (out / "completion.json").write_text(json.dumps({
            "status": "complete_io_pending_accuracy_review", "frame_count": 2}))
        (out / "predictions_raw.jsonl").write_text("explicit artificial predictions fixture\n")
        (out / "full_red_raw.mp4").write_bytes(b"artificial video identity fixture, not playable")
        with (out / "review_coverage.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_id", "pts", "time_base", "status", "reviewer", "notes"])
            writer.writeheader()
            for i in range(2):
                writer.writerow({"frame_id": i, "pts": i * 400, "time_base": "1/11988", "status": "reviewed",
                                 "reviewer": "unit_fixture_only", "notes": "artificial fixture"})
        review = {"schema_version": "action4.video_review.v1", "job_id": event["job_id"],
                  "event_sha256": event["event_sha256"], "status": "reviewed", "decision": "continue",
                  "reviewer": "unit_fixture_only", "evidence": "artificial evidence", "observation": "not accuracy"}
        for key, name in {"run_manifest": "run_manifest.json", "completion": "completion.json",
                          "predictions": "predictions_raw.jsonl", "coverage": "review_coverage.csv",
                          "video": "full_red_raw.mp4"}.items():
            review[key + "_sha256"] = sha256_file(out / name)
        if stage == "C1":
            base = self.root / "outputs/action4/unitrun/initial_eval/eval_metrics.json"
            current = self.root / "outputs/action4/unitrun" / f"epoch_{event['identity']['epoch']:03d}_step_00000005/eval_metrics.json"
            for path in (base, current):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"fixture": true}')
            review["c1_comparison"] = {"baseline_eval_sha256": sha256_file(base),
                                        "current_eval_sha256": sha256_file(current),
                                        "improved": True, "ordinary_scene_review": "unit fixture only"}
        attempt = event.get("attempts", [{"number": 1}])[-1]["number"]
        suffix = "" if attempt == 1 else f"_attempt_{attempt:02d}"
        destination = self.queue.directory / "reviews" / f"epoch_{event['identity']['epoch']:03d}{suffix}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(review))
        return destination

    def mark_failed(self, event, message="RuntimeError: CUDA out of memory"):
        job = self.queue.dispatcher.jobs[event["job_id"]]
        worker = {"pid": 987654321, "start": "unit-test-only", "command_sha256": "a" * 64}
        child = {"pid": 987654322, "start": "unit-test-only", "command_sha256": "b" * 64}
        job.update(status="failed", exitcode=1, token="fixture_token",
                   worker=worker, child=child, tracked_processes=[])
        out = Path(job["spec"]["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "dispatch_exit.json").write_text(json.dumps({"token": job["token"], "exitcode": 1, "finished_at": 12345}))
        (out / "dispatch_worker.json").write_text(json.dumps({"token": job["token"], "worker": worker, "child": child}))
        (out / "stdout.log").write_text(message)
        return job

    def conclusive_profile(self, event):
        spec = self.queue.dispatcher.jobs[event["job_id"]]["spec"]
        path = self.queue.dispatcher.directory / "profiles" / (event["job_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "conclusive", "profile_key": spec["profile_key"], "gpu_count": 1,
                  "recommended_reservation_mib": 10000, "code": spec["code"], "configs": spec["configs"],
                  "effective_workload_sha256": spec["effective_workload_sha256"]}))

    def retry(self, epoch=5, **kwargs):
        with mock.patch("model.training.action4_video_queue.is_live", return_value=False), \
             mock.patch("model.training.action4_video_queue.descendants", return_value=set()):
            return self.queue.retry_failed(epoch, controller="unit-fixture-controller", reason_category="oom",
                                           reason="artificial terminal OOM fixture", **kwargs)

    def test_snapshot_survives_top3_unlink_and_replays_without_source(self):
        source = self.checkpoint()
        event = self.queue.ensure_event("B0", 5, source)
        source.unlink()
        self.assertTrue((self.root / event["identity"]["checkpoint"]).is_file())
        self.assertEqual(self.queue.ensure_event("B0", 5, source), event)

    def test_frozen_config_argv_hashes_and_no_review_fabrication(self):
        event = self.event()
        self.queue.pump()
        spec = self.queue.dispatcher.jobs[event["job_id"]]["spec"]
        self.assertIn("--checkpoint", spec["argv"])
        self.assertIn("--frame-manifest", spec["argv"])
        self.assertEqual(spec["inputs"][event["identity"]["checkpoint"]], event["identity"]["checkpoint_sha256"])
        self.assertEqual(spec["gpu_count"], 1)
        self.assertEqual(spec["cpu_threads"], 9)
        self.assertTrue(spec["profile"])
        self.assertEqual(spec["result_dirs"], [str(self.root / event["result_dir"])])
        self.assertFalse((self.queue.directory / "reviews").exists())
        self.assertEqual(self.queue.state()["outstanding"], [event["job_id"]])

    def test_idempotent_crash_after_enqueue_before_ledger_commit(self):
        event = self.event()
        self.queue.dispatcher.enqueue(self.queue._spec(event, profile=True), profile=True)
        self.assertEqual(len(self.queue.dispatcher.jobs), 1)
        self.queue.pump()
        self.queue.pump()
        self.assertEqual(len(self.queue.dispatcher.jobs), 1)

    def test_two_intervals_wait_for_profile_and_apply_backpressure(self):
        self.event(5)
        self.queue.pump()
        second = self.event(10)
        state = self.queue.state()
        self.assertEqual(state["events"][1]["status"], "waiting_for_profile")
        self.assertEqual(len(state["outstanding"]), 2)
        self.assertNotIn(second["job_id"], self.queue.dispatcher.jobs)
        with self.assertRaisesRegex(ValueError, "truthful"):
            self.queue.require_resume("B0", 10)

    def test_complete_review_relaxes_b0_backpressure_but_no_fake_gt(self):
        first = self.event(5)
        self.queue.pump()
        self.event(10)
        self.mock_completed_review(first)
        state = self.queue.require_resume("B0", 10)
        self.assertEqual(len(state["outstanding"]), 1)
        self.assertFalse((self.root / "data").exists())

    def test_c1_resume_requires_complete_job_review_and_eval_comparison(self):
        event = self.event(5, "C1")
        self.queue.pump()
        with self.assertRaises(ValueError):
            self.queue.require_resume("C1", 5)
        receipt = self.mock_completed_review(event, "C1")
        self.assertEqual(self.queue.require_resume("C1", 5)["outstanding"], [])
        obj = json.loads(receipt.read_text())
        obj["c1_comparison"]["improved"] = False
        receipt.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "improvement"):
            self.queue.require_resume("C1", 5)

    def test_short_duplicate_or_pending_review_coverage_rejected_even_with_new_hash(self):
        event = self.event()
        self.queue.pump()
        receipt = self.mock_completed_review(event)
        out = self.root / event["result_dir"]
        coverage = out / "review_coverage.csv"
        original = coverage.read_text()
        for contents in (original.splitlines()[0] + "\n", original.replace("reviewed", "pending_review"),
                         original.replace("1,400", "0,0")):
            coverage.write_text(contents)
            obj = json.loads(receipt.read_text())
            obj["coverage_sha256"] = sha256_file(coverage)
            receipt.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "coverage"):
                self.queue.state()

    def test_review_receipt_identity_and_stop_reject_continuation(self):
        event = self.event()
        self.queue.pump()
        receipt = self.mock_completed_review(event)
        original = json.loads(receipt.read_text())
        for update in ({"event_sha256": "0" * 64}, {"video_sha256": "0" * 64}, {"decision": "stop"}):
            receipt.write_text(json.dumps({**original, **update}))
            with self.assertRaises(ValueError):
                self.queue.state()

    def test_source_config_manifest_snapshot_changes_are_rejected(self):
        event = self.event()
        candidates = [self.root / "src/example.py", self.root / "frames.json",
                      self.root / event["identity"]["checkpoint"], self.queue.config_path]
        for path in candidates:
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaises((ValueError, json.JSONDecodeError)):
                self.queue.pump()
            path.write_bytes(original)
        changed = copy.deepcopy(self.config)
        changed["runtime"]["device"] = "cuda"
        with self.assertRaisesRegex(ValueError, "configuration"):
            VideoQueue(self.root, "unitrun", changed)

    def test_existing_job_different_identity_not_adopted(self):
        event = self.event()
        self.queue.pump()
        self.queue.dispatcher.jobs[event["job_id"]]["spec"]["inputs"] = {}
        with self.assertRaisesRegex(ValueError, "immutable video intent"):
            self.queue.pump()

    def test_profile_reuse_requires_conclusive_matching_record(self):
        first = self.event(5)
        self.queue.pump()
        self.event(10)
        profile = self.queue.dispatcher.directory / "profiles" / (first["job_id"] + ".json")
        profile.parent.mkdir()
        profile.write_text('{"status": "inconclusive"}')
        state = self.queue.state()
        self.assertEqual(state["events"][1]["profile_status"], "inconclusive")
        self.assertEqual(len(self.queue.dispatcher.jobs), 1)

    def test_production_profile_reuses_same_frozen_config_and_rejects_stale_profile(self):
        first = self.event(5)
        self.queue.pump()
        first_spec = self.queue.dispatcher.jobs[first["job_id"]]["spec"]
        second = self.event(10)
        profile = self.queue.dispatcher.directory / "profiles" / (first["job_id"] + ".json")
        profile.parent.mkdir()
        record = {"status": "conclusive", "profile_key": first_spec["profile_key"], "gpu_count": 1,
                  "recommended_reservation_mib": 10000, "code": first_spec["code"],
                  "configs": first_spec["configs"], "effective_workload_sha256": first_spec["effective_workload_sha256"]}
        profile.write_text(json.dumps({**record, "configs": {}}))
        with self.assertRaisesRegex(ValueError, "identities changed"):
            self.queue.pump()
        profile.write_text(json.dumps(record))
        self.queue.pump()
        second_spec = self.queue.dispatcher.jobs[second["job_id"]]["spec"]
        self.assertFalse(second_spec["profile"])
        self.assertEqual(first_spec["configs"], second_spec["configs"])
        self.assertEqual(second_spec["profile_id"], first["job_id"])

    def test_failed_job_is_not_retried_or_promoted_by_single_interval(self):
        event = self.event()
        self.queue.pump()
        self.queue.dispatcher.jobs[event["job_id"]]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "truthful"):
            self.queue.require_resume("B0", 5)
        self.assertEqual(len(self.queue.dispatcher.jobs), 1)

    def test_amp_scale_changes_gate_binding(self):
        from model.training.action4_gate import build_gate_binding
        config = {"stages": {"D0": {"roi_count": 1}}, "initialization": {}, "runtime": {}}
        records = [{"sample_id": "fixture", "image": "frames.json", "instances": [{"instance_id": "fixture"}]}]
        first = build_gate_binding(self.root, config, "frames.json", records, component=True)
        self.assertEqual(first["method"]["runtime"]["amp_init_scale"], 65536.)
        config["runtime"]["amp_init_scale"] = 8192
        second = build_gate_binding(self.root, config, "frames.json", records, component=True)
        self.assertNotEqual(first, second)

    def test_trainer_queued_backpressure_saves_epoch_and_never_calls_inline_video(self):
        # Explicit artificial component data and a mock gate isolate training
        # control flow. This test cannot issue formal acceptance.
        from test_action4_training_eval import fixture, TinyControlNet
        from model.training.action4_trainer import train
        config = fixture(self.root)
        config["runtime"]["require_cuda"] = False
        config["video"] = copy.deepcopy(self.config["video"])
        config["video"]["every_epochs"] = 1
        config["stages"]["B0"]["epochs"] = 3
        # Existing setUp queue belongs to another config; use a fresh run.
        init = self.root / "a0_fixture.pt"
        torch.save({"schema_version": "action4.checkpoint.v1", "stage": "A0", "config": config,
                    "claim_scope": "accepted_data_model_training", "model": TinyControlNet().state_dict()}, init)
        gate = self.root / "gate_fixture.json"
        gate.write_text("{}")
        with mock.patch("model.action4_network.Action4Net", TinyControlNet), \
             mock.patch("model.training.action4_trainer.require_dispatch_owner"), \
             mock.patch("model.training.action4_trainer.validate_d0_gate"), \
             mock.patch("eval.action4_video.run_video", side_effect=AssertionError("queued must not run inline")):
            result = train(self.root, config, stage="B0", run_id="control_fixture",
                           init_weights=init, d0_gate=gate, allow_diagnostic=True)
        self.assertEqual(result["status"], "paused_at_epoch_boundary_for_video_review")
        self.assertEqual(result["next_epoch"], 2)
        checkpoint = self.root / "models/action4/control_fixture/last.pt"
        state = torch.load(checkpoint, weights_only=True)
        self.assertTrue(state["epoch_boundary"])
        self.assertEqual(state["next_epoch"], 2)
        self.assertEqual(len(result["video_queue"]["outstanding"]), 2)
        with mock.patch("model.training.action4_trainer.require_dispatch_owner"), \
             mock.patch("model.action4_network.Action4Net") as constructor:
            with self.assertRaisesRegex(ValueError, "truthful"):
                train(self.root, config, stage="B0", run_id="control_fixture",
                      resume=checkpoint, d0_gate=gate, allow_diagnostic=True)
            constructor.assert_not_called()

    def test_invalid_dispatch_configuration_rejected_before_creating_new_queue(self):
        for changes in ({"profile_id": "../escape"}, {"reservation_mib": True},
                        {"reservation_mib": 0}, {"allowed_gpu_uuids": ["GPU-unassigned"]},
                        {"cpu_threads": 8}, {"cpu_threads": 33}, {"profile_first": False},
                        {"estimated_remaining_bytes": -1}):
            config = copy.deepcopy(self.config)
            config["video"]["dispatch"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                VideoQueue(self.root, "invalid_new", config)
            self.assertFalse((self.root / "outputs/action4/invalid_new").exists())

    def test_training_owner_requires_live_matching_dispatch_descendant(self):
        with mock.patch("model.training.action4_video_queue.Dispatcher") as factory, \
             mock.patch("model.training.action4_video_queue.descendants", return_value={os.getpid()}):
            dispatcher = factory.return_value
            dispatcher.state.return_value = {"jobs": {}}
            with self.assertRaisesRegex(ValueError, "live child"):
                require_dispatch_owner(self.root, "unitrun")
            job = {"status": "running", "spec": {"kind": "train", "result_dirs": [
                   str(self.root / "models/action4/unitrun")]}, "worker": {"pid": 123}}
            dispatcher.state.return_value = {"jobs": {"trainer": job}}
            require_dispatch_owner(self.root, "unitrun")
            job["status"] = "succeeded"
            with self.assertRaises(ValueError):
                require_dispatch_owner(self.root, "unitrun")

    def test_missing_historical_interval_cannot_be_replaced_by_later_best(self):
        self.event(10)
        with self.assertRaisesRegex(ValueError, "historical video event"):
            self.queue.require_resume("B0", 10)
        self.assertEqual(self.queue.dispatcher.jobs, {})

    def test_two_rank_final_epoch_reports_identical_pending_video_status(self):
        from test_action4_training_eval import fixture, TinyControlNet
        config = fixture(self.root)
        config["runtime"].update(require_cuda=False, cpu_threads=1)
        config["stages"]["B0"]["epochs"] = 1
        config["video"] = copy.deepcopy(self.config["video"])
        (self.root / "ddp_config.json").write_text(json.dumps(config))
        (self.root / "ddp_gate.json").write_text("{}")
        torch.save({"schema_version": "action4.checkpoint.v1", "stage": "A0", "config": config,
                    "claim_scope": "accepted_data_model_training", "model": TinyControlNet().state_dict()},
                   self.root / "ddp_a0.pt")
        with socket.socket() as endpoint:
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
        torch.multiprocessing.spawn(_queued_final_epoch_worker, args=(str(self.root), port),
                                    nprocs=2, join=True)
        results = [json.loads((self.root / f"rank_{rank}_queued_result.json").read_text()) for rank in range(2)]
        self.assertEqual([r["world_size"] for r in results], [2, 2])
        self.assertEqual([r["status"] for r in results],
                         ["training_updates_finished_pending_queued_video"] * 2)
        self.assertEqual(results[0]["video_queue"], results[1]["video_queue"])
        self.assertTrue(results[0]["video_queue"]["outstanding"])
        self.assertTrue(all(r["all_rank_model_states_equal"] for r in results))
        completion = json.loads((self.root / "outputs/action4/ddp_final_fixture/completion.json").read_text())
        self.assertEqual(completion["status"], results[0]["status"])
        print("two_rank_queued_final_statuses=" + json.dumps([r["status"] for r in results]))

    def test_failed_profile_has_readable_state_and_explicit_bounded_retry(self):
        first = self.event(5)
        self.queue.pump()
        self.event(10)
        self.queue.dispatcher.jobs[first["job_id"]]["status"] = "failed"
        profile = self.queue.dispatcher.directory / "profiles" / (first["job_id"] + ".json")
        profile.parent.mkdir()
        profile.write_text('{"status": "inconclusive"}')
        state = self.queue.state()
        self.assertEqual(state["failed"], [first["job_id"]])
        self.assertEqual(state["events"][1]["status"], "waiting_for_profile")
        self.assertTrue(callable(self.queue.retry_failed))

    def test_failed_profile_retry_restores_waiting_events_and_c1_resume(self):
        first = self.event(5, "C1")
        self.queue.pump()
        self.event(10, "C1")
        self.mark_failed(first)
        old_profile = self.queue.dispatcher.directory / "profiles" / (first["job_id"] + ".json")
        old_profile.parent.mkdir()
        old_profile.write_text('{"status": "inconclusive"}')
        self.assertEqual(self.queue.state()["failed"], [first["job_id"]])
        retried = self.retry()
        self.assertNotEqual(retried["job_id"], first["job_id"])
        self.assertEqual(retried["event_sha256"], first["event_sha256"])
        self.assertEqual(retried["identity"], first["identity"])
        self.assertEqual(len(retried["attempts"]), 2)
        self.assertEqual(retried["attempts"][0]["status"], "failed")
        self.assertTrue(self.queue.dispatcher.jobs[retried["job_id"]]["spec"]["profile"])
        self.conclusive_profile(retried)
        self.mock_completed_review(retried, "C1")
        state = self.queue.pump()
        second = state["events"][1]
        self.assertEqual(self.queue.dispatcher.jobs[second["job_id"]]["spec"]["profile_id"], retried["job_id"])
        self.mock_completed_review(second, "C1")
        self.assertEqual(self.queue.require_resume("C1", 10)["outstanding"], [])
        self.assertEqual(json.loads(old_profile.read_text())["status"], "inconclusive")
        self.assertEqual(self.queue.dispatcher.jobs[first["job_id"]]["status"], "failed")

    def test_failed_production_retry_requires_new_receipt_and_is_bounded(self):
        first = self.event(5, "C1")
        self.queue.pump()
        self.conclusive_profile(first)
        self.mock_completed_review(first, "C1")
        second = self.event(10, "C1")
        self.queue.pump()
        self.mark_failed(second)
        old_review = self.queue.directory / "reviews/epoch_010.json"
        old_review.write_text(json.dumps({"job_id": second["job_id"], "status": "reviewed"}))
        retried = self.retry(10)
        self.assertFalse(self.queue.dispatcher.jobs[retried["job_id"]]["spec"]["profile"])
        self.assertIn(retried["job_id"], self.queue.state()["outstanding"])
        with self.assertRaisesRegex(ValueError, "truthful"):
            self.queue.require_resume("C1", 10)
        new_review = self.mock_completed_review(retried, "C1")
        self.assertTrue(old_review.exists())
        self.assertIn("attempt_02", new_review.name)
        self.assertEqual(self.queue.require_resume("C1", 10)["outstanding"], [])
        self.mark_failed(retried)
        with self.assertRaisesRegex(ValueError, "one isolated retry"):
            self.retry(10)
        self.assertEqual(len(self.queue.dispatcher.jobs), 3)

    def test_retry_rejects_live_unknown_identity_deterministic_failure_and_changed_sha(self):
        first = self.event()
        self.queue.pump()
        job = self.mark_failed(first)
        with mock.patch("model.training.action4_video_queue.is_live", return_value=True):
            with self.assertRaisesRegex(ValueError, "live"):
                self.queue.retry_failed(5, controller="fixture", reason_category="oom", reason="fixture")
        with mock.patch("model.training.action4_video_queue.is_live", side_effect=RuntimeError("unknown observation")):
            with self.assertRaisesRegex(RuntimeError, "unknown observation"):
                self.queue.retry_failed(5, controller="fixture", reason_category="oom", reason="fixture")
        job["exitcode"] = None
        with self.assertRaisesRegex(ValueError, "known nonzero"):
            self.retry()
        job["exitcode"] = 1
        out = Path(job["spec"]["output_dir"])
        (out / "stdout.log").write_text("ValueError: shape mismatch; out of memory mentioned only as context")
        with self.assertRaisesRegex(ValueError, "deterministic"):
            self.retry()
        (out / "stdout.log").write_text("RuntimeError: CUDA out of memory")
        for path in (self.root / first["identity"]["checkpoint"], self.root / "frames.json",
                     self.root / "src/example.py", self.queue.config_path):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaises((ValueError, json.JSONDecodeError)):
                self.retry()
            path.write_bytes(original)
        for status in ("running", "queued", "succeeded"):
            job["status"] = status
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "terminal failed"):
                self.retry()
        job["status"] = "failed"
        with mock.patch("model.training.action4_video_queue.is_live", return_value=False), \
             mock.patch("model.training.action4_video_queue.descendants", return_value={987654323}):
            with self.assertRaisesRegex(ValueError, "live"):
                self.queue.retry_failed(5, controller="fixture", reason_category="oom", reason="fixture")
        self.assertEqual(len(self.queue.dispatcher.jobs), 1)

    def test_retry_rejects_missing_owner_proof_and_crash_replays_once(self):
        first = self.event()
        self.queue.pump()
        job = self.mark_failed(first)
        job["tracked_processes"] = [None]
        with self.assertRaisesRegex(ValueError, "identity is unknown"):
            self.retry()
        job["tracked_processes"] = []
        enqueue = self.queue.dispatcher.enqueue
        def crash_after_enqueue(spec, **kwargs):
            enqueue(spec, **kwargs)
            raise RuntimeError("injected crash after durable enqueue")
        with mock.patch.object(self.queue.dispatcher, "enqueue", side_effect=crash_after_enqueue):
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                self.retry()
        self.queue.pump()
        self.queue.pump()
        self.assertEqual(len(self.queue.dispatcher.jobs), 2)
        ledger = json.loads(self.queue.ledger_path.read_text())
        self.assertEqual(len(ledger["events"][0]["attempts"]), 2)
        with self.assertRaisesRegex(ValueError, "one isolated retry"):
            self.retry()

    def test_invalid_checkpoint_role_epoch_or_original_replay_rejected(self):
        source = self.checkpoint()
        data = torch.load(source, weights_only=True)
        for changes in ({"epoch_boundary": False}, {"stage": "A0"}, {"global_step": 0},
                        {"claim_scope": "component_diagnostic"}, {"run_id": "another"}, {"config": {}}):
            torch.save({**data, **changes}, source)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.queue.ensure_event("B0", 5, source)
        torch.save(data, source)
        self.queue.ensure_event("B0", 5, source)
        with self.assertRaisesRegex(ValueError, "original checkpoint"):
            self.queue.ensure_event("B0", 5, self.checkpoint(4))


class ScheduleAndPrecisionTests(unittest.TestCase):
    def test_inline_is_default_and_queued_is_explicit(self):
        self.assertEqual(execution_mode({}), "inline")
        self.assertEqual(execution_mode({"video": {"execution": "queued"}}), "queued")
        with self.assertRaises(ValueError):
            execution_mode({"video": {"execution": "invalid"}})

    def test_complete_epoch_backpressure_and_c1_boundaries(self):
        self.assertFalse(epoch_pause_required("B0", 5, 60, 1, complete_epoch=True))
        self.assertTrue(epoch_pause_required("B0", 10, 60, 2, complete_epoch=True))
        self.assertFalse(epoch_pause_required("B0", 10, 60, 2, complete_epoch=False))
        for epoch in (5, 10, 15):
            self.assertTrue(epoch_pause_required("C1", epoch, 20, 0, complete_epoch=True))
        self.assertFalse(epoch_pause_required("C1", 20, 20, 1, complete_epoch=True))
        config = {"video": {"frame_manifest": "frames.json", "every_epochs": 5}, "stages": {"C1": {}}}
        for epoch in (5, 10, 15, 20):
            self.assertTrue(video_due("C1", epoch, 20, config))
        self.assertFalse(video_due("B0", 5, 60, config, completed=False))

    def test_amp_scale_defaults_and_invalid_values(self):
        self.assertEqual(amp_initial_scale({}), 65536.)
        self.assertEqual(amp_initial_scale({"runtime": {"amp_init_scale": 8192}}), 8192.)
        for value in (0, -1, True, False, "8192", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                amp_initial_scale({"runtime": {"amp_init_scale": value}})


if __name__ == "__main__":
    unittest.main()
