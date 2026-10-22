"""Counterexample-driven checkpoint recovery tests; artificial CPU control only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from test_action4_training_eval import TinyControlNet, fixture
from model.training.action4_trainer import train, build_optimizer
from model.training.action4_runtime import load_checkpoint, save_checkpoint_atomic
from model.training.action4_checkpoint import validate_resume_checkpoint, recover_epoch20


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def _gloo_recovery_worker(rank, root_text, case):
    import datetime
    import time
    from model.training.action4_checkpoint import synchronize_checkpoint_recovery
    root = Path(root_text)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
        init_method="file://" + str(root / ("gloo_init_" + case)), timeout=datetime.timedelta(seconds=20))
    latest = root / "models/action4/joint/last.pt"
    state = load_checkpoint(latest)
    try:
        if rank == 0: time.sleep(.2)  # nonzero arrives while aliases are absent
        context = mock.patch("model.training.action4_checkpoint.link_checkpoint_atomic", side_effect=OSError("injected rank0 publication failure")) if case == "failure" and rank == 0 else mock.patch.dict({}, {})
        with context:
            synchronize_checkpoint_recovery(state, latest, rank=rank, distributed=True)
        result = {"rank": rank, "status": "success", "aliases_present": all((latest.parent / n).is_file() for n in (state["top_checkpoints"][0]["file"], "epoch_020.pt", "best_through_epoch020.pt"))}
    except Exception as exc:
        result = {"rank": rank, "status": "error", "error": str(exc), "error_type": type(exc).__name__}
    finally:
        torch.distributed.destroy_process_group()
    (root / (case + "_rank_" + str(rank) + ".json")).write_text(json.dumps(result))


def _gloo_queued_pause_worker(rank, root_text, ports):
    import os
    from test_action4_video_queue import FakeDispatcher
    from model.training.action4_runtime import sha256_file
    root = Path(root_text)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(ports[0]))
    config = json.loads((root / "queued_config.json").read_text())
    with mock.patch("model.action4_network.Action4Net", TinyControlNet), mock.patch("model.training.action4_trainer.require_dispatch_owner"), mock.patch("model.training.action4_trainer.validate_d0_gate"), mock.patch("model.training.action4_video_queue.Dispatcher", FakeDispatcher):
        paused = train(root, config, stage="B0", run_id="queued_pause", init_weights=root / "queued_a0.pt", d0_gate=root / "gate.json", allow_diagnostic=True)
        last = root / "models/action4/queued_pause/last.pt"
        before = sha256_file(last)
        errors, cleaned, constructors = [], [], []
        for attempt in range(2):
            os.environ["MASTER_PORT"] = str(ports[attempt + 1])
            with mock.patch("model.action4_network.Action4Net") as constructor:
                try:
                    train(root, config, stage="B0", run_id="queued_pause", resume=last, d0_gate=root / "gate.json", allow_diagnostic=True)
                    errors.append("unexpected_success")
                except ValueError as exc: errors.append(str(exc))
                constructors.append(constructor.call_count)
            cleaned.append(not torch.distributed.is_initialized())
        result = {"rank": rank, "paused_status": paused["status"], "paused_epoch": paused["next_epoch"], "queue": paused["video_queue"], "errors": errors, "groups_cleaned": cleaned, "resume_model_constructor_calls": constructors, "checkpoint_unchanged": sha256_file(last) == before}
    (root / ("queued_rank_" + str(rank) + ".json")).write_text(json.dumps(result))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = fixture(self.root)
        self.config["stages"]["A0"]["epochs"] = 3
        self.patch = mock.patch("model.action4_network.Action4Net", TinyControlNet)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.evaluation = mock.patch("model.training.action4_trainer._run_epoch_evaluation", return_value=({"epe_p95_heatmap_px": 1.}, None, []))
        self.evaluation.start(); self.addCleanup(self.evaluation.stop)

    def run_train(self, run="victim", **kwargs):
        return train(self.root, self.config, stage="A0", run_id=run,
                     allow_diagnostic=True, component_smoke=True, **kwargs)

    def prepare(self):
        self.run_train(max_updates=2)
        self.directory = self.root / "models/action4/victim"
        self.latest = self.directory / "last.pt"
        self.state = load_checkpoint(self.latest)
        return self.state

    def resume(self):
        return self.run_train(resume=self.latest)

    def test_foreign_retained_checkpoint_rejected(self):
        self.prepare(); self.run_train("foreign", max_updates=1)
        foreign = load_checkpoint(self.root / "models/action4/foreign/last.pt")
        save_checkpoint_atomic(self.directory / self.state["top_checkpoints"][0]["file"], foreign)
        with self.assertRaisesRegex(ValueError, "Retained checkpoint content identity"):
            self.resume()

    def test_same_run_finite_tensor_drift_and_truncated_checkpoint_rejected(self):
        self.prepare()
        path = self.directory / self.state["top_checkpoints"][0]["file"]
        original = path.read_bytes()
        changed = load_checkpoint(path)
        next(iter(changed["model"].values())).add_(.25)
        save_checkpoint_atomic(path, changed)
        with self.assertRaisesRegex(ValueError, "content identity"):
            self.resume()
        path.write_bytes(original[:len(original)//2])
        with self.assertRaisesRegex(ValueError, "content identity"):
            self.resume()

    def test_current_alias_mismatch_rejected(self):
        self.prepare()
        current = self.directory / "epoch_002_step_00000002.pt"
        changed = load_checkpoint(current); changed["run_id"] = "foreign"
        save_checkpoint_atomic(current, changed)
        with self.assertRaisesRegex(ValueError, "content identity"):
            self.resume()

    def test_missing_current_alias_repaired_but_missing_historical_rejected(self):
        self.prepare()
        current = self.directory / "epoch_002_step_00000002.pt"; current.unlink()
        self.resume()
        self.assertTrue(current.is_file())
        # Use an independent run to keep further updates available.
        self.run_train("missing", max_updates=2)
        self.latest = self.root / "models/action4/missing/last.pt"
        state = load_checkpoint(self.latest)
        (self.latest.parent / state["top_checkpoints"][0]["file"]).unlink()
        with self.assertRaisesRegex(ValueError, "older retained checkpoint is missing"):
            self.run_train("missing", resume=self.latest)

    def test_counter_scheduler_sampler_and_rng_corruption_rejected(self):
        self.prepare()
        corruptions = [
            (lambda s: s.update(global_step=999), "consistency"),
            (lambda s: s["scheduler"].update(last_epoch=0), "scheduler progress"),
            (lambda s: s["sampler"].update(next_epoch=0), "sampler"),
            (lambda s: s.update(rng_by_rank=[]), "RNG topology"),
            (lambda s: s["resume_integrity"].update(nonfinite_updates=1), "consistency"),
        ]
        for mutate, message in corruptions:
            with self.subTest(message=message):
                changed = copy.deepcopy(self.state); mutate(changed)
                save_checkpoint_atomic(self.latest, changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.resume()

    def test_legacy_partial_nonfinite_and_optimizer_shape_rejected(self):
        self.prepare()
        legacy = copy.deepcopy(self.state); legacy.pop("resume_integrity")
        with self.assertRaisesRegex(ValueError, "legacy checkpoints"):
            validate_resume_checkpoint(legacy)
        partial = copy.deepcopy(self.state); partial["epoch_boundary"] = False
        with self.assertRaisesRegex(ValueError, "complete epoch boundary"):
            validate_resume_checkpoint(partial)
        broken = copy.deepcopy(self.state)
        next(iter(broken["model"].values())).flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite tensor"):
            validate_resume_checkpoint(broken)
        optimizer = build_optimizer(TinyControlNet(), self.config["stages"]["A0"])
        broken = copy.deepcopy(self.state)
        next(iter(broken["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "moment shape"):
            validate_resume_checkpoint(broken, optimizer=optimizer)
        with self.assertRaisesRegex(ValueError, "device type"):
            validate_resume_checkpoint(self.state, device=torch.device("cuda"))

    def test_consistently_changed_lr_and_optimizer_hyperparameters_rejected(self):
        from model.training.action4_trainer import learning_rate_factor
        self.prepare()
        optimizer = build_optimizer(TinyControlNet(), self.config["stages"]["A0"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, 3, 0))
        validate_resume_checkpoint(self.state, optimizer=optimizer, scheduler=scheduler)
        changed = copy.deepcopy(self.state)
        changed["scheduler"]["_last_lr"] = [1.0] * len(changed["optimizer"]["param_groups"])
        for group in changed["optimizer"]["param_groups"]: group["lr"] = 1.0
        with self.assertRaisesRegex(ValueError, "declared schedule"):
            validate_resume_checkpoint(changed, optimizer=optimizer, scheduler=scheduler)
        changed = copy.deepcopy(self.state)
        changed["optimizer"]["param_groups"][0]["weight_decay"] = 2.0
        with self.assertRaisesRegex(ValueError, "hyperparameter changed"):
            validate_resume_checkpoint(changed, optimizer=optimizer, scheduler=scheduler)

    def test_freeze_unused_head_and_complete_state_continuation(self):
        self.config["runtime"]["workers"] = 2
        self.config["stages"]["A0"]["freeze_backbone_updates"] = 1
        self.run_train("full")
        self.run_train("split", max_updates=1)
        p = self.root / "models/action4/split/last.pt"
        validate_resume_checkpoint(load_checkpoint(p))
        self.run_train("split", resume=p)
        a = load_checkpoint(self.root / "models/action4/full/last.pt"); b = load_checkpoint(p)
        for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "roi_rng_by_rank", "sampler", "global_step", "next_epoch"):
            self.assertTrue(equal_tree(a[key], b[key]), key)
        steps = {int(value["step"].item()) for value in b["optimizer"]["state"].values()}
        self.assertEqual(steps, {2, 3})

    def test_allowed_skipped_update_retains_cumulative_counter_on_resume(self):
        self.config["stages"]["A0"]["max_nonfinite_updates"] = 1
        calls = 0
        def fail_one_then_interrupt(value, device, distributed):
            nonlocal calls
            calls += 1
            if calls == 3: return False  # two microbatch loss checks then one synchronized gradient rejection
            if calls == 4: raise OSError("interrupt next epoch after zero-success committed boundary")
            return value
        with mock.patch("model.training.action4_trainer._all_ranks_finite", side_effect=fail_one_then_interrupt):
            with self.assertRaises(OSError): self.run_train()
        self.latest = self.root / "models/action4/victim/last.pt"
        state = load_checkpoint(self.latest)
        self.assertEqual((state["next_epoch"], state["global_step"], state["resume_integrity"]["nonfinite_updates"]), (1, 0, 1))
        validate_resume_checkpoint(state)
        self.resume()
        state = load_checkpoint(self.latest)
        self.assertEqual((state["next_epoch"], state["global_step"], state["resume_integrity"]["nonfinite_updates"]), (3, 2, 1))

    def test_epoch20_interruption_recovers_exact_snapshot_and_preserves_it(self):
        from model.training.action4_checkpoint import link_checkpoint_atomic
        self.config["stages"]["B0"]["epochs"] = 22
        self.run_train("init", max_updates=1)
        init = self.root / "models/action4/init/best.pt"
        def interrupted(source, target):
            if target.name == "epoch_020.pt": raise OSError("injected before mandatory publication")
            return link_checkpoint_atomic(source, target)
        with mock.patch("model.training.action4_checkpoint.link_checkpoint_atomic", side_effect=interrupted):
            with self.assertRaises(OSError):
                train(self.root, self.config, stage="B0", run_id="joint", init_weights=init,
                      allow_diagnostic=True, component_smoke=True)
        directory = self.root / "models/action4/joint"; last = directory / "last.pt"
        state20 = load_checkpoint(last)
        self.assertEqual(state20["next_epoch"], 20)
        train(self.root, self.config, stage="B0", run_id="joint", resume=last,
              allow_diagnostic=True, component_smoke=True, max_updates=21)
        epoch20 = load_checkpoint(directory / "epoch_020.pt")
        self.assertEqual(epoch20["next_epoch"], 20)
        self.assertTrue(equal_tree(epoch20["model"], state20["model"]))
        self.assertEqual(load_checkpoint(directory / "best_through_epoch020.pt")["next_epoch"], 1)
        (directory / "epoch_020.pt").unlink()
        with self.assertRaisesRegex(ValueError, "cannot be reconstructed from later weights"):
            train(self.root, self.config, stage="B0", run_id="joint", resume=last,
                  allow_diagnostic=True, component_smoke=True)

    def test_partial_diagnostic_retention_preserves_committed_resume(self):
        self.config["stages"]["A0"].update(epochs=4, global_batch=1)
        self.evaluation.stop()
        scores = iter([float(n) for n in range(30, 0, -1)])
        with mock.patch("model.training.action4_trainer._run_epoch_evaluation", side_effect=lambda *args: ({"epe_p95_heatmap_px": next(scores)}, None, [])):
            self.run_train("full")
            self.run_train("partial", max_updates=7)
            directory = self.root / "models/action4/partial"
            last = directory / "last.pt"
            committed = load_checkpoint(last)
            self.assertEqual((committed["next_epoch"], committed["global_step"]), (3, 6))
            self.assertTrue((directory / "component_partial.pt").is_file())
            for entry in committed["top_checkpoints"]:
                self.assertTrue((directory / entry["file"]).is_file(), entry["file"])
            self.run_train("partial", resume=last)
            full = load_checkpoint(self.root / "models/action4/full/last.pt")
            resumed = load_checkpoint(last)
            self.assertTrue(equal_tree(full["model"], resumed["model"]))
            self.assertTrue(equal_tree(full["optimizer"], resumed["optimizer"]))
            self.assertEqual(resumed["global_step"], 8)

    def test_cuda_rng_completeness_and_current_device_count_before_restore(self):
        self.prepare()
        state = copy.deepcopy(self.state)
        state["resume_integrity"]["device_type"] = "cuda"
        # CPU validation fixtures only: mocked CUDA templates do not use a GPU.
        for bad in ([], None, [torch.empty(0, dtype=torch.uint8)], [torch.zeros(16)], ["not a tensor"]):
            changed = copy.deepcopy(state); changed["rng_by_rank"][0]["cuda"] = bad
            with self.subTest(bad_type=str(type(bad))):
                with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                    validate_resume_checkpoint(changed)
        state["rng_by_rank"][0]["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
        with mock.patch("torch.cuda.device_count", return_value=1), mock.patch("torch.cuda.get_rng_state", return_value=torch.zeros(16, dtype=torch.uint8)):
            validate_resume_checkpoint(state, device=torch.device("cuda"))
            changed = copy.deepcopy(state); changed["rng_by_rank"][0]["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
            with self.assertRaisesRegex(ValueError, "tensor shape"):
                validate_resume_checkpoint(changed, device=torch.device("cuda"))
        with mock.patch("torch.cuda.device_count", return_value=2), mock.patch("torch.cuda.get_rng_state") as rng_read:
            with self.assertRaisesRegex(ValueError, "visible device count"):
                validate_resume_checkpoint(state, device=torch.device("cuda"))
            rng_read.assert_not_called()
        validate_resume_checkpoint(self.state, device=torch.device("cpu"))

    def test_two_rank_epoch20_repair_and_rank0_failure_are_synchronized(self):
        self.config["stages"]["B0"]["epochs"] = 21
        self.run_train("init", max_updates=1)
        scores = iter([float(i) for i in range(30, 0, -1)])
        with mock.patch("model.training.action4_trainer._run_epoch_evaluation", side_effect=lambda *args: ({"epe_p95_heatmap_px": next(scores)}, None, [])):
            train(self.root, self.config, stage="B0", run_id="joint", init_weights=self.root / "models/action4/init/best.pt",
                  allow_diagnostic=True, component_smoke=True, max_updates=20)
        directory = self.root / "models/action4/joint"
        state = load_checkpoint(directory / "last.pt")
        self.assertEqual(state["top_checkpoints"][0]["file"], "epoch_020_step_00000020.pt")
        for case in ("success", "failure"):
            for name in (state["top_checkpoints"][0]["file"], "epoch_020.pt", "best_through_epoch020.pt"):
                (directory / name).unlink()
            torch.multiprocessing.spawn(_gloo_recovery_worker, args=(str(self.root), case), nprocs=2, join=True)
            results = [json.loads((self.root / (case + "_rank_" + str(i) + ".json")).read_text()) for i in range(2)]
            if case == "success":
                self.assertTrue(all(r["status"] == "success" and r["aliases_present"] for r in results))
            else:
                self.assertTrue(all(r["status"] == "error" for r in results))
                self.assertEqual(results[0]["error"], results[1]["error"])
                self.assertIn("injected rank0 publication failure", results[0]["error"])

    def test_two_rank_pending_review_blocks_resume_and_cleans_owned_groups(self):
        import socket
        from model.training.action4_dispatch import ALLOWED_GPU_UUIDS
        config = copy.deepcopy(self.config)
        config["runtime"].update(require_cuda=False, cpu_threads=1)
        config["stages"]["B0"]["epochs"] = 3
        config["video"] = {"execution": "queued", "every_epochs": 1, "frame_manifest": "frames.json",
                           "dispatch": {"allowed_gpu_uuids": [ALLOWED_GPU_UUIDS[0]], "reservation_mib": 12000,
                                        "cpu_threads": 9, "profile_key": "cpu-test-only", "profile_first": True}}
        (self.root / "frames.json").write_text(json.dumps({"complete": True, "source": {"frame_count": 2, "time_base": "1/11988"}, "frames": [{"frame_id": i, "pts": i*400, "duration_pts": 400} for i in range(2)]}))
        for name in ("src/example.py", "scripts/action4_video.py", "scripts/action4_dispatch.py"):
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# Explicit CPU control fixture only")
        (self.root / "queued_config.json").write_text(json.dumps(config))
        (self.root / "gate.json").write_text("{}")
        torch.save({"schema_version": "action4.checkpoint.v1", "stage": "A0", "config": config, "claim_scope": "accepted_data_model_training", "model": TinyControlNet().state_dict()}, self.root / "queued_a0.pt")
        endpoints = [socket.socket() for _ in range(3)]
        try:
            for endpoint in endpoints: endpoint.bind(("127.0.0.1", 0))
            ports = [endpoint.getsockname()[1] for endpoint in endpoints]
        finally:
            for endpoint in endpoints: endpoint.close()
        torch.multiprocessing.spawn(_gloo_queued_pause_worker, args=(str(self.root), ports), nprocs=2, join=True)
        results = [json.loads((self.root / ("queued_rank_" + str(i) + ".json")).read_text()) for i in range(2)]
        self.assertEqual([r["paused_status"] for r in results], ["paused_at_epoch_boundary_for_video_review"] * 2)
        self.assertEqual([r["paused_epoch"] for r in results], [2, 2])
        self.assertEqual(results[0]["queue"], results[1]["queue"])
        self.assertEqual(results[0]["errors"], results[1]["errors"])
        for result in results:
            self.assertTrue(all("truthful" in error for error in result["errors"]))
            self.assertEqual(result["groups_cleaned"], [True, True])
            self.assertEqual(result["resume_model_constructor_calls"], [0, 0])
            self.assertTrue(result["checkpoint_unchanged"])

    def test_process_group_cleanup_preserves_preexisting_group(self):
        from model.training.action4_checkpoint import cleanup_owned_process_group
        @cleanup_owned_process_group
        def failure(): raise OSError("pre-loop failure")
        with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch("torch.distributed.destroy_process_group") as destroy:
            with self.assertRaises(OSError): failure()
            destroy.assert_not_called()
        with mock.patch("torch.distributed.is_initialized", side_effect=[False, True]), mock.patch("torch.distributed.destroy_process_group") as destroy:
            with self.assertRaises(OSError): failure()
            destroy.assert_called_once()

    def test_epoch20_existing_wrong_snapshot_rejected(self):
        # A controlled publication check with the exact commit created above is
        # separately covered; here existing mismatched bytes must never overwrite.
        self.prepare()
        wrong = copy.deepcopy(self.state)
        wrong["stage"] = "B0"; wrong["next_epoch"] = 20
        wrong["mandatory_snapshots"] = {
            "epoch_020.pt": {"source_file": "epoch_020_step_00000002.pt", "global_step": 2, "sha256": None},
            "best_through_epoch020.pt": {"source_file": self.state["top_checkpoints"][0]["file"], "global_step": 1, "sha256": None}}
        target = self.directory / "epoch_020.pt"; target.write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError, "content identity changed"):
            recover_epoch20(wrong, self.latest, repair=True)
        self.assertEqual(target.read_bytes(), b"wrong checkpoint")


if __name__ == "__main__": unittest.main()
