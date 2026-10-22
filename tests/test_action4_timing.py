"""CPU/mock timing contracts only: no model, training data, GPU or queue runs."""
import ast
import copy
import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from model.training.action4_timing import timing_enabled, synchronized_update_boundary, measured_update_window


class MockAsyncTorch:
    """Explicit virtual async work, not a timing/performance claim."""
    def __init__(self, *, remote_marker=None):
        self.events = ["optimizer_tail_queued", "loss_collective_queued", "ordinary_log_closed"]
        self.now = 5
        self.pending_ns = 100
        self.remote_marker = remote_marker
        self.cuda = SimpleNamespace(synchronize=self.sync)
        self.distributed = SimpleNamespace(barrier=self.barrier, broadcast_object_list=self.broadcast)

    def sync(self, device):
        self.events.append("cuda_sync")
        self.now += self.pending_ns
        self.pending_ns = 0

    def barrier(self):
        self.events.append("all_rank_barrier")
        self.now += 20  # virtual wait for a slower other rank

    def clock(self):
        self.events.append("rank0_monotonic_clock")
        return self.now

    def broadcast(self, marker, src, device):
        self.events.append("release_marker_broadcast")
        if marker[0] is None:
            marker[0] = self.remote_marker
        self.now += 3


def _gloo_boundary_worker(rank, root_text, port):
    import torch
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
                                         init_method=f"tcp://127.0.0.1:{port}")
    try:
        if rank == 1:
            time.sleep(.05)  # explicit CPU-only delayed participant, no training
        before = time.perf_counter_ns()
        marker = synchronized_update_boundary(torch, torch.device("cpu"),
                                               distributed=True, rank=rank, enabled=True)
        after = time.perf_counter_ns()
        (Path(root_text) / f"rank_{rank}.json").write_text(json.dumps(
            {"rank": rank, "before": before, "marker": marker, "after": after}))
    finally:
        torch.distributed.destroy_process_group()


class TimingTests(unittest.TestCase):
    def test_async_optimizer_tail_counterexample_and_collective_order(self):
        torch = MockAsyncTorch()
        premature_legacy_marker = torch.now
        marker = synchronized_update_boundary(torch, SimpleNamespace(type="cuda"),
                                               distributed=True, rank=0, enabled=True, clock=torch.clock)
        self.assertEqual(premature_legacy_marker, 5)
        self.assertEqual(marker, 125)  # includes100ns async tail +20ns remote rank wait
        self.assertEqual(torch.events, [
            "optimizer_tail_queued", "loss_collective_queued", "ordinary_log_closed",
            "cuda_sync", "all_rank_barrier", "cuda_sync", "rank0_monotonic_clock",
            "release_marker_broadcast"])
        self.assertLess(marker, torch.now)  # timestamp precedes releasing the next update

    def test_nonzero_rank_uses_rank0_clock_only(self):
        torch = MockAsyncTorch(remote_marker=123456)
        def forbidden_clock():
            raise AssertionError("nonzero rank clocks cannot be compared")
        marker = synchronized_update_boundary(torch, SimpleNamespace(type="cuda"),
                                               distributed=True, rank=1, enabled=True, clock=forbidden_clock)
        self.assertEqual(marker, 123456)
        self.assertNotIn("rank0_monotonic_clock", torch.events)

    def test_default_off_has_no_clock_cuda_collective_or_logging_calls(self):
        class Forbidden:
            def __getattr__(self, name):
                raise AssertionError("Off mode touched " + name)
        def forbidden_clock():
            raise AssertionError("Off mode read clock")
        self.assertFalse(timing_enabled({}))
        self.assertFalse(timing_enabled({"runtime": {"synchronized_update_timing": False}}))
        self.assertIsNone(synchronized_update_boundary(Forbidden(), Forbidden(), distributed=True,
                                                      rank=0, enabled=False, clock=forbidden_clock))

    def test_actual_two_rank_gloo_marker_follows_all_participants_and_precedes_release(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            with socket.socket() as endpoint:
                endpoint.bind(("127.0.0.1", 0))
                port = endpoint.getsockname()[1]
            torch.multiprocessing.spawn(_gloo_boundary_worker, args=(temporary, port), nprocs=2, join=True)
            rows = [json.loads((Path(temporary) / f"rank_{rank}.json").read_text()) for rank in range(2)]
        self.assertEqual(rows[0]["marker"], rows[1]["marker"])
        self.assertGreaterEqual(rows[0]["marker"], max(row["before"] for row in rows))
        self.assertLessEqual(rows[0]["marker"], min(row["after"] for row in rows))

    def test_config_boolean_is_strict(self):
        self.assertTrue(timing_enabled({"runtime": {"synchronized_update_timing": True}}))
        for value in (0, 1, None, "true", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_enabled({"runtime": {"synchronized_update_timing": value}})

    def test_cpu_boundary_does_not_touch_cuda(self):
        torch = MockAsyncTorch()
        marker = synchronized_update_boundary(torch, SimpleNamespace(type="cpu"),
                                               distributed=False, rank=0, enabled=True, clock=torch.clock)
        self.assertEqual(marker, 5)
        self.assertNotIn("cuda_sync", torch.events)
        self.assertNotIn("all_rank_barrier", torch.events)

    @staticmethod
    def records():
        return [{"epoch": 1, "global_step": step, "optimizer_updated": True, "global_actual_count": 8,
                 "completed_monotonic_ns": step * 1_000_000_000} for step in range(1, 31)]

    def test_exact_10_to30_window_has160_images_not_warmup_median(self):
        records = self.records()
        # Dramatically different earlier warmup duration must not bias window.
        for row in records[:9]:
            row["completed_monotonic_ns"] = row["global_step"]
        report = measured_update_window(records, stage="B0")
        self.assertEqual(report["status"], "measured")
        self.assertEqual(report["images"], 160)
        self.assertEqual(report["measured_updates"], 20)
        self.assertEqual(report["seconds"], 20)
        self.assertEqual(report["images_per_second"], 8)
        self.assertEqual(measured_update_window(records, stage="A0")["status"], "not_full_image_stage")

    def test_missing_duplicate_cross_resume_or_incomplete_updates_cannot_be_measured(self):
        original = self.records()
        variants = [original[10:], original[:-1], original + [copy.deepcopy(original[-1])]]
        for records in variants:
            with self.subTest(length=len(records)):
                self.assertNotEqual(measured_update_window(records, stage="B0")["status"], "measured")
        for key, value in (("optimizer_updated", False), ("global_actual_count", 7),
                           ("completed_monotonic_ns", 0), ("completed_monotonic_ns", float("nan")),
                           ("epoch", 2)):
            records = copy.deepcopy(original)
            records[20][key] = value
            with self.subTest(key=key):
                self.assertNotEqual(measured_update_window(records, stage="B0")["status"], "measured")

    def test_trainer_places_boundary_after_original_log_and_before_next_batch(self):
        # This guards the integration point: testing helper order alone would
        # not detect moving it before optimizer/log synchronization in trainer.
        source = Path("src/model/training/action4_trainer.py").read_text()
        tree = ast.parse(source)
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.unparse(node.target) == "(batch_index, items)")
        statements = loop.body
        timing_index = next(i for i, node in enumerate(statements)
                            if isinstance(node, ast.If) and ast.unparse(node.test) == "synchronized_timing")
        prior = statements[timing_index - 1]
        self.assertIsInstance(prior, ast.If)
        self.assertEqual(ast.unparse(prior.test), "rank == 0")
        self.assertIsInstance(prior.body[0], ast.With)
        self.assertIn("train_updates.jsonl", ast.unparse(prior))
        call_line = next(node.lineno for node in ast.walk(statements[timing_index])
                         if isinstance(node, ast.Call) and ast.unparse(node.func) == "synchronized_update_boundary")
        step_lines = [node.lineno for node in ast.walk(loop) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) in ("scaler.step", "scaler.update", "scheduler.step",
                                                     "torch.distributed.all_reduce")]
        self.assertTrue(step_lines and max(step_lines) < call_line)
        self.assertIn("synchronized_update_boundaries.jsonl", ast.unparse(statements[timing_index]))
        # Full config already participates in run identity, so a changed
        # timing flag cannot be silently resumed/profiled as the same config.
        self.assertIn('"config_sha256": canonical_hash(config)', source)


if __name__ == "__main__":
    unittest.main()
