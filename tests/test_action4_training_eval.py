"""Meaningful control-flow/metric/IO tests; fixtures do not claim model accuracy."""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

from data import DataAdapter
from eval.action4_feedback import candidate_events, reviewed_requests
from eval.action4_metrics import maximum_matching, score_image, score_dataset
from eval.action4_runner import freeze_test_selection, run_evaluation, tune_decode
from eval.action4_video import clip_line, draw_cuboids, run_video
from model.training.action4_runtime import load_checkpoint
from model.training.action4_trainer import EpochWindowSampler, epoch_indices, learning_rate_factor, train

torch.set_num_threads(2)


def cuboid(x=10., y=10., *, ignore=False):
    points = np.array([[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10],
                       [x + 2, y + 2], [x + 12, y + 2], [x + 12, y + 12], [x + 2, y + 12]])
    return {"instance_id": "camera", "object_id": 1, "corners_px": points.tolist(),
            "bbox_full_xyxy": [x, y, x + 12, y + 12], "bbox_visible_xyxy": [x, y, x + 12, y + 12],
            "corner_visible": [True] * 8, "ignore": ignore, "score": .9}


class TinyControlNet(nn.Module):
    """A tiny trainable fixture validates runner semantics, not ResNet accuracy."""
    def __init__(self, weights=None, **kwargs):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, 4, 1), nn.ReLU())
        self.corner_head = nn.Conv2d(4, 8, 1)
        self.center_head = nn.Conv2d(4, 1, 1)
        self.offset_head = nn.Conv2d(4, 2, 1)
        self.box_head = nn.Conv2d(4, 4, 1)

    def forward(self, images, roi_images=None, *, roi_builder=None):
        result = {}
        if images is not None:
            features = self.backbone(images)
            result = {"center_logits": self.center_head(features), "offset": self.offset_head(features),
                      "ltrb": self.box_head(features).softplus() if hasattr(torch.Tensor, "softplus") else torch.nn.functional.softplus(self.box_head(features))}
        if roi_builder is not None:
            roi_images, metadata = roi_builder({k: v.detach() for k, v in result.items()})
            result["roi_metadata"] = metadata
        if roi_images is not None:
            if len(roi_images):
                result["corner_logits"] = self.corner_head(self.backbone(roi_images))
            else:
                result["corner_logits"] = torch.empty((0, 8, roi_images.shape[-2] // 4, roi_images.shape[-1] // 4), device=roi_images.device) + self.corner_head.weight.sum() * 0
        else:
            result["corner_logits"] = torch.empty((0, 8, 8, 8), device=images.device) + self.corner_head.weight.sum() * 0
        return result


def fixture(root):
    paths = {}
    for split, count in (("train", 2), ("eval", 1), ("test", 1)):
        rows = []
        for i in range(count):
            sample_id = f"{split}_{i}"
            path = root / f"{sample_id}.png"
            yy, xx = np.mgrid[:48, :64]
            image = np.stack((xx * 3, yy * 4, (xx + yy) * 2), axis=-1).astype(np.uint8)
            Image.fromarray(image).save(path)
            rows.append({"schema_version": "action4.v1", "sample_id": sample_id, "image": path.name,
                         "width": 64, "height": 48, "group_id": sample_id, "background_source_id": sample_id,
                         "scene_family": "artificial_unit_fixture", "instances": [cuboid()], "split": split,
                         "status": "unit_test_fixture"})
        manifest = root / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[f"{split}_manifest"] = str(manifest)
    return {
        "network": {"decoder_channels": 16, "head_channels": 8, "freeze_bn": True},
        "preprocess": {"full_size": [64, 64], "roi_size": 32, "roi_expansion": 1.25},
        "decode": {"score_threshold": .99, "topk": 20, "max_detections": 2,
                   "duplicate_iou": .95, "duplicate_center_ratio": .05, "corner_topk": 4},
        "initialization": {"backbone_weights": None}, "data": paths,
        "runtime": {"device": "cpu", "require_cuda": True, "workers": 0, "cpu_threads": 2,
                    "amp": False, "seed": 20261018, "deterministic": True},
        "eval": {"batch_size": 1}, "video": {"frame_manifest": None, "every_epochs": 5, "line_width": 2},
        "stages": {name: {"epochs": 2, "roi_count": 2, "global_batch": 2, "micro_batch": 1,
                          "warmup_updates": 0, "freeze_backbone_updates": 0, "lr_backbone": 1e-3,
                          "lr_new": 1e-3, "weight_decay": 1e-4, "clip_grad_norm": 1.}
                   for name in ("D0", "A0", "B0", "C1")}}


def _ddp_worker(rank, root_text, port):
    root = Path(root_text)
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    config = json.loads((root / "ddp_config.json").read_text())
    with mock.patch("model.action4_network.Action4Net", TinyControlNet):
        result = train(root, config, stage="B0", run_id="ddp", init_weights=root / "models/action4/a0/best.pt",
                       allow_diagnostic=True, component_smoke=True, max_updates=1)
    (root / f"rank_{rank}_result.json").write_text(json.dumps(result))


class TrainingEvalTests(unittest.TestCase):
    def test_maximum_matching_and_bad_corner_semantics(self):
        self.assertEqual(len(maximum_matching([[0, 1], [0]], 2)), 2)
        gt = cuboid()
        wrong = copy.deepcopy(gt)
        wrong["corners_px"] = wrong["corners_px"][4:] + wrong["corners_px"][:4]
        scored = score_image([wrong], [gt])
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"]), (0, 1, 1))
        self.assertEqual(scored["bad_corner_instances"], 1)
        ignored = copy.deepcopy(gt)
        ignored["ignore"] = True
        scored = score_image([gt], [gt, ignored])
        self.assertEqual((scored["tp"], scored["ignored_predictions"]), (1, 0))
        scored = score_image([gt, gt, gt], [ignored])
        self.assertEqual((scored["fp"], scored["ignored_predictions"]), (2, 1))

    def test_undefined_metrics_and_missing_sample_rejected(self):
        sample = {"sample_id": "empty", "scene_family": "empty", "width": 64, "instances": []}
        result = score_dataset([{"sample_id": "empty", "instances": []}], [sample])
        self.assertIsNone(result["thresholds"]["0.05"]["f1"])
        with self.assertRaises(ValueError):
            score_dataset([], [sample])

    def test_ddp_tail_has_exact_real_coverage_and_mean_weight(self):
        indices = list(range(11))
        ranks = [EpochWindowSampler(indices, 8, 1, rank, 4) for rank in range(4)]
        seen = [index for rank in ranks for batch in rank for index, real in batch if real]
        self.assertEqual(sorted(seen), indices)
        self.assertEqual([len(list(s)) for s in ranks], [4] * 4)
        gradient = 0.
        for sampler in ranks:
            batches = list(sampler)
            for batch in batches[2:]:
                values = [index for index, real in batch if real]
                gradient += sum(values) * 4 / sampler.window_counts[-1] / 4
        self.assertAlmostEqual(gradient, np.mean([8, 9, 10]))

    def test_replay_rotation_quotas_and_schedule(self):
        first = epoch_indices(8000, epoch=0, seed=8, s1_count=2000)
        second = epoch_indices(8000, epoch=1, seed=8, s1_count=2000)
        self.assertEqual((sum(i < 8000 for i in first), sum(i >= 8000 for i in first)), (6000, 2000))
        self.assertEqual(len(set(i for i in first + second if i < 8000)), 8000)
        self.assertEqual(first, epoch_indices(8000, epoch=0, seed=8, s1_count=2000))
        self.assertAlmostEqual(learning_rate_factor(0, 1000, 100), .1)
        self.assertAlmostEqual(learning_rate_factor(100, 1000, 100), 1.)
        self.assertAlmostEqual(learning_rate_factor(1000, 1000, 100), .01)

    def test_epoch_resume_reproduces_uninterrupted_parameters(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("model.action4_network.Action4Net", TinyControlNet):
            root = Path(temporary)
            config = fixture(root)
            full = train(root, config, stage="D0", run_id="complete", allow_diagnostic=True, component_smoke=True)
            partial = train(root, config, stage="D0", run_id="interrupted", allow_diagnostic=True, component_smoke=True, max_updates=1)
            resume_path = root / "models/action4/interrupted/last.pt"
            # Simulate the small publication interruption after last.pt was
            # written but before its same-epoch immutable alias was linked.
            (root / "models/action4/interrupted/epoch_001_step_00000001.pt").unlink()
            resumed = train(root, config, stage="D0", run_id="interrupted", resume=resume_path,
                            allow_diagnostic=True, component_smoke=True, max_updates=2)
            a = load_checkpoint(root / "models/action4/complete/last.pt")
            b = load_checkpoint(resume_path)
            self.assertEqual((a["global_step"], b["global_step"]), (2, 2))
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(full["claim_scope"], "component_diagnostic")
            self.assertTrue(resumed["backbone_parameter_changed"])
            retained_first = root / "models/action4/interrupted/epoch_001_step_00000001.pt"
            with self.assertRaisesRegex(ValueError, "latest committed"):
                train(root, config, stage="D0", run_id="interrupted", resume=retained_first,
                      allow_diagnostic=True, component_smoke=True)
            changed = copy.deepcopy(config)
            changed["runtime"]["seed"] += 1
            with self.assertRaises(ValueError):
                train(root, changed, stage="D0", run_id="interrupted", resume=resume_path,
                      allow_diagnostic=True, component_smoke=True)

    def test_formal_test_freeze_rejects_component_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("model.action4_network.Action4Net", TinyControlNet):
            root = Path(temporary)
            config = fixture(root)
            train(root, config, stage="D0", run_id="component", allow_diagnostic=True, component_smoke=True, max_updates=1)
            checkpoint = root / "models/action4/component/best.pt"
            with self.assertRaisesRegex(ValueError, "completed B0/C1"):
                freeze_test_selection(root, config, checkpoint, root / "final_selection.json")
            with self.assertRaisesRegex(ValueError, "completed B0/C1"):
                run_evaluation(root, config, checkpoint, split="test", run_id="blocked_test", selection="missing.json")
            self.assertFalse((root / "final_selection.json").exists())
            self.assertFalse((root / "outputs/action4/blocked_test").exists())

    def test_two_rank_gloo_joint_empty_roi_and_tail(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("model.action4_network.Action4Net", TinyControlNet):
            root = Path(temporary)
            config = fixture(root)
            train_path = Path(config["data"]["train_manifest"])
            records = [json.loads(row) for row in train_path.read_text().splitlines()]
            empty = copy.deepcopy(records[0])
            empty.update({"sample_id": "train_empty", "group_id": "train_empty", "background_source_id": "train_empty", "instances": []})
            records.append(empty)
            train_path.write_text("".join(json.dumps(row) + "\n" for row in records))
            train(root, config, stage="A0", run_id="a0", allow_diagnostic=True, component_smoke=True, max_updates=1)
            config["stages"]["B0"]["global_batch"] = 4
            (root / "ddp_config.json").write_text(json.dumps(config))
            with socket.socket() as endpoint:
                endpoint.bind(("127.0.0.1", 0))
                port = endpoint.getsockname()[1]
            torch.multiprocessing.spawn(_ddp_worker, args=(str(root), port), nprocs=2, join=True)
            results = [json.loads((root / f"rank_{rank}_result.json").read_text()) for rank in range(2)]
            self.assertEqual([r["global_step"] for r in results], [1, 1])
            self.assertTrue(all(r["all_rank_model_states_equal"] for r in results))
            self.assertEqual(results[0]["final_model_state_sha256"], results[1]["final_model_state_sha256"])
            state = load_checkpoint(root / "models/action4/ddp/last.pt")
            self.assertEqual(len(state["rng_by_rank"]), 2)
            self.assertTrue(state["epoch_boundary"])
            if os.environ.get("ACTION4_TEST_EVIDENCE_DIR"):
                evidence = Path(os.environ["ACTION4_TEST_EVIDENCE_DIR"]).resolve()
                project_root = Path(__file__).resolve().parents[1]
                self.assertTrue(evidence.is_relative_to(project_root))
                evidence.mkdir(parents=True, exist_ok=True)
                (evidence / "two_rank_gloo_control.json").write_text(json.dumps({
                    "scope": "tiny trainable control fixture, not ResNet/model accuracy",
                    "real_training_images": 3, "includes_zero_roi_image": True,
                    "global_batch": 4, "world_size": 2, "actual_optimizer_updates": 1,
                    "tail_padding_has_zero_weight": True, "rank_results": results}, indent=2))

    def test_joint_train_and_weights_only_continuation(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("model.action4_network.Action4Net", TinyControlNet):
            root = Path(temporary)
            config = fixture(root)
            train(root, config, stage="A0", run_id="a0", allow_diagnostic=True, component_smoke=True, max_updates=1)
            init = root / "models/action4/a0/best.pt"
            result = train(root, config, stage="B0", run_id="b0", init_weights=init,
                           allow_diagnostic=True, component_smoke=True, max_updates=1)
            state = load_checkpoint(root / "models/action4/b0/last.pt")
            self.assertEqual(state["initialization"]["mode"], "weights_only_new_optimizer_schedule")
            self.assertEqual(state["global_step"], 1)
            self.assertTrue((root / "models/action4/b0/joint_init.pt").is_file())
            report = json.loads((root / "outputs/action4/b0/epoch_001_step_00000001/eval_metrics.json").read_text())
            self.assertEqual(report["scope"], "full_rgb_pipeline")
            self.assertFalse(report["reference_boxes_enter_inference"])
            tuning = tune_decode(root, config, root / "models/action4/b0/best.pt", {"score_threshold": [.98, .99]},
                                  run_id="threshold_fixture", allow_diagnostic=True)
            self.assertEqual(len(tuning["trials"]), 2)
            self.assertFalse(tuning["test_read_or_scored"])
            source_row = json.loads(Path(config["data"]["train_manifest"]).read_text().splitlines()[0])
            source_row.update({"sample_id": "s1_fixture", "group_id": "s1_fixture", "background_source_id": "s1_fixture"})
            s1_manifest = root / "s1_fixture.jsonl"
            s1_manifest.write_text(json.dumps(source_row) + "\n")
            config["data"]["s1_manifest"] = str(s1_manifest)
            config["stages"]["C1"].update({"replay_s0": 2, "replay_s1": 1, "predicted_roi_after_epoch": 0})
            continued = train(root, config, stage="C1", run_id="c1", init_weights=root / "models/action4/b0/best.pt",
                              allow_diagnostic=True, component_smoke=True, max_updates=2)
            self.assertEqual(continued["global_step"], 2)
            self.assertTrue((root / "outputs/action4/c1/initial_eval/eval_metrics.json").is_file())
            self.assertTrue((root / "outputs/action4/c1/epoch_001_step_00000002/c1_comparison.json").is_file())

    def test_red_drawing_clips_without_mutating_predictions(self):
        gt = cuboid(-5, 10)
        before = copy.deepcopy(gt)
        image = draw_cuboids(Image.new("RGB", (64, 48), "white"), [gt], line_width=2)
        self.assertEqual(image.getpixel((5, 10)), (255, 0, 0))
        self.assertEqual(gt, before)
        self.assertIsNone(clip_line((-20, -20), (-10, -10), 64, 48))

    def test_feedback_candidates_are_not_labels_or_confirmed_failures(self):
        rows = [{"frame_id": i, "pts": i * 400, "duration_pts": 400, "time_base": "1/11988",
                 "instances": [cuboid()] if i else []} for i in range(3)]
        candidates = candidate_events(rows)
        self.assertEqual(candidates[0]["status"], "pending_review")
        self.assertEqual(reviewed_requests(candidates, rows)["allocated_quota"], 0)
        confirmed = {**candidates[0], "status": "confirmed_failure", "category": "wrist_or_cable_occlusion",
                     "reviewer": "unit_fixture_reviewer", "evidence": "artificial review fixture", "synthesis": {
                         "scene_family": "self_forearm_occlusion", "view": {"wrist_delta_xyz_deg": [0, 2, 0]}, "scale": 1.,
                         "occluder": "scene_default", "occluder_location": "linked_geometry", "allowed_perturbations": {
                             "rotation_jitter_deg": 3, "position_jitter_m": [.002, .002, .002],
                             "light_intensity_gain": 1., "exposure_delta_ev": 0.},
                         "reference_frame_ids": [1], "quota": 200}}
        request = reviewed_requests([confirmed], rows)["requests"][0]
        self.assertFalse(request["prediction_coordinates_used_as_training_labels"])
        self.assertNotIn("corners_px", request)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is a required external video dependency")
    def test_actual_model_video_entry_preserves_exact_pts(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("model.action4_network.Action4Net", TinyControlNet):
            root = Path(temporary)
            config = fixture(root)
            train(root, config, stage="D0", run_id="component", allow_diagnostic=True, component_smoke=True, max_updates=1)
            source = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=2997/100",
                            "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-video_track_timescale", "11988", str(source)], check=True)
            DataAdapter(root).extract(source, "cache", all_frames=True)
            checkpoint = root / "models/action4/component/best.pt"
            with self.assertRaises(ValueError):
                run_video(root, config, checkpoint, "cache/frames_manifest.json", run_id="must_reject")
            final_config = copy.deepcopy(config)
            final_config.setdefault("video", {})["annotation_stage"] = "final_inference"
            with self.assertRaisesRegex(ValueError, "Component weights"):
                run_video(root, final_config, checkpoint, "cache/frames_manifest.json", run_id="must_not_promote", allow_component=True)
            self.assertFalse((root / "data/action4_annotations/final_inference/must_not_promote").exists())
            result = run_video(root, config, checkpoint, "cache/frames_manifest.json", run_id="video_component", allow_component=True)
            self.assertEqual(result["frame_count"], 3)
            self.assertEqual(result["annotation_dir"], "outputs/action4/video_component/drawn_frames")
            self.assertEqual(len(list((root / result["annotation_dir"]).glob("frame_*.png"))), 3)
            self.assertFalse((root / "data/action4_annotations").exists())
            self.assertTrue(result["timeline_verification"]["all_pts_equal"])
            self.assertEqual(result["claim_scope"], "component_diagnostic")
            predictions = [json.loads(line) for line in (root / "outputs/action4/video_component/predictions_raw.jsonl").read_text().splitlines()]
            self.assertEqual([p["pts"] for p in predictions], [0, 400, 800])


if __name__ == "__main__":
    unittest.main()
