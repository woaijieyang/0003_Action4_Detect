"""CPU geometry, data leakage, path containment and real FFmpeg timing checks.

Run from the repository root with ``PYTHONPATH=src python -m unittest discover
-s tests -p 'test_action4_adapter.py' -v``. Fixtures are generated under .cache;
no source video, network, renderer or model is required.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib
from pathlib import Path

import numpy as np

from data import DataAdapter


def write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]):
    def chunk(name, payload):
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    path.write_bytes(payload + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path.cwd() / ".cache" / "action4_adapter_tests"
        base.mkdir(parents=True, exist_ok=True)
        cls.run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=base))

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="case-", dir=self.run_dir))
        self.adapter = DataAdapter(self.root)

    def test_pixel_centres_letterbox_crop_and_inverse(self):
        full = self.adapter.crop_affine((-0.5, -0.5, 99.5, 79.5), (100, 80))
        np.testing.assert_allclose(full, np.eye(3), atol=1e-12)
        matrix, config = self.adapter.letterbox_affine((3248, 2464), (1024, 768))
        self.assertEqual(config["resize_size"], [1012, 768])
        points = np.array([[0, 0], [3247, 2463], [-200, 2700], [713.125, 654.25]])
        crop = self.adapter.crop_affine((-300.5, 800.5, 499.5, 1600.5), (256, 256))
        composed = crop @ matrix
        recovered = self.adapter.transform_points(self.adapter.transform_points(points, composed), composed, inverse=True)
        np.testing.assert_allclose(points, recovered, atol=1e-9)
        with self.assertRaises(ValueError):
            self.adapter.transform_points(points, np.zeros((3, 3)))

    def test_fixed_corner_order_depth_and_rigid_projection(self):
        corners = self.adapter.fixed_corners((-1, -2, -3), (1, 2, 3))
        np.testing.assert_equal(corners[[0, 1, 2, 3]], [[-1, -2, -3], [1, -2, -3], [1, 2, -3], [-1, 2, -3]])
        transform = np.eye(4)
        transform[2, 3] = 13
        k = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        pixels, depth = self.adapter.project_corners(corners, k, transform)
        np.testing.assert_allclose(pixels[0], [40, 20])
        np.testing.assert_allclose(depth[:4], 10)
        transform[2, 3] = 0
        with self.assertRaises(ValueError):
            self.adapter.project_corners(corners, k, transform)
        transform[2, 3] = 13
        transform[0, 0] = 2
        with self.assertRaises(ValueError):
            self.adapter.project_corners(corners, k, transform)

    def test_output_escape_symlink_and_existing_source_preserved(self):
        source = self.root / "original.json"
        source.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.adapter.save_json({"changed": True}, source)
        with self.assertRaises(ValueError):
            self.adapter.save_json({}, "../escaped.json")
        outside = Path(tempfile.mkdtemp(prefix="outside-", dir=self.run_dir))
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.adapter.save_json({}, "linked/escaped.json")
        self.assertEqual(source.read_text(), "original")
        self.assertFalse((outside / "escaped.json").exists())

    def test_transitive_group_background_closure_is_deterministic(self):
        rows = [
            {"sample_id": "a", "group_id": "g1", "background_source_id": "bg1"},
            {"sample_id": "b", "group_id": "g1", "background_source_id": "bg2"},
            {"sample_id": "c", "group_id": "g2", "background_source_id": "bg2"},
        ] + [{"sample_id": str(i), "group_id": f"g{i}", "background_source_id": f"bg{i}"} for i in range(3, 12)]
        split = self.adapter.grouped_split(rows, seed=51)
        self.assertEqual(split, self.adapter.grouped_split(list(reversed(rows)), seed=51))
        assigned = {row["sample_id"]: row["split"] for row in split}
        self.assertEqual(len({assigned[key] for key in ("a", "b", "c")}), 1)
        self.assertTrue(all(self.adapter.validate_splits(split).values()))
        broken = copy.deepcopy(split)
        next(row for row in broken if row["sample_id"] == "a")["split"] = "test" if assigned["a"] != "test" else "train"
        with self.assertRaises(ValueError):
            self.adapter.validate_splits(broken)
        with self.assertRaises(ValueError):
            self.adapter.grouped_split(split)
        with self.assertRaises(ValueError):
            self.adapter.grouped_split(rows[:3])

    def test_stale_source_probe_is_rejected_before_any_output(self):
        source = self.root / "changed.mp4"
        source.write_bytes(b"bar")
        metadata = {"source_video": str(source), "source_bytes": 3,
                    "source_sha256": hashlib.sha256(b"foo").hexdigest(), "frame_count": 1}
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.adapter.extract(source, "should_not_exist", frame_ids=[0], probe=metadata)
        self.assertFalse((self.root / "should_not_exist").exists())

    def test_render_delegate_checks_exit_contract_and_complete_manifest(self):
        asset_dir = self.root / "asset"
        asset_dir.mkdir()
        (asset_dir / "mesh_v001.npz").write_bytes(b"contract fixture; renderer is mocked")
        (asset_dir / "asset_manifest.json").write_text(json.dumps({"mesh_data_sha256": hashlib.sha256(
            (asset_dir / "mesh_v001.npz").read_bytes()).hexdigest()}))

        def backend(command, *, check, stdout, stderr, text):
            self.assertEqual(command[command.index("--python-exit-code") + 1], "1")
            self.assertEqual(command[command.index("--threads") + 1], "2")
            self.assertEqual(command[command.index("--start") + 1], "0")
            self.assertEqual(command[command.index("--stop") + 1], "32")
            output = Path(command[command.index("--output-dir") + 1])
            image = output / "fixture.png"
            write_png(image, 64, 48, (20, 30, 40))
            number = 31 if output.name == "partial_pilot" else 32
            records = [{"schema_version": "action4.v1", "sample_id": f"{output.name}_{index:06d}",
                        "image": str(image.relative_to(self.root)), "width": 64, "height": 48,
                        "group_id": f"g{index}", "background_source_id": "fixture", "scene_family": "fixture",
                        "instances": []} for index in range(number)]
            (output / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
            stdout.write("mocked backend contract fixture, not a Blender rendering\n")
            return subprocess.CompletedProcess(command, 0)

        with patch("data.action4_adapter.subprocess.run", side_effect=backend):
            result = self.adapter.render_pilot(asset_dir, "complete_pilot", blender=sys.executable, count=32)
            self.assertEqual(result["sample_count"], 32)
            self.assertFalse(result["training_accepted"])
            with self.assertRaisesRegex(ValueError, "sample IDs/count"):
                self.adapter.render_pilot(asset_dir, "partial_pilot", blender=sys.executable, count=32)
        failure = json.loads((self.root / "partial_pilot/adapter_render_result.json").read_text())
        self.assertEqual(failure["status"], "failed")
        with patch("data.action4_adapter.subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(RuntimeError, "Blender exited 1"):
                self.adapter.render_pilot(asset_dir, "failed_pilot", blender=sys.executable, count=32)
        with self.assertRaisesRegex(ValueError, "requires count=32"):
            self.adapter.render_pilot(asset_dir, "unsupported_recipe", blender=sys.executable, count=2)

    def test_sample_keeps_occluded_and_out_of_frame_corners(self):
        image = self.root / "sample.png"
        write_png(image, 100, 80, (0, 0, 0))
        corners = [[-20, -10], [110, -10], [110, 90], [-20, 90]] * 2
        sample = {"schema_version": "action4.v1", "sample_id": "s", "image": "sample.png", "width": 100,
                  "height": 80, "group_id": "g", "background_source_id": "bg", "scene_family": "wrist",
                  "instances": [{"instance_id": 0, "object_id": 1, "corners_px": corners,
                                 "bbox_full_xyxy": [-20, -10, 110, 90], "corner_visible": [False] * 8,
                                 "occlusion_inframe": 0.3, "truncation": 0.2}]}
        self.adapter.validate_sample(sample)
        undefined = copy.deepcopy(sample)
        undefined["instances"][0].update(occlusion_inframe=None, occlusion_inframe_valid=False)
        self.adapter.validate_sample(undefined)
        del undefined["instances"][0]["occlusion_inframe_valid"]
        with self.assertRaises(ValueError):
            self.adapter.validate_sample(undefined)
        broken = copy.deepcopy(sample)
        broken["instances"][0]["corners_px"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.adapter.validate_sample(broken)
        image.write_bytes(image.read_bytes()[:24])
        with self.assertRaises(ValueError):
            self.adapter.validate_sample(sample)
        broken = copy.deepcopy(sample)
        broken["instances"][0]["bbox_full_xyxy"] = [0, 0, 99, 79]
        with self.assertRaises(ValueError):
            self.adapter.validate_sample(broken)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and FFprobe required")
    def test_real_ffmpeg_exact_pts_extraction_and_reassembly(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        for index in range(7):
            write_png(inputs / f"frame_{index:08d}.png", 64, 48, (index * 25, 50, 100))
        source = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-framerate", "2997/100", "-i", str(inputs / "frame_%08d.png"),
                        "-frames:v", "7", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                        "-video_track_timescale", "11988", str(source)], check=True)
        probe = self.adapter.probe(source)
        self.assertEqual(probe["frame_count"], 7)
        self.assertEqual(probe["time_base"], "1/11988")
        self.assertEqual([f["pts"] for f in probe["frames"]], list(range(0, 2800, 400)))
        selected = self.adapter.extract(source, "selected", frame_ids=[6, 0, 3], probe=probe)
        self.assertEqual([f["frame_id"] for f in selected["frames"]], [0, 3, 6])
        self.assertEqual([f["pts"] for f in selected["frames"]], [0, 1200, 2400])
        with self.assertRaises(ValueError):
            self.adapter.encode_video(selected, "selected/frames", "bad_partial.mp4")
        manifest = self.adapter.extract(source, "complete", all_frames=True, probe=probe)
        report = self.adapter.encode_video(manifest, "complete/frames", "encoded.mp4", fps="2997/100")
        self.assertTrue(report["verified"])
        self.assertTrue(report["all_pts_equal"])
        with self.assertRaises(ValueError):
            self.adapter.encode_video(manifest, "complete/frames", "bad_fps.mp4", fps="30")
        (self.root / "complete/frames/frame_00000006.png").unlink()
        with self.assertRaises(ValueError):
            self.adapter.encode_video(manifest, "complete/frames", "bad_missing.mp4")
        self.assertFalse((self.root / "bad_missing.mp4").exists())


if __name__ == "__main__":
    unittest.main()
