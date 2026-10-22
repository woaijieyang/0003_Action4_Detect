"""Real FFmpeg and negative-path checks for the existing Action4 encoder.

All fixtures/evidence stay in outputs/action4_video_io_validation_v001.
These are seven-frame synthetic interface fixtures, not model/GT accuracy evidence.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
from data.action4_adapter import DataAdapter

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class VideoIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path.cwd()
        cls.base = cls.project / "outputs/action4_video_io_validation_v001"
        cls.base.mkdir(parents=True, exist_ok=True)
        cls.fixture_root = Path(tempfile.mkdtemp(prefix="tests-", dir=cls.base))
        cls.source = cls.fixture_root / "source.mp4"
        inputs = cls.fixture_root / "source_inputs"
        inputs.mkdir()
        for i in range(7):
            Image.new("RGB", (64, 48), (20 + i * 20, 80, 120)).save(inputs / f"frame_{i:08d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-threads", "2",
                        "-framerate", "2997/100", "-i", str(inputs / "frame_%08d.png"),
                        "-frames:v", "7", "-c:v", "libx264", "-threads", "2",
                        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                        "-video_track_timescale", "11988", str(cls.source)], check=True)
        cls.adapter = DataAdapter(cls.fixture_root)
        cls.manifest = cls.adapter.extract(cls.source, "raw", all_frames=True)
        cls.source_hash = sha(cls.source)
        cls.original_hashes = [sha(cls.fixture_root / row["image"]) for row in cls.manifest["frames"]]

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="case-", dir=self.fixture_root))
        self.adapter = DataAdapter(self.root)
        self.frames = self.root / "drawn"
        self.frames.mkdir()
        self.manifest = copy.deepcopy(self.__class__.manifest)
        # The source metadata is provenance; actual drawn pixels intentionally differ.
        for i, row in enumerate(self.manifest["frames"]):
            with Image.open(self.fixture_root / row["image"]) as src:
                im = src.copy()
            ImageDraw.Draw(im).rectangle((4+i, 4, 40+i, 35), outline="red", width=2)
            im.save(self.frames / f"frame_{i:08d}.png")
        self.manifest_path = self.root / "frames_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.out = self.root / "outputs/check.mp4"

    def encode(self, **kwargs):
        return self.adapter.encode_video(self.manifest, self.frames, self.out, threads=2, **kwargs)

    def rejected(self, **kwargs):
        with patch("data.action4_adapter._run") as runner:
            with self.assertRaises((ValueError, FileNotFoundError, FileExistsError)):
                self.encode(**kwargs)
            runner.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_annotated_exact_timeline_identity_and_no_input_mutation(self):
        before = [sha(self.frames / f"frame_{i:08d}.png") for i in range(7)]
        self.assertTrue(all(a != b for a, b in zip(before, self.original_hashes)))
        report = self.adapter.encode_video(self.manifest_path, self.frames, self.out, fps="2997/100", threads=2)
        self.assertTrue(report["verified"])
        self.assertTrue(all(report[key] for key in ("all_pts_equal", "all_durations_equal",
                        "time_base_equal", "fps_equal", "no_audio", "input_frames_unchanged")))
        self.assertEqual(report["frame_count"], 7)
        self.assertEqual((report["width"], report["height"]), (64, 48))
        self.assertEqual(report["time_base"], "1/11988")
        self.assertEqual(report["frame_manifest_sha256"], sha(self.manifest_path))
        self.assertEqual(report["frame_manifest_hash_kind"], "file_bytes")
        self.assertEqual([row["sha256"] for row in report["input_frames"]], before)
        self.assertEqual(report["output"], str(self.out))
        self.assertTrue(Path(report["verification_report"]).is_file())
        probe = self.adapter.probe(self.out)
        self.assertEqual([f["pts"] for f in probe["frames"]], [0, 400, 800, 1200, 1600, 2000, 2400])
        self.assertEqual([f["duration_pts"] for f in probe["frames"]], [400]*7)
        self.assertEqual(probe["audio_stream_indices"], [])
        self.assertEqual([sha(self.frames / f"frame_{i:08d}.png") for i in range(7)], before)
        self.assertEqual(sha(self.source), self.source_hash)
        self.assertEqual([sha(self.fixture_root / row["image"]) for row in self.__class__.manifest["frames"]], self.original_hashes)
        evidence = {"scope": "seven_frame_synthetic_interface_test_not_model_result",
                    "source_before_sha256": self.source_hash, "source_after_sha256": sha(self.source),
                    "raw_before_sha256": self.original_hashes,
                    "raw_after_sha256": [sha(self.fixture_root / row["image"]) for row in self.__class__.manifest["frames"]],
                    "drawn_before_sha256": before,
                    "drawn_after_sha256": [sha(self.frames / f"frame_{i:08d}.png") for i in range(7)],
                    "verification_report": report["verification_report"]}
        (self.root / "outputs/input_preservation.json").write_text(json.dumps(evidence, indent=2))
        with self.assertRaises(FileExistsError):
            self.encode()
        self.assertEqual(sha(self.source), self.source_hash)

    def test_missing_extra_and_uppercase_png_rejected(self):
        p = self.frames / "frame_00000006.png"
        saved = p.read_bytes()
        p.unlink()
        self.rejected()
        p.write_bytes(saved)
        for name in ("frame_00000007.png", "frame_6.png", "frame_00000006.PNG"):
            extra = self.frames / name
            extra.write_bytes(saved)
            self.rejected()
            extra.unlink()

    def test_duplicate_and_unordered_manifest_ids_rejected(self):
        for ids in ([0,1,2,3,4,5,5], [1,0,2,3,4,5,6]):
            self.manifest = copy.deepcopy(self.__class__.manifest)
            for row, value in zip(self.manifest["frames"], ids):
                row["frame_id"] = value
            self.rejected()

    def test_corrupt_wrong_size_and_not_png_rejected(self):
        path = self.frames / "frame_00000000.png"
        path.write_bytes(path.read_bytes()[:30])
        self.rejected()
        Image.new("RGB", (62,48)).save(path)
        self.rejected()
        Image.new("RGB", (64,48)).save(path, format="JPEG")
        self.rejected()

    def test_invalid_fps_and_unsupported_timing_rejected(self):
        for fps in ("30", "0", "-2997/100", "1/0", "nan", "bad"):
            self.rejected(fps=fps)
        for kind in ("partial", "irregular", "last_duration", "audio", "zero_base", "nonunit_base"):
            self.manifest = copy.deepcopy(self.__class__.manifest)
            if kind == "partial":
                self.manifest["complete"] = False
            elif kind == "audio":
                self.manifest["source"]["audio_stream_indices"] = [1]
            elif kind == "zero_base":
                self.manifest["source"]["time_base"] = "0"
            elif kind == "nonunit_base":
                self.manifest["source"]["time_base"] = "2/11989"
            else:
                key, value = ("pts", 2001) if kind == "irregular" else ("duration_pts", 399)
                self.manifest["source"]["frames"][-1][key] = value
                self.manifest["frames"][-1][key] = value
            self.rejected()

    def test_root_escape_and_duplicate_symlink_inputs_rejected(self):
        outside = self.fixture_root / "outside_frames"
        outside.mkdir(exist_ok=True)
        (outside / "frame_00000000.png").write_bytes((self.frames / "frame_00000000.png").read_bytes())
        path = self.frames / "frame_00000000.png"
        path.unlink()
        path.symlink_to(outside / "frame_00000000.png")
        self.rejected()
        path.unlink()
        path.symlink_to(self.frames / "frame_00000001.png")
        self.rejected()
        with self.assertRaises(ValueError):
            self.adapter.encode_video(self.manifest, outside, self.out)
        alias = self.root / "escape"
        alias.symlink_to(self.fixture_root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.adapter.encode_video(self.fixture_root / "raw/frames_manifest.json", self.frames, self.out)
        # Restore valid PNG so output containment, not input validity, is exercised.
        path.unlink()
        shutil.copyfile(self.frames / "frame_00000001.png", path)
        with self.assertRaises(ValueError):
            self.adapter.encode_video(self.manifest, self.frames, alias / "escape.mp4")
        self.assertFalse((self.fixture_root / "escape.mp4").exists())

    def test_existing_sidecar_and_invalid_threads_rejected_before_encoding(self):
        self.out.parent.mkdir()
        sidecar = Path(str(self.out) + ".verification.json")
        sidecar.write_text("keep")
        self.rejected()
        self.assertEqual(sidecar.read_text(), "keep")
        sidecar.unlink()
        for value in (0, 9, True):
            with self.assertRaises(ValueError):
                self.adapter.encode_video(self.manifest, self.frames, self.out, threads=value)

    def test_input_mutation_during_encode_invalidates_report(self):
        from data.action4_adapter import _run
        def change_after_encoding(command):
            result = _run(command)
            if command[0] == self.adapter.ffmpeg:
                Image.new("RGB", (64,48), "green").save(self.frames / "frame_00000003.png")
            return result
        with patch("data.action4_adapter._run", side_effect=change_after_encoding):
            with self.assertRaisesRegex(RuntimeError, "input verification"):
                self.encode()
        report = json.loads(Path(str(self.out) + ".verification.json").read_text())
        self.assertFalse(report["verified"])
        self.assertEqual(report["changed_frame_ids"], [3])
        self.assertTrue(self.out.exists())

    def test_real_encode_forwards_threads_to_every_verification_probe(self):
        from data.action4_adapter import _run
        commands = []
        def capture(command):
            commands.append(list(command))
            return _run(command)
        with patch("data.action4_adapter._run", side_effect=capture):
            report = self.encode()
        (self.root / "outputs/actual_encode_commands.json").write_text(
            json.dumps({"threads": 2, "verified": report["verified"], "commands": commands}, indent=2))
        probes = [command for command in commands if command[0] == self.adapter.ffprobe]
        self.assertEqual(len(probes), 2)
        self.assertTrue(any("-show_frames" in command for command in probes))
        for command in probes:
            self.assertIn("-threads", command)
            self.assertEqual(command[command.index("-threads") + 1], "2")
            self.assertEqual(command[-1], str(self.out))
        self.assertEqual(report["frame_count"], 7)
        self.assertTrue(report["all_pts_equal"])
        self.assertTrue(report["all_durations_equal"])

    def test_probe_optional_threads_validation_and_legacy_default(self):
        for value in (0, 9, True, 2.5):
            with patch("data.action4_adapter._run") as runner:
                with self.assertRaises(ValueError):
                    self.adapter.probe(self.out, threads=value)
                runner.assert_not_called()
        from data.action4_adapter import _run
        commands = []
        def capture(command):
            commands.append(list(command))
            return _run(command)
        # Existing call shape remains valid and does not add a thread option.
        with patch("data.action4_adapter._run", side_effect=capture):
            result = self.__class__.adapter.probe(self.source)
        self.assertEqual(result["frame_count"], 7)
        self.assertTrue(all("-threads" not in command for command in commands))

    def test_cli_outputs_only_and_json_paths(self):
        cmd = [sys.executable, str(self.project / "scripts/action4_data.py"),
               "--root", str(self.root), "encode", "--manifest", str(self.manifest_path),
               "--frames-dir", str(self.frames), "--fps", "2997/100", "--threads", "2", "--output"]
        for output in ("data/forbidden.mp4", "../escape.mp4"):
            result = subprocess.run(cmd + [output], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("outputs/", result.stderr)
        result = subprocess.run(cmd + ["outputs/cli.mp4"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["verified"])
        self.assertTrue(Path(report["output"]).is_file())
        self.assertTrue(Path(report["verification_report"]).is_file())
        result = subprocess.run(cmd + ["outputs/cli.mp4"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        (self.root / "outputs_alias").symlink_to(self.root / "outputs", target_is_directory=True)
        # Resolved legitimate output aliases are safe; alias to data is forbidden.
        (self.root / "data").mkdir()
        (self.root / "outputs/bad_link").symlink_to(self.root / "data", target_is_directory=True)
        result = subprocess.run(cmd + ["outputs/bad_link/forbidden.mp4"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "data/forbidden.mp4").exists())

if __name__ == "__main__":
    unittest.main()
