"""Regression checks for project-only reads; all fixtures stay in project TMPDIR."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.action4_adapter import DataAdapter
from data.action4_dataset import Action4Dataset
from data.action4_paths import PROJECT_ROOT, checked_root, contained_path
from model.training.action4_runtime import input_path, read_config, output_path
from data import action4_render, action4_synthesis


class PathBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        assert self.base.is_relative_to(PROJECT_ROOT)
        self.root = self.base / "root"
        self.root.mkdir()
        self.outside = self.base / "outside"
        self.outside.mkdir()
        (self.outside / "secret.json").write_text('{"fixture":true}')
        (self.root / "valid.json").write_text('{"valid":true}')
        self.adapter = DataAdapter(self.root)

    def test_source_rejects_absolute_relative_and_symlink_escape(self):
        (self.root / "link").symlink_to(self.outside, target_is_directory=True)
        for value in (self.outside / "secret.json", "../outside/secret.json", "link/secret.json"):
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                self.adapter._source(value)
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                input_path(self.root, value)

    def test_normal_inputs_outputs_and_missing_input(self):
        expected = self.root / "valid.json"
        self.assertEqual(self.adapter._source("valid.json"), expected)
        self.assertEqual(input_path(self.root, expected), expected)
        self.assertEqual(output_path(self.root, "new/out.json"), self.root / "new/out.json")
        self.assertEqual(read_config(expected), {"valid": True})
        with self.assertRaises(FileNotFoundError):
            input_path(self.root, "missing.json")

    def test_configuration_escape_rejected_before_read(self):
        outside = PROJECT_ROOT / ".." / "not_authorized.json"
        with patch.object(Path, "read_text", side_effect=AssertionError("must not read")):
            with self.assertRaises(ValueError):
                read_config(outside)

    def test_root_itself_cannot_escape_repository(self):
        with self.assertRaises(ValueError):
            checked_root(PROJECT_ROOT / "..")
        with self.assertRaises(ValueError):
            DataAdapter(PROJECT_ROOT / "..")

    def test_rebound_dataset_image_is_checked_at_read(self):
        from PIL import Image
        img = self.root / "image.png"
        Image.new("RGB", (16,16)).save(img)
        external = self.outside / "image.png"
        Image.new("RGB", (16,16)).save(external)
        row = {"schema_version":"action4.v1", "sample_id":"fixture", "image":"image.png",
               "width":16,"height":16,"group_id":"fixture","background_source_id":"fixture",
               "scene_family":"unit_fixture","instances":[],"status":"unit_test_fixture"}
        (self.root / "samples.jsonl").write_text(json.dumps(row)+"\n")
        dataset = Action4Dataset(self.root, "samples.jsonl", allow_diagnostic=True)
        img.unlink()
        img.symlink_to(external)
        with self.assertRaises(ValueError):
            dataset[0]

    def test_renderer_reads_reject_root_escape(self):
        outside = PROJECT_ROOT / ".." / "not_authorized.bin"
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must not read")):
            with self.assertRaises(ValueError):
                action4_render._read_stl(outside)
            with self.assertRaises(ValueError):
                action4_synthesis._hash(outside)
        with patch.object(Path, "read_text", side_effect=AssertionError("must not read")):
            with self.assertRaises(ValueError):
                action4_synthesis._read(outside)
