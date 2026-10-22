"""Review candidates remain distinct from annotations, GT and formal results."""
import copy
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from eval.action4_feedback import candidate_events, reviewed_requests, run_feedback
from eval.action4_video import validate_video_qualification, run_video


def rows(count, present=()):
    instance = {"score": .9, "bbox_full_xyxy": [0, 0, 20, 20],
                "corners_px": [[5, 5]] * 8}
    return [{"frame_id": i, "pts": i * 400, "time_base": "1/11988",
             "instances": [copy.deepcopy(instance)] if i in present else []} for i in range(count)]


class FeedbackExtensionsTests(unittest.TestCase):
    def test_full_zero_video_and_exact_threshold(self):
        self.assertEqual(candidate_events(rows(29)), [])
        events = candidate_events(rows(30))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["start_frame_id"], events[0]["end_frame_id"]), (0, 29))
        self.assertIn("sustained_zero_detections", events[0]["triggers"])
        self.assertEqual(events[0]["status"], "pending_review")
        self.assertEqual(reviewed_requests(events, rows(30))["requests"], [])

    def test_leading_and_trailing_zero_intervals(self):
        for present, expected in ((range(30, 80), (0, 30)), (range(0, 50), (50, 79))):
            events = candidate_events(rows(80, present))
            zeros = [e for e in events if "sustained_zero_detections" in e["triggers"]]
            self.assertEqual(len(zeros), 1)
            self.assertEqual((zeros[0]["start_frame_id"], zeros[0]["end_frame_id"]), expected)

    def test_separated_zero_runs_remain_distinct(self):
        events = candidate_events(rows(130, range(30, 100)))
        zeros = [e for e in events if "sustained_zero_detections" in e["triggers"]]
        self.assertEqual(len(zeros), 2)
        self.assertEqual([(e["start_frame_id"], e["end_frame_id"]) for e in zeros], [(0, 30), (100, 129)])

    def test_low_confidence_count_and_jump_candidates_retained(self):
        data = rows(5, range(5))
        data[1]["instances"][0]["score"] = .1
        data[2]["instances"][0]["corners_px"] = [[100, 100]] * 8
        data[3]["instances"] = []
        reasons = set().union(*(set(e["triggers"]) for e in candidate_events(data)))
        self.assertTrue({"low_confidence", "instance_count_change", "ordered_corner_motion_jump"} <= reasons)

    def test_invalid_zero_thresholds(self):
        for bad in (0, -1, True, 3.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate_events(rows(2), zero_detection_frames=bad)

    def test_qualified_b0_c1_only_and_diagnostic_separation(self):
        base = {"claim_scope": "accepted_data_model_training", "stage": "B0",
                "epoch_boundary": True, "global_step": 2}
        config = {"video": {"annotation_stage": "final_inference"}}
        for stage in ("B0", "C1"):
            self.assertEqual(validate_video_qualification({**base, "stage": stage}, config), (False, "final_inference"))
        for change in ({"stage": "A0"}, {"stage": "D0"}, {"epoch_boundary": False},
                       {"global_step": 0}, {"global_step": True}, {"claim_scope": "component_diagnostic"}):
            state = {**base, **change}
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    validate_video_qualification(state, config)
                with self.assertRaises(ValueError):
                    validate_video_qualification(state, config, allow_component=True)
                self.assertEqual(validate_video_qualification(state, {}, allow_component=True), (True, "checks"))
        with self.assertRaises(ValueError):
            validate_video_qualification(base, {"video": {"annotation_stage": "formal_training"}})

    def test_reviewed_recipe_preserves_single_target_side(self):
        event = {
            "event_id": "unit_fixture", "status": "confirmed_failure",
            "start_frame_id": 0, "end_frame_id": 29,
            "category": "domain_gap", "reviewer": "unit_test",
            "evidence": "synthetic unit fixture only", "observation": "test export",
            "synthesis": {
                "scene_family": "wall_rear", "view": {"wrist_delta_xyz_deg": [0, 0, 0]},
                "scale": 1., "occluder": "scene_default", "occluder_location": "linked_geometry",
                "allowed_perturbations": {"rotation_jitter_deg": 3, "position_jitter_m": [.002] * 3,
                                          "light_intensity_gain": 1., "exposure_delta_ev": 0.},
                "reference_frame_ids": [1], "quota": 200, "target_presence": [False, True],
            }}
        request = reviewed_requests([event], rows(30))["requests"][0]
        self.assertEqual(request["recipe"]["target_presence"], [False, True])
        self.assertFalse(request["prediction_coordinates_used_as_training_labels"])

    def test_video_cannot_write_predictions_into_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"not loaded, explicit mock state")
            frame = {"frame_id": 0, "pts": 0, "duration_pts": 400}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"complete": True, "frames": [frame],
                                           "source": {"frame_count": 1, "frames": [frame]}}))
            state = {"claim_scope": "accepted_data_model_training", "stage": "B0",
                     "epoch_boundary": True, "global_step": 1}
            with mock.patch("eval.action4_video.load_checkpoint", return_value=state):
                with self.assertRaisesRegex(ValueError, "under outputs"):
                    run_video(root, {}, checkpoint, manifest, run_id="fixture", output_dir="data/video")
            self.assertFalse((root / "data").exists())

    def test_feedback_cannot_write_run_products_into_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            with self.assertRaisesRegex(ValueError, "under outputs"):
                run_feedback(root, "run", "data/feedback")
            self.assertFalse((root / "data").exists())


if __name__ == "__main__":
    unittest.main()
