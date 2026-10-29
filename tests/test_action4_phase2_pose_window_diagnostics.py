import unittest
import numpy as np
from model.action4_phase2_tracking_contracts import GateScale, WindowIdentity
from model.action4_phase2_pose_window_diagnostics import (
    pose_windows_with_commits, split_pose_segments, so3_log,
    same_frame_window_disagreement, temporal_kinematics, material_observability)


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def neighbors():
    return dict(window_a=WindowIdentity("t", 0, "w0", 0, 15), window_b=WindowIdentity("t", 0, "w1", 1, 16))


class PoseWindowTests(unittest.TestCase):
    def test_exact_windows_and_unique_commits_all_tail_lengths(self):
        for length in (0, 1, 2, 15, 16, 17, 21, 27, 101):
            windows = pose_windows_with_commits(7, 7 + length)
            committed = [f for w in windows for f in range(*w["commit"])]
            self.assertEqual(committed, list(range(7, 7 + length)))
            for w in windows:
                self.assertLessEqual(w["window"][1] - w["window"][0], 15)
                self.assertLessEqual(w["window"][0], w["commit"][0])
                self.assertGreaterEqual(w["window"][1], w["commit"][1])
        self.assertEqual(pose_windows_with_commits(0, 17), [{"window": (0, 15), "commit": (0, 10)}, {"window": (5, 17), "commit": (10, 17)}])
        self.assertEqual([x["commit"] for x in pose_windows_with_commits(0, 21)], [(0, 10), (10, 15), (15, 21)])

    def test_split_lost_identity_gap_and_uneven_pts(self):
        result = split_pose_segments(list(range(8)), [0, .03, .065, .1, .13, .3, .33, .36],
                                     ["a", "a", "a", "a", "b", "b", "b", "b"],
                                     ["observed_pose"] * 2 + ["lost"] + ["pose_only"] * 5, .033)
        self.assertEqual(result, [(0, 2), (3, 4), (4, 5), (5, 8)])
        with self.assertRaises(ValueError): split_pose_segments([1, 2], [1, 1], ["a"] * 2, ["pose_only"] * 2, .033)
        self.assertEqual(split_pose_segments([], [], [], [], .033), [])

    def test_so3_zero_fastturn_and_pi(self):
        for angle in (0., 1e-8, .8, np.pi - 1e-7, np.pi):
            self.assertAlmostEqual(np.linalg.norm(so3_log(rz(angle))), angle, places=6)
        with self.assertRaises(ValueError): so3_log(np.ones((3, 3)))
        with self.assertRaises(ValueError): so3_log(np.diag([1.000001, 1., 1.]))

    def disagreement(self, angle=0, shift=0, frame=10):
        ids = [("t", "a", 0), ("t", "b", 0), ("t", "h", 0)]
        return same_frame_window_disagreement(np.eye(3), [0, 0, 1], rz(angle), [shift, 0, 1],
            frame_id_a=frame, frame_id_b=frame, identity_a=("t", 0), identity_b=("t", 0),
            cad_diagonal_m=1, gate_scale=GateScale(400, "full", 0, True, "frozen"),
            material_ids_a=ids, xyz_a=[[0, 0, 0], [.1, 0, 0], [999, 999, 0]],
            material_ids_b=ids, xyz_b=[[0, 0, 0], [.1, 0, 0], [-999, -999, 0]],
            heldout_ids=[ids[-1]], camera_matrix_a=np.diag([100, 100, 1]), camera_matrix_b=np.diag([100, 100, 1]),
            fixed_material_ids=ids[:2], fixed_material_manifest_id="fixed-F-v1", **neighbors())

    def test_same_frame_diagnostic_threshold_and_heldout_projection_exclusion(self):
        r = self.disagreement()
        self.assertEqual(r["projection"]["count"], 2); self.assertEqual(r["projection"]["median_px"], 0)
        self.assertFalse(r["diagnostic_trigger"])
        r = self.disagreement(angle=np.deg2rad(6), shift=.06)
        self.assertTrue(r["diagnostic_trigger"])
        self.assertIn("rotation_gt_5deg", r["trigger_reasons"])
        self.assertIn("translation_gt_0.05_cad_diagonal", r["trigger_reasons"])
        self.assertEqual(r["precision_acceptance"], "not_measured")
        self.assertEqual(r["adjacent_motion"], "not_measured")

    def test_no_common_material_and_same_frame_enforcement(self):
        k = dict(frame_id_a=1, frame_id_b=1, identity_a=("t", 0), identity_b=("t", 0), cad_diagonal_m=1,
                 gate_scale=GateScale(400, "full", 0, True, "frozen"), **neighbors())
        r = same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [0, 0, 1], **k)
        self.assertEqual(r["projection"]["status"], "not_measured")
        k["frame_id_b"] = 2
        with self.assertRaises(ValueError): same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [0, 0, 1], **k)

    def test_projection_requires_exact_material_xyz_and_camera(self):
        k = dict(frame_id_a=1, frame_id_b=1, identity_a=("t", 0), identity_b=("t", 0), cad_diagonal_m=1,
                 gate_scale=GateScale(400, "full", 0, True, "frozen"), material_ids_a=[("t", "a", 0)],
                 material_ids_b=[("t", "a", 0)], xyz_a=[[0, 0, 0]], xyz_b=[[0, 0, 0]],
                 camera_matrix_a=np.diag([100, 100, 1]), camera_matrix_b=np.diag([100, 100, 1]),
                 fixed_material_ids=[("t", "a", 0)], fixed_material_manifest_id="fixed-F-v1", **neighbors())
        r = same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [.1, 0, 1], **k)
        self.assertAlmostEqual(r["projection"]["median_px"], 10)
        k["xyz_b"] = [[.01, 0, 0]]
        with self.assertRaises(ValueError): same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [.1, 0, 1], **k)
        k["xyz_b"] = [[0, 0, 0]]; k["camera_matrix_b"] = np.diag([101, 100, 1])
        with self.assertRaises(ValueError): same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [.1, 0, 1], **k)
        k["camera_matrix_b"] = k["camera_matrix_a"]; k["xyz_b"] = [0, 0, 0]
        with self.assertRaises(ValueError): same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [.1, 0, 1], **k)

    def test_missing_fixed_material_does_not_shrink_projection_domain(self):
        fixed = [("t", "easy", 0), ("t", "difficult", 3)]
        k = dict(frame_id_a=1, frame_id_b=1, identity_a=("t", 0), identity_b=("t", 0), cad_diagonal_m=1,
                 gate_scale=GateScale(400, "full", 0, True, "frozen"), material_ids_a=fixed,
                 material_ids_b=fixed[:1], xyz_a=[[0, 0, 0], [.1, 0, 0]], xyz_b=[[0, 0, 0]],
                 fixed_material_ids=fixed, fixed_material_manifest_id="preregistered-F-v2",
                 camera_matrix_a=np.diag([100, 100, 1]), camera_matrix_b=np.diag([100, 100, 1]), **neighbors())
        r = same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [0, 0, 1], **k)
        self.assertEqual(r["projection"]["status"], "not_measured")
        self.assertEqual(r["projection"]["reason"], "fixed_material_members_missing")
        self.assertEqual(r["projection"]["missing_ids_b"], [fixed[1]])
        self.assertEqual(r["projection"]["required_count"], 2)
        self.assertIsNone(r["projection"]["median_px"])
        k["fixed_material_ids"] = None; k["fixed_material_manifest_id"] = None
        r = same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [0, 0, 1], **k)
        self.assertEqual(r["projection"]["reason"], "fixed_material_set_absent")
        k["fixed_material_ids"] = fixed; k["fixed_material_manifest_id"] = "preregistered-F-v2"
        k["heldout_ids"] = [fixed[1]]
        with self.assertRaises(ValueError): same_frame_window_disagreement(np.eye(3), [0, 0, 1], np.eye(3), [0, 0, 1], **k)

    def test_constant_velocity_uneven_pts_uses_velocity_midpoints(self):
        ts = np.array([0, .1, .25, .6])
        poses = np.array([rz(2 * x) for x in ts]); trans = np.column_stack([3 * ts, ts * 0, ts * 0])
        result = temporal_kinematics(poses, trans, ts)
        for velocity in result["velocities"]:
            np.testing.assert_allclose(velocity["linear_m_s"], [3, 0, 0], atol=1e-12)
            np.testing.assert_allclose(velocity["angular_rad_s"], [0, 0, 2], atol=1e-12)
        for acc in result["accelerations"]:
            np.testing.assert_allclose(acc["linear_m_s2"], 0, atol=1e-12)
            np.testing.assert_allclose(acc["angular_rad_s2"], 0, atol=1e-12)

    def test_known_quadratic_fixture_acceleration(self):
        ts = np.array([0, .1, .25, .4]); trans = np.column_stack([ts ** 2, ts * 0, ts * 0])
        result = temporal_kinematics(np.array([rz(x * x) for x in ts]), trans, ts)
        for acc in result["accelerations"]:
            np.testing.assert_allclose(acc["linear_m_s2"], [2, 0, 0], atol=1e-12)
            np.testing.assert_allclose(acc["angular_rad_s2"], [0, 0, 2], atol=1e-12)

    def test_transient_bad_observation_and_commit_seam_are_diagnostics(self):
        # Fixture candidates only: no model or smoothing is being simulated.
        windows = pose_windows_with_commits(0, 21)
        self.assertEqual(windows[0]["commit"][1], 10)
        for frame in (8, 10):
            self.assertTrue(self.disagreement(shift=.1, frame=frame)["diagnostic_trigger"])
        # A fast actual turn can agree across windows; speed is not disagreement.
        r = same_frame_window_disagreement(rz(1.), [0, 0, 1], rz(1.), [0, 0, 1],
            frame_id_a=10, frame_id_b=10, identity_a=("t", 0), identity_b=("t", 0), cad_diagonal_m=1,
            gate_scale=GateScale(400, "full", 0, True, "frozen"), **neighbors())
        self.assertFalse(r["diagnostic_trigger"])

    def test_temporal_short_invalid_time_and_gap_explicit_skips(self):
        for n in (1, 2):
            result = temporal_kinematics(np.repeat(np.eye(3)[None], n, axis=0), np.zeros((n, 3)), np.arange(n, dtype=float))
            self.assertEqual(result["acceleration_status"], "insufficient_frames")
        result = temporal_kinematics(np.repeat(np.eye(3)[None], 3, axis=0), np.zeros((3, 3)), [0, 0, 2])
        self.assertEqual(result["velocities"][0]["reason"], "invalid_time")
        result = temporal_kinematics(np.repeat(np.eye(3)[None], 3, axis=0), np.zeros((3, 3)), [0, 1, 2], gap_after=np.array([False, True]))
        self.assertEqual(result["accelerations"][0]["status"], "skipped")

    def test_material_only_condition_scaling_and_rank(self):
        scale = GateScale(400, "full", 0, True, "frozen")
        j = np.diag(2 / np.array([np.deg2rad(5)] * 3 + [.05] * 3))
        r = material_observability(j, cad_diagonal_m=1, gate_scale=scale)
        np.testing.assert_allclose(r["h_obs"], np.eye(6)); self.assertEqual(r["rank"], 6)
        self.assertAlmostEqual(r["h_obs_condition"], 1)
        r = material_observability(j[:5], cad_diagonal_m=1, gate_scale=scale)
        self.assertFalse(r["well_conditioned"]); self.assertIsNone(r["h_obs_condition"])


if __name__ == "__main__": unittest.main()
