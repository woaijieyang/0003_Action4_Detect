"""CPU fixtures only; no real-video accuracy or admission claim."""
import unittest
import numpy as np
from model.action4_mask_motion_evidence import (
    MotionThresholds, B0MotionEvidence, diagnose_mask_motion)


class MaskMotionTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = MotionThresholds(.05, .95, .95, 1.)
        self.previous = np.zeros((80, 120), dtype=bool)
        self.previous[20:40, 20:60] = True

    def run_diagnostic(self, current, **kwargs):
        return diagnose_mask_motion(self.previous, current, thresholds=self.thresholds, **kwargs)

    def test_static_occlusion_reproduces_ten_pixel_centroid_bias(self):
        current = self.previous.copy()
        current[:, 20:40] = False
        r = self.run_diagnostic(current, b0_motion=B0MotionEvidence((0, 0), True, True, True))
        self.assertEqual(r["centroid_delta_xy_px"], [10., 0.])
        self.assertEqual(r["area_ratio_current_over_previous"], .5)
        self.assertTrue(r["subset_change_suspected"])
        self.assertFalse(r["translation_shape_compatible"])
        self.assertFalse(r["b0_corroborates_delta"])
        self.assertFalse(r["direct_centroid_translation_sufficient"])

    def test_pure_translation_is_shape_evidence_only(self):
        current = np.zeros_like(self.previous)
        current[23:43, 27:67] = True
        r = self.run_diagnostic(current, b0_motion=B0MotionEvidence((7, 3), True, True, True))
        self.assertEqual(r["centroid_delta_xy_px"], [7., 3.])
        self.assertEqual(r["aligned_iou"], 1.)
        self.assertLess(r["unshifted_iou"], 1.)
        self.assertTrue(r["translation_shape_compatible"])
        self.assertTrue(r["b0_corroborates_delta"])
        self.assertEqual(r["physical_motion"], "unknown")
        self.assertFalse(r["direct_centroid_translation_sufficient"])

    def test_translation_and_occlusion_not_collapsed_into_translation(self):
        current = np.zeros_like(self.previous)
        current[23:43, 47:67] = True
        r = self.run_diagnostic(current, b0_motion=B0MotionEvidence((7, 3), True, True, True))
        self.assertEqual(r["centroid_delta_xy_px"], [17., 3.])
        self.assertFalse(r["translation_shape_compatible"])
        self.assertTrue(r["visible_support_change_suspected"])
        self.assertFalse(r["b0_corroborates_delta"])

    def test_equal_area_shape_change_does_not_pass(self):
        current = np.zeros_like(self.previous)
        current[10:50, 30:50] = True
        r = self.run_diagnostic(current)
        self.assertEqual(r["area_ratio_current_over_previous"], 1.)
        self.assertFalse(r["translation_shape_compatible"])

    def test_empty_masks_have_no_invented_delta(self):
        empty = np.zeros_like(self.previous)
        for first, second in ((empty, self.previous), (self.previous, empty), (empty, empty)):
            r = diagnose_mask_motion(first, second, thresholds=self.thresholds)
            self.assertIsNone(r["centroid_delta_xy_px"])
            self.assertFalse(r["direct_centroid_translation_sufficient"])

    def test_size_and_dtype_validation(self):
        for current in (np.zeros((40, 60), dtype=bool), self.previous.astype(int)):
            with self.assertRaises(ValueError):
                self.run_diagnostic(current)

    def test_copied_shared_masks_do_not_authenticate_identity(self):
        for current in (self.previous, self.previous.copy(), self.previous.view()):
            r = self.run_diagnostic(current)
            self.assertTrue(r["translation_shape_compatible"])
            self.assertFalse(r["external_identity_verified_by_module"])
            self.assertFalse(r["direct_centroid_translation_sufficient"])

    def test_inputs_remain_unchanged_and_outputs_do_not_alias(self):
        current = self.previous.copy()
        first, second = self.previous.copy(), current.copy()
        r = self.run_diagnostic(current)
        r["centroid_delta_xy_px"][0] = 900
        np.testing.assert_array_equal(first, self.previous)
        np.testing.assert_array_equal(second, current)

    def test_b0_provenance_is_required_for_corroboration(self):
        r = self.run_diagnostic(self.previous, b0_motion=B0MotionEvidence((0, 0), False, True, True))
        self.assertFalse(r["b0_corroborates_delta"])

    def test_thresholds_and_evidence_validation(self):
        for args in ((float("nan"), .9, .9, 1), (.1, 2, .9, 1), (.1, .9, .9, -1)):
            with self.assertRaises(ValueError):
                MotionThresholds(*args)
        with self.assertRaises(ValueError):
            B0MotionEvidence((float("inf"), 0), True, True, True)
        with self.assertRaises(TypeError):
            diagnose_mask_motion(self.previous, self.previous, thresholds=None)


if __name__ == "__main__":
    unittest.main()
