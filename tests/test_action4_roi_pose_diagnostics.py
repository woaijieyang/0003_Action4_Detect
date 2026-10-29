import unittest
from eval.action4_roi_pose_diagnostics import summarize
class Diagnostics(unittest.TestCase):
 def test_error_units_and_missing_denominator(self):
  p=[{"sample_id":"s","instances":[{"prediction_id":"p","raw_network_pose":{"coordinate_frame":"object_to_capture_cv_m","finite":True,"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation_m":[.003,.004,1.012]}}]}]
  records=[{"sample_id":"s","instances":[{"instance_id":0,"canonical8_geometry":{"object_to_capture_cv_m":[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}}]}]
  report={"pose_gt_count":2,"pose_output_count":1,"per_gt_failure_reason":[{"sample_id":"s","instance_id":0,"prediction_id":"p","errors_px":[1]*8},{"sample_id":"s","instance_id":1,"prediction_id":None,"errors_px":None}]}
  result=summarize(p,records,report)
  self.assertEqual(result["full_GT_count"],2)
  self.assertEqual(result["matched_pose_with_raw_RT"],1)
  self.assertEqual(result["metrics"]["rotation_error_deg"]["median"],0)
  self.assertAlmostEqual(result["metrics"]["translation_center_error_m"]["median"],.013)
  self.assertAlmostEqual(result["metrics"]["depth_abs_error_m"]["median"],.012)
 def test_missing_pose_has_null_not_zero(self):
  r=summarize([],[],{"pose_gt_count":0,"pose_output_count":0,"per_gt_failure_reason":[]})
  self.assertIsNone(r["metrics"]["rotation_error_deg"]["median"])
  self.assertIsNone(r["corner_EPE_median_native_px"])
