import unittest
from model.action4_sam_tracking_scores import tracking_scores
class ScoreTests(unittest.TestCase):
 def test_raw_logit_is_not_probability(self):
  state={"tracker_metadata":{"obj_id_to_score":{0:1.0},"obj_id_to_sam2_score_frame_wise":{3:{0:-2.0,1:2.0}}}}
  r=tracking_scores(state,3)
  self.assertEqual(r["objects"]["0"]["object_logit_raw"],-2.0)
  self.assertAlmostEqual(r["objects"]["0"]["sigmoid_once"],.119202922,places=8)
  self.assertFalse(r["objects"]["0"]["is_DJI_identity_probability"])
 def test_missing_score_remains_unknown(self):
  self.assertFalse(tracking_scores({},3)["available"])
 def test_nonfinite_not_clamped(self):
  with self.assertRaises(ValueError):tracking_scores({"tracker_metadata":{"obj_id_to_sam2_score_frame_wise":{3:{0:float("nan")}}}},3)
