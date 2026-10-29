"""Analytic CPU distinction: max-IoU regression is not one-to-one TP."""
import unittest
from eval.action4_refined_quality_diagnostic import regression_rows,regression_stats,select_diagnostic_tie,TARGET
from eval.action4_bbox_metrics import score_boxes
class DiagnosticTests(unittest.TestCase):
 def test_duplicate_boxes_share_quality_target_but_not_TP(self):
  ref=[{"sample_id":"a","instances":[{"instance_id":"g","bbox_full_xyxy":[0,0,2,2]}]}]
  base=[{"sample_id":"a","instances":[{"grid_index":i,"score":.8,"bbox_full_xyxy":[0,0,2,2]} for i in (1,2)]}]
  side={"rows":[{"image":{"sample_id":"a"},"qualities":[{"prediction_id":f"b0:a:grid:{i}","refined_quality_score":.9} for i in (1,2)]}]}
  rows=regression_rows(base,ref,side);self.assertEqual([r[TARGET] for r in rows],[1.,1.])
  metric=score_boxes(base,ref);self.assertEqual((metric["overall"]["tp"],metric["overall"]["fp"]),(1,1))
  self.assertAlmostEqual(regression_stats(rows)["bias_q_minus_target"],-.1)
 def test_tie_prioritizes_existing_point5(self):
  rows=[{"quality_threshold":t,"overall":{"f1":1.,"recall":1.,"precision":1.}} for t in (.4,.5,.6)]
  self.assertEqual(select_diagnostic_tie(rows),.5)
 def test_empty_quality_bin_retained(self):
  self.assertEqual(regression_stats([])["candidates"],0);self.assertIsNone(regression_stats([])["MAE"])
if __name__=="__main__":unittest.main()
