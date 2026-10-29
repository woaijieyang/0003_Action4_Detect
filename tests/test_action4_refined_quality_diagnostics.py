import unittest
from eval.action4_refined_quality_diagnostics import summarize_quality,prediction_quality_targets
from data.action4_sam_native_export import canonical

class QualityDiagnosticsTests(unittest.TestCase):
    def test_bias_and_empty(self):
        r=summarize_quality([.9,.9],[.8,.6])
        self.assertAlmostEqual(r["signed_bias"],.2)
        self.assertIsNone(r["pearson"])
        self.assertIsNone(summarize_quality([],[])["MAE"])
    def test_nonfinite_rejected(self):
        for q,i in [([float("nan")],[.4]),([1.1],[.4]),([.1],[])]:
            with self.assertRaises(ValueError):summarize_quality(q,i)
    def fixture(self):
        box=dict(grid_index=1,bbox_full_xyxy=[0,0,10,10],score=.9)
        return ([dict(sample_id="a",instances=[box])],
                [dict(sample_id="a",instances=[dict(bbox_full_xyxy=[0,0,10,10],ignore=False)])],
                [dict(image=dict(sample_id="a"),qualities=[dict(grid_index=1,base_row_sha256=canonical(box),refined_quality_score=.9)])])
    def test_quality_not_assignment_and_unknown_ignore(self):
        p,r,s=self.fixture()
        self.assertEqual(prediction_quality_targets(p,r,s)[0]["max_nonignore_full_box_IoU"],1)
        r[0]["instances"][0]["ignore"]=True
        self.assertIsNone(prediction_quality_targets(p,r,s)[0]["max_nonignore_full_box_IoU"])
    def test_wrong_identity(self):
        p,r,s=self.fixture();p[0]["instances"][0]["score"]=.8
        with self.assertRaises(ValueError):prediction_quality_targets(p,r,s)
