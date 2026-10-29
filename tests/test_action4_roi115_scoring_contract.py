"""Scoring-contract regression; no inference or fabricated accuracy."""
import copy,unittest
from eval.action4_corner_scoring_inputs import _scoring_roi_expansion
from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL
class ROI115ScoringContractTests(unittest.TestCase):
 def contract(self,value):
  return {"input_contract":{"schema":CORNER_NATIVE_PROTOCOL,"roi_source":"frozen_b0_prediction","roi_expansion":value}}
 def test_registered_roi115(self):
  self.assertEqual(_scoring_roi_expansion(self.contract(1.15)),1.15)
 def test_old_native15_and_legacy225_preserved(self):
  self.assertEqual(_scoring_roi_expansion(self.contract(1.5)),1.5)
  self.assertEqual(_scoring_roi_expansion({}),2.25)
  self.assertEqual(_scoring_roi_expansion({"input_contract":{"schema":"action4.corner_legacy_live_B0_roi225.v1","roi_expansion":2.25,"roi_source":"legacy_live_B0_prediction"}}),2.25)
 def test_other_native_scales_rejected(self):
  for value in (1.0,1.1,1.2,1.25,2.25,0.,True,float("nan")):
   with self.subTest(value=value),self.assertRaises(ValueError):_scoring_roi_expansion(self.contract(value))
 def test_roi115_requires_registered_schema_and_frozen_prediction(self):
  for key,value in (("schema","unregistered"),("roi_source","GT")):
   identity=self.contract(1.15);identity["input_contract"][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):_scoring_roi_expansion(identity)
if __name__=="__main__":unittest.main()
