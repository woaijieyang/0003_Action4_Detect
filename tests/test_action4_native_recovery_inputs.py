import unittest,numpy as np
from eval.action4_native_recovery_inputs import current_recovery_inputs
class RecoveryInputsTests(unittest.TestCase):
 def test_sam_only_keeps_absent_B0(self):
  x=dict(frame_id=186,pts=74400,track_id="a",current_support_status={"B0":False,"SAM":True},bbox_b0_raw=None,score=None)
  r=current_recovery_inputs(x,np.ones((3,4),bool),frame_id=186,pts=74400,track_id="a")
  self.assertIsNone(r["b0_box"]);self.assertIsNone(r["b0_score"]);self.assertIsNone(r["mask_inside_box"])
  self.assertTrue(r["current_SAM"])
 def test_missing_actual_mask_rejected(self):
  x=dict(frame_id=1,pts=400,track_id="a",current_support_status={"B0":False,"SAM":True},bbox_b0_raw=None,score=None)
  with self.assertRaises(ValueError):current_recovery_inputs(x,None,frame_id=1,pts=400,track_id="a")
 def test_no_fabricated_B0(self):
  x=dict(frame_id=1,pts=400,track_id="a",current_support_status={"B0":False,"SAM":True},bbox_b0_raw=[0,0,3,4],score=.9)
  with self.assertRaises(ValueError):current_recovery_inputs(x,np.ones((3,4),bool),frame_id=1,pts=400,track_id="a")
