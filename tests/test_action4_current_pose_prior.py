import unittest,copy
import numpy as np
from model.action4_current_pose_prior import current_pose_prior
class PosePriorTests(unittest.TestCase):
 def fixture(self):
  return dict(track_id="a",recovery_evidence=dict(frame_id=4,pts=1600,current_support_status={"B0":True,"SAM":True},
    pose_valid=True,roi_decision_id="roi"),raw_A0_pose=dict(finite=True,coordinate_frame="object_to_capture_cv_m",
    rotation=np.eye(3).tolist(),translation_m=[0,0,1]))
 def call(self,i):return current_pose_prior(i,track_id="a",frame_id=4,pts=1600,canonical8=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.02,.02]]))
 def test_current_target_change_consumed(self):
  i=self.fixture();a=self.call(i);i["raw_A0_pose"]["translation_m"][0]=.01;b=self.call(i)
  self.assertNotEqual(a["pose"],b["pose"]);self.assertTrue(b["available"]);self.assertFalse(b["ground_truth"])
 def test_unknown_not_zero_pose(self):
  i=self.fixture();i["recovery_evidence"]["current_support_status"]={"B0":False,"SAM":False}
  p=self.call(i);self.assertFalse(p["available"]);self.assertIsNone(p["pose"])
 def test_wrong_frame_and_track(self):
  i=self.fixture()
  for k,v in [("frame_id",5),("pts",2000)]:
   bad=copy.deepcopy(i);bad["recovery_evidence"][k]=v
   with self.assertRaises(ValueError):self.call(bad)
  i["track_id"]="b"
  with self.assertRaises(ValueError):self.call(i)
 def test_nonfinite_and_wrong_units(self):
  i=self.fixture();i["raw_A0_pose"]["translation_m"][0]=float("nan")
  self.assertFalse(self.call(i)["available"])
  i=self.fixture();i["raw_A0_pose"]["coordinate_frame"]="world_mm"
  with self.assertRaises(ValueError):self.call(i)

 def test_selected_pose_is_consumed_without_overwriting_raw(self):
  i=self.fixture();original=copy.deepcopy(i['raw_A0_pose']);i['selected_A0_pose']=copy.deepcopy(original)
  i['selected_A0_pose']['translation_m'][0]=.08
  cad=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.02,.02]])
  p=current_pose_prior(i,track_id='a',frame_id=4,pts=1600,canonical8=cad,pose_key='selected_A0_pose')
  self.assertEqual(p['pose'][0][3],.08);self.assertEqual(i['raw_A0_pose'],original)
  self.assertEqual(p['source'],'current_selected_A0_from_resolved_ROI')
  self.assertEqual(self.call(i)['pose'][0][3],0)
