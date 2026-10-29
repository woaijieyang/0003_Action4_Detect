import unittest
import numpy as np
def rz(a):
 return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
from model.action4_native_rotation_history import NativeRotationHistory
class HistoryTests(unittest.TestCase):
 def add(self,h,f,angle=0,matches=20):
  return h.observe("a",f,f,[1,10],rz(angle),matches=matches,current_support=True,native_ok=True)
 def test_variable_time_constant_velocity(self):
  h=NativeRotationHistory("a")
  self.add(h,0,0);self.add(h,1,.1);self.add(h,3,.3)
  r=h.predict("a",4,4,[1,10])
  np.testing.assert_allclose(r["rotation"],rz(.4),atol=1e-12)
 def test_bounded_history(self):
  h=NativeRotationHistory("a")
  for f in range(8):self.add(h,f,.01*f)
  self.assertEqual(h.predict("a",8,8,[1,10])["source_frames"],[3,4,5,6,7])
 def test_unsupported_frames_do_not_refresh_history(self):
  h=NativeRotationHistory("a");self.add(h,0)
  for f in range(1,7):self.assertFalse(self.add(h,f,matches=0)["accepted"])
  self.assertFalse(h.predict("a",7,7,[1,10])["available"])
 def test_identity_and_duplicate_rejected(self):
  h=NativeRotationHistory("a");self.add(h,0)
  with self.assertRaises(ValueError):h.predict("b",1,1,[1,10])
  with self.assertRaises(ValueError):self.add(h,0)
 def test_fast_rotation_not_clamped_into_fake_measurement(self):
  h=NativeRotationHistory("a");self.add(h,0);self.add(h,1,1.2)
  self.assertEqual(h.predict("a",2,2,[1,10])["reason"],"fast_rotation_unmodeled")

 def test_missing_support_still_records_time_domain(self):
  h=NativeRotationHistory("a");self.add(h,1,matches=0)
  with self.assertRaises(ValueError):h.predict("a",2,0,[1,10])
 def test_nonfinite_match_count_not_support(self):
  h=NativeRotationHistory("a")
  with self.assertRaises(ValueError):self.add(h,0,matches=float("nan"))
