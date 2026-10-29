import unittest
from eval.action4_native_clip import keyframe_state
class TelemetryTest(unittest.TestCase):
 def test_old_binary_is_explicitly_unavailable(self):
  self.assertEqual(keyframe_state([dict(track="a",texture=object())]),
   [dict(track_id="a",available=False,age=None,point_counts=None)])
 def test_read_only_getters(self):
  class Texture:
   def audit_keyframe_age(self): return 7
   def audit_keyframe_point_counts(self): return [43,0]
  tex=Texture()
  a=keyframe_state([dict(track="a",texture=tex)])
  self.assertEqual(a,[dict(track_id="a",available=True,age=7,point_counts=[43,0])])
  a[0]["point_counts"].append(99)
  self.assertEqual(tex.audit_keyframe_point_counts(),[43,0])
