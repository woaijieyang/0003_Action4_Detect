import unittest
import numpy as np
from model.action4_face_motion_evidence import FaceMotionConfig,border_support
class ComponentExclusionTests(unittest.TestCase):
 def test_disconnected_island_does_not_fill_static_gap(self):
  body=np.ones((120,160),bool);dynamic=np.zeros_like(body)
  dynamic[55:100,25:135]=True;dynamic[10:14,70:74]=True
  _,old=border_support(body,dynamic,FaceMotionConfig())
  _,new=border_support(body,dynamic,FaceMotionConfig(dynamic_hull_mode="per_component"))
  self.assertTrue(old[35,72]);self.assertFalse(new[35,72])
  self.assertTrue(new[dynamic].all());self.assertTrue((new<=old).all())
 def test_single_component_exact_legacy(self):
  body=np.ones((120,160),bool);dynamic=np.zeros_like(body);dynamic[30:90,20:130]=True;dynamic[45:60,45:60]=False
  a=border_support(body,dynamic,FaceMotionConfig());b=border_support(body,dynamic,FaceMotionConfig(dynamic_hull_mode="per_component"))
  for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
 def test_all_islands_preserved_not_area_filtered(self):
  body=np.ones((120,160),bool);dynamic=np.zeros_like(body)
  for y,x in [(5,5),(20,100),(100,150)]:dynamic[y,x]=True
  _,mask=border_support(body,dynamic,FaceMotionConfig(dynamic_hull_mode="per_component"))
  self.assertTrue(mask[dynamic].all())
 def test_unknown_policy_rejected(self):
  with self.assertRaises(ValueError):FaceMotionConfig(dynamic_hull_mode="drop_small_masks")
