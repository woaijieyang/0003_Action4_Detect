import unittest
import numpy as np
from eval.action4_face_texture_probe import match_features,features
class TextureProbeTests(unittest.TestCase):
 def test_missing_descriptors_retained_as_no_matches(self):
  self.assertEqual(match_features(None,np.zeros((3,32),np.uint8)),[])
 def test_ambiguous_identical_descriptors_fail_ratio(self):
  d=np.zeros((4,32),np.uint8);self.assertEqual(match_features(d,d),[])
 def test_exact_distinct_bidirectional_matches(self):
  d=np.random.default_rng(17).integers(0,256,(20,32),dtype=np.uint8)
  m=match_features(d,d);self.assertEqual(len(m),20)
  self.assertTrue(all(x.queryIdx==x.trainIdx for x in m))
 def test_keypoints_stay_in_current_instance_and_box(self):
  rgb=np.random.default_rng(19).integers(0,256,(160,160,3),dtype=np.uint8)
  mask=np.zeros((160,160),bool);mask[20:140,20:140]=True
  f=features(rgb,mask,[50,50,110,110]);self.assertGreater(len(f["xy"]),0)
  self.assertTrue(((f["xy"]>=54)&(f["xy"]<106)).all())
  self.assertIsNone(features(rgb,np.zeros_like(mask),[0,0,160,160]))
