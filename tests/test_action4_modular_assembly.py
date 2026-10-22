import unittest
import numpy as np
from data import action4_modular_assembly as m
class AssemblyMathTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c,_=m.load_config('configs/data/action4_modular_assembly_probe_v001.json')
  cls.A,cls.C,cls.B,cls.W,cls.K,cls.candidate=m.frames(cls.c)
 def test_shared_world_frame_preserves_C_BA(self):
  np.testing.assert_allclose(self.W@self.B@self.A,self.W@self.C,atol=1e-14)
  x=np.array([[.01,.02,-.01],[-.02,.01,.01]])
  direct=m.project(m.transform(self.C,x),self.K)
  via=m.project(m.transform(np.linalg.inv(self.W),m.transform(self.W@self.B@self.A,x)),self.K)
  np.testing.assert_allclose(direct,via,atol=1e-10)
 def test_different_world_frame_breaks_relation(self):
  wrong=self.W.copy();wrong[0,3]+=.01
  self.assertGreater(np.max(abs(wrong@self.B@self.A-self.W@self.C)),.009)
 def test_scale_mirror_nonfinite_rejected(self):
  for matrix in [np.diag([1.1,1,1,1]),np.diag([-1,1,1,1]),np.full((4,4),np.nan),np.zeros((4,4))]:
   with self.assertRaises(ValueError):m.rigid(matrix)
 def test_zero_or_behind_depth_rejected(self):
  for z in [0,-.1,float('nan')]:
   with self.assertRaises(ValueError):m.project(np.array([[0,0,z]]),self.K)
 def test_failed_candidate_not_relabelled(self):
  self.assertFalse(self.candidate['production_accepted'])
  self.assertFalse(self.candidate['all_original_observation_thresholds_pass'])
  self.assertEqual(self.c['status'],m.STATUS)
 def test_old_source_and_candidate_A_are_distinct(self):
  source=m.read(self.c['sources']['camera_source_contract']['path'])['original_invariants']['fixed_A']['right']
  self.assertGreater(np.max(abs(self.A-source)),.1)
if __name__=='__main__':unittest.main(verbosity=2)
