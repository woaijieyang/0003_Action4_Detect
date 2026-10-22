"""Actual-cache BVH candidates and conservative AABB counterexamples; no Blender."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_modular_bilateral_scene as m
class CandidateFilterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.proof=m.read('outputs/action4_bilateral_bvh_candidate_diagnosis_v001/diagnosis.json');cls.cache=np.load(m.bounded(cls.proof['sources']['actual_geometry']['path']),allow_pickle=False);cls.zone={**cls.proof['original_joint_zone'],'module_to_zone':cls.proof['world_to_original_zone']}
 def test_two_real_double_cache_pairs_proven_separated(self):
  d=self.cache;pairs=[tuple(row['pair']) for row in self.proof['pairs']];keep,proof=m.filter_separated_bvh_candidates(d['right_ball_vertices_m'],d['right_ball_triangles'],d['right_band_vertices_m'],d['right_band_triangles'],pairs,self.zone)
  self.assertEqual(keep,[]);self.assertEqual([row['pair'] for row in proof],[list(x) for x in pairs])
  for actual,expected in zip(proof,self.proof['pairs']):
   self.assertEqual(actual['frame'],'existing_joint_zone');np.testing.assert_allclose(actual['axis_gaps_m'],expected['axis_gaps_m'],rtol=0,atol=1e-15);self.assertGreater(actual['axis_gaps_m'][0],1e-12);self.assertTrue(actual['not_surface_intersection']);self.assertEqual(actual['existing_numeric_margin_m'],1e-12)
 def test_real_cache_overlapping_and_boundary_touch_triangles_retained(self):
  V=self.cache['right_ball_vertices_m'];F=self.cache['right_ball_triangles'];keep,proof=m.filter_separated_bvh_candidates(V,F,V,F,[(4,4)])
  self.assertEqual(keep,[(4,4)]);self.assertEqual(proof,[]) # Identical actual triangle has true coincident overlap.
  touching=None
  for i in range(len(F)):
   for j in range(i+1,len(F)):
    if not set(F[i])&set(F[j]):continue
    lower=np.maximum(V[F[i]].min(0),V[F[j]].min(0));upper=np.minimum(V[F[i]].max(0),V[F[j]].max(0))
    if np.all(lower<=upper) and np.any(lower==upper):touching=(i,j);break
   if touching is not None:break
  self.assertIsNotNone(touching);keep,proof=m.filter_separated_bvh_candidates(V,F,V,F,[touching]);self.assertEqual(keep,[touching]);self.assertEqual(proof,[]) # Shared actual vertex guarantees boundary contact.
 def test_touch_and_one_e_minus12_uncertainty_not_pruned(self):
  A=np.array([[-1.,0,0],[0,0,0],[0,1,0]]);F=np.array([[0,1,2]])
  for gap in [0.,.5e-12,1e-12]:
   B=np.array([[gap,0,0],[1,0,0],[gap,1,0]]);keep,proof=m.filter_separated_bvh_candidates(A,F,B,F,[(0,0)]);self.assertEqual(keep,[(0,0)]);self.assertEqual(proof,[])
  B=np.array([[2e-12,0,0],[1,0,0],[2e-12,1,0]]);keep,proof=m.filter_separated_bvh_candidates(A,F,B,F,[(0,0)]);self.assertEqual(keep,[]);self.assertEqual(len(proof),1);self.assertEqual(proof[0]['frame'],'world')
 def test_retained_overlap_outside_zone_stays_unresolved(self):
  A=np.array([[0.,0,0],[1,0,0],[0,1,0]]);F=np.array([[0,1,2]]);zone={'module_to_zone':np.eye(4).tolist(),'min_m':[2]*3,'max_m':[3]*3}
  keep,proof=m.filter_separated_bvh_candidates(A,F,A,F,[(0,0)],zone);self.assertEqual(proof,[]);self.assertEqual(keep,[(0,0)]);row=m.mount_algorithms.classify_pairs(A,F,A,F,keep,zone);self.assertEqual(row['outside_or_unresolved_pairs'],[[0,0]]);self.assertFalse(row['mechanical_acceptance'])
if __name__=='__main__':unittest.main(verbosity=2)
