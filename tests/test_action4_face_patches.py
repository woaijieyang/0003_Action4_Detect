import unittest
import numpy as np
from data.action4_face_patches import PatchThresholds,build_patch_candidates
class PatchTests(unittest.TestCase):
 def run_mesh(self,v,f,views=None):
  return build_patch_candidates(np.array(v,float),np.array(f,int),views or {i:[] for i in range(6)},
   mesh_sha256="a"*64,thresholds=PatchThresholds(3,.1,.01,.2,0))
 def test_adjacent_coplanar_and_flipped_winding(self):
  v=[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[-1,-1,-1]]
  r=self.run_mesh(v,[[0,1,2],[0,3,2]],{i:([0,1] if i==5 else []) for i in range(6)})
  self.assertEqual(len(r["patches"]),1);p=r["patches"][0]
  self.assertEqual(p["winding_negative_count"],1);self.assertEqual(p["logical_face_candidates"],[5])
  self.assertFalse(p["semantic_gt"]);self.assertFalse(p["exterior_verified"])
 def test_two_depth_planes_and_unknown_retained(self):
  v=[[-1,-1,1],[1,-1,1],[1,1,1],[-1,-1,-1],[1,-1,-1],[1,1,-1]]
  r=self.run_mesh(v,[[0,1,2],[3,4,5]])
  self.assertEqual(len(r["patches"]),2)
  self.assertTrue(all(x["sampling_status"]=="unhit_unknown" for x in r["triangle_mapping"]))
 def test_nonplanar_neighbor_not_merged(self):
  v=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
  r=self.run_mesh(v,[[0,1,2],[0,3,1]])
  self.assertEqual(len(r["patches"]),2)
 def test_internal_and_opposite_halfspace_not_promoted(self):
  v=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,-1,-2],[1,1,2]]
  r=self.run_mesh(v,[[0,1,2]],{i:([0] if i in (4,5) else []) for i in range(6)})
  self.assertEqual(r["patches"][0]["logical_face_candidates"],[])
  self.assertEqual(set(r["patches"][0]["excluded_view_reasons"]),{"4","5"})
 def test_multiview_not_forced_single_and_inputs_unchanged(self):
  v=np.array([[.9,.9,1],[1,.9,1],[1,1,1],[-1,-1,-1]])
  f=np.array([[0,1,2]]);before=v.copy()
  r=self.run_mesh(v,f,{i:([0] if i in (1,3,5) else []) for i in range(6)})
  self.assertEqual(r["patches"][0]["logical_face_candidates"],[5])
  self.assertEqual(r["patches"][0]["observed_first_hit_views"],[1,3,5])
  self.assertEqual(r["patches"][0]["excluded_view_reasons"]["1"],"view_not_parallel_to_patch_plane_normal")
  np.testing.assert_array_equal(v,before)
 def test_nonnearaxis_and_degenerate_unknown(self):
  r=self.run_mesh([[0,0,0],[1,0,1],[0,1,1]],[[0,1,2],[0,0,1]])
  self.assertEqual(len(r["triangle_mapping"]),2)
  self.assertEqual(r["triangle_mapping"][0]["reason"],"non_near_axis_triangle")
  self.assertEqual(r["triangle_mapping"][1]["reason"],"degenerate_triangle")
 def test_bad_threshold_and_index(self):
  with self.assertRaises(ValueError):PatchThresholds(60,.1,.01,.2,0)
  with self.assertRaises(ValueError):
   self.run_mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],{i:([3] if i==0 else []) for i in range(6)})
 def test_view_ids_must_be_exact_ints(self):
  for badkey in [False,0.0]:
   views={i:[] for i in range(1,6)};views[badkey]=[]
   with self.assertRaises(ValueError):
    self.run_mesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],views)

if __name__=="__main__":unittest.main()
