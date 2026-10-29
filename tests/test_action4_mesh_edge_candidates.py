import unittest
import numpy as np
from model.action4_mesh_edge_candidates import MeshEdgePolicy,candidate_curves,ray_visible

class MeshEdges(unittest.TestCase):
 def setUp(self):
  self.v=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)*.01
  self.f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
  self.cfg=MeshEdgePolicy(spacing_m=.002)
 def test_current_view_and_no_virtual_corner_substitution(self):
  r=candidate_curves(self.v,self.f,np.arange(12),np.eye(3),[0,0,.1],policy=self.cfg)
  self.assertGreater(len(r["curves_m"]),0);self.assertFalse(r["observed"])
  pts=np.concatenate(r["curves_m"]);self.assertTrue((pts[:,2]<=-.009999).all())
  self.assertTrue(all(tuple(x) in set(tuple(sorted(e)) for f in self.f for e in zip(f,np.roll(f,-1))) for x in r["curve_vertex_ids"]))
 def test_global_occluder_is_not_bridged(self):
  v=np.vstack([self.v,[[-.1,-.1,-.04],[.1,-.1,-.04],[.1,.1,-.04],[-.1,.1,-.04]]])
  f=np.vstack([self.f,[[8,9,10],[8,10,11]]])
  r=candidate_curves(v,f,np.arange(12),np.eye(3),[0,0,.1],policy=self.cfg)
  self.assertEqual(r["curves_m"],[])
 def test_pose_and_component_validation(self):
  with self.assertRaises(ValueError):candidate_curves(self.v,self.f,[0,0],np.eye(3),[0,0,.1],policy=self.cfg)
  with self.assertRaises(ValueError):candidate_curves(self.v,self.f,np.arange(12),np.eye(3),[0,np.nan,.1],policy=self.cfg)
