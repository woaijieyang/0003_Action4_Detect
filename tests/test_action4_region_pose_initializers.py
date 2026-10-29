import unittest
import numpy as np
import cv2
from model.action4_region_pose_initializers import propose_initializers
class RegionInitializerTests(unittest.TestCase):
 def setUp(self):
  self.K=np.array([[300.,0,64],[0,300.,64],[0,0,1.]])
  self.v=np.array([[-.03,-.02,0],[.03,-.02,0],[.03,.02,0],[-.03,.02,0]])
  uv=(self.v+[0,0,.3])@self.K.T;uv=uv[:,:2]/uv[:,2:]
  m=np.zeros((128,128),np.uint8);cv2.fillConvexPoly(m,np.rint(uv).astype(np.int32),1)
  self.part=dict(part_id="fixture",vertices_m=self.v,outward_normal_object=[0,0,-1],mask=m.astype(bool))
  self.kw=dict(rotation_grid_rad=[[0],[0],[0]],translation_offsets_m=[-.01,0,.01],depth_factors=[1.],
   max_points_per_part=256,rotation_prior_weight=0.,translation_prior_weight=1.,top_k=2,min_facing_cos=.01)
 def test_observed_support_changes_initializer_not_final_pose(self):
  r=propose_initializers(self.K,np.eye(3),[.01,0,.3],[self.part],**self.kw)
  self.assertAlmostEqual(r["retained"][0]["t_m"][0],0.,places=5)
  self.assertFalse(r["final_pose_available"]);self.assertFalse(r["geometry_acceptance"])
  shifted=dict(self.part);shifted["mask"]=np.roll(self.part["mask"],10,axis=1)
  other=propose_initializers(self.K,np.eye(3),[.01,0,.3],[shifted],**self.kw)
  self.assertNotEqual(r["retained"][0]["t_m"],other["retained"][0]["t_m"])
 def test_no_duplicate_or_missing_evidence(self):
  with self.assertRaises(ValueError):propose_initializers(self.K,np.eye(3),[0,0,.3],[self.part,self.part],**self.kw)
  missing=dict(self.part);missing["mask"]=np.zeros_like(self.part["mask"])
  with self.assertRaises(ValueError):propose_initializers(self.K,np.eye(3),[0,0,.3],[missing],**self.kw)
 def test_facing_mismatch_retains_failure(self):
  p=dict(self.part);p["outward_normal_object"]=[0,0,1]
  r=propose_initializers(self.K,np.eye(3),[0,0,.3],[p],**self.kw)
  self.assertEqual(r["retained"],[]);self.assertEqual(r["facing_rejections"],r["total_grid"])
