import unittest
import numpy as np
import cv2
from model.action4_planar_landmark_pose import planar_pose_candidates
class PlanarCandidates(unittest.TestCase):
 def test_known_pose_and_fixed_corner_order(self):
  x=np.array([[-.02,.01,0],[.02,.01,0],[.02,-.01,0],[-.02,-.01,0]])
  c=np.array([[a,b,z] for a in [-.03,.03] for b in [-.02,.02] for z in [-.01,.01]])
  K=np.array([[500.,0,200],[0,500.,150],[0,0,1.]])
  R=cv2.Rodrigues(np.array([.2,.3,.1]))[0];t=np.array([.01,.02,.5])
  q=(x@R.T+t)@K.T;u=q[:,:2]/q[:,2:]
  result=planar_pose_candidates(x,u,K,c);good=[p for p in result["candidates"] if p["valid"]]
  best=min(good,key=lambda p:p["correspondence_rms_native_px"])
  np.testing.assert_allclose(best["R"],R,atol=1e-7)
  self.assertEqual(best["corner_ids"],list(range(8)));self.assertFalse(result["final_prediction_replaced"])
 def test_bad_geometry_fails(self):
  with self.assertRaises(ValueError):planar_pose_candidates(np.zeros((4,3)),np.zeros((4,2)),np.eye(3),np.zeros((8,3)))
