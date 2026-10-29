"""Counterexamples for image evidence gating, not pose-accuracy acceptance."""
import unittest
import numpy as np
from model.action4_native_rotation_history import rotation_exp
from model.action4_video_motion_evidence import bind_visible_box_points, score_pose_motion
K=np.array([[800.,0,256],[0,800,256],[0,0,1]])
CAD=np.array([[x,y,z] for x in [-.03,.03] for y in [-.02,.02] for z in [-.015,.015]])
T=np.array([0.,0.,.5])
UV=np.array([[x,y] for y in [235.,250.,265.,277.] for x in [220.,240.,260.,280.]])
def project(x,r,t):
 h=(x@r.T+t)@K.T
 return h[:,:2]/h[:,2,None]
class MotionEvidenceTests(unittest.TestCase):
 def test_binding_reprojects_and_hits_nearest_visible_face(self):
  x,ok=bind_visible_box_points(UV,K,CAD,np.eye(3),T)
  self.assertTrue(ok.all());np.testing.assert_allclose(x[:,2],-.015,atol=1e-10)
  np.testing.assert_allclose(project(x,np.eye(3),T),UV,atol=1e-10)
 def test_outside_rays_are_not_fabricated_surface(self):
  _,ok=bind_visible_box_points(np.array([[0.,0.],[256,256]]),K,CAD,np.eye(3),T)
  self.assertEqual(ok.tolist(),[False,True])
 def test_stationary_texture_rejects_large_rotation_candidate(self):
  result=score_pose_motion(UV,UV,K,CAD,np.eye(3),T,rotation_exp([0,0,2.]),T,100.)
  self.assertTrue(result['reject_candidate'])
  self.assertGreater(result['candidate_median_px'],result['hold_median_px']+2)
 def test_real_turn_not_rejected_when_correspondences_explain_it(self):
  x,ok=bind_visible_box_points(UV,K,CAD,np.eye(3),T)
  r=rotation_exp([0,0,.7])
  result=score_pose_motion(UV,project(x,r,T),K,CAD,np.eye(3),T,r,T,100.)
  self.assertFalse(result['reject_candidate']);self.assertLess(result['candidate_median_px'],1e-7)
 def test_current_translation_allowed(self):
  x,_=bind_visible_box_points(UV,K,CAD,np.eye(3),T);t=T+np.array([.01,.005,0.])
  result=score_pose_motion(UV,project(x,np.eye(3),t),K,CAD,np.eye(3),T,np.eye(3),t,100.)
  self.assertFalse(result['reject_candidate']);self.assertLess(result['hold_median_px'],1e-7)
 def test_insufficient_points_unknown_not_stationary(self):
  result=score_pose_motion(UV[:3],UV[:3],K,CAD,np.eye(3),T,np.eye(3),T,100.)
  self.assertFalse(result['available']);self.assertFalse(result['reject_candidate'])
 def test_nan_is_error_not_normal_rejection(self):
  with self.assertRaises(ValueError):
   score_pose_motion(UV,UV*np.nan,K,CAD,np.eye(3),T,np.eye(3),T,100.)
 def test_non_rigid_prior_rejected(self):
  with self.assertRaises(ValueError):bind_visible_box_points(UV,K,CAD,2*np.eye(3),T)
