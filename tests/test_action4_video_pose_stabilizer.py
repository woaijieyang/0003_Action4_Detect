import copy,unittest
import numpy as np
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_native_rotation_history import rotation_exp,rotation_log
K=np.array([[1000.,0,500],[0,1000,400],[0,0,1]])
CAD=np.array([[x,y,z] for x in [-.03,.03] for y in [-.015,.015] for z in [-.02,.02]])
def packet(fid,angle=0,center=500.,size=100.,depth=1.,track="a",support=True):
 r=rotation_exp([0,0,angle]);t=np.array([(center-500)*depth/1000,0,depth]);camera=CAD@r.T+t
 h=camera@K.T;corners=(h[:,:2]/h[:,2,None]).tolist()
 return dict(instances=[dict(track_id=track,prediction_id=f"video:{track}:frame:{fid}",bbox_b0_raw=None,bbox_sam_visible=[center-size/2,400-size/2,center+size/2,400+size/2],roi_decision_id=f"decision-{fid}-{track}",frame_id=fid,pts=fid,time_base=[1,30],corner_ids=list(range(8)),
   published=True,pose_valid=True,post_A0_geometry="none",roi_source="SAM_tracking_geometry",current_support_status={"B0":False,"SAM":support},
   bbox_for_a0=[center-size/2,400-size/2,center+size/2,400+size/2],
   corners_px=corners,corners_final=copy.deepcopy(corners),corners_raw_from_resolved_roi=copy.deepcopy(corners),
   raw_network_pose=dict(finite=True,rotation=r.tolist(),translation_m=t.tolist()))])
def apply(s,p,f,mode="video"):
 return s.apply_prediction(p,frame_id=f,pts=f,time_base=[1,30],K_native=K,canonical_m=CAD,mode=mode)
class StabilizerTests(unittest.TestCase):
 def test_stationary_exact_and_raw_unchanged(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(10):
   p=packet(f);before=copy.deepcopy(p);o=apply(s,p,f)
   self.assertEqual(p,before);np.testing.assert_allclose(o["instances"][0]["corners_final"],p["instances"][0]["corners_final"],atol=1e-10)
 def test_current_sam_ROI_motion_has_no_added_center_lag(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(10):
   o=apply(s,packet(f,center=500+20*f),f)["instances"][0];t=np.array(o["final_pose"]["translation_m"]);h=K@t
   self.assertAlmostEqual(h[0]/h[2],500+20*f,places=9)
 def test_current_scale_follows_without_depth_lag(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(10):
   scale=1+.1*f;o=apply(s,packet(f,size=100*scale,depth=1/scale),f)["instances"][0]
   self.assertAlmostEqual(o["final_pose"]["translation_m"][2],1/scale,places=10)
 def test_single_rotation_flip_rejected_by_history(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(5):apply(s,packet(f),f)
  o=apply(s,packet(5,angle=np.pi),5)["instances"][0]
  self.assertLess(np.linalg.norm(rotation_log(np.array(o["final_pose"]["rotation"]))),1e-8)
  self.assertNotEqual(o["pose_stabilization"]["selected_rotation_frame"],5)
 def test_two_bodies_never_share_history(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(8):
   p=packet(f);p["instances"]+=packet(f,angle=1,track="b")["instances"];o=apply(s,p,f)
   for i,a in enumerate([0,1]):np.testing.assert_allclose(o["instances"][i]["final_pose"]["rotation"],rotation_exp([0,0,a]),atol=1e-10)
 def test_rigid_pose_and_positive_depth(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(30):
   o=apply(s,packet(f,angle=.05*f),f)["instances"][0];r=np.array(o["final_pose"]["rotation"]);t=np.array(o["final_pose"]["translation_m"])
   np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10);self.assertTrue(((CAD@r.T+t)[:,2]>0).all())
   self.assertEqual(o["corner_ids"],list(range(8)))
 def test_gap_resets_no_unlimited_extrapolation(self):
  s=VideoPoseStabilizer(StabilizerConfig());apply(s,packet(0),0)
  o=apply(s,packet(30,angle=1),30)["instances"][0]
  self.assertEqual(o["pose_stabilization"]["status"],"initialized")
  self.assertEqual(o["pose_stabilization"]["history_frames"],[])
  self.assertEqual(o["pose_stabilization"]["candidate_frames"],[30])
 def test_missing_support_does_not_publish_synthetic_pose(self):
  s=VideoPoseStabilizer(StabilizerConfig());p=packet(0,support=False);o=apply(s,p,0)
  self.assertEqual(o["instances"][0]["corners_final"],p["instances"][0]["corners_final"]);self.assertNotIn("a",s.states)
 def test_image_mode_and_reversed_pts_rejected(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  with self.assertRaises(ValueError):apply(s,packet(0),0,"image")
  apply(s,packet(0),0)
  with self.assertRaises(ValueError):apply(s,packet(0),0)
 def test_step_by_step_equals_saved_sequence_processing(self):
  a=VideoPoseStabilizer(StabilizerConfig());b=VideoPoseStabilizer(StabilizerConfig())
  for f in range(10):
   p=packet(f,angle=.02*f);self.assertEqual(apply(a,p,f),apply(b,copy.deepcopy(p),f))
 def test_constant_turn_lag_bounded_in_fixture(self):
  s=VideoPoseStabilizer(StabilizerConfig());errors=[]
  for f in range(60):
   r=rotation_exp([0,0,2*f/30]);o=apply(s,packet(f,angle=2*f/30),f)["instances"][0]
   errors.append(np.linalg.norm(rotation_log(r@np.array(o["final_pose"]["rotation"]).T)))
  self.assertLess(max(errors[20:]),.3)
 def test_noise_reduced_in_stationary_fixture(self):
  s=VideoPoseStabilizer(StabilizerConfig());angles=[]
  for f in range(80):
   o=apply(s,packet(f,angle=.1*(-1)**f),f)["instances"][0];angles.append(rotation_log(np.array(o["final_pose"]["rotation"]))[2])
  self.assertLess(np.std(angles[20:]),.05)
 def test_nonfinite_beta_rejected(self):
  with self.assertRaises(ValueError):StabilizerConfig(rotation_beta=float("nan"))
