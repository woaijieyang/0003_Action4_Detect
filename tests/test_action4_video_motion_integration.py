import copy,unittest
import numpy as np
from test_action4_video_pose_stabilizer import packet,K,CAD
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_video_motion_observation import measure_motion
from eval.action4_staged_pipeline import StagedPipeline
from types import SimpleNamespace

class MotionIntegrationTests(unittest.TestCase):
 def test_enabled_gate_consumed_by_existing_pipeline_and_preserves_raw(self):
  pipe=object.__new__(StagedPipeline);pipe.a0_kind="joint_pose"
  s=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True))
  uv=np.array([[x,y] for y in [390.,396.,402.,408.] for x in [480.,490.,500.,510.,520.]])
  rejects=0
  for f in range(12):
   raw=packet(f,angle=1.7 if f>=6 else 0);pipe.predict=lambda *a,**k:copy.deepcopy(raw)
   obs=dict(track_id="a",frame_id=f,previous_frame_id=f-1,pts=f,time_base=[1,30],same_generation=True,available=True,
       points=len(uv),inlier_ratio=1.,coverage=.6,previous_uv=uv.tolist(),current_uv=uv.tolist())
   out=pipe.predict_video(None,K,CAD,video_context=SimpleNamespace(frame_id=f),pose_stabilizer=s,
        frame_clock=dict(frame_id=f,pts=f,time_base=[1,30]),motion_observations={"a":obs} if f else {})
   item=out["instances"][0];rejects+=int(item["pose_stabilization"]["image_motion_decision"]["reject_candidate"])
   self.assertEqual(item["raw_network_pose"],raw["instances"][0]["raw_network_pose"])
  self.assertGreater(rejects,0)
  np.testing.assert_allclose(out["instances"][0]["final_pose"]["rotation"],np.eye(3),atol=1e-6)
 def test_cross_identity_or_frame_rejected(self):
  s=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True))
  with self.assertRaises(ValueError):
   s.apply_prediction(packet(0),frame_id=0,pts=0,time_base=[1,30],K_native=K,canonical_m=CAD,mode="video",
     motion_observations={"a":dict(track_id="b",frame_id=0,pts=0,time_base=[1,30])})
 def test_missing_evidence_does_not_imply_stationary(self):
  a=VideoPoseStabilizer(StabilizerConfig());b=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True))
  for f in range(10):
   kwargs=dict(frame_id=f,pts=f,time_base=[1,30],K_native=K,canonical_m=CAD,mode="video")
   x=a.apply_prediction(packet(f,angle=.2*f),**kwargs)
   y=b.apply_prediction(packet(f,angle=.2*f),motion_observations={},**kwargs)
   np.testing.assert_array_equal(x["instances"][0]["corners_final"],y["instances"][0]["corners_final"])
 def test_actual_pixels_and_other_mask_exclusion(self):
  import cv2
  cv2.setNumThreads(1)
  rng=np.random.default_rng(47)
  a=(rng.random((240,320))*255).astype(np.uint8)
  a=cv2.GaussianBlur(a,(3,3),.7)
  b=cv2.warpAffine(a,np.float32([[1,0,5],[0,1,3]]),(320,240))
  ma=np.zeros_like(a,bool);ma[40:200,60:260]=1
  mb=cv2.warpAffine(ma.astype(np.uint8),np.float32([[1,0,5],[0,1,3]]),(320,240))>0
  o=measure_motion(a,b,ma,mb,[-20,-20,340,260],max_side=192)
  self.assertTrue(o["available"],o)
  flow=np.array(o["current_uv"])-np.array(o["previous_uv"])
  np.testing.assert_allclose(np.median(flow,axis=0),[5,3],atol=.35)
  none=measure_motion(a,b,ma,mb,[0,0,320,240],current_other=mb)
  self.assertFalse(none["available"])
 def test_blank_texture_unknown(self):
  im=np.zeros((100,100),np.uint8);mask=np.ones_like(im,bool)
  o=measure_motion(im,im,mask,mask,[0,0,100,100])
  self.assertFalse(o["available"])

class BoundedVetoTests(unittest.TestCase):
 def test_after_veto_missing_texture_has_bounded_not_unlimited_hold(self):
  s=VideoPoseStabilizer(StabilizerConfig(motion_veto_enabled=True,motion_veto_hold_frames=5))
  uv=np.array([[x,y] for y in [390.,396.,402.,408.] for x in [480.,490.,500.,510.,520.]])
  last_veto=None;prior_rotation=None;held=0;expired=False
  for f in range(22):
   obs=dict(track_id="a",frame_id=f,previous_frame_id=f-1,pts=f,time_base=[1,30],
      same_generation=True,available=True,points=len(uv),inlier_ratio=1.,coverage=.6,
      previous_uv=uv.tolist(),current_uv=uv.tolist())
   evidence={"a":obs} if f and last_veto is None else {}
   out=s.apply_prediction(packet(f,angle=1.7 if f>=6 else 0),frame_id=f,pts=f,time_base=[1,30],
      K_native=K,canonical_m=CAD,mode="video",motion_observations=evidence)["instances"][0]
   d=out["pose_stabilization"]["image_motion_decision"]
   if d.get("reject_candidate"):last_veto=f
   if d.get("bounded_rotation_hold"):
    held+=1;np.testing.assert_array_equal(out["final_pose"]["rotation"],prior_rotation)
    self.assertFalse(d["current_rotation_observed"]);self.assertLessEqual(f-last_veto,5)
   expired|=d.get("hold_expired",False);prior_rotation=out["final_pose"]["rotation"]
  self.assertEqual(held,5);self.assertTrue(expired)
 def test_invalid_unbounded_hold_rejected(self):
  with self.assertRaises(ValueError):StabilizerConfig(motion_veto_enabled=True,motion_veto_hold_frames=99)
