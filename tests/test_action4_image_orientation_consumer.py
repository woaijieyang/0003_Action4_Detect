import copy,unittest
import numpy as np
from test_action4_video_pose_stabilizer import packet,K,CAD
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_native_rotation_history import rotation_log
from model.action4_image_orientation_evidence import DEFAULT_POLICY,policy_identity
POLICY_ID=policy_identity(DEFAULT_POLICY)
class ConsumerTests(unittest.TestCase):
 def call(self,s,f,angle=0,hold=False,available=True,center=500,change=None):
  p=packet(f,angle=angle,center=center);p['RGB_sha256']='a'*64
  ob=dict(track_id='a',frame_id=f,pts=f,time_base=[1,30],current_RGB_sha256='a'*64,available=available,
          current_mask_authenticated=available,hold_rotation=hold,previous_frame_id=f-1,reference_frame_id=0,reason='fixture',policy_id=POLICY_ID)
  if change:ob.update(change)
  before=copy.deepcopy(p)
  r=s.apply_prediction(p,frame_id=f,pts=f,time_base=[1,30],K_native=K,canonical_m=CAD,mode='video',image_orientation_observations={'a':ob})
  self.assertEqual(p,before);return r['instances'][0]
 def test_observed_stationarity_holds_rotation_but_current_ROI_translates(self):
  s=VideoPoseStabilizer(StabilizerConfig(image_orientation_enabled=True,image_orientation_policy_id=POLICY_ID))
  self.call(s,0)
  for f in range(1,15):
   o=self.call(s,f,angle=1,hold=True,center=500+10*f)
   self.assertLess(np.linalg.norm(rotation_log(np.array(o['final_pose']['rotation']))),1e-10)
   t=np.array(o['final_pose']['translation_m']);h=K@t;self.assertAlmostEqual(h[0]/h[2],500+10*f)
   self.assertEqual(o['pose_stabilization']['image_hold_run'],f)
 def test_unknown_evidence_releases_and_no_geometry_claim(self):
  s=VideoPoseStabilizer(StabilizerConfig(image_orientation_enabled=True,image_orientation_policy_id=POLICY_ID,rotation_max_speed_deg_s=180))
  self.call(s,0)
  for f in range(1,10):o=self.call(s,f,angle=1,hold=False,available=False)
  self.assertGreater(np.linalg.norm(rotation_log(np.array(o['final_pose']['rotation']))),.1)
  self.assertFalse(o['pose_stabilization']['image_rotation_held'])
 def test_invalid_or_stale_evidence_cannot_hold(self):
  for changed in [dict(track_id='b'),dict(frame_id=3),dict(current_RGB_sha256='b'*64),dict(current_mask_authenticated=False),dict(previous_frame_id=9),dict(policy_id='b'*64)]:
   s=VideoPoseStabilizer(StabilizerConfig(image_orientation_enabled=True,image_orientation_policy_id=POLICY_ID));self.call(s,0)
   with self.assertRaises(ValueError):self.call(s,1,hold=True,change=changed)
 def test_disabled_consumer_rejects_unused_evidence(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  with self.assertRaises(ValueError):self.call(s,0)

 def test_missing_previous_pose_declines_hold_without_losing_frame(self):
  s=VideoPoseStabilizer(StabilizerConfig(image_orientation_enabled=True,image_orientation_policy_id=POLICY_ID))
  self.call(s,0)
  bad=packet(1,support=False);bad['RGB_sha256']='a'*64
  ob=dict(track_id='a',frame_id=1,pts=1,time_base='1/30',current_RGB_sha256='a'*64,available=False,hold_rotation=False,current_mask_authenticated=False,policy_id=POLICY_ID,previous_frame_id=0)
  s.apply_prediction(bad,frame_id=1,pts=1,time_base=[1,30],K_native=K,canonical_m=CAD,mode='video',image_orientation_observations={'a':ob})
  o=self.call(s,2,angle=.4,hold=True)
  self.assertEqual(o['pose_stabilization']['status'],'initialized')
  self.assertFalse(o['pose_stabilization']['image_rotation_held'])
  self.assertFalse(o['pose_stabilization']['image_hold_state_ready'])
