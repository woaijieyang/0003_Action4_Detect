import copy,unittest
import numpy as np
from test_action4_video_pose_stabilizer import packet,K,CAD
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_image_orientation_evidence import DEFAULT_POLICY,policy_identity
from model.action4_native_rotation_history import rotation_log
P=policy_identity(DEFAULT_POLICY);M='b'*64
class MaskGainTests(unittest.TestCase):
 def engine(self,on=True):
  kw=dict(image_orientation_enabled=True,image_orientation_policy_id=P,rotation_min_cutoff_hz=2.,rotation_max_cutoff_hz=3.,rotation_max_speed_deg_s=180.)
  if on:kw.update(mask_motion_gain_enabled=True,mask_motion_policy_id=M,weak_rotation_max_cutoff_hz=.75,weak_rotation_max_speed_deg_s=90.)
  return VideoPoseStabilizer(StabilizerConfig(**kw))
 def call(self,s,f,eligible,*,known=False,change=None):
  p=packet(f,angle=0 if f<4 else 1.4);p['RGB_sha256']='a'*64;before=copy.deepcopy(p)
  hint=dict(track_id='a',frame_id=f,pts=f,time_base='1/30',current_RGB_sha256='a'*64,previous_frame_id=f-1,
    policy_id=M,eligible_for_gain=eligible,weak_low_motion=eligible,current_mask_authenticated=True)
  if change:hint.update(change)
  ob=dict(track_id='a',frame_id=f,pts=f,time_base='1/30',current_RGB_sha256='a'*64,previous_frame_id=f-1,policy_id=P,
     available=known,hold_rotation=False,current_mask_authenticated=True,reason='fixture',mask_motion_hint=hint)
  r=s.apply_prediction(p,frame_id=f,pts=f,time_base=[1,30],K_native=K,canonical_m=CAD,mode='video',image_orientation_observations={'a':ob})
  self.assertEqual(p,before);return r['instances'][0]
 def test_low_gain_reduces_updates_without_freezing(self):
  slow=self.engine();normal=self.engine(False);prev=None;changes=[]
  for f in range(20):
   a=self.call(slow,f,f>0);b=self.call(normal,f,False);r=np.array(a['final_pose']['rotation'])
   if prev is not None:changes.append(np.rad2deg(np.linalg.norm(rotation_log(r@prev.T))))
   prev=r
  self.assertLessEqual(max(changes),3+1e-7);self.assertGreater(max(changes),.1)
  self.assertLess(np.linalg.norm(rotation_log(r)),np.linalg.norm(rotation_log(np.array(b['final_pose']['rotation']))))
  self.assertFalse(a['pose_stabilization']['image_rotation_held']);self.assertTrue(a['pose_stabilization']['weak_silhouette_gain_applied'])
 def test_strong_evidence_cannot_be_overridden(self):
  s=self.engine();self.call(s,0,False)
  with self.assertRaises(ValueError):self.call(s,1,True,known=True)
 def test_wrong_source_and_missing_masks_rejected(self):
  for ch in [dict(track_id='b'),dict(current_RGB_sha256='c'*64),dict(policy_id='c'*64),dict(previous_frame_id=9),dict(current_mask_authenticated=False)]:
   s=self.engine();self.call(s,0,False)
   with self.assertRaises(ValueError):self.call(s,1,True,change=ch)
 def test_ineligible_hint_is_exact_no_gain_parity(self):
  a=self.engine();b=self.engine(False)
  for f in range(12):self.assertEqual(self.call(a,f,False)['final_pose'],self.call(b,f,False)['final_pose'])
