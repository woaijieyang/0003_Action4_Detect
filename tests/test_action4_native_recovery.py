import unittest,copy
import numpy as np
from model.action4_native_recovery import NativeRecoveryController,DEFAULT_POLICY
class RecoveryTests(unittest.TestCase):
 def setup(self):
  cad=np.array([[x,y,z] for x in [-.05,.05] for y in [-.03,.03] for z in [-.01,.01]])
  T=np.eye(4);T[2,3]=1
  K=np.array([[100,0,50],[0,100,50],[0,0,1]])
  return NativeRecoveryController("a",0,copy.deepcopy(DEFAULT_POLICY)),dict(track_id="a",generation=0,texture_points=0,current_B0=True,current_SAM=True,b0_score=.9,native_box=[70,70,80,80],b0_box=[44,46,56,54],seed_pose=T,canonical8=cad,K=K,mask_inside_box=.9)
 def test_two_frames_and_ack(self):
  c,k=self.setup()
  self.assertEqual(c.observe(frame_id=1,pts=400,**k)["action"],"withhold")
  self.assertEqual(c.observe(frame_id=2,pts=800,**k)["action"],"reseed")
  with self.assertRaises(ValueError):c.observe(frame_id=3,pts=1200,**k)
  c.acknowledge(2,True)
  self.assertEqual(c.recovery_count,1)
  self.assertEqual(c.observe(frame_id=3,pts=1200,**k)["action"],"withhold")
 def test_good_tracking_never_resets(self):
  c,k=self.setup();k["native_box"]=k["b0_box"]
  for f in range(1,8):self.assertEqual(c.observe(frame_id=f,pts=f*400,**k)["action"],"continue")
 def test_missing_or_bad_seed_not_reseeded(self):
  c,k=self.setup();k["current_B0"]=False
  for f in range(1,4):self.assertEqual(c.observe(frame_id=f,pts=f*400,**k)["action"],"withhold")
  c,k=self.setup();k["seed_pose"][2,3]=-1
  c.observe(frame_id=1,pts=400,**k)
  self.assertEqual(c.observe(frame_id=2,pts=800,**k)["reason"],"nonpositive_current_A0_depth")
 def test_identity_and_gap(self):
  c,k=self.setup()
  with self.assertRaises(ValueError):c.observe(frame_id=1,pts=400,**dict(k,track_id="b"))
  c.observe(frame_id=1,pts=400,**k)
  self.assertEqual(c.observe(frame_id=3,pts=1200,**k)["bad_streak"],1)

 def test_nonfinite_native_projection_is_withheld(self):
  c,k=self.setup();k["native_box"]=None
  self.assertEqual(c.observe(frame_id=1,pts=400,**k)["action"],"withhold")
  self.assertEqual(c.observe(frame_id=2,pts=800,**k)["action"],"reseed")

 def test_explicit_sam_only_continuation_no_reseed(self):
  c,k=self.setup();c.allow_sam_only_continuation=True
  k.update(current_B0=False,current_SAM=True,texture_points=20,b0_box=None)
  for f in range(1,25):
   r=c.observe(frame_id=f,pts=400*f,**k)
   self.assertEqual(r["action"],"continue")
   self.assertFalse(r["recovery_allowed"])
  self.assertEqual(c.recovery_count,0)
 def test_sam_only_weak_or_missing_does_not_continue(self):
  for support,points in [(False,20),(True,0)]:
   c,k=self.setup();c.allow_sam_only_continuation=True
   k.update(current_B0=False,current_SAM=support,texture_points=points,b0_box=None)
   self.assertEqual(c.observe(frame_id=1,pts=400,**k)["action"],"withhold")

 def test_support_count_kind_is_explicit(self):
  c,k=self.setup();c.support_count_kind="unregistered_count"
  with self.assertRaises(ValueError):c.observe(frame_id=1,pts=400,**k)
  self.assertEqual(c.last_frame,-1)

 def test_earlier_disagreement_does_not_change_seed_gate(self):
  from model.action4_native_recovery import EARLY_DISAGREEMENT_POLICY
  c,k=self.setup();k["native_box"]=[51,46,63,54]
  c.policy=copy.deepcopy(EARLY_DISAGREEMENT_POLICY)
  self.assertEqual(c.observe(frame_id=1,pts=400,**k)["action"],"withhold")
  self.assertEqual(c.observe(frame_id=2,pts=800,**k)["action"],"reseed")
  legacy,k=self.setup();k["native_box"]=[51,46,63,54]
  self.assertEqual(legacy.observe(frame_id=1,pts=400,**k)["action"],"continue")
  c,k=self.setup();c.policy=copy.deepcopy(EARLY_DISAGREEMENT_POLICY);k["b0_score"]=.1
  c.observe(frame_id=1,pts=400,**k)
  self.assertEqual(c.observe(frame_id=2,pts=800,**k)["reason"],"current_seed_not_consistent_with_B0_SAM")
 def test_unregistered_threshold_rejected(self):
  c,k=self.setup();c.policy["disagreement_iou"]=.2
  with self.assertRaises(ValueError):c.observe(frame_id=1,pts=400,**k)
