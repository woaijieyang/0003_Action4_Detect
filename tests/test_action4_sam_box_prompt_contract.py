import unittest
from dataclasses import asdict,replace,FrozenInstanceError
from unittest.mock import patch
from model.action4_sam_session_contracts import (RegionSpec,SeedBinding,PromptEvent,PVSBoxPromptEvent,
 ReplayPlan,lock_replay_plan,SessionLedger,_fingerprint)
from model import action4_sam_seed_extent as extent
import test_action4_sam_session_contracts as legacy

class BoxPromptContractTests(unittest.TestCase):
 def region(self):return RegionSpec('r',tuple(range(100)),tuple(i*10 for i in range(100)),(1,1000),'manifest','pixels',100,80)
 def seed(self,region,track='t'):return SeedBinding(track,region.identity,'b0:0',0,0,(.1,.1,.4,.4),('b0-weight','candidate-sha'),'frozen_b0')
 def event(self,region,seed,index=20,event_id='box',segment='segment'):
  return PVSBoxPromptEvent(event_id,seed.track_id,region.identity,seed.fingerprint,index,index,
   ((10,8),(50,40)),((.1,.1),(.5,.5)),(2,3),('frozen-b0-weight-sha','candidate-sha'),'extent-collapse',segment,'frozen_b0')
 def test_new_frozen_subclass_and_common_request_converter(self):
  r=self.region();s=self.seed(r);e=self.event(r,s)
  self.assertIsInstance(e,PromptEvent);self.assertEqual(e.protocol,extent.PROTOCOL)
  plan=lock_replay_plan(r,[s],[e]);self.assertIsInstance(plan,ReplayPlan)
  with patch.object(extent,'box_corner_prompt',wraps=extent.box_corner_prompt) as converter:
   req=e.to_request(r,s,session_id='existingSID',obj_id=1)
   converter.assert_called_once_with('existingSID',20,1,(10,8,50,40),100,80)
  self.assertEqual(req['point_labels'],[2,3]);self.assertEqual(req['points'],[[.1,.1],[.5,.5]])
  self.assertEqual(req['session_id'],'existingSID');self.assertEqual(req['obj_id'],1)
  self.assertNotIn('text',req);self.assertNotIn('bounding_boxes',req);self.assertTrue(req['clear_old_points'])
  with self.assertRaises(FrozenInstanceError):e.labels=(1,0)
 def test_coordinate_roundtrip_and_frame_binding(self):
  r=self.region();s=self.seed(r);e=self.event(r,s)
  bad=replace(e,points_xy01=((.11,.1),(.5,.5)))
  with self.assertRaises(ValueError):lock_replay_plan(r,[s],[bad])
  with self.assertRaises(ValueError):bad.to_request(r,s,session_id='sid',obj_id=0)
  for bad in [replace(e,global_frame_id=21),replace(e,region_identity='other'),replace(e,seed_fingerprint='other')]:
   with self.assertRaises(ValueError):lock_replay_plan(r,[s],[bad])
   with self.assertRaises(ValueError):bad.to_request(r,s,session_id='sid',obj_id=0)
 def test_frozen_B0_only_evidence(self):
  r=self.region();s=self.seed(r);e=self.event(r,s)
  for source in ('GT','A0','frozen_b0_yolo','manual'):
   with self.assertRaises(ValueError):replace(e,evidence_source=source)
  with self.assertRaises(ValueError):replace(e,evidence_ids=())
  mixed_seed=replace(s,source='frozen_b0_yolo');mixed_event=replace(e,seed_fingerprint=mixed_seed.fingerprint)
  with self.assertRaises(ValueError):lock_replay_plan(r,[mixed_seed],[mixed_event])
 def test_mixed_labels_bad_shape_order_protocol_rejected(self):
  r=self.region();s=self.seed(r);e=self.event(r,s)
  for labels in ((1,1),(3,2),(2,3,1),(2.,3),(-1,3),(2,0)):
   with self.assertRaises(ValueError):replace(e,labels=labels)
  for points in (((50,8),(10,40)),((10,8),(50,8)),((10,8),),((10,8),(float('nan'),40))):
   with self.assertRaises(ValueError):replace(e,points_xy_native=points)
  with self.assertRaises(ValueError):replace(e,protocol='unregistered')
  with self.assertRaises(ValueError):replace(e,points_xy01=((.1,.1),(1.1,.5)))
 def test_existing_event_budgets_and_order_remain_enforced(self):
  r=self.region();s=self.seed(r);a=self.event(r,s,20,'a','a');b=self.event(r,s,40,'b','b');c=self.event(r,s,60,'c','c')
  lock_replay_plan(r,[s],[a,b])
  for events in ([a,b,c],[b,a],[a,replace(b,local_index=25,global_frame_id=25)],[a,replace(b,error_segment_id='a')]):
   with self.assertRaises(ValueError):lock_replay_plan(r,[s],events)
  s2=self.seed(r,'t2');s3=self.seed(r,'t3')
  events=[a,self.event(r,s2,30,'d','d'),b,self.event(r,s2,50,'e','e'),self.event(r,s3,60,'f','f')]
  with self.assertRaises(ValueError):lock_replay_plan(r,[s,s2,s3],events)
 def test_legacy_fingerprint_and_asdict_byte_behavior_unchanged(self):
  r=self.region();s=self.seed(r)
  e=PromptEvent('e','t',r.identity,s.fingerprint,20,20,((10,10),(20,20),(70,60),(80,70)),((.1,.125),(.2,.25),(.7,.75),(.8,.875)),(1,1,0,0),('predicted-only',),'drift','seg','frozen_b0')
  self.assertNotIn('protocol',asdict(e))
  self.assertEqual(_fingerprint(asdict(e)),'74e94d0092d36f2b6364b5400f0e94da3b170a3cc8888234dcbb28aab0599830')
  self.assertEqual(lock_replay_plan(r,[s],[e]).identity,'87486407e3012c79b6b4c5d3d0532095f83ab83bdbd56064ce24e529be521ea4')
  with self.assertRaises(ValueError):PromptEvent('e','t',r.identity,s.fingerprint,20,20,((10,8),(50,40)),((.1,.1),(.5,.5)),(2,3),('b0',),'drift','seg','frozen_b0')
 def test_existing_gen1_correction_request_uses_converter_for_subclass(self):
  helper=legacy.SamSessionTests();ledger=SessionLedger();r=self.region();s=self.seed(r)
  record=ledger.plan(r,s,worker_id='w',device_identity='CPU-fixture',dependency_identity='frozen-source')
  old=helper.seeded(ledger,record,obj=0,sid='old');helper.complete(ledger,old)
  ledger.publish(old.record_id,identity_confirmed=True);ledger.close(old.record_id,released=True)
  event=self.event(r,s);plan=lock_replay_plan(r,[s],[event]);new=ledger.replacement(old.record_id,plan)
  new=helper.seeded(ledger,new,obj=1,sid='new');helper.complete(ledger,new)
  with patch.object(extent,'box_corner_prompt',wraps=extent.box_corner_prompt) as converter:
   req=ledger.correction_request(new.record_id,event.event_id);self.assertEqual(converter.call_count,1)
  self.assertEqual(req['point_labels'],[2,3]);self.assertEqual(req['session_id'],'new')
  ledger.correction_applied(new.record_id,event.event_id)
 def test_shared_SID_rejection_is_unchanged_not_silently_supported(self):
  r=self.region();ledger=SessionLedger()
  a=ledger.plan(r,self.seed(r,'a'),worker_id='w',device_identity='cpu',dependency_identity='d')
  b=ledger.plan(r,self.seed(r,'b'),worker_id='w',device_identity='cpu',dependency_identity='d')
  ledger.begin_create(a.record_id);ledger.created(a.record_id,'sameSID');ledger.begin_create(b.record_id)
  with self.assertRaisesRegex(ValueError,'reused'):ledger.created(b.record_id,'sameSID')
  self.assertEqual(ledger.active_count,2)

if __name__=='__main__':unittest.main()
