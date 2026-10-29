import unittest
from dataclasses import replace
import numpy as np
from model.action4_sam_face_contract import FaceSeedBinding,FaceMaskThresholds,face_point_request,validate_face_raw
from model.action4_sam_session_contracts import RegionSpec,SessionLedger
from model.action4_sam_response_adapter import ResponseEnvelope

class FaceTests(unittest.TestCase):
 def setUp(self):
  self.region=RegionSpec("face-clip",(10,11),(100,110),(1,100),"a"*64,"native",20,20)
  ref={"path":"fixture.json","sha256":"b"*64}
  self.seed=FaceSeedBinding("body",0,0,0,self.region.identity,0,10,100,(1,100),"native","c"*64,"d"*64,
    (1,0,0,0,1,0,0,0,1),dict(ref),dict(ref),dict(ref),(.1,.1,.8,.8))
  self.points=[[.25,.25],[.35,.35],[.7,.7],[.8,.8]];self.labels=[1,1,0,0]
  self.ledger=SessionLedger();self.record=self.plan(self.seed,"session0")
  self.body=np.zeros((20,20),bool);self.body[2:18,2:18]=True
  self.face=np.zeros_like(self.body);self.face[3:9,3:9]=True
  self.thresholds=FaceMaskThresholds(1,0,.9,.85,.1,.05,.9)
 def plan(self,seed,sid):
  r=self.ledger.plan(self.region,seed,worker_id="worker",device_identity="fixtureGPU",dependency_identity="fixture-source")
  self.ledger.begin_create(r.record_id);self.ledger.created(r.record_id,sid)
  request=face_point_request(self.ledger,r.record_id,seed,points_xy01=self.points,point_labels=self.labels)
  self.assertEqual(request["obj_id"],0);self.assertEqual(request["point_labels"],self.labels)
  return self.ledger.get(r.record_id)
 def validate(self,mask=None,**kwargs):
  raw={"frame_index":0,"outputs":{"out_obj_ids":np.array([0]),"out_probs":np.array([.9]),
      "out_boxes_xywh":np.array([[.1,.1,.8,.8]]),"out_binary_masks":np.array([self.face if mask is None else mask])}}
  return validate_face_raw(raw,seed=kwargs.pop("seed",self.seed),record=self.record,region=self.region,
   envelope=ResponseEnvelope("session0",self.region.identity,0,0,"seed","seed",0),
   points_xy01=self.points,point_labels=self.labels,body_support=self.body,
   neighbor_supports=kwargs.pop("neighbor_supports",[]),other_body_supports=kwargs.pop("other_body_supports",[]),
   existing_face_masks=kwargs.pop("existing_face_masks",[]),thresholds=self.thresholds,
   raw_receipt_ref={"path":"raw.json","sha256":"e"*64},**kwargs)
 def test_real_ledger_obj0_native_binding(self):
  result=self.validate();self.assertEqual(result["status"],"predicted_face_support_only")
  self.assertFalse(result["semantic_face_verified"]);self.assertFalse(result["pose_accuracy_verified"])
  self.ledger.bind_seed(self.record.record_id,[result["native_obj_id"]])
 def test_separate_face_sessions_can_both_obj0(self):
  other=replace(self.seed,face_id=1);r=self.plan(other,"session1")
  self.assertNotEqual(r.track_id,self.record.track_id)
  self.ledger.bind_seed(r.record_id,[0]);self.ledger.bind_seed(self.record.record_id,[0])
 def test_source_and_generation_tamper(self):
  with self.assertRaises(ValueError):replace(self.seed,source="frozen_b0")
  with self.assertRaises(ValueError):self.validate(seed=replace(self.seed,body_generation=1))
  with self.assertRaises(ValueError):self.validate(seed=replace(self.seed,face_generation=1))
 def test_frame_pts_tamper(self):
  with self.assertRaises(ValueError):self.validate(seed=replace(self.seed,pts=101))
  with self.assertRaises(ValueError):self.validate(seed=replace(self.seed,global_frame_id=11))
 def test_insufficient_points(self):
  l=SessionLedger();r=l.plan(self.region,self.seed,worker_id="w",device_identity="g",dependency_identity="d")
  l.begin_create(r.record_id);l.created(r.record_id,"sid")
  with self.assertRaises(ValueError):
   face_point_request(l,r.record_id,self.seed,points_xy01=[[.2,.2],[.8,.8]],point_labels=[1,0])
 def test_whole_body_and_empty_unknown(self):
  r=self.validate(self.body);self.assertEqual(r["status"],"unknown")
  self.assertIn("whole_body_swallowed_not_face",r["reasons"])
  self.assertEqual(self.validate(np.zeros_like(self.body))["status"],"unknown")
 def test_shared_neighbor_crossbody_unknown(self):
  self.assertIn("duplicate_shared_face_mask",self.validate(existing_face_masks=[self.face])["reasons"])
  self.assertIn("multiple_faces_swallowed",self.validate(neighbor_supports=[self.face])["reasons"])
  self.assertIn("cross_body_support",self.validate(other_body_supports=[self.face])["reasons"])
 def test_prompt_change_or_duplicate_points_rejected(self):
  self.points[0]=[.3,.25]
  with self.assertRaises(ValueError):self.validate()
  self.points[0]=self.points[1]
  with self.assertRaises(ValueError):self.validate()
 def test_inputs_preserved(self):
  before=self.face.copy();body=self.body.copy();self.validate()
  np.testing.assert_array_equal(before,self.face);np.testing.assert_array_equal(body,self.body)
if __name__=="__main__":unittest.main()
