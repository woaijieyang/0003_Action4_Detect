"""Pinned official removed-object sentinel, never clamp into a valid probability."""
import unittest
import numpy as np
from model.action4_sam_session_contracts import RegionSpec,SeedBinding,SessionLedger
from model.action4_sam_response_adapter import ResponseEnvelope,validate_raw_response,parse_target_response,ResponseSchemaError
from eval.action4_sam_image_eval import bind_native_output
class RemovedTests(unittest.TestCase):
 def setUp(self):
  self.region=RegionSpec("r",(0,1),(0,400),(1,11988),"manifest","native",10,10)
  seed=SeedBinding("track",self.region.identity,"candidate",0,0,(.1,.1,.4,.4),("B0",),"frozen_b0")
  ledger=SessionLedger();r=ledger.plan(self.region,seed,worker_id="w",device_identity="cpu-fixture",dependency_identity="source")
  ledger.begin_create(r.record_id);ledger.created(r.record_id,"session");ledger.seed_request(r.record_id);ledger.bind_seed(r.record_id,[7])
  self.record=ledger.begin_propagation(r.record_id).record
  self.env=ResponseEnvelope("session",self.region.identity,0,0,"propagate_both","response")
  mask=np.zeros((1,10,10),bool);mask[:,1:5,1:5]=True
  self.raw={"frame_index":1,"outputs":{"out_obj_ids":np.array([7]),"out_probs":np.array([-1e4],np.float32),
             "out_boxes_xywh":np.array([[.1,.1,.4,.4]],np.float32),"out_binary_masks":mask}}
 def test_residual_mask_not_registered_or_used(self):
  parsed=parse_target_response(self.raw,record=self.record,region=self.region,envelope=self.env,
       register_mask=lambda req:(_ for _ in ()).throw(AssertionError("removed mask must not register")))
  self.assertEqual(parsed.status,"target_visibility_unknown");self.assertFalse(parsed.available)
  self.assertTrue(parsed.public_mask_nonempty);self.assertIsNone(parsed.registration)
 def test_other_invalid_probabilities_still_fail(self):
  for value in [-1.,-9999.,1.1,float("nan")]:
   self.raw["outputs"]["out_probs"][:]=value
   with self.assertRaises(ResponseSchemaError):validate_raw_response(self.raw,record=self.record,region=self.region,envelope=self.env)
 def test_removed_mask_cannot_bind_a_seed(self):
  output=validate_raw_response(self.raw,record=self.record,region=self.region,envelope=self.env)
  candidate={"candidate_id":"candidate","bbox_sam_native_xyxy":[1,1,5,5],"eligible_for_SAM_diagnostic_prompt":True}
  obj,_,rows=bind_native_output(output,{"native_shape_hw":[10,10],"candidates":[candidate]},candidate,
            {"mask_in_box_min":.8,"unique_margin":.15,"multi_target_support_min":.7})
  self.assertIsNone(obj);self.assertTrue(rows[0]["native_object_removed"]);self.assertIsNone(rows[0]["probability"])
if __name__=="__main__":unittest.main(verbosity=2)
