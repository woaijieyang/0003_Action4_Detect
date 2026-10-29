"""Predicted-mask instance prompts and optional ledger handoff."""
import unittest
import numpy as np
from model.action4_sam_instance_points import predicted_instance_points
from model.action4_sam_session_contracts import RegionSpec,SeedBinding,SessionLedger
class PointTests(unittest.TestCase):
 def test_points_use_unique_foreground_and_background(self):
  mask=np.zeros((50,60),bool);mask[10:30,15:40]=True
  points,labels,trace=predicted_instance_points(mask,[14,9,41,31])
  self.assertEqual(labels,[1,1,0,0]);self.assertFalse(trace["GT_used"])
  for (x,y),label in zip(trace["points_native"],labels):
   self.assertEqual(bool(mask[int(y),int(x)]),bool(label))
  self.assertTrue(np.all((np.asarray(points)>=0)&(np.asarray(points)<=1)))
 def test_no_unique_support_refused(self):
  mask=np.zeros((50,60),bool);mask[10:30,15:40]=True
  with self.assertRaises(ValueError):predicted_instance_points(mask,[14,9,41,31],[[0,0,60,50]])
 def test_point_seed_is_separate_from_text_box_and_binds_log(self):
  region=RegionSpec("r",(0,1),(0,400),(1,11988),"manifest","native",50,50)
  seed=SeedBinding("t",region.identity,"b0",0,0,(.1,.1,.5,.5),("predicted-mask",),"frozen_b0")
  ledger=SessionLedger();record=ledger.plan(region,seed,worker_id="w",device_identity="fixture",dependency_identity="d")
  ledger.begin_create(record.record_id);ledger.created(record.record_id,"session")
  before=ledger.get(record.record_id).prompt_log_identity
  req=ledger.seed_request(record.record_id,points_xy01=[[.2,.2],[.3,.3],[.8,.8],[.9,.9]],
      point_labels=[1,1,0,0],obj_id=0,point_evidence_id="bound-predicted-mask")
  self.assertNotIn("text",req);self.assertNotIn("bounding_boxes",req);self.assertEqual(req["obj_id"],0)
  self.assertNotEqual(before,ledger.get(record.record_id).prompt_log_identity)
  self.assertEqual(ledger.get(record.record_id).active_operation,"seed")
if __name__=="__main__":unittest.main(verbosity=2)
