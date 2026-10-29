import unittest
import numpy as np
from model.action4_sam_supported_boxes import SamSupportedBoxes
class SupportedTests(unittest.TestCase):
 def masks(self):
  a=np.zeros((50,70),bool);a[10:20,10:20]=True
  b=np.zeros_like(a);b[10:20,35:45]=True
  return {"a":a,"b":b}
 def tracker(self):return SamSupportedBoxes([{"track_id":"a","bbox_native":[8,8,22,22]},{"track_id":"b","bbox_native":[33,8,47,22]}])
 def test_current_SAM_keeps_two_IDs_when_B0_has_no_candidate(self):
  t=self.tracker();r=t.update(0,0,[],self.masks())
  self.assertEqual([x["status"] for x in r["tracks"]],["sam_supported","sam_supported"])
  self.assertEqual({x["track_id"] for x in r["tracks"]},{"a","b"})
  self.assertNotEqual(r["tracks"][0]["bbox_for_roi"],[10,10,20,20])
 def test_shirt_candidate_does_not_override_instance_mask(self):
  t=self.tracker();c={"candidate_id":"shirt","score":.99,"refined_quality_score":.99,"bbox_full_xyxy":[45,30,65,45]}
  r=t.update(0,0,[c],self.masks())
  self.assertTrue(all(x["candidate_id"] is None for x in r["tracks"]))
 def test_shared_large_box_not_assigned_to_both(self):
  t=self.tracker();c={"candidate_id":"merged","score":.99,"refined_quality_score":.99,"bbox_full_xyxy":[8,8,47,22]}
  r=t.update(0,0,[c],self.masks());self.assertTrue(all(x["candidate_id"] is None for x in r["tracks"]))
 def test_copied_mask_is_not_two_independent_supports(self):
  t=self.tracker();m=self.masks();m["b"]=m["a"]
  r=t.update(0,0,[],m);self.assertTrue(all(not x["sam_current_support"] for x in r["tracks"]))
 def test_empty_masks_eventually_lost_not_fabricated(self):
  t=self.tracker()
  for f in range(6):r=t.update(f,f/30,[],{})
  self.assertTrue(all(x["status"]=="lost" for x in r["tracks"]))

class SizeStabilityTests(unittest.TestCase):
 def test_visible_mask_aspect_does_not_rescale_full_body_box(self):
  masks={"a":np.zeros((50,70),bool)};masks["a"][10:20,10:20]=True
  tracker=SamSupportedBoxes([{"track_id":"a","bbox_native":[8,8,22,22]}])
  tracker.update(0,0,[],masks)
  thin=np.zeros((50,70),bool);thin[13:16,0:30]=True
  row=tracker.update(1,1/30,[],{"a":thin})["tracks"][0]
  self.assertTrue(row["sam_current_support"])
  box=np.array(row["bbox_for_roi"])
  np.testing.assert_allclose(box[2:]-box[:2],[14,14])
class InvalidSupportedTests(unittest.TestCase):
 def test_nonfinite_candidate_rejected(self):
  t=SamSupportedBoxes([{"track_id":"a","bbox_native":[0,0,10,10]}])
  with self.assertRaises(ValueError):t.update(0,0,[{"candidate_id":"bad","score":.9,"bbox_full_xyxy":[0,0,float("nan"),10]}],{})
 def test_nonfinite_time_rejected(self):
  t=SamSupportedBoxes([])
  with self.assertRaises(ValueError):t.update(0,float("nan"),[],{})
