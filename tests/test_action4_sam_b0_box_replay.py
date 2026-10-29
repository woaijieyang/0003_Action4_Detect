import unittest
import numpy as np
from eval.action4_sam_b0_box_replay import support_box
class SupportTests(unittest.TestCase):
 def test_continuous_bounds(self):
  mask=np.zeros((4,5),bool);mask[1:3,2:4]=True
  self.assertEqual(support_box(mask,(4,5)),[2.,1.,4.,3.])
 def test_wrong_shape(self):
  with self.assertRaises(ValueError):support_box(np.ones((3,4),bool),(4,3))
 def test_wrong_dtype(self):
  with self.assertRaises(ValueError):support_box(np.ones((3,4),np.float32),(3,4))
 def test_empty_not_absent(self):
  with self.assertRaises(ValueError):support_box(np.zeros((3,4),bool),(3,4))

class PrefixTests(unittest.TestCase):
 def row(self,fid):
  p={"grid_index":7,"score":.95,"bbox_full_xyxy":[1,2,6,8]}
  return {"frame":{"global_frame_id":fid},"pool":{"association_candidates":[p],"single_frame_selected":[p]}}
 def seeds(self):return [{"track_id":"left","candidate_id":"b0:frame4:grid:7","global_frame_id":4,"bbox_native":[1,2,6,8]}]
 def test_preconfirmation_keeps_frame_without_fake_track(self):
  from eval.action4_sam_b0_box_replay import prefix_record
  r=prefix_record(self.row(0),self.seeds(),4)
  self.assertEqual(r["frame"]["global_frame_id"],0);self.assertEqual(r["fused"]["tracks"],[])
  self.assertEqual(r["fused"]["initialization_status"],"unconfirmed")
  self.assertEqual(len(r["fused"]["baseline_detection_fallback"]),1)
 def test_seed_uses_current_candidate(self):
  from eval.action4_sam_b0_box_replay import prefix_record
  r=prefix_record(self.row(4),self.seeds(),4)
  self.assertEqual(r["fused"]["tracks"][0]["candidate_id"],"b0:frame4:grid:7")
 def test_prefix_cannot_replace_later_tracking(self):
  from eval.action4_sam_b0_box_replay import prefix_record
  with self.assertRaises(ValueError):prefix_record(self.row(5),self.seeds(),4)
 def test_wrong_seed_not_silently_remapped(self):
  from eval.action4_sam_b0_box_replay import prefix_record
  s=self.seeds();s[0]["bbox_native"]=[0,0,10,10]
  with self.assertRaises(ValueError):prefix_record(self.row(4),s,4)

class IndependentSupportTests(unittest.TestCase):
 def test_copied_pixels_under_different_ids_are_not_two_supports(self):
  from eval.action4_sam_b0_box_replay import independent_supports
  supports={"left":[0,0,10,10],"right":[0,0,10,10]}
  evidence={"left":{"pixel_sha256":"a"*64,"mask_identity":"one"},
            "right":{"pixel_sha256":"a"*64,"mask_identity":"two"}}
  accepted,rejected=independent_supports(supports,evidence)
  self.assertEqual(accepted,{})
  self.assertEqual({r["track_id"] for r in rejected},{"left","right"})
 def test_overlapping_boxes_with_independent_pixels_are_not_merged(self):
  from eval.action4_sam_b0_box_replay import independent_supports
  supports={"left":[0,0,10,10],"right":[1,0,11,10]}
  accepted,rejected=independent_supports(supports,{"left":{"pixel_sha256":"a"*64},"right":{"pixel_sha256":"b"*64}})
  self.assertEqual(accepted,supports);self.assertEqual(rejected,[])
 def test_missing_pixel_provenance_is_rejected(self):
  from eval.action4_sam_b0_box_replay import independent_supports
  with self.assertRaises(ValueError):independent_supports({"left":[0,0,1,1]},{"left":{}})

class PublishedBaselineTests(unittest.TestCase):
 def test_selector_low_score_is_not_published(self):
  from eval.action4_sam_b0_box_replay import published_baseline
  pool={"single_frame_selected":[{"score":.96},{"score":.2898503541946411}]}
  self.assertEqual(len(published_baseline(pool)),1)
  self.assertEqual(len(pool["single_frame_selected"]),2)
 def test_exact_threshold_is_published(self):
  from eval.action4_sam_b0_box_replay import published_baseline
  self.assertEqual(len(published_baseline({"single_frame_selected":[{"score":.3}]})),1)

class OrderedDisplayTests(unittest.TestCase):
 def test_prefetch_keeps_order_and_values(self):
  from eval.action4_sam_b0_box_replay import ordered_display
  self.assertEqual(list(ordered_display(range(12),4,loader=lambda x:x*x)),[(x,x*x) for x in range(12)])
 def test_failed_decode_is_not_a_successful_empty_frame(self):
  from eval.action4_sam_b0_box_replay import ordered_display
  def load(x):
   if x==3:raise ValueError("bad image")
   return x
  with self.assertRaisesRegex(ValueError,"bad image"):list(ordered_display(range(8),2,loader=load))
 def test_worker_budget(self):
  from eval.action4_sam_b0_box_replay import ordered_display
  with self.assertRaises(ValueError):list(ordered_display([],100))
