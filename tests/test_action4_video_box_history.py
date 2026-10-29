"""Five-frame video boxes and image-mode isolation, no model accuracy claim."""
import unittest
from unittest.mock import patch
import numpy as np
from model.action4_video_box_history import VideoBoxHistory
from data.action4_image_mode_contract import ensure_image_mode
class HistoryTests(unittest.TestCase):
 def candidate(self,cid,x,score=.7):return {"candidate_id":cid,"score":score,"bbox_full_xyxy":[x,10,x+20,30]}
 def tracker(self):
  tracker=VideoBoxHistory(mode="video_inference")
  tracker.initialize(0,0,[dict(self.candidate("seed",10),track_id="camera")]);return tracker
 def test_lower_score_near_history_wins(self):
  tracker=self.tracker()
  result=tracker.update(1,1/30,[self.candidate("shirt",100,.99),self.candidate("camera",11,.6)])
  row=result["tracks"][0];self.assertEqual(row["candidate_id"],"camera")
  self.assertTrue(10<row["bbox_for_roi"][0]<11)
  self.assertEqual(row["raw_detection"]["bbox_full_xyxy"][0],11)
 def test_only_five_recent_observations_and_less_jitter(self):
  tracker=self.tracker();xs=[12,9,13,10,14,11,15,12];output=[]
  for i,x in enumerate(xs,1):output.append(tracker.update(i,i/30,[self.candidate(str(i),x)])["tracks"][0]["bbox_for_roi"][0])
  self.assertEqual(len(tracker.tracks["camera"]["history"]),5)
  self.assertLess(np.std(np.diff(output)),np.std(np.diff(xs)))
 def test_missing_is_prediction_then_lost_not_false_detection(self):
  tracker=self.tracker()
  for i in range(1,6):row=tracker.update(i,i/30,[])["tracks"][0]
  self.assertEqual(row["status"],"lost");self.assertIsNone(row["bbox_for_roi"])
  self.assertTrue(row["redetection_required"])
  row=tracker.update(6,.2,[self.candidate("later",10)])["tracks"][0]
  self.assertEqual(row["status"],"lost")
 def test_time_gap_is_not_accepted_as_continuous_track(self):
  tracker=self.tracker();row=tracker.update(20,2,[self.candidate("later",10)])["tracks"][0]
  self.assertEqual(row["status"],"lost")
 def test_sam_jump_does_not_move_box_to_shirt(self):
  tracker=self.tracker();row=tracker.update(1,1/30,[self.candidate("camera",11)],sam_support_boxes={"camera":[100,100,140,140]})["tracks"][0]
  self.assertEqual(row["sam_support"],"history_conflict_ignored");self.assertLess(row["bbox_for_roi"][0],12)
 def pair_tracker(self):
  tracker=VideoBoxHistory(mode="video_inference")
  tracker.initialize(0,0,[dict(self.candidate("seedL",10),track_id="left"),
                          dict(self.candidate("seedR",28),track_id="right")])
  return tracker
 def test_close_overlapping_distinct_candidates_keep_two_ids(self):
  tracker=self.pair_tracker()
  out=tracker.update(1,1/30,[self.candidate("right",28),self.candidate("left",10)])
  self.assertEqual({t["track_id"]:t["candidate_id"] for t in out["tracks"]},{"left":"left","right":"right"})
  self.assertTrue(all(t["status"]=="observed" for t in out["tracks"]))
 def test_one_large_box_covering_both_is_not_one_confirmed_camera(self):
  tracker=self.pair_tracker()
  out=tracker.update(1,1/30,[{"candidate_id":"merged","score":.99,"bbox_full_xyxy":[9,9,49,31]}])
  self.assertTrue(all(t["status"]!="observed" for t in out["tracks"]))
 def test_single_candidate_never_copied_to_two_tracks(self):
  tracker=self.pair_tracker();out=tracker.update(1,1/30,[self.candidate("left",10)])
  assigned=[t["candidate_id"] for t in out["tracks"] if t["candidate_id"] is not None]
  self.assertEqual(assigned,["left"])
 def test_current_observation_at_fifth_frame_is_checked_before_lost(self):
  tracker=self.tracker()
  for i in range(1,5):tracker.update(i,i/30,[])
  row=tracker.update(5,5/30,[self.candidate("returned",11,.8)])["tracks"][0]
  self.assertEqual(row["status"],"observed")
  self.assertEqual(row["candidate_id"],"returned")
 def test_image_training_and_eval_refuse_video_state(self):
  with self.assertRaises(ValueError):VideoBoxHistory(mode="image_training")
  ensure_image_mode({"sam_enabled":True}) # per-image SAM remains legal
  for cfg in [{"video_tracking":{"enabled":True}},{"cotracker_enabled":True},{"sam_video_enabled":True}]:
   with self.assertRaises(ValueError):ensure_image_mode(cfg)
  from data.action4_stage2_data import Stage2Units
  from model.training.action4_stage2_core import construct
  from eval.action4_stage2_eval import evaluate
  cfg={"video_tracking":{"enabled":True}}
  with patch("data.action4_stage2_data.ComponentData",side_effect=AssertionError("dataset must not load")):
   with self.assertRaises(ValueError):Stage2Units(cfg)
  with self.assertRaises(ValueError):construct(cfg,"cpu")
  with self.assertRaises(ValueError):evaluate(cfg,None,None,None,"cpu")
if __name__=="__main__":unittest.main(verbosity=2)
