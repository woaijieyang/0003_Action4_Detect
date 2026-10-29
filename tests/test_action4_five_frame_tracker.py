import unittest
from model.action4_five_frame_tracker import FiveFrameTracker
class TrackerTests(unittest.TestCase):
 def candidate(self,f,x=200,name="camera",score=.95,q=.9):
  return {"candidate_id":f"{name}:{f}","score":score,"refined_quality_score":q,"bbox_full_xyxy":[x,10,x+20,30]}
 def tracker(self,two=False):
  t=FiveFrameTracker();seeds=[{"track_id":"left","bbox_full_xyxy":[10,10,30,30]}]
  if two:seeds.append({"track_id":"right","bbox_full_xyxy":[28,10,48,30]})
  t.initialize(0,0,seeds)
  for f in range(1,6):t.update(f,f/30,[])
  return t
 def test_ten_frame_history_is_explicit_and_bounded(self):
  t=FiveFrameTracker(window=10);t.initialize(0,0,[{"track_id":"left","bbox_full_xyxy":[10,10,30,30]}])
  for f in range(1,15):t.update(f,f/30,[self.candidate(f,10)])
  self.assertEqual(len(t.core.tracks["left"]["history"]),10)
  with self.assertRaises(ValueError):FiveFrameTracker(window=20)
 def test_lost_recovers_only_after_five_new_observations(self):
  t=self.tracker()
  for f in range(6,10):
   r=t.update(f,f/30,[self.candidate(f)])
   self.assertEqual(r["tracks"][0]["status"],"lost")
  r=t.update(10,10/30,[self.candidate(10)])
  self.assertEqual(r["tracks"][0]["status"],"observed")
  self.assertEqual(r["tracks"][0]["association_epoch"],1)
  self.assertEqual(r["tracks"][0]["bbox_for_roi"],[200,10,220,30])
 def test_weak_frame_breaks_reacquisition(self):
  t=self.tracker()
  for f in range(6,13):
   r=t.update(f,f/30,[self.candidate(f,score=.8 if f==9 else .95)])
  self.assertEqual(r["tracks"][0]["status"],"lost")
 def test_no_target_stays_lost(self):
  t=self.tracker()
  for f in range(6,15):r=t.update(f,f/30,[])
  self.assertTrue(all(x["status"]=="lost" for x in r["tracks"]))
 def test_one_candidate_not_copied_to_two_slots(self):
  t=self.tracker(two=True)
  for f in range(6,11):r=t.update(f,f/30,[self.candidate(f)])
  observed=[x for x in r["tracks"] if x["status"]=="observed"]
  self.assertEqual(len(observed),1)
 def test_two_close_overlapping_new_candidates_stay_separate(self):
  t=self.tracker(two=True)
  for f in range(6,11):
   r=t.update(f,f/30,[self.candidate(f,200,"a"),self.candidate(f,210,"b")])
  observed=[x for x in r["tracks"] if x["status"]=="observed"]
  self.assertEqual(len(observed),2)
  self.assertEqual(len({x["candidate_id"] for x in observed}),2)
 def test_large_shared_box_not_reinitialized_as_one_camera(self):
  t=self.tracker(two=True)
  for f in range(6,11):
   c=self.candidate(f);c["bbox_full_xyxy"]=[9,9,49,31];r=t.update(f,f/30,[c])
  self.assertTrue(all(x["status"]=="lost" for x in r["tracks"]))
