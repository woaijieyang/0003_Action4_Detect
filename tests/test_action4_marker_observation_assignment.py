import unittest
import numpy as np
from model.action4_marker_observation_assignment import assign_markers
class MarkerOwnership(unittest.TestCase):
 def setUp(self):
  a=np.zeros((80,100),bool);a[5:50,5:45]=True
  b=np.zeros_like(a);b[5:50,55:95]=True
  self.m={"a":a,"b":b};self.q=dict(marker_id=18,corners_native=[[60,10],[80,10],[80,30],[60,30]])
  self.kw=dict(minimum_coverage=.8,minimum_margin=.2,duplicate_corner_tolerance_px=2.)
 def test_overlapping_crops_are_one_observation(self):
  r=assign_markers(4,[dict(self.q,crop_track="a"),dict(self.q,crop_track="b")],self.m,{"a":19,"b":18},**self.kw)
  self.assertEqual(r["unique_observation_count"],1);self.assertEqual(r["observations"][0]["track_id"],"b")
  self.assertEqual(r["observations"][0]["duplicate_crop_observation_count"],1)
 def test_same_id_distinct_locations_and_shared_masks_remain_unknown(self):
  q=dict(self.q,corners_native=[[10,10],[30,10],[30,30],[10,30]])
  r=assign_markers(4,[q,self.q],self.m,{"a":18,"b":18},**self.kw)
  self.assertTrue(all(x["track_id"] is None for x in r["observations"]))
  r=assign_markers(4,[self.q],{"a":self.m["b"],"b":self.m["b"]},{"a":19,"b":18},**self.kw)
  self.assertEqual(r["observations"][0]["reason"],"mixed_or_duplicate_body_support")
 def test_identity_conflict_and_missing_support_not_absence(self):
  r=assign_markers(4,[self.q],self.m,{"a":18,"b":19},**self.kw)
  self.assertEqual(r["observations"][0]["reason"],"decoded_ID_conflicts_with_registered_body")
  r=assign_markers(4,[self.q],{},{"a":19,"b":18},**self.kw)
  self.assertEqual(r["observations"][0]["reason"],"current_native_support_missing")
