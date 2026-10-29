import unittest,copy
from eval.action4_native_candidate_bridge import join_candidate_rows

class CandidateBridgeTests(unittest.TestCase):
 def fixture(self):
  pose=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_m=[0,0,1])
  corners=[[1,2]]*8
  inst=dict(track_id="a",raw_network_pose=pose,corners_raw_from_resolved_roi=corners,
    corners_final=corners,bbox_b0_raw=[0,0,10,10],roi_decision_id="roi1",post_A0_geometry="none")
  base=[dict(frame_id=4,pts=1600,RGB_sha256="a"*64,prediction=dict(instances=[inst]))]
  source=dict(path="frame.png",sha256="a"*64)
  record=dict(frame_id=4,track_id="a",pts=1600,time_base=[1,11988],source_RGB=source,
    raw_baseline_pose=pose,raw_baseline_corners=corners,final_corners=corners,
    source_B0_box=[0,0,10,10],published=False,status="native_numeric_invalid",native_call_ok=True,
    SAM_consumed_by_native=True,native_pose=[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]],
    numeric_error="numeric_rejected",corners_candidate=None)
  manifest=dict(tracks=["a"],rows=[dict(frame_id=4,pts=1600,time_base=[1,11988],source_png=source)])
  return base,[record],manifest
 def test_invalid_stays_invalid_and_raw_preserved(self):
  b,n,m=self.fixture();old=copy.deepcopy(b);out=join_candidate_rows(b,n,m)
  self.assertEqual(b,old)
  candidate=out[0]["prediction"]["instances"][0].pop("native_tracking_candidate")
  self.assertEqual(out,b);self.assertFalse(candidate["numerical_candidate_eligible"])
  self.assertFalse(candidate["deployment_admitted"])
 def test_wrong_roi_output_or_frame_rejected(self):
  for key in ["pts","source_B0_box","raw_baseline_corners"]:
   b,n,m=self.fixture()
   n[0][key]=1601 if key=="pts" else [99]
   with self.assertRaises(ValueError):join_candidate_rows(b,n,m)
 def test_partial_or_duplicate_native_domain_rejected(self):
  b,n,m=self.fixture()
  for records in [[],n+n]:
   with self.assertRaises(ValueError):join_candidate_rows(b,records,m)
 def test_withheld_numeric_candidate_cannot_enter_eligible(self):
  b,n,m=self.fixture();n[0]["status"]="native_numeric_candidate_not_accuracy";n[0]["recovery_decision"]={"action":"withhold"}
  out=join_candidate_rows(b,n,m)
  self.assertFalse(out[0]["prediction"]["instances"][0]["native_tracking_candidate"]["numerical_candidate_eligible"])
