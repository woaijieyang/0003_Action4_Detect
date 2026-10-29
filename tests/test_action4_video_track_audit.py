import unittest,copy
from eval.action4_video_track_audit import audit
class AuditTests(unittest.TestCase):
 def row(self):
  return {"frame":{"global_frame_id":9,"pts":3600},"fused":{"frame_id":9,"mode":"video_inference","tracks":[
   {"track_id":"left","status":"observed","bbox_for_roi":[0,0,10,10],"candidate_id":"b0:frame9:grid:4",
    "raw_detection":{"candidate_id":"b0:frame9:grid:4","grid_index":4},"sam_support":"unavailable"}]},"sam_evidence":{}}
 def test_valid_is_not_accuracy(self):
  r=self.row();out=audit([r],[r["frame"]])
  self.assertEqual(out["counts"]["one_observed"],1);self.assertTrue(out["two_observed_is_not_two_correct"])
 def test_missing_frame_rejected(self):
  with self.assertRaises(ValueError):audit([],[self.row()["frame"]])
 def test_copied_detection_rejected(self):
  r=self.row();p=copy.deepcopy(r["fused"]["tracks"][0]);p["track_id"]="right";r["fused"]["tracks"].append(p)
  with self.assertRaisesRegex(ValueError,"multiple"):audit([r],[r["frame"]])
 def test_wrong_time_rejected(self):
  r=self.row()
  with self.assertRaises(ValueError):audit([r],[{"global_frame_id":9,"pts":3601}])
 def test_copied_mask_rejected(self):
  r=self.row();p=copy.deepcopy(r["fused"]["tracks"][0]);p["track_id"]="right"
  p["candidate_id"]="b0:frame9:grid:5";p["raw_detection"]={"candidate_id":p["candidate_id"],"grid_index":5}
  r["fused"]["tracks"].append(p)
  for p in r["fused"]["tracks"]:
   p["sam_support"]="used_for_association_only";r["sam_evidence"][p["track_id"]]={"pixel_sha256":"a"*64}
  with self.assertRaisesRegex(ValueError,"Copied mask"):audit([r],[r["frame"]])
