import copy,unittest
from eval.action4_reanchor_roi_provider import prediction_calibration_provenance
from data.action4_video_roi_context import VideoROIContext
from test_action4_sam_supported_roi import fixture,observed
class Provenance(unittest.TestCase):
 def test_prediction_calibration_does_not_become_trusted_extent(self):
  f=fixture();row=observed(f[0]);row["fused"]["tracks"][0]["calibration_frames"]=[3]
  rows=[{"frame":{"global_frame_id":3,"pts":1200}},row]
  row["_roi_last_prediction_calibration"]=prediction_calibration_provenance(row,rows)
  c=VideoROIContext.from_replay(row,f[1],f[2],b0_sha256="a"*64,replay_identity="b"*64)
  p=c.consume(f[1],f[2],b0_sha256="a"*64)[0]
  self.assertIsNone(p["last_trusted_full_frame_id"]);self.assertEqual(p["last_prediction_calibration"]["frame_id"],3)
  self.assertFalse(p["last_prediction_calibration"]["verified_full_body_extent"])
 def test_future_or_unknown_calibration_rejected(self):
  for value in [5,3]:
   row=fixture()[0];row["fused"]["tracks"][0]["calibration_frames"]=[value]
   with self.assertRaises(ValueError):prediction_calibration_provenance(row,[row])
 def test_nonzero_global_frame_mapping_not_list_index(self):
  row={"frame":{"global_frame_id":822,"pts":328800},"fused":{"tracks":[{"track_id":"t","calibration_frames":[820]}]}}
  r=prediction_calibration_provenance(row,[{"frame":{"global_frame_id":820,"pts":328000}},row])
  self.assertEqual(r["t"]["pts"],328000)
