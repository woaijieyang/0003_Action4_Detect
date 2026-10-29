import unittest,copy
import numpy as np
from data.action4_video_roi_context import VideoROIContext
class ContextTests(unittest.TestCase):
 def setUp(self):
  self.rgb=np.zeros((3,8,10),np.uint8);self.K=np.eye(3)
  self.row={"frame":{"global_frame_id":9},"fused":{"mode":"video_inference","frame_id":9,"tracks":[
   {"track_id":"left","status":"observed","candidate_id":"b0:frame9:grid:4","bbox_for_roi":[1,2,7,6],
    "raw_detection":{"grid_index":4,"candidate_id":"b0:frame9:grid:4","score":.8,"bbox_full_xyxy":[1,1,8,7]}}]}}
 def build(self):return VideoROIContext.from_replay(self.row,self.rgb,self.K,b0_sha256="a"*64,replay_identity="b"*64)
 def test_consumes_smoothed_box_not_raw(self):
  c=self.build();d=c.consume(self.rgb,self.K,b0_sha256="a"*64)[0]
  self.assertEqual(d["bbox_full_xyxy"],[1.,2.,7.,6.]);self.assertEqual(d["raw_bbox_full_xyxy"],[1,1,8,7])
 def test_rgb_mismatch(self):
  c=self.build();self.rgb[0,0,0]=1
  with self.assertRaises(ValueError):c.consume(self.rgb,self.K,b0_sha256="a"*64)
 def test_K_mismatch(self):
  c=self.build()
  with self.assertRaises(ValueError):c.consume(self.rgb,self.K*2,b0_sha256="a"*64)
 def test_model_mismatch(self):
  with self.assertRaises(ValueError):self.build().consume(self.rgb,self.K,b0_sha256="c"*64)
 def test_wrong_frame_candidate(self):
  self.row["fused"]["tracks"][0]["candidate_id"]="b0:frame8:grid:4"
  with self.assertRaises(ValueError):self.build()
 def test_missing_observation_not_fabricated(self):
  self.row["fused"]["tracks"][0]["status"]="lost"
  self.assertEqual(self.build().consume(self.rgb,self.K,b0_sha256="a"*64),[])
