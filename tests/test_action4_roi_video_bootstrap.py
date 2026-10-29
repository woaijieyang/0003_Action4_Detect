import unittest
from types import SimpleNamespace
import numpy as np
import torch
from eval.action4_roi_video import bootstrap_context
class Bootstrap(unittest.TestCase):
 def test_actual_B0_only_and_no_old_corners_consumed(self):
  provider=SimpleNamespace(rows=[{'frame':{'global_frame_id':0,'pts':0}}],region={'time_base':[1,11988]},b0_sha='a'*64)
  rgb=torch.zeros((3,50,60),dtype=torch.uint8);K=np.eye(3)
  base={'prediction':{'detections':[{'grid_index':7,'score':.9,'bbox_full_xyxy':[10,10,30,40]}],'instances':'not_consumed'}}
  seeds=[{'track_id':'track_00','evidence_ids':['b0:frame0:grid:7']},{'track_id':'track_01','evidence_ids':['b0:frame1:grid:8']}]
  c=bootstrap_context(provider,0,rgb,K,base,seeds,'b'*64);rows=c.consume(rgb,K,b0_sha256='a'*64)
  self.assertEqual(len(rows),1);self.assertEqual(rows[0]['track_id'],'track_00')
  self.assertEqual(rows[0]['bbox_for_a0'],[10.,10.,30.,40.])
 def test_absent_detection_never_filled_from_later_frame(self):
  provider=SimpleNamespace(rows=[{'frame':{'global_frame_id':0,'pts':0}}],region={'time_base':[1,11988]},b0_sha='a'*64)
  rgb=torch.zeros((3,50,60),dtype=torch.uint8);K=np.eye(3)
  c=bootstrap_context(provider,0,rgb,K,{'prediction':{'detections':[]}},[{'track_id':'a','evidence_ids':['b0:frame0:grid:7']}],'b'*64)
  self.assertEqual(c.consume(rgb,K,b0_sha256='a'*64),[])
