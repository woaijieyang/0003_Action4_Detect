import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from eval.action4_roi_pair_inference import predict_core_frame_pair
from data.action4_video_roi_context import VideoROIContext
from test_action4_sam_supported_roi import fixture,context,DummyPipeline

class PairInference(unittest.TestCase):
 def fixture(self):
  torch.set_num_threads(1)
  f=fixture();rgb=torch.from_numpy(f[1]);ctx=context(f)
  candidate={"score":.9,"grid_index":5,"bbox_full_xyxy":[4.,10.,34.,40.]}
  frame={"global_frame_id":4,"local_index":0,"pts":1600,"sha256":"c"*64}
  replay=SimpleNamespace(clip={"evaluation_frame_ids":[4]},causal=True,by_global={4:0},
   doc={"frames":[frame]},pools=[{"frame":frame,"pool":{"single_frame_selected":[candidate]}}],
   status="native_complete_not_accuracy",load=lambda fid,K:(rgb,ctx),
   region=SimpleNamespace(pixel_identity="native_integer_centers"),
   cfg={"b0":{"sha256":"a"*64,"scoring_policy":"center"}},b0_pool_identity="e"*64,
   rows={4:{"frame":frame,"native_owners":{"track_00":{"session_id":"session-A","obj_id":0}}}})
  probe=DummyPipeline();probe.pipeline.b0_scoring_policy="center"
  def forbidden(*args):raise AssertionError("B0 must not rerun")
  probe.pipeline._detect=forbidden
  return f,replay,probe,candidate
 def run_pair(self,f,replay,probe):
  with patch("eval.action4_surface_snapshot.model_identities",return_value={"B0":{"sha256":"a"*64}}),patch(
   "model.action4_geometry_forward.instancewise_geometry",side_effect=probe.geometry):
   return predict_core_frame_pair(probe.pipeline,replay,4,f[2],np.zeros((8,3)))
 def test_same_frozen_pool_and_actual_different_ROI_consumption(self):
  f,replay,probe,candidate=self.fixture();before=copy.deepcopy(candidate);out=self.run_pair(f,replay,probe)
  self.assertEqual(candidate,before);self.assertEqual(len(probe.rois),2)
  self.assertFalse(torch.equal(probe.rois[0],probe.rois[1]));self.assertFalse(torch.equal(probe.K[0],probe.K[1]))
  self.assertEqual(out["b0_only"]["prediction"]["instances"][0]["bbox_for_a0"],before["bbox_full_xyxy"])
  self.assertEqual(out["track_assisted"]["prediction"]["instances"][0]["bbox_for_a0"],[10.,12.,40.,42.])
  self.assertEqual(out["b0_only"]["frozen_B0_pool_identity"],out["track_assisted"]["frozen_B0_pool_identity"])
 def test_identical_box_has_identical_tensor_and_K(self):
  f,replay,probe,candidate=self.fixture();candidate["bbox_full_xyxy"]=[10.,12.,40.,42.]
  self.run_pair(f,replay,probe)
  self.assertTrue(torch.equal(probe.rois[0],probe.rois[1]));self.assertTrue(torch.equal(probe.K[0],probe.K[1]))
 def test_initializing_no_A0_track_but_B0_baseline_kept(self):
  f,replay,probe,candidate=self.fixture()
  row=copy.deepcopy(f[0]);row["fused"]["tracks"]=[]
  ctx=VideoROIContext.from_replay(row,torch.from_numpy(f[1]),f[2],b0_sha256="a"*64,replay_identity="b"*64)
  replay.load=lambda fid,K:(torch.from_numpy(f[1]),ctx)
  out=self.run_pair(f,replay,probe)
  self.assertEqual(len(probe.rois),1);self.assertEqual(len(out["b0_only"]["prediction"]["instances"]),1)
  self.assertEqual(out["track_assisted"]["prediction"]["instances"],[])
 def test_causal_and_core_boundary_enforced(self):
  f,replay,probe,_=self.fixture();replay.causal=False
  with self.assertRaises(ValueError):self.run_pair(f,replay,probe)
  replay.causal=True;replay.clip["evaluation_frame_ids"]=[]
  with self.assertRaises(ValueError):self.run_pair(f,replay,probe)
