import copy,unittest
from types import SimpleNamespace
from test_action4_video_pose_stabilizer import packet,K,CAD
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from eval.action4_video_stabilization import validate_processed_row
from eval.action4_staged_pipeline import StagedPipeline
class IntegrationTests(unittest.TestCase):
 def make(self):
  pipe=object.__new__(StagedPipeline);pipe.a0_kind="joint_pose"
  return pipe
 def test_existing_video_entry_uses_real_returned_pose(self):
  pipe=self.make();s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(7):
   raw=packet(f,angle=3.14 if f==6 else 0);pipe.predict=lambda *args,**kw:copy.deepcopy(raw)
   out=pipe.predict_video(None,K,CAD,video_context=SimpleNamespace(frame_id=f),pose_stabilizer=s,
    frame_clock=dict(frame_id=f,pts=f,time_base=[1,30]))
  self.assertNotEqual(out["instances"][0]["corners_final"],raw["instances"][0]["corners_final"])
  self.assertEqual(out["instances"][0]["corners_raw_from_resolved_roi"],raw["instances"][0]["corners_raw_from_resolved_roi"])
 def test_off_path_is_exact(self):
  pipe=self.make();raw=packet(0);pipe.predict=lambda *a,**k:raw
  self.assertIs(pipe.predict_video(None,K,CAD,video_context=SimpleNamespace(frame_id=0)),raw)
 def test_mismatched_clock_rejected_before_forward(self):
  pipe=self.make();pipe.predict=lambda *a,**k:self.fail("Forward must not run")
  with self.assertRaises(ValueError):pipe.predict_video(None,K,CAD,video_context=SimpleNamespace(frame_id=0),
   pose_stabilizer=VideoPoseStabilizer(StabilizerConfig()),frame_clock=dict(frame_id=1,pts=1,time_base=[1,30]))
 def test_saved_row_rejects_render_geometry_or_roi_tampering(self):
  raw=dict(frame_id=0,pts=0,RGB_sha256="a"*64,roi_expansion=1.5,prediction=packet(0))
  s=VideoPoseStabilizer(StabilizerConfig());good=copy.deepcopy(raw)
  good["prediction"]=s.apply_prediction(raw["prediction"],frame_id=0,pts=0,time_base=[1,30],K_native=K,canonical_m=CAD,mode="video")
  self.assertTrue(validate_processed_row(raw,good,K,CAD,s.config.identity))
  bad=copy.deepcopy(good);bad["prediction"]["instances"][0]["corners_final"][0][0]+=1
  with self.assertRaises(ValueError):validate_processed_row(raw,bad,K,CAD,s.config.identity)
  bad=copy.deepcopy(good);bad["prediction"]["instances"][0]["bbox_for_a0"][0]+=1
  with self.assertRaises(ValueError):validate_processed_row(raw,bad,K,CAD,s.config.identity)
