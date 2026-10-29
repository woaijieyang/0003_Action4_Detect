import json,unittest
import numpy as np
from pathlib import Path
from test_action4_video_pose_stabilizer import packet,apply
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig,alpha
from model.action4_native_rotation_history import rotation_log,rotation_exp
from data.action4_paths import PROJECT_ROOT as R
class SourceClockTests(unittest.TestCase):
 def test_V1_filter_identity_remains_readable(self):
  cfg=json.loads((R/"configs/model/action4_video_stable_v1.json").read_text())
  receipt=json.loads((R/"outputs/action4_stage5_stable_v1/STABILIZATION_RECEIPT.json").read_text())
  self.assertEqual(StabilizerConfig(**cfg["filter"]).identity,receipt["filter_id"])
 def test_source_skip_uses_actual_measurement_interval(self):
  s=VideoPoseStabilizer(StabilizerConfig(rotation_speed_clock="selected_measurement"))
  for f in range(7):o=apply(s,packet(f,angle=0 if f<4 else .3),f)["instances"][0]
  meta=o["pose_stabilization"]
  self.assertGreater(meta["rotation_speed_source_dt_seconds"],meta["dt_seconds"])
  self.assertAlmostEqual(s.states["a"]["omega"][2],alpha(1,1/30)*.3/meta["rotation_speed_source_dt_seconds"],places=10)
 def test_source_reference_never_moves_backwards(self):
  s=VideoPoseStabilizer(StabilizerConfig(rotation_speed_clock="selected_measurement"))
  rng=np.random.default_rng(51);times=[]
  for f in range(100):
   apply(s,packet(f,angle=float(rng.normal())),f);times.append(s.states["a"]["last_forward_time"])
  self.assertTrue(np.all(np.diff(times)>=0))
 def test_translation_and_target_selection_unchanged(self):
  a=VideoPoseStabilizer(StabilizerConfig());b=VideoPoseStabilizer(StabilizerConfig(rotation_speed_clock="selected_measurement"))
  rng=np.random.default_rng(21)
  for f in range(80):
   p=packet(f,angle=float(rng.normal()),center=500+f,size=100+.1*f,depth=1+.01*np.sin(f))
   x=apply(a,p,f)["instances"][0];y=apply(b,p,f)["instances"][0]
   self.assertEqual(x["final_pose"]["translation_m"],y["final_pose"]["translation_m"])
   self.assertEqual(x["pose_stabilization"]["selected_rotation_frame"],y["pose_stabilization"]["selected_rotation_frame"])
 def test_same_fast_turn_fixture_gate(self):
  s=VideoPoseStabilizer(StabilizerConfig(rotation_speed_clock="selected_measurement"));errors=[]
  for f in range(60):
   r=rotation_exp([0,0,2*f/30]);o=apply(s,packet(f,angle=2*f/30),f)["instances"][0]
   errors.append(np.linalg.norm(rotation_log(r@np.asarray(o["final_pose"]["rotation"]).T)))
  self.assertLess(max(errors[20:]),.3)
