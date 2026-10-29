import copy,unittest
import numpy as np
from test_action4_video_pose_stabilizer import packet,apply,CAD,K
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_native_rotation_history import rotation_log,rotation_exp
class BoundedTests(unittest.TestCase):
 def test_selected_branch_is_consumed_and_raw_is_preserved(self):
  p=packet(0);i=p['instances'][0];raw=copy.deepcopy(i['raw_network_pose'])
  i['post_A0_geometry']='corner_candidate_research_only';i['final_pose']=dict(raw,rotation=rotation_exp([0,0,.2]).tolist())
  s=VideoPoseStabilizer(StabilizerConfig(measurement_source='selected_A0_pose'))
  o=apply(s,p,0)['instances'][0]
  np.testing.assert_allclose(o['final_pose']['rotation'],rotation_exp([0,0,.2]),atol=1e-10)
  self.assertEqual(o['raw_network_pose'],raw);self.assertEqual(p['instances'][0]['raw_network_pose'],raw)
 def test_persistent_large_innovation_is_rate_bounded(self):
  s=VideoPoseStabilizer(StabilizerConfig(rotation_max_cutoff_hz=3,rotation_max_speed_deg_s=180))
  last=None;limited=0
  for f in range(50):
   o=apply(s,packet(f,angle=0 if f<5 else 2.6),f)['instances'][0];r=np.array(o['final_pose']['rotation'])
   if last is not None:self.assertLessEqual(np.linalg.norm(rotation_log(r@last.T))*180/np.pi,6+1e-7)
   last=r;limited+=o['pose_stabilization']['rotation_rate_limited']
  self.assertGreater(limited,0)
 def test_depth_jump_is_bounded_without_deforming_cuboid(self):
  s=VideoPoseStabilizer(StabilizerConfig(coordinate_max_speed=(.6,.6,1.2)));last=None
  for f in range(30):
   o=apply(s,packet(f,depth=1 if f<5 else 3),f)['instances'][0];t=np.array(o['final_pose']['translation_m']);r=np.array(o['final_pose']['rotation'])
   if last is not None:self.assertLessEqual(abs(np.log(t[2]/last)),1.2/30+1e-9)
   last=t[2];camera=CAD@r.T+t;h=camera@K.T
   np.testing.assert_allclose(h[:,:2]/h[:,2:],o['corners_final'],atol=1e-8)
 def test_bad_bounds_and_unknown_pose_stage_rejected(self):
  for kw in [dict(rotation_max_speed_deg_s=-1),dict(coordinate_max_speed=(1,0,1)),dict(rotation_max_cutoff_hz=float('nan'))]:
   with self.assertRaises(ValueError):StabilizerConfig(**kw)
  s=VideoPoseStabilizer(StabilizerConfig(measurement_source='selected_A0_pose'))
  with self.assertRaises(ValueError):apply(s,packet(0),0)
