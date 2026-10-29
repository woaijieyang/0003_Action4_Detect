import unittest,copy
import numpy as np
from test_action4_video_pose_stabilizer import packet,apply
from model.action4_video_pose_stabilizer import StabilizerConfig,VideoPoseStabilizer
from model.action4_native_rotation_history import rotation_log

def run(angles,on):
 s=VideoPoseStabilizer(StabilizerConfig(rotation_coherence_gain=on,rotation_min_cutoff_hz=2,rotation_beta=.1,rotation_max_cutoff_hz=3,rotation_max_speed_deg_s=180));out=[]
 for f,a in enumerate(angles):out.append(apply(s,packet(f,angle=a),f)['instances'][0])
 return out
class CoherenceTests(unittest.TestCase):
 def test_default_identity_unchanged(self):
  import hashlib,json
  from dataclasses import asdict
  cfg=StabilizerConfig();d=asdict(cfg)
  for k in ('rotation_coherence_gain','incoherent_cutoff_hz','incoherent_speed_deg_s'):d.pop(k)
  self.assertEqual(cfg.identity,StabilizerConfig(**d).identity)
 def test_constant_turn_exact_parity(self):
  angles=np.arange(90)*.04
  a,b=run(angles,False),run(angles,True)
  for x,y in zip(a,b):np.testing.assert_allclose(x['final_pose']['rotation'],y['final_pose']['rotation'],atol=1e-12)
 def test_alternating_jitter_is_damped(self):
  a,b=run([.35*(-1)**f for f in range(100)],False),run([.35*(-1)**f for f in range(100)],True)
  amp=lambda rows:np.std([rotation_log(np.asarray(x['final_pose']['rotation']))[2] for x in rows[30:]])
  self.assertLess(amp(b),amp(a)*.6)
 def test_reversal_recovers_not_frozen(self):
  angles=np.r_[np.arange(40)*.04,1.56-np.arange(40)*.04]
  b=run(angles,True)
  self.assertLess(abs(rotation_log(np.asarray(b[-1]['final_pose']['rotation']))[2]-angles[-1]),.25)
 def test_no_mutation_of_raw_roi_or_ids(self):
  a,b=run([.2*(-1)**f for f in range(20)],False),run([.2*(-1)**f for f in range(20)],True)
  for x,y in zip(a,b):
   for k in ('raw_network_pose','bbox_for_a0','corner_ids','translation_m'):self.assertEqual(x[k],y[k])
