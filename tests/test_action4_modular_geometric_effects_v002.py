from pathlib import Path
import json,sys,unittest,numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'));sys.dont_write_bytecode=True
from data import action4_modular_geometric_effects_v002 as e
class Tests(unittest.TestCase):
 def test_real_close_clusters_pass_f0_rejected(self):
  c=json.loads((R/'outputs/action4_modular_bilateral_prepare_v003/runs/cpu_probe_v003/scene_contract.json').read_text());corners=c['canonical_definition']['corners_3d_m'];K=json.loads((R/'outputs/action4_modular_relation_actual_v001/config.json').read_text())['K_native']
  for frame in [540,1200,1590]:
   p=json.loads((R/f'outputs/action4_modular_layout_proposals_prepare_v001/layout_cluster_{frame}.json').read_text());r=e.propose_near_double(p,corners,K);self.assertTrue(r['effect_evidence']['projection_condition_pass'])
  p=json.loads((R/'outputs/action4_modular_layout_proposals_prepare_v001/layout_cluster_0.json').read_text())
  with self.assertRaises(ValueError):e.propose_near_double(p,corners,K)
 def test_fixed9_geometry_sections_rigid_and_distance(self):
  p=json.loads((R/'outputs/action4_modular_geometric_effects_prepare_v002/engineering9_preregistered.json').read_text());self.assertEqual(len(p['candidates']),9);A=np.array(json.loads((R/'outputs/action4_modular_relation_actual_v001/config.json').read_text())['A_shared'])
  for c in p['candidates']:
   B=e.rigid(c['B_capture']);C=e.rigid(c['C_capture']);np.testing.assert_allclose(B@A,C,atol=1e-12);np.testing.assert_allclose(C[:3,3],[0,0,.55],atol=1e-12);self.assertEqual(c['section']['source_region'],102);self.assertFalse(c['counts_toward_reference'])
 def test_fullocclusion_never_eligible(self):
  r=json.loads((R/'outputs/action4_modular_geometric_effects_prepare_v002/engineering9_result.json').read_text());self.assertEqual(len(r['rows']),9)
  for row in r['rows']:
   if row['counts']['visible_core']==0:self.assertFalse(row['eligible_engineering_partial_candidate']);self.assertIn('full_or_nearfull_target_occlusion',row['rejections'])
 def test_no_3D_collision_bypass(self):
  with self.assertRaises(ValueError):e.actual_near_double_check({'instances':[{'instance_id':1},{'instance_id':2}]},{'outside_or_unresolved_surface_pairs':1})
if __name__=='__main__':unittest.main(verbosity=2)
