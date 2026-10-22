"""Pure effect/proposal checks. No Blender, scene or new orientation search."""
from pathlib import Path
import sys,json,unittest,numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'));sys.dont_write_bytecode=True
from data import action4_modular_geometric_effects as e
class GeometricEffectsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.r=json.loads((R/'outputs/action4_modular_relation_actual_v001/config.json').read_text());cls.l=json.loads((R/'outputs/action4_left_hand_y25_v001/assembly.json').read_text());c=json.loads((R/'outputs/action4_modular_bilateral_prepare_v003/runs/cpu_probe_v003/scene_contract.json').read_text());cls.corners=c['canonical_definition']['corners_3d_m'];cls.K=cls.r['K_native'];cls.layout={'schema':'action4.modular_layout_proposal.v1','mode':'layout_conditioned_preview','target_presence':[True,True],'layout_conditioning':{'source_cluster_frame':0},'sides':{}}
  for side,j in [('left',cls.l),('right',cls.r)]:
   C=np.array(j['C_per_frame']['0']);A=np.array(j['A_shared']);cls.layout['sides'][side]={'present':True,'core_C_capture':C.tolist(),'brace_B_capture':(C@np.linalg.inv(A)).tolist(),'A_brace_from_CAD':A.tolist(),'S_skin_relative_to_brace':j['S_skin_relative_to_brace']}
 def test_real_f0_near_double_without_false_difficulty_claim(self):
  r=e.propose_near_double(self.layout,self.corners,self.K);self.assertTrue(r['effect_evidence']['projection_condition_pass']);self.assertFalse(r['clears_family_pending']);self.assertFalse(r['training_accepted'])
 def test_all_four_truncation_edges_keep_rigidA_and_input(self):
  before=json.dumps(self.layout,sort_keys=True)
  for edge in ['left','right','top','bottom']:
   r=e.propose_truncation(self.layout,self.corners,self.K,edge=edge);ctrl=r['effect_controls'];newA=np.linalg.inv(ctrl['brace_B_capture'])@ctrl['core_C_capture'];np.testing.assert_allclose(newA,self.layout['sides']['right']['A_brace_from_CAD'],atol=1e-12);self.assertTrue(any(r['effect_evidence']['after_projection_only']['corner_inframe']));self.assertFalse(all(r['effect_evidence']['after_projection_only']['corner_inframe']));self.assertTrue(r['effect_evidence']['actual_mask_truncation_unknown'])
  self.assertEqual(before,json.dumps(self.layout,sort_keys=True))
 def test_near_plane_scaling_and_unbounded_fraction_rejected(self):
  M=np.eye(4);M[2,3]=.001
  with self.assertRaises(ValueError):e.project8(M,self.corners,self.K)
  M[0,0]=2
  with self.assertRaises(ValueError):e.rigid(M)
  with self.assertRaises(ValueError):e.propose_truncation(self.layout,self.corners,self.K,outside_bbox_fraction=.8)
 def test_rays_distance_region_nearest_without_backface_cull(self):
  V=np.array([[-1,-1,1],[1,-1,1],[0,1,1],[-1,-1,2],[1,-1,2],[0,1,2]],float);F=np.array([[0,2,1],[3,4,5]]);distance,i=e.nearest_ray_hit([0,0,1],V,F);self.assertAlmostEqual(distance,1);self.assertEqual(i,0);self.assertEqual(e.nearest_ray_hit([0,0,-1],V,F)[1],None)
 def test_actual_effect_needs_mask_counts_and_actual_attribution(self):
  record={'instances':[{'instance_id':2,'area_visible_px':30,'area_inframe_amodal_px':60,'area_full_amodal_px':100,'occlusion_inframe':.5}]};self.assertTrue(e.mask_effect_check('truncation',record,2)['effect_condition_pass'])
  with self.assertRaises(ValueError):e.mask_effect_check('self_forearm',record,2,{'actual_scene_rays':False,'same_actor_forearm_first_hits':5})
  self.assertFalse(e.mask_effect_check('self_forearm',record,2,{'actual_scene_rays':True,'same_actor_forearm_first_hits':0})['effect_condition_pass'])
 def test_orientation_request_rejects_relative_installation_change(self):
  side=self.layout['sides']['right'];C=np.array(side['core_C_capture']);B=np.array(side['brace_B_capture']);C[0,3]+=.01
  with self.assertRaises(ValueError):e.orientation_effect_request(self.layout,'right',C,B,{})
if __name__=='__main__':unittest.main(verbosity=2)
