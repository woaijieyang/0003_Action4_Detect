"""Finite cloth candidate/source/shape CPU checks; no Blender or render."""
from pathlib import Path
from copy import deepcopy
import sys,unittest,numpy as np
R=Path('/data/zy/vln/a-test');sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
from data import action4_modular_cloth_effects as cloth
from data import action4_modular_scene_sampling as sampling
class ClothTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.context=sampling.prepare_base('configs/data/action4_modular_bilateral_preview_v003.json')
  cls.parameters={'W_from_capture':cls.context['source_context']['W_from_capture'],'K_native':cls.context['source_context']['K_native'],'sides':{s:{'present':True,'brace_B_capture':cls.context['source_context']['sides'][s]['base_B_capture']} for s in ['left','right']}}
 def test_exact_three_profiles_and_real_source_refs(self):
  c,ref=cloth.prepare_candidate();self.assertEqual(set(c['profiles']),{'table_fastening','table_turn','cloth_lift'});self.assertEqual(c['sources']['source_video']['sha256'],'016628eca2887220080d823f65d4f9c520ded71be363dbbbee95e175df7bd3e8');self.assertNotIn('bpy',sys.modules)
 def test_three_proposals_have_source_closed_rigid_pose_and_depth_clearance(self):
  for family in ['table_fastening','table_turn','cloth_lift']:
   with self.subTest(family=family):
    p=cloth.propose(family,self.context,self.parameters);self.assertEqual(cloth.validate_proposal(p,self.context),p)
    row=p['coarse_occluders'][0];pose=np.asarray(p['parent_B_capture'])@np.asarray(row['pose_brace']);self.assertTrue(np.allclose(pose,p['initial_pose_capture'],atol=1e-12));self.assertFalse(p['family_effect_render_verified']);self.assertFalse(p['occlusion_layer_covered'])
    lower=pose[2,3]-.008-row['fold_lift_m']-row['dimensions_m'][2]/2;maxactor=max(v[1][2] for v in p['cached_actor_bounds_capture'].values());self.assertGreaterEqual(lower-maxactor,.012-1e-12)
 def test_absent_side_retains_only_non_target_skin_band_bounds(self):
  parameters=deepcopy(self.parameters);parameters['sides']['left']={'present':False,'brace_B_capture':None}
  p=cloth.propose('table_fastening',self.context,parameters);self.assertEqual({name for name in p['cached_actor_bounds_capture'] if name.startswith('left/')},{'left/skin','left/band'});self.assertEqual(p['parent_B_capture'],self.context['source_context']['sides']['left']['base_B_capture'])
 def test_wrong_profile_pose_or_source_ref_rejected(self):
  original=cloth.propose('table_turn',self.context,self.parameters)
  for change in ['dimensions','lift','pose','ref','source','flag']:
   with self.subTest(change=change):
    p=deepcopy(original)
    if change=='dimensions':p['coarse_occluders'][0]['dimensions_m'][0]=.14
    elif change=='lift':p['coarse_occluders'][0]['fold_lift_m']=.03
    elif change=='pose':p['coarse_occluders'][0]['pose_brace'][0][3]+=.01
    elif change=='ref':p['candidate_config']['sha256']='0'*64
    elif change=='source':p['source_context_sha256']='0'*64
    else:p['family_effect_render_verified']=True
    with self.assertRaises(ValueError):cloth.validate_proposal(p,self.context)
 def test_finite_lift_changes_only_new_array_and_is_bounded(self):
  V=np.array([[0,-.06,0],[0,0,0],[0,.06,0]],float);original=V.copy();result=cloth.finite_lift(V,[.22,.12,.0012],.025)
  self.assertTrue(np.array_equal(V,original));self.assertTrue(np.array_equal(result[:,:2],V[:,:2]));self.assertEqual(result[0,2],-.025);self.assertEqual(result[-1,2],0);self.assertTrue(np.all(result[:,2]>=-.025))
  for bad in [-.001,.026,float('nan')]:
   with self.assertRaises(ValueError):cloth.finite_lift(V,[.22,.12,.0012],bad)
 def test_background_patch_and_original_table_are_not_effect_coverage(self):
  c,_=cloth.prepare_candidate();self.assertFalse(c['effect_limits']['occlusion_layer_covered']);self.assertFalse(c['effect_limits']['family_table_semantics_verified']);self.assertTrue(c['effect_limits']['foreground_fold_variant_pending'])
if __name__=='__main__':unittest.main(verbosity=2)
