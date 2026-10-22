"""Finite actual7cluster projection proposals and independent consumer negatives."""
import sys,json,copy,unittest,ast
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_modular_layout_proposals as l
from data import action4_modular_scene_sampling as consumer
class LayoutTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=l.load_config();cls.pc=l._parameter_config(cls.c);cls.base=consumer.prepare_base(cls.pc['source_refs']['scene_config']['path']);cls.source=cls.base['source_context'];cls.entry=next(e for e in l.params._entries(cls.pc).values() if e['scene_family']=='wall_rear' and e['target_count']==2);cls.report=json.loads(l.params.verified_bytes(cls.c['audit_ref']));cls.proposals={f:l.propose_layout(cls.entry,l.CONFIG,cls.source,f) for f in cls.report['frames_used']}
 def test_all7_actual_layout_clusters_have_finite_nonzero_proposals(self):
  self.assertEqual(len(self.proposals),7);centers=[]
  for frame,p in self.proposals.items():
   self.assertEqual(p['schema'],l.SCHEMA);self.assertEqual(p['status'],l.STATUS);self.assertEqual(p['entry'],self.entry);self.assertFalse(p['augmentation']['counts_as_formal_coverage']);self.assertTrue(p['estimated']);self.assertLessEqual(len(p['layout_conditioning']['attempts']),12);consumer.validate_parameters(p,self.base);self.assertEqual(p['safety_prior']['pair']['piece_AABB_checked_pairs'],81);self.assertEqual(p['safety_prior']['pair']['piece_AABB_unresolved_count'],0);self.assertTrue(p['safety_prior']['near_plane_pass'])
   for side,row in p['sides'].items():
    C=np.array(row['core_C_capture']);B=np.array(row['brace_B_capture']);A=np.array(row['A_brace_from_CAD']);np.testing.assert_allclose(C,B@A,atol=1e-12,rtol=0);self.assertTrue(.15<=C[2,3]<=1.2);metric=p['derived']['by_side'][side];target=p['layout_conditioning']['targets_by_side'][side];self.assertLess(np.max(abs(np.array(metric['bbox_center_normalized'])-target['center_norm'])*[3248,2464]),.01);self.assertLess(abs(metric['bbox_area_over_image']-target['bbox_area_fraction'])/target['bbox_area_fraction'],1e-6)
   centers.append(tuple(np.array(p['sides']['left']['core_C_capture'])[:3,3]))
  self.assertEqual(len(set(centers)),7)
 def test_no_scale_or_camera_or_A_S_F_change_and_source_R_only(self):
  for p in self.proposals.values():
   self.assertEqual(p['W_from_capture'],self.source['W_from_capture']);self.assertEqual(p['K_native'],self.source['K_native'])
   for side,row in p['sides'].items():
    for field in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']:self.assertEqual(row[field],self.source['sides'][side][field])
    template=p['layout_conditioning']['orientation_templates_by_side'][side];expected=np.array(self.source['sides'][template['source_side']]['anchor_C_capture_by_frame'][str(template['source_frame'])])[:3,:3];np.testing.assert_array_equal(np.array(row['core_C_capture'])[:3,:3],expected);self.assertEqual(row['rotation_brace_local_xyz_deg'],[0.,0.,0.])
 def test_cross_side_is_explicit_necessary_fallback_not_left_GT(self):
  p=self.proposals[1200];self.assertTrue(p['layout_conditioning']['orientation_templates_by_side']['left']['engineering_cross_side']);self.assertFalse(p['layout_conditioning']['same_real_frame_3D_pair_claim']);self.assertTrue(p['layout_conditioning']['orientation_source_is_not_target_video_pose_GT']);self.assertTrue(all(r['failure_reasons'] for r in p['layout_conditioning']['attempts'][:3]))
  for frame,p in self.proposals.items():
   if p['layout_conditioning']['orientation_templates_by_side']['left']['engineering_cross_side']:self.assertTrue(all(r['failure_reasons'] for r in p['layout_conditioning']['attempts'][:3]))
   else:self.assertEqual(len(p['layout_conditioning']['attempts']),3)
 def test_deterministic_bound_jitter_preserves_paired_cluster(self):
  p=l.propose_layout(self.entry,l.CONFIG,self.source,0);self.assertEqual(p,self.proposals[0]);row=next(r for r in self.report['rows'] if r['frame']==0);observed={r['side']:r for r in row['body_envelope_metrics']}
  for side,letter in [('left','L'),('right','R')]:
   t=p['layout_conditioning']['targets_by_side'][side];self.assertLessEqual(max(abs(np.array(t['center_norm'])-observed[letter]['center_norm'])),.01);self.assertTrue(.95<=t['bbox_linear_factor_image_target_only']<=1.05);self.assertAlmostEqual(t['bbox_area_fraction'],observed[letter]['bbox_area_fraction']*t['bbox_linear_factor_image_target_only']**2)
 def test_large_translation_cannot_disguise_as_old_small_delta(self):
  p=copy.deepcopy(self.proposals[2250]);self.assertGreater(max(np.max(abs(np.array(row['translation_capture_m']))) for row in p['sides'].values()),.004)
  p['schema']='action4.modular_parameters.v1'
  with self.assertRaises(ValueError):consumer.validate_parameters(p,self.base)
  p['status']='BOUNDED_AUGMENTATION_CANDIDATE'
  with self.assertRaises(ValueError):consumer.validate_parameters(p,self.base)
 def test_frozen_K_R_A_S_and_derived_tamper_rejected(self):
  for kind in ['K','R','A','S','derived','depth']:
   p=copy.deepcopy(self.proposals[0])
   if kind=='K':p['K_native'][0][0]+=1
   elif kind=='R':
    C=np.array(p['sides']['left']['core_C_capture']);C[:3,:3]=C[:3,:3]@l.params.rotation_xyz_degrees([1,0,0]);p['sides']['left']['core_C_capture']=C.tolist()
   elif kind=='A':p['sides']['left']['A_brace_from_CAD'][0][3]+=.001
   elif kind=='S':p['sides']['left']['S_skin_relative_to_brace'][0][3]+=.001
   elif kind=='derived':p['derived']['by_side']['left']['bbox_area_over_image']+=.01
   else:p['sides']['left']['core_C_capture'][2][3]=.1
   with self.assertRaises(ValueError):consumer.validate_parameters(p,self.base)
 def test_unreachable_area_center_depth_and_budget_fail_without_retry(self):
  target,guard=l._targets(self.entry,self.c,self.pc,0,False);R=np.array(self.source['sides']['left']['anchor_C_capture_by_frame']['0'])[:3,:3];bad=copy.deepcopy(target['left']);bad['bbox_area_fraction']=10.
  with self.assertRaisesRegex(ValueError,'unreachable'):l.solve_translation(R,bad,np.array(self.source['K_native']),self.pc['fixed']['canonical8_m'],guard,'L',self.c['search_domain'])
  bad=copy.deepcopy(target['left']);bad['center_norm']=[-.1,-.1]
  with self.assertRaisesRegex(ValueError,'center'):l.solve_translation(R,bad,np.array(self.source['K_native']),self.pc['fixed']['canonical8_m'],guard,'L',self.c['search_domain'])
  for field,value in [('max_iterations_z',65),('max_orientation_proposals_per_cluster',13),('C_translation_z_m',[.1,1.2])]:
   c=copy.deepcopy(self.c);c['search_domain'][field]=value
   with self.assertRaises(ValueError):l.validate_config(c)
 def test_consumer_transaction_and_restore_AST_unchanged(self):
  old=(l.ROOT/'outputs/action4_modular_layout_proposals_prepare_v001/scene_sampling_v001.py.txt').read_text();new=Path(consumer.__file__).read_text();a={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};b={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  for name in a:
   if name not in ('validate_parameters','prepare_base'):self.assertEqual(a[name],b[name])
  self.assertNotIn('bpy',sys.modules)
 def test_visible_wall_not_hidden_to_force_layout_success(self):
  wall=self.pc['fixed']['baseline_environment']['wall_front_capture_z_m']
  for p in self.proposals.values():
   self.assertTrue(p['environment_controls']['wall_visible'])
   for row in p['derived']['by_side'].values():self.assertLess(np.max(np.array(row['canonical8_capture_m'])[:,2]),wall)
  fake={'left':{'canonical8_capture_m':[[0,0,wall+.01]]*8}}
  self.assertTrue(l._environment_prior_failures(fake,self.pc,'wall_rear'))
 def test_derived_layout_and_pair_diagnostics_recomputed_after_translation(self):
  for p in self.proposals.values():
   self.assertEqual(p['derived']['reference_image_layout_diagnostic'],l.params.check_reference_layout(p,l.params.reference_layout_constraints(self.pc)))
   left,right=[p['derived']['by_side'][s] for s in ['left','right']];self.assertAlmostEqual(p['derived']['paired_horizontal_center_separation_fraction'],abs(right['bbox_center_normalized'][0]-left['bbox_center_normalized'][0]));self.assertAlmostEqual(p['derived']['canonical_center_pair_distance_capture_m'],np.linalg.norm(np.mean(left['canonical8_capture_m'],axis=0)-np.mean(right['canonical8_capture_m'],axis=0)))
if __name__=='__main__':unittest.main(verbosity=2)
