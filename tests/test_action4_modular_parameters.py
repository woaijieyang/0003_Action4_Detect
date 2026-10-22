"""Bounded candidate parameters; no Blender, GPU, scene or generated dataset."""
import sys,json,copy,unittest
from pathlib import Path
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_modular_parameters as m
from data import action4_modular_scene_sampling as consumer
class ParameterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=m.load_config();cls.plan=json.loads(m.verified_bytes(cls.c['source_refs']['source_plan']));cls.entries=cls.plan['entries'];cls.base=consumer.prepare_base(cls.c['source_refs']['scene_config']['path']);cls.ctx=cls.base['source_context'];cls.reg=copy.deepcopy(cls.c);cls.reg['sampling_mode']='baseline_regression'
 def sample(self,entry,config=None):return m.sample_parameters(entry,config or self.c,self.ctx)
 def test_original_plan_quotas_groups_and_200_augmentation_partition(self):
  audit=m.audit_plan(self.plan);self.assertEqual(audit['split_counts'],m.SPLITS);self.assertEqual(audit['instance_counts_by_split'],m.QUOTAS);self.assertEqual(audit['groups'],500);self.assertEqual(audit['group_size'],4);self.assertIn('not four',audit['group_semantics']);self.assertEqual(Counter(self.entries[i]['split'] for i in self.c['augmentation_indices']),Counter(train=160,eval=20,test=20))
 def test_deterministic_order_independent_identity_and_JSON(self):
  e=self.entries[0];original=copy.deepcopy(e);first=self.sample(e,self.reg);self.sample(self.entries[1],self.reg);second=self.sample(e,self.reg);self.assertEqual(first,second);self.assertEqual(e,original);self.assertEqual(first['entry'],original);json.dumps(first,allow_nan=False);self.assertEqual(first['provenance']['source_context_sha256'],m.digest(self.ctx))
 def test_reference_unknown_is_pending_not_1800_zero_coverage(self):
  e=next(e for e in self.entries if e['scene_family']=='wall_rear' and e['index'] not in self.c['augmentation_indices']);p=self.sample(e);self.assertEqual(p['status'],'REFERENCE_BOUNDS_UNKNOWN_PENDING');self.assertIsNone(p['augmentation']['reference_bounds']);self.assertFalse(p['augmentation']['counts_as_formal_coverage'])
  with self.assertRaises(m.PendingParametersError):m.require_consumable_preview(p)
  with self.assertRaises(consumer.PendingFamilyError):consumer.validate_parameters(p,self.base)
  deficit=m.coverage_deficit(self.c);self.assertEqual(deficit['reference_bounds_missing_entries'],1800);self.assertEqual(deficit['formal_acceptance_deficit'],2000);self.assertEqual(set(deficit['required128_preview_family_set']),set(m.FAMILIES));self.assertEqual(len(deficit['pending_family_entry_counts']),6)
 def test_baseline_regression_explicit_never_formal_coverage(self):
  e=next(e for e in self.entries if e['scene_family']=='wall_rear');p=self.sample(e,self.reg);self.assertEqual(p['status'],'BASELINE_REGRESSION_ENGINEERING_ONLY');self.assertFalse(p['augmentation']['counts_as_formal_coverage']);self.assertEqual(p['augmentation']['reference_or_augmentation'],'baseline_regression');self.assertTrue(m.require_consumable_preview(p));consumer.validate_parameters(p,self.base)
  with self.assertRaises(m.PendingParametersError):m.require_production_ready(p)
 def test_nonzero_reference_interval_API_with_evidence_fixture(self):
  # API fixture only: this unit test does not certify these as measured bounds.
  c=copy.deepcopy(self.c);c['reference_sampling']={'status':'evidence_checked_candidate','bounds':{'translation_capture_each_axis_m':[-.001,.001],'rotation_brace_local_each_axis_deg':[0.,0.],'area_light_gain':[1.,1.],'exposure_offset_ev':[0.,0.]},'evidence_refs':[c['source_refs']['right_anchor_relation']],'primary_control_choices':['whole_actor_translation']}
  e=next(e for e in self.entries if e['scene_family']=='wall_rear' and e['index'] not in c['augmentation_indices']);p=self.sample(e,c);self.assertEqual(p['status'],'BOUNDED_REFERENCE_CANDIDATE');self.assertEqual(p['augmentation']['primary_control'],'whole_actor_translation');self.assertGreater(sum(np.linalg.norm(r['translation_capture_m']) for r in p['sides'].values()),0);self.assertTrue(p['augmentation']['reference_bounds_known']);self.assertFalse(p['augmentation']['counts_as_formal_coverage']);consumer.validate_parameters(p,self.base)
  c['reference_sampling']['bounds']['translation_capture_each_axis_m']=[0.,0.]
  with self.assertRaises(ValueError):m.validate_config(c)
 def test_augmentation_one_primary_bounded_control_preserves_C_B_A(self):
  for index in self.c['augmentation_indices'][:24]:
   p=self.sample(self.entries[index]);self.assertEqual(p['status'],'BOUNDED_AUGMENTATION_CANDIDATE');nonzero=0
   for side,row in p['sides'].items():
    if not row['present']:continue
    B=np.array(row['brace_B_capture']);C=np.array(row['core_C_capture']);A=np.array(row['A_brace_from_CAD']);np.testing.assert_allclose(C,B@A,rtol=0,atol=1e-12);m.rigid(B);m.rigid(C);self.assertLessEqual(np.max(abs(np.array(row['translation_capture_m']))),.004);self.assertLessEqual(np.max(abs(np.array(row['rotation_brace_local_xyz_deg']))),8);nonzero+=np.count_nonzero(row['translation_capture_m'])+np.count_nonzero(row['rotation_brace_local_xyz_deg'])
   nonzero+=int(p['environment_controls']['area_light_gain']!=1)+int(p['environment_controls']['exposure_offset_ev']!=0);self.assertLessEqual(nonzero,1);self.assertFalse(p['augmentation']['counts_as_formal_coverage'])
 def test_anchor_pairing_and_sparse_presence_original_IDS(self):
  single_right=[e for e in self.entries if e['target_presence']==[False,True]][:20];frames=set()
  for e in single_right:
   p=self.sample(e);frames.add(p['source_anchor_frames_by_side']['right']);self.assertIsNone(p['sides']['left']['core_C_capture']);self.assertIsNone(p['source_anchor_frames_by_side']['left']);self.assertFalse(p['derived']['mixed_frame_pair'])
  self.assertEqual(frames,{0,540,960})
  for presence in [[True,False],[True,True],[False,False]]:
   p=self.sample(next(e for e in self.entries if e['target_presence']==presence));self.assertEqual(p['target_presence'],presence)
   if all(presence):self.assertEqual(p['source_anchor_frames_by_side'],{'left':0,'right':0})
   if presence[0]:self.assertEqual(p['source_anchor_frames_by_side']['left'],0)
 def test_frozen_A_S_F_K_scale_and_entry_overrides_rejected(self):
  for field in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']:
   ctx=copy.deepcopy(self.ctx);ctx['sides']['left'][field][0][3]+=.001
   with self.assertRaises(ValueError):m.sample_parameters(self.entries[0],self.c,ctx)
  ctx=copy.deepcopy(self.ctx);ctx['K_native'][0][0]+=1
  with self.assertRaises(ValueError):m.sample_parameters(self.entries[0],self.c,ctx)
  c=copy.deepcopy(self.c);c['scale']=1.1
  with self.assertRaises(ValueError):m.validate_config(c)
  for field,value in [('split','test'),('seed',0),('target_presence',[False,True])]:
   e=copy.deepcopy(self.entries[0]);e[field]=value
   with self.assertRaises(ValueError):self.sample(e)
 def test_six_pending_families_and_two_real_coarse_families(self):
  for family in m.FAMILIES:
   p=self.sample(next(e for e in self.entries if e['scene_family']==family),self.reg)
   if family in ['wall_rear','hard_negative']:
    self.assertEqual(p['scene_family_controls']['pending_reasons'],[]);m.require_consumable_preview(p);consumer.validate_parameters(p,self.base)
   else:
    self.assertTrue(p['scene_family_controls']['pending_reasons'])
    with self.assertRaises(m.PendingParametersError):m.require_consumable_preview(p)
    with self.assertRaises(consumer.PendingFamilyError):consumer.validate_parameters(p,self.base)
 def test_original8_occupancy_positive_depth_and_mask_unknown(self):
  p=self.sample(self.entries[0],self.reg)
  for side,row in p['derived']['by_side'].items():
   self.assertTrue(row['valid_positive_depth']);self.assertEqual(np.array(row['canonical8_native_px']).shape,(8,2));self.assertGreater(row['bbox_width_over_W'],0);self.assertTrue(row['bbox_fraction_is_not_mask_occlusion_or_truncation']);self.assertGreater(row['canonical_center_distance_to_capture_eye_m'],0)
  C=np.eye(4);C[2,3]=.0001;self.assertFalse(m.project_canonical(C,p['K_native'],self.c['fixed']['canonical8_m'])['valid_positive_depth'])
 def test_background_actor_AABB_and_no_collision_acceptance(self):
  p=self.sample(next(e for e in self.entries if e['target_count']==0),self.reg);self.assertEqual(set(p['safety_prior']['actor_AABB_by_side_capture_m']),{'left','right'});self.assertIsNone(p['safety_prior']['pair']['core_center_distance_m']);self.assertIsNone(p['safety_prior']['pair']['minimum_surface_distance_m']);self.assertTrue(p['safety_prior']['requires_actual_geometry_contact_containment_and_masks']);self.assertFalse(p['training_accepted']);self.assertFalse(p['dataset_generation_allowed'])
 def test_import_and_sampling_do_not_load_Blender(self):self.assertNotIn('bpy',sys.modules)
 def test_exact_paired_image_audit_constraints_not_3D_GT(self):
  constraints=m.reference_layout_constraints(self.c);report=json.loads(m.verified_bytes(self.c['source_refs']['reference_image_layout_audit']));self.assertEqual([r['source_frame'] for r in constraints['clusters']],report['frames_used']);self.assertEqual(constraints['observed_envelope'],report['observed_envelope']);self.assertFalse(constraints['physical_pose_K_or_anatomy_GT']);self.assertFalse(constraints['production_frozen'])
  p=self.sample(self.entries[0],self.reg);request=p['derived']['reference_image_layout_request'];self.assertIsNone(request['conditional_capture_pose_bounds']);self.assertTrue(request['target_C_or_scale_not_modified']);self.assertTrue(request['not_a_3D_pose_or_camera_GT']);self.assertEqual({r['side'] for r in request['paired_cluster']['body_envelope_metrics']},{'L','R'})
 def test_layout_check_uses_actual_derived_metrics_without_overwriting(self):
  p=self.sample(self.entries[0],self.reg);original=copy.deepcopy(p);constraints=m.reference_layout_constraints(self.c);m.check_reference_layout(p,constraints);self.assertEqual(p,original)
  fixture=copy.deepcopy(p);fixture['derived']['by_side']['left']['bbox_width_over_W']=constraints['observed_envelope']['bbox_width_fraction'][1]+1;report=m.check_reference_layout(fixture,constraints);self.assertEqual(report['status'],'OUTSIDE_DIAGNOSTIC_IMAGE_LAYOUT');self.assertFalse(report['production_acceptance']);self.assertTrue(any(isinstance(row,dict) and row['field']=='left/width' for row in report['violations']))
if __name__=='__main__':unittest.main(verbosity=2)
