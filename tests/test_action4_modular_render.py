import unittest,copy,sys,ast
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from data import action4_modular_render as m
class ModularPreviewTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c,cls.rel,cls.rc,cls.oldcfg,cls.oldcontract,cls.registry=m.prepare()
 def test_prepare_no_Blender_GPU_subprocess(self):
  with patch.object(m,'worker_environment',side_effect=AssertionError('No devices in prepare')),patch.object(m.subprocess,'run',side_effect=AssertionError('No child process')):
   m.prepare()
  self.assertNotIn('bpy',sys.modules)
 def test_one_right_f0_and_no_training(self):
  self.assertEqual(self.c['frame_id'],0);self.assertEqual(self.c['instance_id'],2)
  self.assertFalse(self.c['dataset_generation_allowed']);self.assertEqual(self.c['threads'],2)
  for key,value in [('frame_id',540),('instance_id',1),('dataset_generation_allowed',True)]:
   c=copy.deepcopy(self.c);c[key]=value
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_source_identity_rejected(self):
  c=copy.deepcopy(self.c);c['sources']['baseline_asset']['sha256']='0'*64
  with self.assertRaises(ValueError):m.validate_config(c)
 def test_render_scope_fixed(self):
  for key,value in [('samples',16),('mask_samples',8),('amodal_mode','full')]:
   c=copy.deepcopy(self.c);c['render'][key]=value
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_environment_positive_and_color_bounds(self):
  for key,value in [('table_width_m',-1),('grid_line_fraction',2),('wall_linear_rgb',[2,0,0]),('area_target_capture_m',self.c['environment']['area_location_capture_m'])]:
   c=copy.deepcopy(self.c);c['environment'][key]=value
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_table_prior_projects_to_observed_bottomstrip(self):
  K=np.array(self.rc['K_native']);g=m.environment_geometry(self.c,K)
  p=np.array(g['table_back_top_capture_points_m']);uv=m.assembly.project(p,K)
  np.testing.assert_allclose(uv[:,1],2365,rtol=0,atol=1e-10)
  self.assertAlmostEqual(g['grid_pitch_at_back_px'],77.61764705882354)
  m.assembly.rigid(g['wall_capture_matrix']);m.assembly.rigid(g['table_capture_matrix'])
 def test_background_geometry_separate_from_actual_modules(self):
  c=self.c;d=np.load(m.bounded(c['sources']['relation_actual_geometry']['path']));inv=np.linalg.inv(self.rc['W_from_capture_engineering_embedding'])
  for k in d.files:
   if k.endswith('_vertices_m'):
    p=m.assembly.transform(inv,d[k]);self.assertLess(float(p[:,2].max()),c['environment']['table_front_capture_z_m'])
 def test_same_A_S_C_and_fixed8(self):
  basis,world,B=self.rel.expected_world(self.rc,self.oldcontract,0);A=np.array(self.rc['A_shared']);S=np.array(self.rc['S_skin_relative_to_brace']);W=np.array(self.rc['W_from_capture_engineering_embedding']);C=np.array(self.rc['C_per_frame']['0'])
  np.testing.assert_allclose(world['right_Action4_core'],W@C,rtol=0,atol=1e-12)
  np.testing.assert_allclose(world['right_wrist_module_root'],W@B@S,rtol=0,atol=1e-12)
  can=self.oldcontract['canonical_definition'];ex=m.canonical.build_export(canonical=can,instance_id=2,object_to_world_m=world['right_Action4_core'].tolist(),camera_to_world_blender_m=world['assembly_observer'].tolist(),K=self.rc['K_native'],image_size_wh=self.c['resolution_wh'],near_plane_m=.001)
  self.assertEqual(ex['corner_ids'],list(range(8)));self.assertEqual(ex['corner_order'],can['corner_order']);self.assertTrue(all(ex['point_near_plane_projectable']));self.assertFalse(ex['training_accepted'])
 def test_fullcore_and_decal_target_only(self):
  decal=SimpleNamespace(name=m.TARGET_NAMES[1],type='MESH',pass_index=2)
  core=SimpleNamespace(name=m.TARGET_NAMES[0],type='MESH',pass_index=2,children_recursive=[decal])
  other=SimpleNamespace(name='brace_band',type='MESH',pass_index=0)
  class Objects(dict):
   def __iter__(self):return iter(self.values())
  scene=SimpleNamespace(objects=Objects({o.name:o for o in [core,decal,other]}))
  self.assertEqual(set(m.targets(scene)),{2});other.pass_index=2
  with self.assertRaises(ValueError):m.targets(scene)
 def test_existing_mask_measurements_and_empty_target(self):
  v=np.zeros((4,6),bool);v[1:3,2:4]=True;full=np.zeros((12,18),bool);full[4:8,6:12]=True
  a=m.render_helpers.mask_measurements(v,full,6,4)
  self.assertAlmostEqual(a['occlusion_inframe'],1-4/24);self.assertEqual(a['mask_nonnested_boundary_pixels'],0)
  full[:]=False;full[0:2,0:2]=True
  self.assertTrue(m.render_helpers.mask_measurements(np.zeros_like(v),full,6,4)['excluded'])
 def test_approval_failure_prevents_any_launch(self):
  real=m.read
  with patch.object(m,'read',side_effect=lambda p:{'status':'not_approved'} if m.bounded(p)==m.APPROVAL else real(p)),patch.object(m,'worker_environment',side_effect=AssertionError('No GPU query')),patch.object(m.subprocess,'run',side_effect=AssertionError('No process')):
   with self.assertRaises(ValueError):m.launch(m.CONFIG,m.OWN/'runs'/'unapproved_test',2,None,True)
  self.assertFalse((m.OWN/'runs'/'unapproved_test').exists())
 def test_output_scope_and_no_formal_data(self):
  for p in [m.ROOT/'data/new_preview',m.OWN,m.ROOT/'outputs/other']:
   with self.assertRaises(ValueError):m.output_path(p)
 def test_original_helpers_used_no_global_K_monkeypatch(self):
  source=Path(m.__file__).read_text();tree=ast.parse(source)
  calls=[ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)]
  for name in ['render_helpers.project_canonical','render_helpers.isolated_expanded','render_helpers.crop_geometry_preflight','render_helpers.mask_measurements','amodal.render_crop','base._mask_nodes','base._mask_array','synth._render_device','synth._set_denoiser','assembly.verify_scene','canonical.build_export']:
   self.assertIn(name,calls)
  self.assertNotIn('env.actual_K',calls);self.assertNotIn('base._reset_scene',calls)
  self.assertFalse(any(x.startswith('bpy.ops.wm.save') for x in calls))
  self.assertEqual(calls.count('bpy.ops.render.render'),2) # third render is originalamodal.render_crop
if __name__=='__main__':unittest.main(verbosity=2)
