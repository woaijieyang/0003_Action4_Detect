"""Bilateral interface pure checks; rejected real LEFT v001 is never scene-accepted."""
import ast,copy,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_modular_bilateral_scene as m
class Objects(dict):
 def __iter__(self):return iter(self.values())
class BilateralSceneTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.known=m.read(m.OWN/'known_source_refs.json');cls.rejected=m.read(m.OWN/'rejected_mount_test_refs.json')['refs'];cls.c={'schema':'action4.modular_bilateral_scene.v1','status':'diagnostic_pending_scene_review','dataset_generation_allowed':False,'frame_id':0,'instance_ids':[1,2],'resolution_wh':[3248,2464],'sources':{**cls.known,**cls.rejected},'output_scope':str((m.OWN/'runs').relative_to(m.ROOT))};cls.rc,cls.rctx=m.composed.prepare(cls.known['composed_config']['path']);cls.left=m.left_geometry_inputs(cls.c,cls.rctx)
 def test_real_rejected_unresolved_mount_blocks_prepare_despite_build_success(self):
  receipt=m.read(self.rejected['left_mount_receipt']['path']);self.assertEqual(receipt['status'],'CPU_MODULE_BUILD_RELOAD_DIAGNOSTIC_ONLY');self.assertTrue(receipt['reloaded_geometry_exact'])
  with self.assertRaisesRegex(ValueError,'unresolved contacts'):m.prepare(self.c)
  self.assertFalse((m.OWN/'runs').exists());self.assertNotIn('bpy',sys.modules)
 def test_common_fixed_W_K_leftneutral_A_S_F_and_rightY25(self):
  g=self.left;np.testing.assert_array_equal(g['W'],self.rctx['geometry']['W']);np.testing.assert_array_equal(g['K'],self.rctx['rc']['K_native']);self.assertEqual(g['row']['pose'],{'left_hand':[0,0,0]});self.assertEqual(self.rctx['Y25_row']['pose'],{'right_hand':[0,25,0]});self.assertEqual(g['row']['canonical8']['instance_id'],1);self.assertEqual(self.rctx['Y25_row']['canonical8']['instance_id'],2)
  real=m.read;path=self.known['left_neutral_row']['path'];bad=copy.deepcopy(real(path));bad['pose']={'left_hand':[0,25,0]}
  with patch.object(m,'read',side_effect=lambda p:bad if p==path else real(p)):
   with self.assertRaises(ValueError):m.left_geometry_inputs(self.c,self.rctx)
  wrong=copy.copy(self.rctx);wrong['rc']=copy.deepcopy(self.rctx['rc']);wrong['rc']['K_native'][0][0]+=1
  with self.assertRaises(ValueError):m.left_geometry_inputs(self.c,wrong)
 def test_complete36_contact_identities_and_receipt_bindings_required(self):
  real=m.read;path=self.rejected['left_mount_collision']['path'];bad=copy.deepcopy(real(path));key=next(iter(bad['pairs']));bad['pairs']['unlisted__pair']=bad['pairs'].pop(key)
  with patch.object(m,'read',side_effect=lambda p:bad if p==path else real(p)):
   with self.assertRaisesRegex(ValueError,'exact36'):m.validate_mount(self.c,self.left['j'])
  wrong=copy.deepcopy(self.c);wrong['sources']['left_mount_asset']['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'receipt binding'):m.validate_mount(wrong,self.left['j'])
 def test_known_source_bindings_and_future_version_explicit_refs(self):
  m.frozen_sources(self.c);required=set(m.read(m.OWN/'required_actual_source_roles.json')['roles']);self.assertEqual(required,{'left_mount_asset','left_mount_contract','left_mount_geometry','left_mount_receipt','left_mount_collision'});source=Path(m.__file__).read_text();self.assertNotIn('outputs/action4_left_mount_modules_v001',source);self.assertNotIn('action4_left_mount_modules as',source);self.assertNotIn('action4_left_mount_phase_modules',source)
  bad=copy.deepcopy(self.c);bad['sources']['left_mount_receipt']={'path':self.rejected['left_mount_receipt']['path'],'sha256':''}
  with self.assertRaises(ValueError):m.frozen_sources(bad)
 def test_target_ID1_own_decal_ID2_own_decal_non_targets0(self):
  objects=Objects()
  for i,names in [(1,m.LEFT_TARGETS),(2,m.RIGHT_TARGETS)]:
   decal=SimpleNamespace(name=names[1],type='MESH',pass_index=i,children_recursive=[]);core=SimpleNamespace(name=names[0],type='MESH',pass_index=i,children_recursive=[decal]);objects[core.name]=core;objects[decal.name]=decal
  dummy=SimpleNamespace(name='left_brace_band',type='MESH',pass_index=0,children_recursive=[]);objects[dummy.name]=dummy;scene=SimpleNamespace(objects=objects)
  self.assertEqual(set(m.targets(scene)),{1,2});dummy.pass_index=1
  with self.assertRaises(ValueError):m.targets(scene)
  dummy.pass_index=0;objects[m.LEFT_TARGETS[1]].pass_index=2
  with self.assertRaises(ValueError):m.targets(scene)
 def test_right_snapshot_detects_changes_and_ignores_added_left_only(self):
  state={'objects':{'right':{'mesh':'sha','world':[1]},'left':{'mesh':'left'}},'K':[[1]],'modular_preview_status':'RIGHT','world':{'state':'same'}}
  with patch.object(m.composed.neutral,'new_scene_state',return_value=state):snapshot=m._right_state(None,['right'])
  changed=copy.deepcopy(state);changed['modular_preview_status']='BILATERAL';changed['objects']['left']['mesh']='addedsource'
  with patch.object(m.composed.neutral,'new_scene_state',return_value=changed):self.assertEqual(snapshot,m._right_state(None,['right']))
  changed['objects']['right']['world'][0]=2
  with patch.object(m.composed.neutral,'new_scene_state',return_value=changed):self.assertNotEqual(snapshot,m._right_state(None,['right']))
 def test_aabb_lower_bound_touch_overlap_and_invalid_inputs(self):
  a=np.array([[0.,0,0],[1,1,1]]);b=np.array([[1.0138996,0,0],[2,1,1]]);r=m.aabb_separation(a,b);self.assertTrue(r['strictly_disjoint']);self.assertAlmostEqual(r['distance_lower_bound_m'],.0138996)
  self.assertFalse(m.aabb_separation(a,a+np.array([1,0,0]))['strictly_disjoint']);self.assertFalse(m.aabb_separation(a,a+.5)['strictly_disjoint'])
  for bad in [np.array([]),np.array([[float('nan'),0,0]])]:
   with self.assertRaises(ValueError):m.aabb_separation(a,bad)
 def test_all_separated_actor_pairs_skip_BVH_no_scene_operation(self):
  lmc=m.read(self.rejected['left_mount_contract']['path']);ctx={'left':self.left,'left_mount_contract':lmc,'right_context':self.rctx};names=[]
  for side,g,mc in [('left',self.left,lmc),('right',self.rctx['geometry'],self.rctx['mount_contract'])]:names += [side+'_Action4_core',g['wc']['object_name'],'left_brace_band' if side=='left' else 'brace_band']+[row['name'] for row in mc['objects'].values()]
  objects=Objects({n:SimpleNamespace(name=n) for n in names});index={n:i for i,n in enumerate(names)}
  def geometry(o):
   base=index[o.name]*10.;return np.array([[base,0,0],[base+.1,.1,0],[base,.1,.1]]),np.array([[0,1,2]])
  fake=SimpleNamespace(FromPolygons=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Separated AABB must skip BVH')))
  with patch.dict(sys.modules,{'mathutils':SimpleNamespace(),'mathutils.bvhtree':SimpleNamespace(BVHTree=fake)}),patch.object(m,'verify_bilateral_scene',return_value={}),patch.object(m.assembly,'world_geometry',side_effect=geometry):report,_=m.geometry_diagnostics(SimpleNamespace(objects=objects),self.c,ctx,{})
  self.assertEqual(report['cross_pair_count'],81);self.assertEqual(len(report['cross_pairs']),81);self.assertEqual([len(x) for x in report['internal_pairs'].values()],[36,36]);self.assertEqual(report['outside_or_unresolved_surface_pairs'],0);self.assertTrue(all(not r['BVH_evaluated'] for r in report['cross_pairs'].values()))
 def test_static_preimport_right_verify_source_materials_and_no_pixels(self):
  source=Path(m.__file__).read_text();tree=ast.parse(source);create=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_scene');body=ast.get_source_segment(source,create);self.assertLess(body.index('composed.verify_composed_scene'),body.index('_append_objects'))
  self.assertIn('assembly.share_exact_materials',source);self.assertIn('Original LEFT source target data/local matrix/pass',source);self.assertIn("'left_original_decal_relative_to_core'",source);self.assertIn('geometry_diagnostics',source);self.assertIn('epsilon=0',source)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotEqual(n.func.attr,'render')
  self.assertNotIn('bpy',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
