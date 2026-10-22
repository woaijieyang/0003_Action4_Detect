"""LEFT mount pure/source/guard checks only; no Blender, scene, build or GPU."""
import ast,copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_left_mount_modules as m
CONFIG=m.CONFIG_PATH
class LeftMountTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=m.load_config(CONFIG)
 def test_exact_degenerate_transition_omitted_preserving_all_stations(self):
  r=m.routes(self.c);cable=np.array(r['derived_cable_path_m']);old=np.array(r['source_cable_path_reexpressed_m'])
  self.assertEqual(cable.shape,(6,3));np.testing.assert_array_equal(cable[-4:,1:],old[1:,1:]);np.testing.assert_array_equal(abs(cable[-4:,0]),abs(old[1:,0]));self.assertGreaterEqual(np.linalg.norm(np.diff(cable,axis=0),axis=1).min(),1e-7)
 def test_near_but_not_exact_transition_is_not_merged(self):
  real=m.read;sourcepath=self.c['sources']['source_contract']['path'];bad=copy.deepcopy(real(sourcepath));lead=np.array(m.read(m.OUTPUT/'route_guard_diagnosis.json')['derived_cable_path_m'])[1]
  bad['assembly']['assembly_priors']['left']['cable_points_brace_m'][1][2]=float(lead[2]-1e-8)
  with patch.object(m,'read',side_effect=lambda p:bad if p==sourcepath else real(p)):
   with self.assertRaisesRegex(ValueError,'Finite nonduplicate'):m.routes(self.c)
 def test_nondegenerate_transition_retains_original_rule(self):
  real=m.read;sourcepath=self.c['sources']['source_contract']['path'];bad=copy.deepcopy(real(sourcepath));lead=np.array(m.read(m.OUTPUT/'route_guard_diagnosis.json')['derived_cable_path_m'])[1];bad['assembly']['assembly_priors']['left']['cable_points_brace_m'][1][2]=float(lead[2]-1e-6)
  with patch.object(m,'read',side_effect=lambda p:bad if p==sourcepath else real(p)):r=m.routes(self.c)
  cable=np.array(r['derived_cable_path_m']);self.assertEqual(cable.shape,(7,3));self.assertEqual(cable[2,2],lead[2]);self.assertGreater(cable[2,2],cable[3,2])
 def test_original_dimensions_radii_shift_and_endpoint_binding(self):
  r=m.routes(self.c);p=m.read(self.c['sources']['source_contract']['path'])['config']['initial_priors'];A=np.array(r['A_new_brace_from_CAD']);link=np.array(r['derived_link_path_m']);cable=np.array(r['derived_cable_path_m'])
  for k in ['link_radius_m','cable_radius_m','ball_radius_m']:self.assertEqual(r[k],p[k])
  self.assertEqual(r['source_dimensions_m'],{k:p[k] for k in ['quick_release_dimensions_m','usb_dimensions_m','white_module_dimensions_m']});np.testing.assert_array_equal(np.array(r['old_brace_to_new_brace'])[:3,3],[0,-.003,0]);np.testing.assert_array_equal(link[0],r['band_anchor_newbrace_m']);np.testing.assert_array_equal(link[1],r['ball_center_newbrace_m']);np.testing.assert_array_equal(link[-1],m.transform(A,r['quick_release_center_CAD_m']));np.testing.assert_array_equal(cable[0],m.transform(A,r['USB_exit_CAD_m']));self.assertLess(m.transform(np.linalg.inv(A),cable[1])[2],r['USB_exit_CAD_m'][2])
 def test_neutral_S375_context_exact_and_identity_or_Y25_rejected(self):
  V,F,meta=m.collision_skin_context(self.c);skin=np.load(m.path(self.c['sources']['wrist_geometry']['path']),allow_pickle=False);j=m.read(self.c['sources']['joint_result']['path']);np.testing.assert_array_equal(V,m.transform(np.array(j['S_skin_relative_to_brace']),skin['brace_vertices_m']));self.assertEqual(meta['pose'],{'left_hand':[0,0,0]});self.assertEqual((meta['baseline_polygon_pairs'],meta['baseline_triangle_pairs']),(99,173));self.assertLess(meta['maximum_cached_analytic_roundoff_m'],1e-14)
  with self.assertRaises(ValueError):m.collision_skin_context(self.c,(skin['brace_vertices_m'],F))
  real=m.read;baseline=self.c['sources']['neutral_baseline']['path'];bad=copy.deepcopy(real(baseline));bad['pose']={'left_hand':[0,25,0]}
  with patch.object(m,'read',side_effect=lambda p:bad if p==baseline else real(p)):
   with self.assertRaises(ValueError):m.collision_skin_context(self.c)
 def test_left_source_names_original_ball_basis_and_material_identity(self):
  plan=m.source_plan(self.c);self.assertEqual(set(plan),{'quick_release','usb_connector','white_module','ball'})
  for key,row in plan.items():self.assertEqual(row['name'],'integrated_left_'+key);self.assertEqual(row['side'],'left');self.assertTrue(row['materials']);self.assertTrue(row['mesh_data_vertices_sha256'])
  np.testing.assert_allclose(plan['ball']['object_scale'],[.006]*3,rtol=0,atol=1e-9)
 def test_original_joint_zone_classifier_not_relaxed(self):
  self.assertIs(m.joint_zones,m.original.joint_zones);self.assertIs(m.classify_pairs,m.original.classify_pairs);self.assertEqual(set(m.joint_zones(self.c,m.routes(self.c))),{'core__quick_release','link__quick_release','ball__link','ball__band','band__link','cable__usb_connector'})
  v=np.array([[1,0,0],[1.1,0,0],[1,.1,0]]);t=np.array([[0,1,2]]);zone={'module_to_zone':np.eye(4).tolist(),'min_m':[-.2]*3,'max_m':[.2]*3}
  for z in [zone,None]:self.assertEqual(m.classify_pairs(v,t,v,t,[(0,0)],z)['outside_or_unresolved_pairs'],[[0,0]])
 def test_preexisting_left_names_rejected_not_suffix_renamed(self):
  bpy=SimpleNamespace(data=SimpleNamespace(objects={},collections={},texts={},scenes={}))
  self.assertEqual(len(m.guard_source_names(bpy)),11);bpy.data.objects['mount_left_ball']=object()
  with self.assertRaises(ValueError):m.guard_source_names(bpy)
 def test_unapproved_build_no_attempt_or_Blender(self):
  real=m.read
  with patch.object(m,'read',side_effect=lambda p:{'status':'NOT_APPROVED'} if str(p).endswith('no_approval.json') else real(p)),patch.object(m,'_build_once',side_effect=AssertionError('forbidden')):
   with self.assertRaises(ValueError):m.build(CONFIG,m.OUTPUT/'no_approval.json')
  self.assertFalse((m.OUTPUT/'actual_build_v001').exists());self.assertNotIn('bpy',sys.modules)
 def test_source_and_config_guard_and_frozen18(self):
  self.assertEqual(m.sha(self.c['sources']['parent_mount_helper']['path']),m.PARENT_SHA);reg=m.read(self.c['sources']['registry']['path']);files={p:h for item in reg['components'] for p,h in item['files'].items()};self.assertEqual(len(files),18)
  for p,h in files.items():self.assertEqual(m.sha(p),h)
  real=m.read;bad=copy.deepcopy(self.c);bad['routing']['nominal_joint_zone_margin_m']=.002
  with patch.object(m,'read',side_effect=lambda p:bad if p==CONFIG else real(p)):
   with self.assertRaises(ValueError):m.load_config(CONFIG)
 def test_static_backend_no_dynamic_source_rewrite_or_right_names(self):
  src=Path(m.__file__).read_text();tree=ast.parse(src);calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)};self.assertNotIn('exec',calls);self.assertNotIn('eval',calls);self.assertNotIn('compile',calls)
  self.assertNotIn('integrated_right_',src);self.assertNotIn('mount_right_',src);self.assertIn('copy.matrix_basis=o.matrix_basis.copy()',src);self.assertIn('copy.matrix_parent_inverse=o.matrix_parent_inverse.copy()',src);self.assertIn('all_triangles=True,epsilon=0',src)
if __name__=='__main__':unittest.main(verbosity=2)
