"""One LEFT sourcebrace180phase candidate, CPU/source checks only."""
import sys,unittest,copy,ast
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_left_mount_phase_modules as m
from data import action4_left_mount_modules as old
class PhaseMountTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=m.load_config(m.CONFIG_PATH);cls.r=m.routes(cls.c);cls.oldc=old.load_config(old.CONFIG_PATH)
 def test_single_phase_exact_proper_transform_no_geometry_mirror(self):
  S=np.array(self.r['old_brace_to_new_brace']);expected=np.diag([-1.,1.,-1.,1.]);expected[1,3]=-.003;np.testing.assert_array_equal(S,expected);self.assertEqual(np.linalg.det(S[:3,:3]),1);np.testing.assert_array_equal(m.rigid(S),S);self.assertFalse(self.r['sourcebrace_phase_measured']);self.assertFalse(self.r['white_location_known'])
 def test_all_original_source_points_uniformly_reexpressed(self):
  source=m.read(self.c['sources']['source_contract']['path']);p=source['assembly']['assembly_priors']['left'];S=np.array(self.r['old_brace_to_new_brace'])
  np.testing.assert_array_equal(self.r['source_link_path_reexpressed_m'],m.transform(S,p['mount_link_brace_m']));np.testing.assert_array_equal(self.r['source_cable_path_reexpressed_m'],m.transform(S,p['cable_points_brace_m']));np.testing.assert_array_equal(self.r['ball_center_newbrace_m'],m.transform(S,p['ball_center_brace_m']));self.assertLess(self.r['ball_center_newbrace_m'][2],0)
  white=next(r for r in m.source_plan(self.c).values() if r['name']=='integrated_left_white_module');self.assertGreater(m.transform(S,np.array(white['basis_matrix'])[:3,3])[2],0)
 def test_short_leg_and_original_guard_retained(self):
  link=np.array(self.r['derived_link_path_m']);length=np.linalg.norm(np.diff(link,axis=0),axis=1);self.assertEqual(link.shape,(5,3));self.assertGreater(length.min(),1e-7);self.assertLess(length.min(),.00014);self.assertGreater(length.min(),.00013)
  real=m.read;path=self.c['sources']['source_contract']['path'];bad=copy.deepcopy(real(path));lead=np.array(self.r['derived_cable_path_m'])[1];bad['assembly']['assembly_priors']['left']['cable_points_brace_m'][1][2]=-float(lead[2]-1e-8)
  with patch.object(m,'read',side_effect=lambda p:bad if p==path else real(p)):
   with self.assertRaisesRegex(ValueError,'Finite nonduplicate'):m.routes(self.c)
 def test_original_fixed_dimensions_endpoints_and_all_cable_stations(self):
  previous=old.routes(self.oldc)
  for k in ['link_radius_m','cable_radius_m','ball_radius_m','source_dimensions_m','A_new_brace_from_CAD','quick_release_center_CAD_m','USB_exit_CAD_m']:self.assertEqual(self.r[k],previous[k])
  cable=np.array(self.r['derived_cable_path_m']);stations=np.array(self.r['source_cable_path_reexpressed_m'])[1:];self.assertEqual(cable.shape,(6,3));np.testing.assert_array_equal(cable[-4:,1:],stations[:,1:]);np.testing.assert_array_equal(abs(cable[-4:,0]),abs(stations[:,0]));np.testing.assert_array_equal(cable[0],previous['derived_cable_path_m'][0]);np.testing.assert_array_equal(cable[1],previous['derived_cable_path_m'][1])
 def test_neutral_context_A_C_K_S_F_unchanged(self):
  V,F,meta=m.collision_skin_context(self.c);oldV,oldF,oldmeta=old.collision_skin_context(self.oldc);np.testing.assert_array_equal(V,oldV);np.testing.assert_array_equal(F,oldF)
  for role in ['joint_result','left_axes','wrist_geometry','wrist_contract','brace_geometry','brace_contract','neutral_baseline','neutral_geometry']:self.assertEqual(self.c['sources'][role],self.oldc['sources'][role])
  self.assertEqual(meta['pose'],{'left_hand':[0,0,0]});self.assertEqual((meta['baseline_polygon_pairs'],meta['baseline_triangle_pairs']),(99,173))
 def test_fixedpieces_localbasis_ballscale_zones_and_no_whitelist_change(self):
  self.assertEqual(m.source_plan(self.c),old.source_plan(self.oldc));self.assertIs(m.joint_zones,old.joint_zones);self.assertIs(m.classify_pairs,old.classify_pairs);np.testing.assert_allclose(m.source_plan(self.c)['ball']['object_scale'],[.006]*3,rtol=0,atol=1e-9)
  source=Path(m.__file__).read_text();self.assertIn('copy.matrix_basis=o.matrix_basis.copy()',source);self.assertIn('copy.matrix_parent_inverse=o.matrix_parent_inverse.copy()',source);self.assertIn('epsilon=0',source);self.assertNotIn('import bpy',source[:source.index('def _build_once')])
 def test_v001_snapshot_and_original_parent_unchanged(self):
  snapshot=m.OUTPUT/'v001_source_snapshot.py.txt';self.assertEqual(snapshot.read_bytes(),Path(old.__file__).read_bytes());self.assertEqual(m.sha(old.__file__),self.c['sources']['v001_left_mount_helper']['sha256']);self.assertEqual(m.sha(self.c['sources']['parent_mount_helper']['path']),m.PARENT_SHA)
  handoff=old.read(old.OUTPUT/'preparation_handoff.json')
  for p,h in handoff['files'].items():self.assertEqual(old.sha(p),h)
 def test_fresh_approval_required_no_build_or_Blender(self):
  real=m.read
  with patch.object(m,'read',side_effect=lambda p:{'status':'APPROVED_SINGLE_CPU_LEFT_MOUNT_BUILD'} if str(p).endswith('bad_approval.json') else real(p)),patch.object(m,'_build_once',side_effect=AssertionError('forbidden')):
   with self.assertRaises(ValueError):m.build(m.CONFIG_PATH,m.OUTPUT/'bad_approval.json')
  self.assertFalse((m.OUTPUT/'actual_build_v001').exists());self.assertNotIn('bpy',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
