"""Single rail-support estimate. No scene, geometry build, BVH or GPU."""
import sys,unittest,copy,ast
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_left_mount_rail_modules as m
from data import action4_left_mount_phase_modules as old
class RailMountTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=m.load_config(m.CONFIG_PATH);cls.r=m.routes(cls.c);cls.oldc=old.load_config(old.CONFIG_PATH);cls.previous=old.routes(cls.oldc)
 def test_full_actual_neutral_slab_all_triangles_and_witness(self):
  support=self.r['rail_slab_support_estimate'];actual=np.load(m.path(self.c['sources']['v002_rejected_geometry']['path']),allow_pickle=False);V,F=actual['skin_vertices_m'],actual['skin_triangles'];lo,hi=support['slab_y_m'];Y=V[F,1];selected=np.flatnonzero((Y.min(1)<=hi)&(Y.max(1)>=lo));np.testing.assert_array_equal(support['selected_triangle_ids'],selected);self.assertEqual(support['full_triangle_count'],17280);self.assertEqual(support['selected_triangle_count'],2731);self.assertFalse(support['only_collision_hit_faces_used']);self.assertIn(support['support_witness_triangle_id'],selected);self.assertLessEqual(lo,support['support_witness_m'][1]);self.assertLessEqual(support['support_witness_m'][1],hi);self.assertAlmostEqual(support['maximum_side_x_support_m'],support['lateral_side']*support['support_witness_m'][0])
 def test_clipping_excludes_out_of_slab_vertex_and_keeps_other_faces(self):
  V=np.array([[100.,2,0],[1,-1,0],[1,1,0],[-4,-.2,0],[-3,.2,0],[-3,0,1]]);F=np.array([[0,1,2],[3,4,5]])
  result=m.slab_lateral_support(V,F,-.5,.5,1);self.assertAlmostEqual(result['maximum_side_x_support_m'],50.5);self.assertEqual(result['selected_triangle_ids'],[0,1]);self.assertLess(result['maximum_side_x_support_m'],100)
  result=m.slab_lateral_support(V,F,-.5,.5,-1);self.assertEqual(result['maximum_side_x_support_m'],4);self.assertEqual(result['support_witness_triangle_id'],1)
 def test_coplanar_boundary_and_invalid_or_empty_slab(self):
  V=np.array([[1.,0,0],[3,0,0],[2,0,1]]);F=np.array([[0,1,2]]);self.assertEqual(m.slab_lateral_support(V,F,0,1,1)['maximum_side_x_support_m'],3)
  for lo,hi,side in [(1,2,1),(0,1,0),(0,0,1)]:
   with self.assertRaises(ValueError):m.slab_lateral_support(V,F,lo,hi,side)
 def test_only_common_rail_x_changes_USB_exit_and_all_yz_unchanged(self):
  r,oldr=self.r,self.previous;new=np.array(r['derived_cable_path_m']);previous=np.array(oldr['derived_cable_path_m']);np.testing.assert_array_equal(new[:2],previous[:2]);np.testing.assert_array_equal(new[-4:,1:],previous[-4:,1:]);self.assertEqual(new.shape,previous.shape);self.assertTrue(np.all(new[-4:,0]==new[-1,0]));self.assertEqual(r['derived_link_path_m'],oldr['derived_link_path_m'])
  s=r['rail_slab_support_estimate'];expected=max(s['original_station_abs_x_m'][0],s['maximum_side_x_support_m']+r['cable_radius_m']+.001);self.assertEqual(s['derived_common_rail_abs_x_m'],expected);np.testing.assert_array_equal(new[-4:,0],np.full(4,r['cable_lateral_side']*expected));self.assertGreaterEqual(expected,s['original_station_abs_x_m'][0]);self.assertTrue(s['derived_safe_routing_estimate_not_measured'])
 def test_slab_includes_all_station_and_exit_lead_y_range(self):
  points=np.array(self.r['derived_cable_path_m']);support=self.r['rail_slab_support_estimate'];expected=np.r_[points[-4:,1],points[1,1]];self.assertEqual(support['slab_y_m'],[float(expected.min()),float(expected.max())]);self.assertGreater(support['slab_y_m'][1],points[-4:,1].max())
 def test_phase_dimensions_A_S_F_and_original_guards_fixed(self):
  for key in ['old_brace_to_new_brace','A_new_brace_from_CAD','ball_center_newbrace_m','band_anchor_newbrace_m','source_dimensions_m','link_radius_m','cable_radius_m','ball_radius_m']:self.assertEqual(self.r[key],self.previous[key])
  for role in ['joint_result','left_axes','wrist_geometry','wrist_contract','brace_geometry','brace_contract','neutral_baseline','neutral_geometry']:self.assertEqual(self.c['sources'][role],self.oldc['sources'][role])
  self.assertIs(m.joint_zones,old.joint_zones);self.assertIs(m.classify_pairs,old.classify_pairs);length=np.linalg.norm(np.diff(self.r['derived_link_path_m'],axis=0),axis=1);self.assertGreater(length.min(),1e-7);self.assertLess(length.min(),.00014)
  source=Path(m.__file__).read_text();self.assertIn('np.array_equal(transition,stations[0])',source);self.assertIn('<1e-7',source);self.assertIn('copy.matrix_basis=o.matrix_basis.copy()',source)
 def test_actual_source_binding_and_single_margin_no_override(self):
  receipt=m.read(self.c['sources']['v002_rejected_receipt']['path']);self.assertEqual(receipt['geometry'],self.c['sources']['v002_rejected_geometry']);real=m.read;bad=copy.deepcopy(self.c);bad['routing']['rail_clearance_margin_m']=.002
  with patch.object(m,'read',side_effect=lambda p:bad if p==m.CONFIG_PATH else real(p)):
   with self.assertRaises(ValueError):m.load_config(m.CONFIG_PATH)
 def test_v002_snapshot_and_all_sealed_files_unchanged(self):
  self.assertEqual((m.OUTPUT/'v002_source_snapshot.py.txt').read_bytes(),Path(old.__file__).read_bytes());self.assertEqual(m.sha(old.__file__),self.c['sources']['v002_phase_helper']['sha256']);handoff=old.read(old.OUTPUT/'preparation_handoff.json')
  for p,h in handoff['files'].items():self.assertEqual(old.sha(p),h)
 def test_prior_approval_cannot_build_or_load_Blender(self):
  real=m.read
  with patch.object(m,'read',side_effect=lambda p:{'status':'APPROVED_SINGLE_CPU_LEFT_PHASE_MOUNT_BUILD'} if str(p).endswith('bad_approval.json') else real(p)),patch.object(m,'_build_once',side_effect=AssertionError('forbidden')):
   with self.assertRaises(ValueError):m.build(m.CONFIG_PATH,m.OUTPUT/'bad_approval.json')
  self.assertFalse((m.OUTPUT/'actual_build_v001').exists());self.assertNotIn('bpy',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
