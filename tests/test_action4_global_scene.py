import copy,json,unittest
from data import action4_global_scene as m
class IntegratedConfigTests(unittest.TestCase):
 def setUp(self):self.c=json.loads((m.ROOT/'configs/data/action4_global_scene_init_v001.json').read_text())
 def test_verified_actual_input(self):self.assertEqual(m.validate_config(self.c)['sources']['finger_asset']['path'],'data/action4_scene_assets/finger_rig_init_v001/build_v007/finger_rig_initial.blend')
 def test_scope_not_generation(self):
  for k,v in [('dataset_generation_allowed',True),('status','fitted')]:
   c=copy.deepcopy(self.c);c[k]=v
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_no_scene_scale_or_focal_override(self):
  for k in ('camera_scale','hand_scale','focal_mm'):
   c=copy.deepcopy(self.c);c['initial_priors'][k]=.78
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_hash_mismatch_rejected(self):
  c=copy.deepcopy(self.c);c['sources']['finger_asset']['sha256']='0'*64
  with self.assertRaises(ValueError):m.validate_config(c)
 def test_bad_metric_rejected(self):
  for x in (-1,True,float('nan'),1e100):
   c=copy.deepcopy(self.c);c['initial_priors']['cable_radius_m']=x
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_source_patch_binding(self):
  c=copy.deepcopy(self.c);c['sources']['skin_forearm']=c['sources']['wristband_weave']
  with self.assertRaises(ValueError):m.validate_config(c)
 def test_original_canonical_order_and_units(self):
  d=m.canonical_definition(self.c)
  self.assertEqual(d['corner_order'][1],'xmax_ymin_zmin')
  self.assertEqual(d['corners_3d_m'],[[x*.001 for x in p] for p in d['corners_3d_mm']])
  self.assertTrue(m.verify_canonical_definition(self.c,d))
 def test_identical_point_set_permutation_rejected(self):
  d=m.canonical_definition(self.c);bad=copy.deepcopy(d);perm=[0,4,3,7,1,5,2,6]
  for key in ('corner_order','corners_3d_mm','corners_3d_m'):bad[key]=[d[key][i] for i in perm]
  self.assertEqual(sorted(map(tuple,bad['corners_3d_m'])),sorted(map(tuple,d['corners_3d_m'])))
  with self.assertRaises(ValueError):m.verify_canonical_definition(self.c,bad)
 def test_points_only_or_names_only_permutation_rejected(self):
  d=m.canonical_definition(self.c)
  for key in ('corner_order','corners_3d_mm','corners_3d_m'):
   bad=copy.deepcopy(d);bad[key][0],bad[key][1]=bad[key][1],bad[key][0]
   with self.assertRaises(ValueError):m.verify_canonical_definition(self.c,bad)
class BodyShadingContractTests(unittest.TestCase):
 def setUp(self):
  self.shading=json.loads((m.ROOT/'data/action4_scene_assets/global_scene_init_v001/build_v006/integrated_contract.json').read_text())['body_shading']
 def test_all_control_and_evaluated_smooth_with_original_normal_metadata(self):
  self.assertTrue(m.validate_body_shading_contract(self.shading))
  for key in ('control','evaluated'):
   self.assertEqual(self.shading['current'][key]['flat_polygons'],0)
   self.assertEqual(self.shading['source'][key]['smooth_polygons'],0)
 def test_flat_candidate_cannot_redefine_acceptance(self):
  bad=copy.deepcopy(self.shading)
  bad['current']=copy.deepcopy(bad['source'])
  with self.assertRaisesRegex(ValueError,'must be smooth'):m.validate_body_shading_contract(bad)
 def test_normal_metadata_is_not_silently_changed(self):
  for key,value in [('use_auto_smooth',True),('has_custom_normals',True),('sharp_edges',1),('auto_smooth_angle_rad',.1)]:
   bad=copy.deepcopy(self.shading);bad['current']['control'][key]=value
   with self.assertRaises(ValueError):m.validate_body_shading_contract(bad)
 def test_missing_explicit_contract_rejected(self):
  with self.assertRaises(ValueError):m.validate_body_shading_contract(None)

if __name__=='__main__':unittest.main()

