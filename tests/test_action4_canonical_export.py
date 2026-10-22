"""Pure canonical8 export checks; no scene, GPU or filesystem fixtures."""
import copy,unittest
import numpy as np
from data import action4_canonical_export as m
from data.action4_adapter import DataAdapter


def context():
 mm=DataAdapter.fixed_corners([-35.,-16.,-22.],[35.,16.,22.]);ow=np.eye(4);ow[2,3]=.5
 c={'canonical':{'source_manifest':{'path':'fixture/canonical.json','sha256':'0'*64},'corner_order':list(m.ORDER),'corners_3d_mm':mm.tolist(),'corners_3d_m':(mm*.001).tolist()},'instance_id':1,'object_to_world_m':ow.tolist(),'camera_to_world_blender_m':m.BLENDER_CAMERA_TO_CV.tolist(),'K':[[100.,0,49.5],[0,100.,39.5],[0,0,1]],'image_size_wh':[100,80],'near_plane_m':.001}
 return c


class CanonicalExportTests(unittest.TestCase):
 def test_all8_order_units_transforms_and_independent_projection(self):
  c=context();v=m.build_export(**c);p=np.array(v['points_capture_m']);uv=p@np.array(c['K']).T;uv=uv[:,:2]/uv[:,2,None]
  np.testing.assert_allclose(v['points_native_px'],uv,atol=1e-12);self.assertEqual(v['corner_ids'],list(range(8)));self.assertTrue(m.verify_export(v,**c));self.assertEqual(v['point_near_plane_projectable'],[True]*8)
 def test_legacy_pixels_and_flags_retained_no_supervision_claim(self):
  c=context();v=m.build_export(**c);c.update(legacy_corners_px=v['points_native_px'],legacy_corner_visible=[False]*8,ray_clear_to_virtual_point=[False]*8);v=m.build_export(**c)
  self.assertEqual(v['legacy_corners_px'],c['legacy_corners_px']);self.assertEqual(v['legacy_corner_visible'],[False]*8);self.assertFalse(v['training_accepted']);self.assertEqual(v['point_near_plane_projectable'],[True]*8)
 def test_outside_keeps8_geometries_and_unclamped_pixels(self):
  c=context();c['object_to_world_m'][0][3]=10;v=m.build_export(**c);self.assertEqual(len(v['points_world_m']),8);self.assertEqual(v['point_inframe'],[False]*8);self.assertTrue(all(p[0]>100 for p in v['points_native_px']))
 def test_positive_near_and_behind_states_separate(self):
  c=context();c['object_to_world_m'][2][3]=.023;c['near_plane_m']=.01;v=m.build_export(**c)
  self.assertEqual(v['point_positive_depth'],[True]*8);self.assertEqual(v['point_near_plane_projectable'],[False]*4+[True]*4);self.assertEqual(v['points_native_px'][:4],[None]*4)
  c['object_to_world_m'][2][3]=0;v=m.build_export(**c);self.assertEqual(v['point_positive_depth'],[False]*4+[True]*4)
 def test_pixel_extent_boundary_differs_from_center_domain(self):
  c=context();c['K']=[[1.,0,0],[0,1.,0],[0,0,1]];v=m.build_export(**c);self.assertTrue(all(v['point_inframe']));self.assertTrue(any(p[0]<0 for p in v['points_native_px']))
 def test_manifest_order_and_mm_m_permutations_rejected(self):
  for kind in ['order','points','units']:
   c=context()
   if kind=='order':c['canonical']['corner_order'][0],c['canonical']['corner_order'][1]=c['canonical']['corner_order'][1],c['canonical']['corner_order'][0]
   if kind=='points':
    for key in ['corners_3d_mm','corners_3d_m']:c['canonical'][key][0],c['canonical'][key][1]=c['canonical'][key][1],c['canonical'][key][0]
   if kind=='units':c['canonical']['corners_3d_m']=c['canonical']['corners_3d_mm']
   with self.assertRaises(ValueError):m.build_export(**c)
 def test_trusted_pose_or_camera_mismatch_rejected(self):
  c=context();v=m.build_export(**c)
  for key in ['object_to_world_m','camera_to_world_blender_m','K']:
   changed=copy.deepcopy(c);changed[key][0][-1]+=.1
   with self.assertRaises(ValueError):m.verify_export(v,**changed)
 def test_actual_row_pixel_permutation_rejected_even_same_envelope(self):
  c=context();v=m.build_export(**c);pts=v['points_native_px'];c['legacy_corners_px']=[pts[i] for i in [0,4,3,7,1,5,2,6]]
  with self.assertRaisesRegex(ValueError,'reprojection'):m.build_export(**c)
 def test_metadata_mutation_caught_independent_of_bbox(self):
  c=context();v=m.build_export(**c)
  for key in ['points_world_m','points_capture_m','points_native_px']:
   changed=copy.deepcopy(v);changed[key][0][0]+=.1
   with self.assertRaises(ValueError):m.verify_export(changed,**c)
 def test_invalid_scale_reflection_and_nonfinite_rejected(self):
  for number in [2.,-1.,float('nan')]:
   c=context();c['object_to_world_m'][0][0]=number
   with self.assertRaises(ValueError):m.build_export(**c)
 def test_cross_numpy_roundoff_only_derived_values_tolerated(self):
  c=context();v=m.build_export(**c);v['points_native_px'][0][0]+=1e-12;v['points_capture_m'][0][0]+=1e-16
  self.assertTrue(m.verify_export(v,**c));v['points_native_px'][0][0]+=.001
  with self.assertRaises(ValueError):m.verify_export(v,**c)
 def test_status_bool_and_identity_types_are_exact(self):
  c=context();v=m.build_export(**c)
  for key,value in [('instance_id',True),('training_accepted',0),('point_positive_depth',[1]*8)]:
   bad=copy.deepcopy(v);bad[key]=value
   with self.assertRaises(ValueError):m.verify_export(bad,**c)
 def test_visibility_semantic_contradiction_rejected(self):
  c=context();v=m.build_export(**c);c.update(legacy_corners_px=v['points_native_px'],legacy_corner_visible=[True]*8,ray_clear_to_virtual_point=[False]*8)
  with self.assertRaisesRegex(ValueError,'contradicts'):m.build_export(**c)

if __name__=='__main__':unittest.main()
