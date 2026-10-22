"""Focused source/pose/weight contracts; CPU only."""
import unittest,json,copy
import numpy as np
from data.action4_finger_rig import ROOT,validate_config,validate_pose,source_mapping,skin_fractions,bounded,DIGITS
class FingerContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=json.loads((ROOT/'configs/data/action4_finger_rig_init_v001.json').read_text())
  cls.geo=json.loads((ROOT/'outputs/action4_realism_reassessment_v001/finger_rig_init_v001/source_geometry.json').read_text())
  cls.mapping=source_mapping(cls.geo['vertices'],cls.geo['faces'],cls.geo['face_sets'])
 def test_exact_source_configuration(self):validate_config(self.cfg)
 def test_both_side_source_chains(self):
  self.assertEqual(len(self.mapping),10)
  for k,m in self.mapping.items():
   self.assertEqual(len(m['lengths_m']),2 if m['digit']=='thumb' else 3)
   self.assertTrue(all(len(r)==12 for r in m['joint_boundary_vertex_ids']))
   for axis in m['rest_axes']:
    M=np.column_stack([axis['local_X_flexion_world'],axis['local_Y_longitudinal_world'],axis['local_Z_inward_world']]);np.testing.assert_allclose(M.T@M,np.eye(3),atol=1e-10);self.assertAlmostEqual(np.linalg.det(M),1)
 def test_opposition_semantics_toward_index_both_sides(self):
  for side in ['left','right']:
   thumb=self.mapping[side+'_thumb'];root=np.array(thumb['joints_m'][0]);tip=np.array(thumb['joints_m'][-1]);target=np.array(self.mapping[side+'_index']['joints_m'][0])-root;axis=np.array(thumb['root_opposition_axis_world'])
   self.assertGreater(float(np.dot(np.cross(axis,tip-root),target)),0)
 def test_mirrored_source_lengths_not_mirrored_camera(self):
  for d in DIGITS:np.testing.assert_allclose(self.mapping['left_'+d]['lengths_m'],self.mapping['right_'+d]['lengths_m'],atol=1e-7)
 def test_no_fitted_claim_or_ranges(self):
  for key,v in [('dataset_generation_allowed',True),('status','fitted'),('calibrated_sampling_ranges',{})]:
   c=copy.deepcopy(self.cfg);c[key]=v
   with self.assertRaises(ValueError):validate_config(c,False)
 def test_no_shape_scale_or_mount_inputs(self):
  for k in ['camera_scale','left_hand','bone_lengths','left_index_root_xyz']:
   with self.assertRaises(ValueError):validate_pose({k:[0,0,0]})
 def test_nonfinite_and_bound_rejections(self):
  for v in [[26,0,0],[True,0,0],[float('nan'),0,0],[float('inf'),0,0],[0,0]]:
   with self.assertRaises(ValueError):validate_pose({'left_index':v})
 def test_weight_partition_and_root_transition(self):
  for m in self.mapping.values():
   joints=np.array(m['joints_m']);root=skin_fractions(joints[0],joints);self.assertAlmostEqual(root[0],1)
   for a,b in zip(joints[:-1],joints[1:]):
    for t in np.linspace(0,1,25):
     w=skin_fractions(a*(1-t)+b*t,joints);self.assertTrue(all(0<=x<=1 for x in w));self.assertAlmostEqual(sum(w),1)
 def test_source_hash_change_rejects(self):
  c=copy.deepcopy(self.cfg);c['source_asset']['sha256']='0'*64
  with self.assertRaises(ValueError):validate_config(c)
 def test_boundary_escape_rejects(self):
  with self.assertRaises(ValueError):bounded('../outside')
if __name__=='__main__':unittest.main(verbosity=2)
