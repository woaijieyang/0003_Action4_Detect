import copy,unittest
import numpy as np
from test_action4_video_pose_stabilizer import packet,apply,CAD,K
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_final_pose_record import resolve_final_pose,CONTRACT
from model.action4_current_pose_prior import current_pose_prior
class FinalPoseTests(unittest.TestCase):
 def item(self):
  s=VideoPoseStabilizer(StabilizerConfig())
  for f in range(5):apply(s,packet(f),f)
  return apply(s,packet(5,angle=2.6),5)['instances'][0]
 def test_new_producer_has_one_authoritative_pose(self):
  i=self.item();p,r=resolve_final_pose(i,CAD,K)
  self.assertEqual(i['pose_output_contract'],CONTRACT);self.assertTrue(p['finite'])
  self.assertEqual(i['rotation'],p['rotation']);self.assertEqual(i['translation_m'],p['translation_m'])
  self.assertFalse(r['geometry_changed']);self.assertFalse(r['pose_accuracy_admitted'])
 def test_legacy_final_is_used_without_touching_raw(self):
  i=self.item();i.pop('pose_output_contract');i['final_pose'].pop('finite');i['rotation']=copy.deepcopy(i['raw_network_pose']['rotation']);i['translation_m']=copy.deepcopy(i['raw_network_pose']['translation_m']);old=copy.deepcopy(i)
  p,r=resolve_final_pose(i,CAD,K);self.assertTrue(r['finite_inferred_from_numeric_checks']);self.assertTrue(r['legacy_aliases_ignored']);self.assertEqual(i,old)
  inp=dict(track_id='a',raw_A0_pose=i['raw_network_pose'],selected_A0_pose=p,recovery_evidence={k:i[k] for k in ['frame_id','pts','current_support_status','pose_valid','roi_decision_id']})
  prior=current_pose_prior(inp,track_id='a',frame_id=5,pts=5,canonical8=CAD,pose_key='selected_A0_pose')
  self.assertTrue(prior['available']);np.testing.assert_array_equal(np.array(prior['pose'])[:3,:3],p['rotation'])
 def test_v2_mismatch_and_tampered_corners_rejected(self):
  for field in ['alias','corners','nonfinite','invalid_flag','units','ids']:
   i=self.item()
   if field=='alias':i['rotation']=i['raw_network_pose']['rotation']
   if field=='corners':i['corners_final'][0][0]+=1
   if field=='nonfinite':i['final_pose']['translation_m'][0]=float('nan')
   if field=='invalid_flag':i['final_pose']['finite']=False
   if field=='units':i['final_pose']['coordinate_frame']='world_mm'
   if field=='ids':i['corner_ids'][0]=False
   with self.assertRaises(ValueError):resolve_final_pose(i,CAD,K)
 def test_unknown_missing_flag_is_not_silently_upgraded(self):
  i=self.item();i.pop('pose_output_contract');i['post_A0_geometry']='unknown';i['final_pose'].pop('finite')
  with self.assertRaises(ValueError):resolve_final_pose(i,CAD,K)
