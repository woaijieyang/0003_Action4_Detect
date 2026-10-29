"""Selection branch fixtures, not actual model/seal accuracy evidence."""
import unittest,copy
from types import SimpleNamespace
from unittest.mock import patch
from eval.action4_corner_selected_inference import selected_output,SealedCornerSelection,predict_selected_corner
class SelectedCornerTests(unittest.TestCase):
 def payload(self):
  raw=[];candidate=[]
  for i in range(2):
   r={'prediction_id':str(i),'corner_ids':list(range(8)),'corners_px':[[i,k] for k in range(8)],'pose_valid':True,'published':True}
   raw.append(r);c=copy.deepcopy(r);c['pose_diagnostic']={'geometric_candidate_accepted':i==0,'deployed':False}
   if i==0:c['corners_px']=[[i+.25,k+.5] for k in range(8)]
   candidate.append(c)
  return {'schema':'action4.corner_prediction_bridge.v1','raw_instances':raw,'candidate_instances':candidate,'instances':copy.deepcopy(raw),'GT_used':False,'diagnostic_candidate':True}
 def contract(self):return {'quality_ready':True,'global_branch':'candidate','parent_stage_fallback':{'path':'parent','sha256':'a'*64,'selected_step':1}}
 def test_global_selection_keeps_raw_and_local_fallback(self):
  p=self.payload();before=copy.deepcopy(p);r=selected_output(p,self.contract(),{})
  self.assertEqual(p,before);self.assertEqual([x['corners_px'] for x in r['instances']],[x['corners_px'] for x in p['candidate_instances']])
  self.assertEqual(r['instances'][0]['corners_raw'],p['raw_instances'][0]['corners_px']);self.assertEqual(r['instances'][1]['selected_pose_source'],'same_checkpoint_raw_pose_fallback')
  self.assertFalse(r['accuracy_accepted']);self.assertFalse(r['diagnostic_candidate']);self.assertEqual(r['schema'],'action4.selected_corner_prediction.v1')
 def test_unready_rejected(self):
  c=self.contract();c['quality_ready']=False
  with self.assertRaises(ValueError):selected_output(self.payload(),c,{})
 def test_missing_reordered_or_renumbered_instances_rejected(self):
  for mode in ['missing','reordered','renumbered']:
   p=self.payload()
   if mode=='missing':p['candidate_instances'].pop()
   elif mode=='reordered':p['candidate_instances'].reverse()
   else:p['candidate_instances'][0]['corner_ids'].reverse()
   with self.assertRaises(ValueError):selected_output(p,self.contract(),{})
 def test_GT_payload_rejected(self):
  p=self.payload();p['GT_used']=True
  with self.assertRaises(ValueError):selected_output(p,self.contract(),{})
 def test_unverified_selection_rejected(self):
  with self.assertRaises(ValueError):SealedCornerSelection('{}','{}','{}','{}',()).consume(SimpleNamespace())
  with self.assertRaises(ValueError):predict_selected_corner(SimpleNamespace(),None,None,None,sam_context=None)
 def test_baseline_header_not_corner_seal(self):
  with patch('eval.action4_corner_selected_inference._v2_checked',return_value={'schema':'action4.sealed_independent_role.v2','role':'A0','strict_load':True}):
   with self.assertRaisesRegex(ValueError,'calibrated corner'):SealedCornerSelection.load({'path':'x','sha256':'a'*64})
 def test_local_raw_fallback_cannot_change_coordinates(self):
  p=self.payload();p['candidate_instances'][1]['corners_px'][0][0]+=10
  with self.assertRaisesRegex(ValueError,'fallback'):selected_output(p,self.contract(),{})
 def test_accepted_nonfinite_geometry_refused(self):
  p=self.payload();p['candidate_instances'][0]['corners_px'][0][0]=float('nan')
  with self.assertRaisesRegex(ValueError,'finite'):selected_output(p,self.contract(),{})
