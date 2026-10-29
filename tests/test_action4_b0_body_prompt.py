import copy,unittest
import numpy as np
from model import action4_sam_seed_extent as api

def inputs(box=None):
 box=[20.,30.,80.,90.] if box is None else box
 d={'candidate_id':'b0:frame4:grid12','bbox_full_xyxy':box,'score':.95,'refined_quality_score':.94}
 c={'accepted':True,'candidate_id':d['candidate_id'],'raw_B0_box':box,'score':d['score'],'refined_quality_score':d['refined_quality_score']}
 return d,c
class BodyScopePromptTests(unittest.TestCase):
 def fn(self):
  self.assertTrue(callable(getattr(api,'b0_body_scope_prompt',None)),'B0-derived native body scope API missing')
  return api.b0_body_scope_prompt
 def call(self,d=None,c=None):
  if d is None:d,c=inputs()
  return self.fn()('sid',4,1,d,c,200,120)
 def test_native_prompt_no_semantic_reset_and_model_provenance(self):
  q,proof=self.call()
  self.assertEqual(q['point_labels'],[2,3,1,0,0,0,0])
  self.assertFalse({'text','bounding_boxes'}&q.keys())
  np.testing.assert_allclose(np.array(q['points'])*[200,120],[[20,30],[80,90],[50,60],[11,60],[89,60],[50,21],[50,99]],rtol=0,atol=1e-12)
  self.assertEqual(proof['candidate_id'],'b0:frame4:grid12')
  self.assertFalse(proof['body_scope_certified']);self.assertFalse(proof['manual_pixel_inputs'])
 def test_stale_or_untrusted_calibration_rejected(self):
  for key,val in [('accepted',False),('candidate_id','other'),('raw_B0_box',[0,0,1,1]),('score',.1),('refined_quality_score',.1)]:
   d,c=inputs();c[key]=val
   with self.assertRaises(ValueError):self.call(d,c)
 def test_low_nonfinite_or_missing_quality_rejected(self):
  for val in [.899,float('nan'),float('inf'),None]:
   d,c=inputs();d['refined_quality_score']=val;c['refined_quality_score']=val
   with self.assertRaises(ValueError):self.call(d,c)
 def test_border_negatives_never_land_inside_declared_body(self):
  d,c=inputs([0.,0.,80.,90.]);q,p=self.call(d,c)
  negative=np.array(q['points'])[np.array(q['point_labels'])==0]*[200,120]
  self.assertTrue(len(negative)>0)
  self.assertTrue(all(x>80 or y>90 for x,y in negative))
  self.assertGreater(p['dropped_negative_points'],0)
 def test_full_image_box_has_no_legal_negative_support(self):
  d,c=inputs([0.,0.,200.,120.])
  with self.assertRaises(ValueError):self.call(d,c)
 def test_source_unmodified_and_targetID_preserved(self):
  d,c=inputs();before=copy.deepcopy((d,c));q,p=self.call(d,c)
  self.assertEqual((d,c),before);self.assertEqual(q['obj_id'],1);self.assertEqual(q['frame_index'],4)
if __name__=='__main__':unittest.main()
