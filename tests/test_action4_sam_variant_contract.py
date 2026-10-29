import importlib,importlib.util,copy,unittest
from eval.action4_rgb_variant_b0 import pack_row
def entry():
 return {'image_id':'sampleA','variant':'light','variant_id':'light:fixed','actual_input':{'path':'outputs/x.png','sha256':'a'*64},'actual_pixels_sha256':'b'*64,'native_shape':[100,200,3],'original_source':{'path':'data/original.png','sha256':'c'*64},'recipe':{'variant_id':'light:fixed','variant_sha256':'b'*64}}
def prediction(e,boxes=None):
 boxes=[[-10.,20.,80.,120.]] if boxes is None else boxes
 return pack_row(e,[{'bbox_full_xyxy':b,'grid_index':i,'score':.9} for i,b in enumerate(boxes)])
class VariantPromptTests(unittest.TestCase):
 def fn(self):
  self.assertIsNotNone(importlib.util.find_spec('eval.action4_sam_variant_contract'),'variant prompt contract missing')
  return importlib.import_module('eval.action4_sam_variant_contract').prompt_row
 def test_actual_variant_pixels_and_original_provenance_stay_distinct(self):
  e=entry();r=self.fn()(e,prediction(e),{'proof':'fixture'})
  self.assertEqual(r['image_path'],'outputs/x.png');self.assertEqual(r['image_sha256'],'a'*64)
  self.assertEqual(r['original_source']['sha256'],'c'*64);self.assertEqual(r['variant_pixels_sha256'],'b'*64)
  self.assertEqual(r['rgb_variant'],'light');self.assertEqual(r['variant_id'],'light:fixed')
  c=r['candidates'][0];self.assertEqual(c['candidate_id'],'b0:sampleA:light:fixed:grid:0')
  self.assertEqual(c['bbox_sam_native_xyxy'],[0.,20.,80.,100.])
  self.assertEqual(c['bbox_sam_normalized_xywh'],[0.,.2,.4,.8])
 def test_original_alias_wrong_variant_or_wrong_image_rejected(self):
  e=entry();p=prediction(e)
  for field,v in [('variant_id','native_original'),('sample_id','other'),('actual_pixels_sha256','d'*64)]:
   q=copy.deepcopy(p);q[field]=v
   with self.assertRaises(ValueError):self.fn()(e,q,{'proof':'fixture'})
 def test_zero_candidate_images_remain_in_domain(self):
  e=entry();r=self.fn()(e,pack_row(e,[]),{'proof':'fixture'})
  self.assertEqual(r['candidates'],[]);self.assertEqual(r['image_id'],'sampleA')
 def test_original_boxes_and_low_score_candidates_are_not_dropped(self):
  e=entry();p=prediction(e,[[-10,20,80,120],[300,0,320,10]])
  p['instances'][1]['score']=.1;p=pack_row(e,p['instances'])
  r=self.fn()(e,p,{'proof':'fixture'});self.assertEqual(len(r['candidates']),2)
  self.assertEqual(r['candidates'][0]['bbox_full_native_xyxy'],[-10,20,80,120])
  self.assertFalse(r['candidates'][1]['eligible_for_SAM_diagnostic_prompt'])
 def test_duplicate_grid_nan_box_and_foreign_id_are_rejected(self):
  e=entry();p=prediction(e)
  p['candidates'][0]['prediction_id']='b0:other'
  with self.assertRaises(ValueError):self.fn()(e,p,{'proof':'fixture'})
  p=prediction(e);p['instances'][0]['bbox_full_xyxy'][0]=float('nan')
  with self.assertRaises(ValueError):self.fn()(e,p,{'proof':'fixture'})
 def test_no_mutation_and_no_GT_inputs_in_prompt_row(self):
  e=entry();p=prediction(e);before=copy.deepcopy((e,p))
  r=self.fn()(e,p,{'proof':'fixture'});self.assertEqual((e,p),before)
  self.assertNotIn('annotations',r);self.assertNotIn('gt_instances',r)
if __name__=='__main__':unittest.main()
