import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import eval.action4_sam_image_eval as native
from eval.action4_rgb_variant_b0 import pack_row
from eval.action4_sam_variant_contract import prompt_row
R=Path('/data/zy/vln/a-test')
spec=importlib.util.spec_from_file_location('image_fixture',R/'tests/test_action4_sam_image_eval.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
class NativeVariantTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=R/'outputs');self.out=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_variant_identity_reaches_existing_native_job_and_receipt(self):
  self.assertEqual(getattr(native,'SCHEMA_VARIANT',None),'action4.sam.single_image_rgb_variant.v1')
  image=self.out/'light.png';Image.fromarray(np.full((8,10,3),70,np.uint8)).save(image)
  entry={'image_id':'sampleA','variant':'light','variant_id':'native_light:fixture','actual_input':{'path':str(image.relative_to(R)),'sha256':native.file_sha(image)},'actual_pixels_sha256':'b'*64,'native_shape':[8,10,3],'original_source':{'path':'data/not_consumed','sha256':'c'*64},'recipe':{'variant_id':'native_light:fixture','variant_sha256':'b'*64}}
  pred=pack_row(entry,[{'bbox_full_xyxy':[1,1,4,5],'grid_index':1,'score':.9}])
  row=prompt_row(entry,pred,{'B0_bundle':{'path':'models/fixture','sha256':'d'*64}})
  cfg={'schema':native.SCHEMA_VARIANT,'manifest_sha256':'e'*64,'upstream_role':'frozen_completed_B0_rgb_variant','K_contract':{'K':None,'status':'not_consumed'},'native':{'text':'camera'},'split':'train','rgb_variant':'light','binding':{'mask_in_box_min':.8,'multi_target_support_min':.7,'unique_margin':.15}}
  job={'image':row,'candidate':row['candidates'][0],'task_key':'fixture-variant-job'}
  result=native.execute_job(job,cfg,{'identity':'f'*64},fixture.FakeBackend(),self.out)
  self.assertTrue(result['available']);self.assertTrue(result['close_success'])
  self.assertEqual(result['variant_id'],entry['variant_id']);self.assertEqual(result['variant_pixels_sha256'],'b'*64)
  self.assertEqual(result['original_source'],entry['original_source']);self.assertEqual(result['rgb_variant'],'light')
  self.assertEqual(result['prediction_id'],row['candidates'][0]['candidate_id'])
 def test_bad_variant_alias_is_rejected_before_backend(self):
  self.assertTrue(callable(getattr(native,'variant_job_fields',None)))
  with self.assertRaises(ValueError):
   native.variant_job_fields({'schema':'action4.sam.prediction_prompt_image.variant.v1','rgb_variant':'light','variant_id':'native_original','variant_pixels_sha256':'b'*64},{'rgb_variant':'light','split':'train'})

 def test_existing_v1_and_v2_execute_job_preserve_legacy_receipts(self):
  image=self.out/'legacy.png';Image.fromarray(np.full((8,10,3),50,np.uint8)).save(image)
  candidate={'candidate_id':'legacy:image:one','bbox_full_native_xyxy':[1,1,4,5],'bbox_sam_native_xyxy':[1,1,4,5],'bbox_sam_normalized_xywh':[.1,.125,.3,.5],'eligible_for_SAM_diagnostic_prompt':True,'identity_binding':{'sample_id':'legacy'}}
  for schema in [native.SCHEMA,native.SCHEMA_V2]:
   row={'image_id':'legacy','image_path':str(image.relative_to(R)),'image_sha256':native.file_sha(image),'native_shape_hw':[8,10],'pixel_space_id':'fixture','source_receipt':{'parent_sha256':'a'*64} if schema==native.SCHEMA else {'b0_bundle':{'sha256':'a'*64}},'candidates':[candidate]}
   cfg={'schema':schema,'manifest_sha256':'e'*64,'upstream_role':'legacy_fixture','K_contract':{'K':None},'native':{'text':'camera'},'split':'eval','rgb_variant':'native_original','binding':{'mask_in_box_min':.8,'multi_target_support_min':.7,'unique_margin':.15}}
   dest=self.out/('v1' if schema==native.SCHEMA else 'v2');dest.mkdir()
   result=native.execute_job({'image':row,'candidate':candidate,'task_key':'legacy-fixture'},cfg,{'identity':'f'*64},fixture.FakeBackend(),dest)
   self.assertTrue(result['available']);self.assertNotIn('variant_pixels_sha256',result);self.assertNotIn('original_source',result)
 def test_manifest_dispatch_retains_empty_images_and_checks_counts(self):
  from unittest.mock import patch
  from eval.action4_sam_variant_contract import load_native_manifest
  rows=[{'image_id':'empty','candidates':[]},{'image_id':'one','candidates':[{'eligible_for_SAM_diagnostic_prompt':True}]}]
  prepared={'images':2,'variant':'light','saved_candidates':1}
  cfg={'prepared_variant':{},'expected_images':2,'rgb_variant':'light','expected_saved_candidates':1,'expected_prompts':1}
  with patch('eval.action4_sam_variant_contract.verify_prepared',return_value=(prepared,rows)):
   actual,jobs=load_native_manifest(cfg);self.assertEqual(actual,rows);self.assertEqual(len(jobs),1)
   with self.assertRaises(ValueError):load_native_manifest(dict(cfg,expected_prompts=2))

if __name__=='__main__':unittest.main()
