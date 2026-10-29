import unittest,copy
from data.action4_prediction_roi_data import PROTOCOL,SAM_NATIVE_PROTOCOL,enabled,validate_config
from model.training.action4_roi_adaptation import validate_parent_input_contract
class Stage3ProtocolTests(unittest.TestCase):
 def config(self):
  return dict(training_protocol=SAM_NATIVE_PROTOCOL,role="A0",mode="integration",roi_source="frozen_b0_prediction",
   roi_expansion=1.5,roi_size=448,preprocess={"roi_expansion":1.5},train_backbone=False,sam_enabled=True,
   a0_variant="a0_sam_image",prediction_roi_manifest={},sam_training_associations={},evaluation_counterpart_sha256="a"*64,
   sam_cache={"manifest":{},"expected_sources":{"upstream_mode":"b0_completed_native_receipt_v2","b0_model_sha256":"a"*64}})
 def test_registered_data_consumer_is_not_baseline_extension(self):
  c=self.config();self.assertTrue(validate_config(c,"train"));self.assertFalse(enabled(c))
  validate_parent_input_contract(dict(training_protocol=PROTOCOL,roi_source="frozen_b0_prediction",roi_expansion=1.5),c)
 def test_baseline_guard_still_rejects_SAM(self):
  c=self.config();c["training_protocol"]=PROTOCOL
  with self.assertRaises(ValueError):validate_config(c,"train")
 def test_old_ROI_variants_and_synthetic_proofs_do_not_enter_native_protocol(self):
  for key,value in [("roi_expansion",2.25),("sam_rgb_variants",{}),("roi_jitter",{}),("continuation_segment",{}),("sam_allow_synthetic_fixture",True),("a0_variant","a0_baseline"),("sam_enabled",False)]:
   c=self.config();c[key]=value
   with self.assertRaises(ValueError,msg=key):validate_config(c,"train")
 def test_wrong_B0_cache_and_split_rejected(self):
  c=self.config();c["sam_cache"]["expected_sources"]["b0_model_sha256"]="b"*64
  with self.assertRaises(ValueError):validate_config(c,"train")
  with self.assertRaises(ValueError):validate_config(self.config(),"eval")
if __name__=="__main__":unittest.main()
