import unittest,copy
import numpy as np
from data.action4_sam_roi_inputs import ImageFrame,array_digest
from data.action4_frozen_detection_context import FrozenDetectionContext
class FrozenSamTests(unittest.TestCase):
 def setUp(self):
  self.rgb=np.zeros((8,8,3),np.uint8);self.frame=ImageFrame("im",self.rgb,"registered_native_domain","b"*64,"native_original",array_digest(self.rgb));self.rows=[{"grid_index":7,"score":.9,"bbox_full_xyxy":[1.,1.,7.,7.]}]
  self.args=dict(sam_predictions={("im","b0:im:grid:7","registered_native_domain"):None},sam_sources={"b0_model_sha256":"a"*64},b0_sha256="a"*64,scoring_policy="center",source_cache_identity="c"*64)
 def test_same_frame_frozen_rows_retained(self):
  x=FrozenDetectionContext.from_sam_frame(self.frame,np.eye(3),self.rows,**self.args)
  self.assertEqual(x.consume(self.rgb,np.eye(3),b0_sha256="a"*64,scoring_policy="center"),self.rows)
 def test_foreign_missing_ID_or_pixel_namespace_rejected(self):
  for mapping in [{},{("other","b0:im:grid:7","registered_native_domain"):None},{("im","b0:im:grid:7","other"):None}]:
   a=dict(self.args,sam_predictions=mapping)
   with self.assertRaises(ValueError):FrozenDetectionContext.from_sam_frame(self.frame,np.eye(3),self.rows,**a)
 def test_false_prediction_ID_and_wrong_model_rejected(self):
  rows=copy.deepcopy(self.rows);rows[0]["prediction_id"]="b0:other:grid:7"
  with self.assertRaises(ValueError):FrozenDetectionContext.from_sam_frame(self.frame,np.eye(3),rows,**self.args)
  a=dict(self.args,sam_sources={"b0_model_sha256":"d"*64})
  with self.assertRaises(ValueError):FrozenDetectionContext.from_sam_frame(self.frame,np.eye(3),self.rows,**a)
 def test_image_track_fields_remain_forbidden(self):
  rows=copy.deepcopy(self.rows);rows[0]["track_id"]="track_00"
  with self.assertRaises(ValueError):FrozenDetectionContext.from_sam_frame(self.frame,np.eye(3),rows,**self.args)
class AsyncProfileTests(unittest.TestCase):
 def test_stage3_profile_keeps_frozen_B0_ROI_and_cache(self):
  from eval.action4_stage2_eval import bind_frozen_image_inputs
  from data.action4_prediction_roi_data import SAM_NATIVE_PROTOCOL
  c=dict(training_protocol=SAM_NATIVE_PROTOCOL,roi_expansion=1.5,frozen_eval_b0_predictions={"path":"frozen","sha256":"c"*64},evaluation_counterpart_sha256="a"*64,sam_eval_cache={"manifest":"fixture"},sam_enabled=True)
  p=bind_frozen_image_inputs({"sam_cache":c["sam_eval_cache"]},c)
  self.assertEqual(p["prediction_input_protocol"],"action4.frozen_B0_SAM_original_eval.v1")
  self.assertEqual(p["roi_expansion"],1.5);self.assertEqual(p["frozen_b0_predictions"],c["frozen_eval_b0_predictions"])
  with self.assertRaises(ValueError):bind_frozen_image_inputs({},c)

if __name__=="__main__":unittest.main()
