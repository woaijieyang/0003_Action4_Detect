import unittest
from types import SimpleNamespace
from unittest.mock import patch
from eval.action4_face_pilot_inputs import load_current_face_inputs
class CurrentFaceInputTests(unittest.TestCase):
 def test_sam_enhanced_parent_uses_native_feature_bridge(self):
  pipe=SimpleNamespace(a0_kind="joint_pose_sam",identity={"B0_bundle_sha256":"b"*64},
    acfg={"sam_cache":{"expected_sources":{"sam_model_sha256":"s"*64}}})
  provider=object();camera=object()
  with patch("eval.action4_video_sam_inputs.VideoSamFeatureContext.load",return_value=("rgb","roi","mask_features")) as load:
   self.assertEqual(load_current_face_inputs(pipe,provider,2778,camera),("rgb","roi","mask_features"))
   load.assert_called_once_with(provider,2778,camera,expected_b0_sha256="b"*64,expected_sam_model_sha256="s"*64)
 def test_baseline_does_not_fabricate_sam_features(self):
  calls=[]
  provider=SimpleNamespace(load=lambda f,k:(calls.append((f,k)) or ("rgb","roi")))
  result=load_current_face_inputs(SimpleNamespace(a0_kind="joint_pose"),provider,4,"K")
  self.assertEqual(result,("rgb","roi",None));self.assertEqual(calls,[(4,"K")])
