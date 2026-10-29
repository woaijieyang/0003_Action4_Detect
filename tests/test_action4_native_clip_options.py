import unittest
from model.action4_current_pose_prior import current_pose_prior
from eval.action4_native_clip import runtime_options
from eval.action4_native_clip_prepare import prepare_clips

class NativeClipOptions(unittest.TestCase):
 def test_legacy_defaults_and_sparse_debug(self):
  self.assertEqual(runtime_options({"frame_ids":[4,5,6]}),("continuous",{4,5,6}))
  self.assertEqual(runtime_options({"frame_ids":[4,5,6],"A0_prior_mode":"disabled_continuous","native_debug_frame_ids":[4]}),("disabled_continuous",{4}))
 def test_unsupported_mode_or_debug_id_rejected(self):
  for patch in [{"A0_prior_mode":"guess"},{"native_debug_frame_ids":[3]},{"native_debug_frame_ids":[4,4]},{"native_debug_frame_ids":[True]}]:
   with self.assertRaises(ValueError):runtime_options(dict({"frame_ids":[4,5,6]},**patch))
 def test_prepare_cannot_mix_explicit_and_legacy_source(self):
  with self.assertRaises(ValueError):prepare_clips("outputs/invalid_unused",{"x":(4,7)},input_lock="unused")
  with self.assertRaises(ValueError):prepare_clips("outputs/invalid_unused",{"x":(4,7)},max_clip_frames=2850)
