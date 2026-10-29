import unittest
from model.action4_current_pose_prior import native_prior_directions
from eval.action4_native_clip import runtime_options
class PriorDirectionsTest(unittest.TestCase):
 def test_translation_only_does_not_enable_rotation(self):
  self.assertEqual(native_prior_directions("translation_only",True),[False]*3+[True]*3)
 def test_existing_modes_preserved(self):
  self.assertEqual(native_prior_directions("continuous",True),[True]*6)
  self.assertEqual(native_prior_directions("disabled_continuous",True),[False]*6)
 def test_absent_current_support_disables_all_modes(self):
  for mode in ("continuous","disabled_continuous","translation_only"):
   self.assertEqual(native_prior_directions(mode,False),[False]*6)
 def test_unknown_mode_and_implicit_availability_rejected(self):
  with self.assertRaises(ValueError):native_prior_directions("guess",True)
  with self.assertRaises(ValueError):native_prior_directions("continuous",1)
 def test_manifest_selects_registered_mode(self):
  self.assertEqual(runtime_options(dict(frame_ids=[4],A0_prior_mode="translation_only")),("translation_only",{4}))
