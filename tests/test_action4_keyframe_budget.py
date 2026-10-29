import unittest
from eval.action4_native_clip import texture_keyframe_budget
class BudgetTest(unittest.TestCase):
 def test_default_preserves_native_one(self):self.assertEqual(texture_keyframe_budget({}),1)
 def test_registered_three(self):self.assertEqual(texture_keyframe_budget({"texture_keyframes":3}),3)
 def test_invalid_and_unbounded_rejected(self):
  for x in [0,4,True,3.0,"3"]:
   with self.assertRaises(ValueError):texture_keyframe_budget({"texture_keyframes":x})

 def test_guard_default_off_and_bool_only(self):
  from eval.action4_native_clip import keyframe_protection
  self.assertFalse(keyframe_protection({}))
  self.assertTrue(keyframe_protection({"retain_nonempty_keyframe":True}))
  for value in [1,"yes",None]:
   with self.assertRaises(ValueError):keyframe_protection({"retain_nonempty_keyframe":value})

 def test_pose_cache_policy_is_explicit(self):
  from eval.action4_native_clip import pose_cache_refresh
  self.assertFalse(pose_cache_refresh({}))
  self.assertTrue(pose_cache_refresh({"refresh_result_pose_cache":True}))
  for value in [1,"true",None]:
   with self.assertRaises(ValueError):pose_cache_refresh({"refresh_result_pose_cache":value})
