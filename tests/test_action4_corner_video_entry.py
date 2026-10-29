import unittest
from unittest.mock import patch
from types import SimpleNamespace
from eval.action4_staged_pipeline import StagedPipeline

class CornerVideoEntryTests(unittest.TestCase):
 def test_existing_entry_dispatches_explicit_trial_without_B0_detection(self):
  p=object.__new__(StagedPipeline);p.a0_kind="joint_pose_sam_corner_candidate";p.roi_expansion=1.5
  p.predict=lambda *a,**k:self.fail("Must not re-enter image/B0 path")
  with patch("eval.action4_corner_video.predict_corner_video",return_value={"actual":"video_corner"}) as call:
   out=p.predict_video("rgb","K","CAD",video_context="resolved",sam_video_context="mask",roi_expansion=1.5,corner_trial_binding={"path":"x","sha256":"s"})
   self.assertEqual(out,{"actual":"video_corner"})
   self.assertEqual(call.call_args.kwargs["video_context"],"resolved")
   self.assertEqual(call.call_args.kwargs["sam_video_context"],"mask")
 def test_trial_cannot_silently_stack_old_smoothing(self):
  p=object.__new__(StagedPipeline)
  with self.assertRaises(ValueError):
   p.predict_video(None,None,None,video_context=None,corner_trial_binding={},pose_stabilizer=object())
 def test_trial_other_crop_rejected(self):
  p=object.__new__(StagedPipeline)
  with self.assertRaises(ValueError):
   p.predict_video(None,None,None,video_context=None,corner_trial_binding={},roi_expansion=2.25)
 def test_raw_entry_remains_same_object(self):
  p=object.__new__(StagedPipeline);p.a0_kind="joint_pose";expected={}
  p.predict=lambda *a,**k:expected
  self.assertIs(p.predict_video(None,None,None,video_context=None),expected)

 def test_completed_model115_dispatch_and_wrong_override(self):
  p=object.__new__(StagedPipeline);p.a0_kind='joint_pose_sam_corner_candidate';p.roi_expansion=1.15
  with patch('eval.action4_corner_video.predict_corner_video',return_value={'actual':'roi115'}) as call:
   self.assertEqual(p.predict_video(None,None,None,video_context='r',sam_video_context='s',roi_expansion=1.15,corner_trial_binding={}),{'actual':'roi115'})
  with self.assertRaises(ValueError):p.predict_video(None,None,None,video_context=None,roi_expansion=1.5,corner_trial_binding={})
