import unittest
from model.action4_sam_fixed_forward import fixed_forward_parser
class FixedForwardTests(unittest.TestCase):
 def state(self):return {"_action4_fixed_forward_ids":(0,1),"tracker_metadata":{"obj_ids_all_gpu":[0,1]}}
 def test_other_object_not_lost_after_single_object_refinement(self):
  f=fixed_forward_parser(lambda s:("propagation_partial",[0]))
  self.assertEqual(f(self.state()),("propagation_partial",[0,1]))
 def test_new_forward_frames_not_read_from_missing_cache(self):
  f=fixed_forward_parser(lambda s:("propagation_fetch",None))
  self.assertEqual(f(self.state()),("propagation_partial",[0,1]))
 def test_native_replay_unchanged_without_explicit_marker(self):
  f=fixed_forward_parser(lambda s:("propagation_fetch",None))
  self.assertEqual(f({}),("propagation_fetch",None))
 def test_missing_registered_object_refused(self):
  s=self.state();s["tracker_metadata"]["obj_ids_all_gpu"]=[0]
  with self.assertRaises(ValueError):fixed_forward_parser(lambda s:("propagation_partial",[0]))(s)
 def test_unseeded_concept_route_refused(self):
  with self.assertRaises(ValueError):fixed_forward_parser(lambda s:("propagation_full",None))(self.state())
