import unittest
from model.action4_sam_state_offload import state_offload_bridge
class StateOffloadTests(unittest.TestCase):
 def original(self,video_height,video_width,num_frames,cached_features=None,offload_video_to_cpu=False,offload_state_to_cpu=False):
  return {"storage_cpu":offload_state_to_cpu,"video_cpu":offload_video_to_cpu,"features":cached_features,"frames":num_frames}
 def test_explicit_candidate_uses_official_cpu_storage(self):
  f=state_offload_bridge(self.original)
  self.assertTrue(f(10,20,30)["storage_cpu"])
 def test_other_arguments_and_shared_features_preserved(self):
  feature=object();r=state_offload_bridge(self.original)(10,20,30,cached_features=feature,offload_video_to_cpu=True)
  self.assertIs(r["features"],feature);self.assertTrue(r["video_cpu"]);self.assertEqual(r["frames"],30)
 def test_unknown_options_not_swallowed(self):
  with self.assertRaises(TypeError):state_offload_bridge(self.original)(10,20,30,unknown=True)
