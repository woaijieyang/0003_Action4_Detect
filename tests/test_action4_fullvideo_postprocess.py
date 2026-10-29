import unittest,copy
from eval.action4_fullvideo_postprocess import validate_native_complete
class GateTests(unittest.TestCase):
 def done(self):return {"status":"CONTINUOUS_NATIVE_SAM_RESPONSES_COMPLETE_NOT_ASSOCIATION_EFFECT_ACCEPTANCE",
  "source_identity":"source","tracks":[{"track_id":"left","frames":2850,"available":2800},{"track_id":"right","frames":2850,"available":0}]}
 def test_missing_masks_do_not_fake_success_or_drop_frames(self):
  self.assertEqual(validate_native_complete(self.done(),"source",["left","right"]),{"left":2800,"right":0})
 def test_partial_frames_refused(self):
  d=self.done();d["tracks"][0]["frames"]=2849
  with self.assertRaises(ValueError):validate_native_complete(d,"source",["left","right"])
 def test_wrong_source_refused(self):
  with self.assertRaises(ValueError):validate_native_complete(self.done(),"other",["left","right"])
 def test_duplicate_tracks_refused(self):
  d=self.done();d["tracks"][1]["track_id"]="left"
  with self.assertRaises(ValueError):validate_native_complete(d,"source",["left","right"])
 def test_impossible_availability_refused(self):
  d=self.done();d["tracks"][1]["available"]=2851
  with self.assertRaises(ValueError):validate_native_complete(d,"source",["left","right"])
