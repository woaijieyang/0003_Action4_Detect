import unittest,copy,inspect
import numpy as np
from eval.action4_reanchor_roi_provider import native_presence,project_body_events,_canonical,ReanchorROIProvider
from data.action4_video_roi_context import validate_project_body_registration
class RegistrationTests(unittest.TestCase):
 def test_registration_default_is_opt_in(self):
  self.assertIs(inspect.signature(ReanchorROIProvider).parameters["register_project_bodies"].default,False)
 def test_current_presence_is_actual_raw_not_nonempty_removed(self):
  masks=np.ones((2,4,4),bool)
  raw={"frame_index":20,"outputs":{"out_obj_ids":np.array([0,1]),"out_probs":np.array([.8,-1e4]),
                                 "out_binary_masks":masks}}
  self.assertEqual(native_presence(raw,20,(4,4)),{"track_00":"target_present","track_01":"target_absent"})
  masks[0]=False
  self.assertEqual(native_presence(raw,20,(4,4))["track_00"],"target_visibility_unknown")
  self.assertEqual(native_presence(None,20,(4,4))["track_00"],"target_visibility_unknown")
  with self.assertRaises(ValueError):native_presence(raw,19,(4,4))
  raw["outputs"]["out_obj_ids"]=np.array([0,0])
  with self.assertRaises(ValueError):native_presence(raw,20,(4,4))
 def test_project_zero_independent_from_native_reanchor(self):
  scope={"clip_id":"full_original_video_development2850","region_identity":"a"*64,"producer_identity":"b"*64}
  scope=dict(scope,schema="action4.project_body_scope.v1",sequence_identity=_canonical(scope))
  ref={"path":"fixture.json","sha256":"c"*64}
  seed={"seed_fingerprint":"d"*64,"creation_global_frame":4,"creation_pts":40,
        "seed_binding_ref":ref,"seeds_ref":ref}
  event=project_body_events(scope,{"track_00":seed},"e"*64,[1,100],"native")["track_00"]
  owner={"generation":9,"session_id":"new-native","native_obj_id":0,"project_seed":seed}
  validate_project_body_registration(event,scope,track_id="track_00",frame_id=20,pts=200,time_base=[1,100],
       pixel_space_id="native",replay_identity="e"*64,native_owner=owner)
  self.assertEqual(event["project_generation"],0);self.assertFalse(event["physical_left_right_verified"])
  with self.assertRaises(ValueError):
   validate_project_body_registration(event,scope,track_id="track_00",frame_id=3,pts=30,time_base=[1,100],
       pixel_space_id="native",replay_identity="e"*64,native_owner=owner)
if __name__=="__main__":unittest.main()
