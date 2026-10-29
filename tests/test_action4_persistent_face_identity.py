import unittest
from dataclasses import replace,asdict
from model.action4_sam_face_contract import FaceSeedBinding,persistent_face_identity,_hash

class PersistentFaceIdentityTests(unittest.TestCase):
 def seed(self):
  ref=dict(path="owned/receipt.json",sha256="a"*64)
  return FaceSeedBinding(body_track_id="body_scope_A",body_generation=0,face_id=3,face_generation=0,
   region_identity="region",local_index=0,global_frame_id=4,pts=1600,time_base=(1,11988),
   pixel_space_id="raw_video_native_integer_centers",a0_model_sha256="b"*64,rgb_sha256="c"*64,
   K_native=(100,0,50,0,100,50,0,0,1),pose_receipt=ref,patch_ref=ref,prompt_ref=ref,
   box_xywh01=(.1,.1,.2,.2))
 def test_session_restart_preserves_entity(self):
  a=self.seed();b=replace(a,face_generation=1,global_frame_id=100,pts=40000)
  self.assertEqual(a.persistent_face_id,b.persistent_face_id)
  self.assertNotEqual(a.session_track_id,b.session_track_id)
  self.assertEqual(a.track_id,"face:"+_hash([a.body_track_id,a.body_generation,a.face_id,a.face_generation]))
  self.assertEqual(a.fingerprint,_hash(asdict(a)))
 def test_two_bodies_have_twelve_unique_entities(self):
  ids={persistent_face_identity(body,0,face) for body in ["body_A","body_B"] for face in range(6)}
  self.assertEqual(len(ids),12)
 def test_new_body_identity_or_face_not_alias(self):
  a=self.seed()
  for b in [replace(a,body_generation=1),replace(a,body_track_id="body_B"),replace(a,face_id=2)]:
   self.assertNotEqual(a.persistent_face_id,b.persistent_face_id)
 def test_bad_identity_rejected(self):
  for args in [("",0,0),("a",-1,0),("a",0,6),("a",0,True)]:
   with self.assertRaises(ValueError):persistent_face_identity(*args)
