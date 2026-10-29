import unittest,numpy as np
from eval.action4_native_m3t_adapter import OwnedNativeCamera
class Camera:
 def __init__(self):self.image=None
class CopyCamera:
 @property
 def image(self):return self._image
 @image.setter
 def image(self,x):self._image=np.array(x,copy=True)
class OwnedChecksumTests(unittest.TestCase):
 def packet(self,cam=None):return OwnedNativeCamera(cam or Camera(),width=8,height=6),np.arange(144,dtype=np.uint8).reshape(6,8,3)
 def test_unchanged_frame_verified(self):
  a,x=self.packet()
  with a.frame(x,frame_id=1,source_sha256='a'*64):np.testing.assert_array_equal(a.snapshot(),x)
  self.assertTrue(a.audit[-1]['unchanged'])
 def test_borrowed_native_write_still_detected(self):
  a,x=self.packet()
  with self.assertRaises(RuntimeError):
   with a.frame(x,frame_id=1,source_sha256='a'*64):a.camera.image.setflags(write=True);a.camera.image[0,0,0]^=1
  self.assertFalse(a.audit[-1]['unchanged']);self.assertFalse(a._active)
 def test_independent_owner_write_still_detected(self):
  a,x=self.packet(CopyCamera())
  with self.assertRaises(RuntimeError):
   with a.frame(x,frame_id=1,source_sha256='a'*64):a._owner.setflags(write=True);a._owner[0,0,0]^=1
  self.assertFalse(a.audit[-1]['unchanged'])
 def test_camera_replacement_still_detected(self):
  a,x=self.packet()
  with self.assertRaises(RuntimeError):
   with a.frame(x,frame_id=1,source_sha256='a'*64):a.camera.image=np.zeros_like(x)
 def test_input_mutation_does_not_mutate_owned_image(self):
  a,x=self.packet();before=x.copy()
  with a.frame(x,frame_id=1,source_sha256='a'*64):x[:]=0;np.testing.assert_array_equal(a.snapshot(),before)
  self.assertTrue(a.audit[-1]['unchanged'])
