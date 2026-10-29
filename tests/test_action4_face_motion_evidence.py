import unittest
import numpy as np
import cv2
from model.action4_face_motion_evidence import FaceMotionConfig,measure_border_motion,border_support
from data.action4_sam_roi_inputs import array_digest

class FaceMotionEvidence(unittest.TestCase):
 def setUp(self):
  cv2.setNumThreads(1);cv2.setRNGSeed(12)
  rng=np.random.default_rng(13)
  self.rgb=rng.integers(0,255,(256,320,3),dtype=np.uint8)
  self.body=np.zeros((256,320),bool);self.body[35:220,35:285]=True
  self.dynamic=np.zeros_like(self.body);self.dynamic[90:175,100:225]=True
  self.M=np.float32([[1,0,5],[0,1,3]])
  self.next=cv2.warpAffine(self.rgb,self.M,(320,256))
  self.nb=cv2.warpAffine(self.body.astype(np.uint8),self.M,(320,256),flags=cv2.INTER_NEAREST).astype(bool)
  self.nd=cv2.warpAffine(self.dynamic.astype(np.uint8),self.M,(320,256),flags=cv2.INTER_NEAREST).astype(bool)
  # Moving LCD contents must not provide the measurement.
  self.next[self.nd]=rng.integers(0,255,(int(self.nd.sum()),3),dtype=np.uint8)
  self.cfg=FaceMotionConfig()
 def ident(self,rgb,body,dynamic,frame):
  return dict(body_id="body00",body_generation=0,face_id=3,face_generation=0,pixel_space_id="native",
   time_base=[1,11988],frame_id=frame,pts=frame*400,rgb_sha256="a"*64,
   rgb_tensor_sha256=array_digest(rgb),body_mask_sha256=array_digest(body),dynamic_mask_sha256=array_digest(dynamic))
 def call(self,nb=None,current_identity=None):
  nb=self.nb if nb is None else nb
  return measure_border_motion(self.rgb,self.next,previous_body=self.body,current_body=nb,
   previous_dynamic=self.dynamic,current_dynamic=self.nd,previous_identity=self.ident(self.rgb,self.body,self.dynamic,0),
   current_identity=current_identity or self.ident(self.next,nb,self.nd,1),config=self.cfg)
 def test_actual_RGB_translation_and_dynamic_exclusion(self):
  r=self.call();self.assertTrue(r["accepted_image_registration"]);self.assertFalse(r["geometry_fit_eligible"])
  H=np.asarray(r["H_native"]);uv=np.array([[[80.,80.],[240.,190.],[150.,60.]]])
  got=cv2.perspectiveTransform(uv,H);np.testing.assert_allclose(got,uv+[5,3],atol=.2)
  support,excluded=border_support(self.body,self.dynamic,self.cfg)
  for t in r["tracks"]:
   x,y=np.rint(t["previous_native"]).astype(int);self.assertTrue(support[y,x]);self.assertFalse(excluded[y,x])
 def test_partial_occlusion_does_not_become_centroid_motion(self):
  nb=self.nb.copy();nb[:,220:]=False
  r=self.call(nb);self.assertTrue(r["accepted_image_registration"])
  H=np.asarray(r["H_native"]);uv=np.array([[[100.,70.],[120.,200.]]])
  np.testing.assert_allclose(cv2.perspectiveTransform(uv,H),uv+[5,3],atol=.5)
 def test_generation_and_pixel_tamper_rejected(self):
  ident=self.ident(self.next,self.nb,self.nd,1);ident["body_id"]="other"
  with self.assertRaises(ValueError):self.call(current_identity=ident)
  ident=self.ident(self.next,self.nb,self.nd,1);ident["dynamic_mask_sha256"]="b"*64
  with self.assertRaises(ValueError):self.call(current_identity=ident)
 def test_uniform_no_support_not_success(self):
  self.rgb[:]=50;self.next[:]=50
  r=self.call();self.assertFalse(r["accepted_image_registration"]);self.assertIsNone(r["H_native"])
 def test_gap_and_bad_config(self):
  ident=self.ident(self.next,self.nb,self.nd,1);ident["pts"]=100000
  with self.assertRaises(ValueError):self.call(current_identity=ident)
  with self.assertRaises(ValueError):FaceMotionConfig(forward_backward_px=float("nan"))
