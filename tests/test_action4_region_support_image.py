import unittest,numpy as np
from model.action4_region_support_image import region_support_images
from eval.action4_native_clip import region_input_mode
class RegionSupportTests(unittest.TestCase):
 def test_default_and_invalid_modes(self):
  self.assertEqual(region_input_mode({}),'raw_RGB')
  with self.assertRaises(ValueError):region_input_mode({'region_input_mode':'guessed'})
 def test_instance_images_are_separate_and_inputs_unchanged(self):
  a=np.zeros((20,30),np.uint8);b=a.copy();a[2:8,3:10]=255;b[10:15,20:28]=255;before=a.copy();images,rec=region_support_images({'a':a,'b':b},track_ids=['a','b'])
  np.testing.assert_array_equal(a,before);np.testing.assert_array_equal(images['a'][:,:,0],a);np.testing.assert_array_equal(images['b'][:,:,0],b);self.assertFalse(rec['a']['complete_body_mask'])
  self.assertTrue(images['a'].flags.c_contiguous);self.assertTrue(np.all(images['a'][b==255]==0))
 def test_overlap_or_wrong_domain_rejected(self):
  m=np.ones((5,5),np.uint8)*255
  with self.assertRaises(ValueError):region_support_images({'a':m,'b':m},track_ids=['a','b'])
  with self.assertRaises(ValueError):region_support_images({'a':m},track_ids=['a','b'])
 def test_missing_support_is_unavailable_not_foreground(self):
  images,r=region_support_images({'a':np.zeros((4,4),np.uint8)},track_ids=['a']);self.assertFalse(r['a']['available']);self.assertFalse(images['a'].any())

 def test_ambiguous_support_not_relabelled_background(self):
  with self.assertRaises(ValueError):region_support_images({'a':np.zeros((4,4),np.uint8)},track_ids=['a'],ambiguous_pixels=1)

 def test_initialization_does_not_read_uninitialized_gradient(self):
  from eval.action4_native_clip import region_gradient_audit
  class Region:
   @property
   def gradient(self):raise AssertionError('uninitialized access')
   @property
   def hessian(self):raise AssertionError('uninitialized access')
  r=region_gradient_audit(Region(),tracking_step=False);self.assertFalse(r['gradient_computed']);self.assertIsNone(r['actual_gradient_norm'])
 def test_tracking_step_reads_actual_gradient(self):
  from eval.action4_native_clip import region_gradient_audit
  class Region:gradient=np.array([3.,4.]);hessian=np.eye(2)
  r=region_gradient_audit(Region(),tracking_step=True);self.assertTrue(r['gradient_computed']);self.assertEqual(r['actual_gradient_norm'],5.)

 def test_audit_domain_and_separate_camera_recovery_guard(self):
  from eval.action4_native_clip import region_line_audit_frames
  self.assertEqual(region_line_audit_frames({'frame_ids':[4,5]}),set())
  with self.assertRaises(ValueError):region_line_audit_frames({'frame_ids':[4],'region_line_audit_frames':[5]})
  with self.assertRaises(ValueError):region_line_audit_frames({'frame_ids':[4],'region_input_mode':'SAM_visible_support_image','recovery_mode':'current_evidence_optional_reseed_and_SAM_only_continuation'})

 def test_unknown_truncation_pixels_not_negative(self):
  a=np.zeros((30,30),np.uint8);a[12:16,12:16]=255
  images,rec=region_support_images({'a':a},track_ids=['a'],unknown_boxes={'a':[10,10,20,20]})
  self.assertEqual(images['a'][12,12,0],255);self.assertEqual(images['a'][18,18,0],128);self.assertEqual(images['a'][0,0,0],0);self.assertGreater(rec['a']['unknown_pixels'],0)
 def test_unknown_requires_explicit_direct_likelihood(self):
  from eval.action4_native_clip import external_region_options
  with self.assertRaises(ValueError):external_region_options({'region_input_mode':'SAM_visible_support_image','region_unknown_full_box':True})
  with self.assertRaises(ValueError):external_region_options({'external_mask_likelihood':True})
  self.assertEqual(external_region_options({}),(False,False,None))
