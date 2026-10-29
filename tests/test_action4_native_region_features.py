import unittest
import numpy as np
from data.action4_sam_roi_inputs import array_digest
from data.action4_native_region_features import region_feature
class RegionFeatureTests(unittest.TestCase):
 def test_pixels_correspond_to_exact_raw_mask(self):
  mask=np.zeros((20,30),bool);mask[2:9,4:12]=True
  image,trace=region_feature(mask,expected_mask_sha256=array_digest(mask),shape_hw=mask.shape)
  self.assertTrue(image.flags.c_contiguous);self.assertEqual(image.dtype,np.uint8)
  for c in range(3):np.testing.assert_array_equal(image[:,:,c],mask.astype(np.uint8)*255)
  self.assertFalse(trace["full_body_extent_verified"]);self.assertFalse(trace["raw_RGB_modified"])
  image[:]=0;self.assertEqual(int(mask.sum()),56)
 def test_missing_or_wrong_identity_rejected(self):
  mask=np.ones((3,4),bool)
  with self.assertRaises(ValueError):region_feature(mask,expected_mask_sha256="0"*64,shape_hw=mask.shape)
  with self.assertRaises(ValueError):region_feature(mask,expected_mask_sha256=array_digest(mask),shape_hw=(4,3))
  mask[:]=False
  with self.assertRaises(ValueError):region_feature(mask,expected_mask_sha256=array_digest(mask),shape_hw=mask.shape)
