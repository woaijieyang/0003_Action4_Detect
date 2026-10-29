import unittest
import numpy as np
from data.action4_region_support import associate_supported_component
class RegionSupportTests(unittest.TestCase):
 def fixture(self):
  raw=np.zeros((32,48),bool);raw[8:24,20:40]=True
  a=np.zeros_like(raw);a[10:14,24:28]=True
  b=np.zeros_like(raw);b[18:22,32:36]=True
  return raw,{"current_button":a,"current_marker":b}
 def test_remote_tiny_pixels_do_not_veto_current_component(self):
  raw,s=self.fixture();raw[1,1:3]=True;raw[29,4:6]=True;before=raw.copy()
  selected,r=associate_supported_component(raw,s)
  self.assertEqual(int(selected.sum()),320);self.assertEqual(r["unknown_unselected_pixels"],4)
  np.testing.assert_array_equal(raw,before);self.assertFalse(r["full_visibility_verified"])
 def test_never_pick_largest_component_without_support(self):
  raw,s=self.fixture();raw[:30,:18]=True
  selected,r=associate_supported_component(raw,s);self.assertEqual(int(selected.sum()),320)
  self.assertGreater(r["unknown_unselected_pixels"],r["selected_pixels"])
 def test_split_support_no_merge_or_arbitrary_selection(self):
  raw,s=self.fixture();raw[1:5,1:5]=True;s["current_marker"][:]=False;s["current_marker"][1:5,1:5]=True
  selected,r=associate_supported_component(raw,s);self.assertIsNone(selected);self.assertIn("disagree",r["reason"])
 def test_missing_support_not_absence(self):
  raw,s=self.fixture();s["current_marker"][:]=False
  selected,r=associate_supported_component(raw,s);self.assertIsNone(selected);self.assertFalse(r["native_mask_modified"])
 def test_different_pixel_domains_rejected(self):
  raw,s=self.fixture();s["current_marker"]=np.zeros((4,4),bool)
  with self.assertRaises(ValueError):associate_supported_component(raw,s)
 def test_cloned_support_cannot_count_as_two_cues(self):
  raw,s=self.fixture();s["current_marker"]=s["current_button"].copy()
  selected,r=associate_supported_component(raw,s);self.assertIsNone(selected);self.assertIn("duplicate_support",r["reason"])

if __name__=="__main__":unittest.main()
