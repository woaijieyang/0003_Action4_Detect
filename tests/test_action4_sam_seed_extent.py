import unittest
import numpy as np
from model.action4_sam_seed_extent import box_corner_prompt,extent_gate,discovery_box
class SeedExtentTests(unittest.TestCase):
 def test_native_box_corner_roundtrip_and_no_reset_fields(self):
  q=box_corner_prompt("sid",0,1,[20,30,80,90],200,100)
  self.assertEqual(q["point_labels"],[2,3]);self.assertFalse({"text","bounding_boxes"}&set(q))
  np.testing.assert_array_equal(np.array(q["points"])*[200,100],[[20,30],[80,90]])
 def test_screen_part_high_purity_cannot_pass(self):
  full=np.ones((100,100),bool);part=np.zeros_like(full);part[:13]=True
  r=extent_gate(full,part);self.assertEqual(r["seed_purity"],1);self.assertFalse(r["extent_preserved"])
 def test_whole_mask_and_empty_prediction(self):
  m=np.zeros((30,40),bool);m[4:24,5:35]=True
  self.assertEqual(discovery_box(m),[5,4,35,24]);self.assertTrue(extent_gate(m,m)["extent_preserved"])
  self.assertFalse(extent_gate(m,np.zeros_like(m))["extent_preserved"])
 def test_invalid_scope_and_geometry(self):
  with self.assertRaises(ValueError):box_corner_prompt("s",4,0,[-1,0,10,20],20,20)
  with self.assertRaises(ValueError):extent_gate(np.zeros((2,2),bool),np.ones((2,2),bool))
  with self.assertRaises(ValueError):extent_gate(np.ones((2,2),bool),np.ones((3,2),bool))
if __name__=="__main__":unittest.main()
