import unittest
import numpy as np
from model.action4_prompt_component_evidence import seed_component_evidence
class PromptComponentTests(unittest.TestCase):
 def test_select_actual_prompt_component_not_largest_and_preserve_raw(self):
  m=np.zeros((50,60),bool);m[3:10,3:10]=True;m[20:45,20:55]=True;before=m.copy()
  selected,r=seed_component_evidence(m,[[5,5],[7,7],[30,30]],[1,1,0])
  self.assertEqual(int(selected.sum()),49);self.assertEqual(r["unknown_unprompted_pixels"],875)
  np.testing.assert_array_equal(m,before);self.assertFalse(r["full_face_verified"])
 def test_split_positive_or_negative_contamination_is_unknown(self):
  m=np.zeros((50,60),bool);m[3:10,3:10]=True;m[20:45,20:55]=True
  a,r=seed_component_evidence(m,[[5,5],[30,30]],[1,1]);self.assertFalse(a.any())
  a,r=seed_component_evidence(m,[[5,5],[7,7]],[1,0]);self.assertFalse(a.any())
 def test_missing_positive_no_fake_support(self):
  m=np.zeros((20,20),bool)
  a,r=seed_component_evidence(m,[[5,5]],[1]);self.assertFalse(a.any())
