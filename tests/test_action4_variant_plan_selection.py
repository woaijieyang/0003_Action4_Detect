import unittest
from data.action4_sam_variant_selection import choose_leaf
from data.action4_rgb_variants import select_variant
class PlanSelection(unittest.TestCase):
 def test_exact_plan_epoch_mapping(self):
  for image in ('p2_train_00000','p2_train_05199'):
   for epoch in range(80):
    result=choose_leaf(dict(original=1/3,light=1/3,blur=1/3),seed=761,epoch=epoch,image_id=image,selection_rule='native_policy_one_based_v1')
    self.assertEqual(result['selected_leaf'],select_variant(image,epoch+1,seed=761))
    self.assertEqual(result['epoch_one_based'],epoch+1)
 def test_missing_or_weighted_leaves_refused(self):
  for weights in (dict(light=1.),dict(original=.5,light=.3,blur=.2)):
   with self.assertRaises(ValueError):
    choose_leaf(weights,seed=1,epoch=0,image_id='x',selection_rule='native_policy_one_based_v1')
 def test_unknown_rule_refused(self):
  with self.assertRaises(ValueError):choose_leaf(dict(light=1.),seed=1,epoch=0,image_id='x',selection_rule='unknown')
