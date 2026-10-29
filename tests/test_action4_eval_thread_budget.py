import unittest
from model.training.action4_resource_policy import evaluation_threads
class Threads(unittest.TestCase):
 def test_default_and_registered_counts(self):
  self.assertEqual(evaluation_threads({}),2)
  self.assertEqual(evaluation_threads({"eval_threads":1}),1)
 def test_invalid_counts_rejected(self):
  for v in [True,0,3,"1",None]:
   with self.assertRaises(ValueError):evaluation_threads({"eval_threads":v})
 def test_four_train_sixteen_eval_budget(self):
  cfg={"eval_threads":1}
  self.assertEqual(4*(2+7)+16*(evaluation_threads(cfg)+1)+1,69)
