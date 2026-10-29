import unittest
from eval.action4_stage2_eval import rank_result
class CandidateSelection(unittest.TestCase):
 def row(self,rate,deploy,contract=True):
  r={'role':'A0','step':10,'deployment_eligible':deploy,'report':{'score_schema':'action4.pose_projection_all8_002.v1','pose_all8_at_002_rate':rate,'pose_fn':4,'pose_fp':4,'pose_corner_p95_valid_matched_only':5.}}
  if contract:r['a0_selection_contract']='full_all8_training_candidate_v1'
  return r
 def test_global_candidate_rank_separate_from_deployment(self):
  self.assertGreater(rank_result(self.row(.6,False)),rank_result(self.row(.5,True)))
 def test_historical_unspecified_contract_keeps_its_semantics(self):
  self.assertLess(rank_result(self.row(.6,False,False)),rank_result(self.row(.5,True,False)))
 def test_unknown_contract_rejected(self):
  r=self.row(.6,False);r['a0_selection_contract']='unknown'
  with self.assertRaises(ValueError):rank_result(r)
