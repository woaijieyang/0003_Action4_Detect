import copy,unittest
from eval.action4_roi_pair_selection import validate_pair,SCORE_SCHEMA
class PairSelection(unittest.TestCase):
 def fixture(self):
  cfg=dict(training_protocol="action4.fixed_roi_prediction_adaptation.v1",roi_expansion=1.5,preprocess={"roi_expansion":1.5},
   roi_source="frozen_b0_prediction",a0_variant="a0_baseline",a0_selection_contract="full_all8_training_candidate_v1",
   evaluation_counterpart_sha256="b"*64,frozen_eval_b0_predictions={"path":"fixture","sha256":"c"*64})
  a=dict(role="A0",strict_load=True,epochs=50,score_schema=SCORE_SCHEMA,
   completion={"inference_contract":{"roi_expansion":1.5,"B0_sha256":"b"*64,"frozen_eval_b0_predictions":cfg["frozen_eval_b0_predictions"]}})
  return a,cfg,dict(role="B0",strict_load=True,sha256="b"*64),dict(role="B0")
 def test_matching_pair(self):validate_pair(*self.fixture())
 def test_wrong_parent_roi_or_legacy_A0_refused(self):
  for key,val in [("roi_expansion",2.25),("evaluation_counterpart_sha256","d"*64),("roi_source","GT"),("a0_variant","a0_sam_image")]:
   a,c,b,bc=self.fixture();c[key]=val
   with self.assertRaises(ValueError):validate_pair(a,c,b,bc)
 def test_not_strictly_reloaded_or_wrong_epochs_refused(self):
  for key,val in [("strict_load",False),("epochs",80)]:
   a,c,b,bc=self.fixture();a[key]=val
   with self.assertRaises(ValueError):validate_pair(a,c,b,bc)
 def test_sealed_contract_cannot_silently_restore_old_ROI(self):
  a,c,b,bc=self.fixture();a["completion"]["inference_contract"]["roi_expansion"]=2.25
  with self.assertRaises(ValueError):validate_pair(a,c,b,bc)


class SamPairSelection(unittest.TestCase):
 def fixture(self):
  import json
  from data.action4_paths import PROJECT_ROOT as R
  c=json.loads((R/"runtime/action4_runs/p2_a0_sam_prediction15_formal80_v001/source/resolved_config.json").read_text())
  keys=("training_protocol","parent_bundle_sha256","parent_best_step","evaluation_counterpart_sha256",
        "roi_source","roi_expansion","roi_size","prediction_roi_manifest","frozen_eval_b0_predictions",
        "sam_protocol_id","sam_cache","sam_eval_cache","sam_training_associations","sam_mask_augmentation","stage3_parent_seal")
  contract={k:copy.deepcopy(c[k]) for k in keys};contract["roi_sampling_device"]="cpu"
  a=dict(role="A0",strict_load=True,frozen_backbone_verified=True,epochs=80,
      completed_optimizer_steps=17360,score_schema=SCORE_SCHEMA,
      completion=dict(status="COMPLETION_METADATA_READY",evaluations=1346,inference_contract=contract))
  return a,c,dict(role="B0",strict_load=True,sha256=c["evaluation_counterpart_sha256"]),dict(role="B0")
 def test_native_SAM_pair_preserves_separate_train_eval_sources(self):
  a,c,b,bc=self.fixture();validate_pair(a,c,b,bc)
  self.assertEqual(a["completion"]["inference_contract"]["sam_cache"],c["sam_cache"])
 def test_wrong_seal_and_incomplete_training_rejected(self):
  for k,v in [("epochs",79),("strict_load",False),("frozen_backbone_verified",False),("completed_optimizer_steps",0)]:
   a,c,b,bc=self.fixture();a[k]=v
   with self.assertRaises(ValueError):validate_pair(a,c,b,bc)
 def test_changed_roi_feature_mode_and_counterpart_rejected(self):
  for k,v in [("roi_expansion",2.25),("a0_variant","a0_baseline"),("sam_enabled",False),("corner_enabled",True),("evaluation_counterpart_sha256","f"*64)]:
   a,c,b,bc=self.fixture();c[k]=v
   with self.assertRaises(ValueError):validate_pair(a,c,b,bc)
 def test_sealed_mask_contract_cannot_change(self):
  a,c,b,bc=self.fixture();a["completion"]["inference_contract"]["sam_eval_cache"]["manifest"]["sha256"]="f"*64
  with self.assertRaises(ValueError):validate_pair(a,c,b,bc)


class ParallelPairVerification(unittest.TestCase):
 def test_independent_checks_are_concurrent_and_roles_ordered(self):
  import threading
  from unittest.mock import patch
  from eval.action4_roi_pair_selection import _verify_pair_seals
  barrier=threading.Barrier(2)
  def verify(value):
   barrier.wait(timeout=5)
   return {"role":value}
  with patch("eval.action4_roi_pair_selection._verify_seal",side_effect=verify):
   a,b=_verify_pair_seals("A0","B0")
  self.assertEqual((a["role"],b["role"]),("A0","B0"))
 def test_any_validation_failure_is_not_a_pair_success(self):
  from unittest.mock import patch
  from eval.action4_roi_pair_selection import _verify_pair_seals
  def verify(value):
   if value=="B0":raise ValueError("invalid B0 seal")
   return {"role":value}
  with patch("eval.action4_roi_pair_selection._verify_seal",side_effect=verify):
   with self.assertRaisesRegex(ValueError,"invalid B0"):_verify_pair_seals("A0","B0")
