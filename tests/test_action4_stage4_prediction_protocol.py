"""Admission-only fixtures; no launch, GPU, or real completion claims."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_stage4_prediction_protocol import validate_recipe,validate_admission,validate_parent_seal,INPUT_FIELDS
from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL

class Stage4GateTests(unittest.TestCase):
 def cfg(self):
  c=json.loads((R/"outputs/action4_stage4_real_input_updates_v002/CONFIG.json").read_text())
  c.update(batch=48,micro=2,eval_every=25,expected_eval_images=650,eta_min=1e-7,grad_clip_norm=1.,world_size=4,
   expected_units=216,train_indices=list(range(216)),epochs=6,training_data_mode="independent_images",corner_softmax_bias_frozen=True,eval_schedule=None,
   stage4_parent_seal={"path":"fixture-parent","sha256":"a"*64},
   corner_refined_quality_bridge={"path":"fixture-bridge","sha256":"b"*64},
   b0_full_box_quality_receipt={"path":"fixture-quality","sha256":"c"*64})
  return c
 def gate(self,c):return dict(schema="action4.stage4_prediction_roi_admission.v1",launch_allowed=True,mode=c['mode'],role='A0',training_protocol=CORNER_NATIVE_PROTOCOL,**{k:c[k] for k in INPUT_FIELDS})
 def test_actual_engineering_config_cannot_schedule_without_sealed_parent(self):
  c=self.cfg();c.pop('stage4_parent_seal')
  with self.assertRaisesRegex(ValueError,'sealed SAM80'):validate_parent_seal(c)
 def test_bounded_integration_recipe_and_distinct_gate(self):
  c=self.cfg()
  with patch('model.training.action4_stage4_prediction_protocol.validate_parent_seal',return_value={}):
   validate_admission(c,self.gate(c))
   old=self.gate(c);old['schema']='action4.stage3_prediction_roi_admission.v1'
   with self.assertRaises(ValueError):validate_admission(c,old)
 def test_unregistered_cadence_and_bias_policy_rejected(self):
  for key,value in [('eval_schedule',{'switch_after_epochs':30}),('corner_softmax_bias_frozen',False),('epochs',7),('expected_units',257)]:
   c=self.cfg();c[key]=value
   with patch('model.training.action4_stage4_prediction_protocol.validate_parent_seal',return_value={}):
    with self.assertRaises(ValueError):validate_recipe(c)
 def test_formal_missing_execution_evidence_rejected(self):
  c=self.cfg();c.update(mode='full_training',epochs=80,train_indices=None,expected_units=10400,formal_training_units=10400,project_code_sha256='d'*64,stage4_execution_gate={'path':'fixture-execution','sha256':'e'*64})
  proof=dict(status='CORNER_PREDICTION_ROI_STAGE4_EXECUTION_PASS',project_code_sha256=c['project_code_sha256'],formal_training_units=10400,**{k:c[k] for k in INPUT_FIELDS if k!='expected_units'})
  with patch('model.training.action4_stage4_prediction_protocol.validate_parent_seal',return_value={}),patch('model.training.action4_stage4_prediction_protocol._v2_checked',return_value=proof):
   with self.assertRaisesRegex(ValueError,'Incomplete'):validate_admission(c,self.gate(c))
 def test_gate_input_change_rejected(self):
  c=self.cfg();g=self.gate(c);g['roi_expansion']=2.25
  with patch('model.training.action4_stage4_prediction_protocol.validate_parent_seal',return_value={}):
   with self.assertRaises(ValueError):validate_admission(c,g)


class Stage4PrepareTests(unittest.TestCase):
 def test_prepare_preserves_recipe_and_never_uses_default_parent(self):
  import uuid
  from model.training.action4_stage1_prepare import prepare_corner
  name="corner_prepare_fixture_"+uuid.uuid4().hex
  c=Stage4GateTests().cfg()
  path=R/"outputs"/(name+".json");path.write_text(json.dumps(c))
  with patch("model.training.action4_stage4_prediction_protocol.validate_parent_seal",
             return_value={"sha256":c["parent_bundle_sha256"]}):
   result=prepare_corner(str(path.relative_to(R)),name)
  saved=json.loads((R/result).read_text())
  for field in ("parent_bundle","parent_best_step","parent_bundle_sha256","evaluation_counterpart_bundle",
                "sam_cache","prediction_roi_manifest","loss_weights","epochs","roi_expansion"):
   self.assertEqual(saved[field],c[field])
  registry=json.loads((R/saved["parent_registry"]).read_text())
  self.assertEqual(registry["entries"]["A0"]["sha256"],c["parent_bundle_sha256"])
  self.assertFalse((R/saved["output"]/"TRAIN_PROCESS.json").exists())
  with patch("model.training.action4_stage4_prediction_protocol.validate_parent_seal",
             return_value={"sha256":c["parent_bundle_sha256"]}):
   with self.assertRaises(FileExistsError):prepare_corner(str(path.relative_to(R)),name)

 def test_missing_real_seal_creates_no_output(self):
  import uuid
  from model.training.action4_stage1_prepare import prepare_corner
  name="corner_prepare_rejected_"+uuid.uuid4().hex
  c=Stage4GateTests().cfg();c.pop("stage4_parent_seal")
  path=R/"outputs"/(name+".json");path.write_text(json.dumps(c))
  with self.assertRaisesRegex(ValueError,"sealed SAM80"):
   prepare_corner(str(path.relative_to(R)),name)
  self.assertFalse((R/"outputs/action4_improvement/stage4_corner"/name).exists())
