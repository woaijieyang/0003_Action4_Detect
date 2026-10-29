"""SAM seal guards: CPU fixtures and real stored bundle, no inference/training."""
import copy,json,unittest
from unittest.mock import patch
from test_action4_role_seal_metadata import seal,PROJECT

class SamSealReadiness(unittest.TestCase):
 def cfg(self):
  return json.loads((PROJECT/"runtime/action4_runs/p2_a0_sam_prediction15_integration28_v001/source/resolved_config.json").read_text())
 def test_integration_never_formal(self):
  with self.assertRaises(ValueError):seal.validate_formal_recipe(self.cfg())
 def test_short_or_subset_formal_rejected(self):
  cfg=self.cfg();cfg.update(mode="full_training",epochs=80,train_indices=None,expected_units=10400)
  self.assertFalse(seal.validate_formal_recipe(cfg))
  for key,value in [("epochs",79),("epochs",81),("formal_epochs",2),("train_indices",[0]),("expected_units",632),("train_backbone",True)]:
   bad=copy.deepcopy(cfg);bad[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):seal.validate_formal_recipe(bad)
 def test_cache_or_sampling_change_rejected(self):
  cfg=self.cfg();proof=seal.sam_input_contract(cfg,{})
  self.assertEqual(proof["roi_sampling_device"],"cpu")
  for key,value in [("roi_expansion",2.25),("roi_sampling_device","cuda")]:
   bad=copy.deepcopy(cfg);bad[key]=value
   with self.assertRaises(ValueError):seal.sam_input_contract(bad,{})
  bad=copy.deepcopy(cfg);bad["sam_cache"]["manifest"]["sha256"]="0"*64
  with self.assertRaises(ValueError):seal.sam_input_contract(bad,{})
 def test_all_real_integration_evals_have_native_cache(self):
  cfg=self.cfg()
  for f in (PROJECT/cfg["output"]/"eval").glob("*.done.json"):
   done=json.loads(f.read_text());seal.validate_sam_eval(done,cfg)
   bad=copy.deepcopy(done);bad["report"]["sam_cache_inputs"]["engineering_fixture"]=True
   with self.assertRaises(ValueError):seal.validate_sam_eval(bad,cfg)
 def test_frozen_every_tensor_and_buffer(self):
  import torch
  parent={"backbone_config":{},"backbone_step":1,"backbone":{"weight":torch.ones(1),"running_mean":torch.zeros(1),"num_batches_tracked":torch.tensor(2)}}
  state=dict(copy.deepcopy(parent),backbone_frozen=True,backbone_parent_step=1)
  self.assertTrue(seal.verify_frozen_backbone(state,parent))
  for key in parent["backbone"]:
   bad=copy.deepcopy(state);bad["backbone"][key]+=1
   with self.assertRaises(ValueError):seal.verify_frozen_backbone(bad,parent)
 def test_actual_strict_loader_rejects_missing_adapter_parameter(self):
  import torch
  from model.action4_release_bundle import load_bundle
  source=PROJECT/"outputs/action4_stage3_native_prediction_v001/real_gpu_updates_v001/ENGINEERING_REAL_STEP2.pt"
  state=torch.load(source,map_location="cpu",weights_only=True)
  state["sam_adapter"].pop(next(iter(state["sam_adapter"])))
  # Deserialization is substituted only to avoid writing another large copy;
  # the production load_bundle and strict adapter load execute unchanged.
  with patch("model.action4_release_bundle.torch.load",return_value=state):
   with self.assertRaises(RuntimeError):load_bundle(str(source),"A0","cpu")
