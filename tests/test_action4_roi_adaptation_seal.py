import copy,json,unittest
from test_action4_role_seal_metadata import seal,PROJECT
class AdaptationSeal(unittest.TestCase):
 def cfg(self):
  return json.loads((PROJECT/"runtime/action4_runs/p2_a0_roi15_prediction50_v001/source/resolved_config.json").read_text())
 def test_actual_frozen50_config_recognized(self):
  self.assertTrue(seal.validate_formal_recipe(self.cfg()))
 def test_49_80_roi_or_source_change_rejected(self):
  for key,value in [("epochs",49),("epochs",80),("roi_expansion",2.25),("roi_source","GT"),("eval_every",10),("train_backbone",True),("sam_enabled",True)]:
   cfg=self.cfg();cfg[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):seal.validate_formal_recipe(cfg)
 def test_unregistered_short_run_rejected(self):
  cfg=self.cfg();cfg.pop("training_protocol")
  with self.assertRaises(ValueError):seal.validate_formal_recipe(cfg)
 def test_legacy80_still_accepted(self):
  self.assertFalse(seal.validate_formal_recipe({"mode":"full_training","role":"B0","epochs":80}))
 def test_missing_terminal_cannot_be_sealed(self):
  from unittest.mock import patch
  with patch.object(seal,"TERMINALS",(*seal.TERMINALS,"CPU_FIXTURE_MISSING_TERMINAL.json")):
   result=seal.inspect("runtime/action4_runs/p2_a0_roi15_prediction50_v001/RUN_SPEC.json")
  self.assertEqual(result[3]["status"],"WAITING_FOR_COMPLETION")
  self.assertIn("CPU_FIXTURE_MISSING_TERMINAL.json",result[3]["missing"])

 def actual_done(self):
  cfg=self.cfg()
  root=PROJECT/cfg["output"]/"eval"
  return cfg,json.loads(next(root.glob("*.done.json")).read_text())
 def test_actual_done_contract_and_missing_or_wrong_contract(self):
  cfg,done=self.actual_done();ids={r["sample_id"] for r in done["report"]["bbox_only"]["per_image"]}
  seal.validate_adaptation_eval(done,cfg,ids)
  for value in [None,"legacy"]:
   bad=copy.deepcopy(done);bad["a0_selection_contract"]=value
   with self.assertRaises(ValueError):seal.validate_adaptation_eval(bad,cfg,ids)
 def test_wrong_roi_model_cache_and_domain_rejected(self):
  cfg,done=self.actual_done();ids={r["sample_id"] for r in done["report"]["bbox_only"]["per_image"]}
  for field,value in [("roi_expansion",2.25),("frozen_b0_predictions",{}),("B0_forward_calls",1)]:
   bad=copy.deepcopy(done);bad["report"][field]=value
   with self.assertRaises(ValueError):seal.validate_adaptation_eval(bad,cfg,ids)
  bad=copy.deepcopy(done);bad["report"]["bbox_only"]["per_image"][0]["sample_id"]="wrong_image"
  with self.assertRaises(ValueError):seal.validate_adaptation_eval(bad,cfg,ids)
  cfg["expected_eval_images"]=649
  with self.assertRaises(ValueError):seal.validate_formal_recipe(cfg)

 def test_frozen_backbone_tensors_and_buffers_checked(self):
  import torch
  parent={"backbone_config":{"shape":"fixture"},"backbone_step":3,"backbone":{"weight":torch.ones(2),"running_mean":torch.zeros(2),"num_batches_tracked":torch.tensor(0)}}
  current=dict(copy.deepcopy(parent),backbone_frozen=True,backbone_parent_step=3)
  self.assertTrue(seal.verify_frozen_backbone(current,parent))
  for name in parent["backbone"]:
   changed=copy.deepcopy(current);changed["backbone"][name]+=1
   with self.assertRaises(ValueError):seal.verify_frozen_backbone(changed,parent)
  changed=copy.deepcopy(current);changed["backbone"]["weight"]=changed["backbone"]["weight"].double()
  with self.assertRaises(ValueError):seal.verify_frozen_backbone(changed,parent)
