import json,unittest
import torch
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_stage2_core import construct
class SamParentIdentity(unittest.TestCase):
 def cfg(self):
  return json.loads((R/'outputs/action4_variant_actual_updates_v001/CONFIG.json').read_text())
 def test_wrong_configured_SHA_fails_before_training(self):
  cfg=self.cfg();cfg['parent_bundle_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'Consumed parent checkpoint SHA mismatch'):construct(cfg,'cpu')
 def test_exact_sealed_parent_constructs(self):
  torch.set_num_threads(1);model,parent=construct(self.cfg(),'cpu')
  self.assertEqual(parent['selected_step'],15770)
  self.assertTrue(model.enable_sam)
  self.assertFalse(any(p.requires_grad for p in model.base.parameters()))
