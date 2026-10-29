"""Actual data consumer wiring with declared CPU native-format fixtures."""
import copy,unittest
from unittest.mock import patch
import torch
from tests.test_action4_sam_roi_routing import RoiRoutingTests
from data.action4_stage2_data import Stage2Units
from data.action4_sam_roi_routing import validate_config
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from model.training.action4_corner_targets import corner_targets_from_items
from model.training.action4_supervision import local_counts
from model.training.action4_stage2_core import validate_run_gate

class CornerDataVariantTests(unittest.TestCase):
 def test_corner_route_actual_unit_targets_and_counts(self):
  fixture=RoiRoutingTests();fixture.setUp();_,item,cfg,_,_=fixture.configured()
  cfg=copy.deepcopy(cfg);cfg.update(a0_variant="a0_corner_image",corner_enabled=True)
  class Dataset:
   records=[item["record"]];units=[(0,0)]
   def __len__(self):return 1
   def __getitem__(self,index):return item
  with patch("data.action4_stage2_data.ComponentData",return_value=Dataset()):
   data=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
  unit=data[(0,0,True,1)]
  target=corner_targets_from_items([unit])
  self.assertEqual(target.identities,(unit["identity"],))
  self.assertEqual(unit["metadata"]["corner_gt224"]["image_id"],unit["identity"].image_id)
  self.assertEqual(unit["sam_input"]["training_gt_instance_id"],"1")
  counts=local_counts([unit],sam_enabled=True,corner_enabled=True)
  self.assertGreater(counts["corner_heatmap_kl"],0)
  self.assertEqual(counts["corner_valid_bce"],8)
  dummy=data[(0,0,False,1)]
  dummy_counts=local_counts([dummy],sam_enabled=True,corner_enabled=True)
  self.assertTrue(all(v==0 for v in dummy_counts.values()))
  self.assertFalse(dummy["metadata"]["sam_roi_routing"]["active"])
  # Data support does not silently lift the still-closed scheduled-run gate.
  with self.assertRaisesRegex(ValueError,"scheduled training refused"):
   validate_run_gate(cfg,{"mode":"integration","launch_allowed":True})
 def test_inconsistent_flags_rejected_before_dataset(self):
  base={"role":"A0","mode":"integration","a0_variant":"a0_corner_image","corner_enabled":True,"sam_enabled":True}
  bads=[dict(base,corner_enabled=False),dict(base,corner_enabled=1),dict(base,sam_enabled=False),
        dict(base,role="B0"),dict(base,a0_variant="a0_sam_image")]
  for cfg in bads:
   with self.subTest(cfg=cfg),patch("data.action4_stage2_data.ComponentData",side_effect=AssertionError("no data work")):
    with self.assertRaises(ValueError):Stage2Units(cfg)
 def test_route_gate_accepts_only_integration_corner_agreement(self):
  from data.action4_sam_roi_routing import POLICY
  cfg={"role":"A0","mode":"integration","a0_variant":"a0_corner_image","corner_enabled":True,"sam_enabled":True,
       "roi_jitter":copy.deepcopy(POLICY["gt_jitter"]),"sam_roi_routing":{"path":"not_consumed_here","sha256":"a"*64}}
  validate_config(cfg,"train")
  with self.assertRaises(ValueError):validate_config(dict(cfg,mode="full_training"),"train")
  with self.assertRaises(ValueError):validate_config(cfg,"eval")

if __name__=="__main__":unittest.main()
