import unittest
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_native_export import sha
from data.action4_sam_roi_routing import decision
from data.action4_stage2_data import Stage2Units
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from tests.test_action4_stage2_variant_selection import actual_config
class ActualVariantRouting(unittest.TestCase):
 def test_selected_variant_provider_routes_and_augments_real_RGB(self):
  torch.set_num_threads(1);cfg=actual_config()
  for key,name in [('sam_roi_routing','action4_sam_roi_routing_v001.json'),('sam_mask_augmentation','action4_sam_mask_augmentation_v001.json')]:
   p=R/'configs/data'/name;cfg[key]={'path':str(p.relative_to(R)),'sha256':sha(p)}
  ds=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
  record=ds.data.records[0];sid=record['sample_id'];gid=str(record['instances'][0]['instance_id'])
  epochs={}
  for epoch in range(200):
   route=decision(cfg['seed'],epoch,sid,gid)['requested_route'];epochs.setdefault(route,epoch)
   if len(epochs)==4:break
  self.assertEqual(len(epochs),4)
  for requested,epoch in epochs.items():
   with self.subTest(route=requested):
    unit=ds[(0,epoch,True,1)];trace=unit['metadata']['sam_roi_routing']
    self.assertEqual(trace['requested_route'],requested)
    self.assertFalse(trace['GT_used_for_native_prompt'])
    self.assertEqual(unit['metadata']['sam_rgb_variant_selection']['selected_leaf'],'light')
    self.assertIn('native_light:',trace['fixed_SAM_prediction_id'])
    self.assertIn('mask_augmentation',unit['sam_input']['dependencies'])
    if requested.startswith('prediction'):
     self.assertEqual(trace['effective_route'],'prediction_full')
     self.assertIn('native_light:',trace['selected_crop_prediction']['prediction_id'])
    else:self.assertEqual(trace['effective_route'],requested)
