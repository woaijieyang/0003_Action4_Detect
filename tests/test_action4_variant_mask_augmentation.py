import copy,json,unittest
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_sam_native_export import sha
from data.action4_stage2_data import Stage2Units
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from tests.test_action4_stage2_variant_selection import actual_config
class ActualVariantMaskAugmentation(unittest.TestCase):
 def test_predicted_mask_augmentation_preserves_RGB_geometry_and_ownership(self):
  torch.set_num_threads(1)
  base=actual_config();aug=copy.deepcopy(base)
  p=R/'configs/data/action4_sam_mask_augmentation_v001.json'
  aug['sam_mask_augmentation']={'path':str(p.relative_to(R)),'sha256':sha(p)}
  clean=Stage2Units(base,sam_factories=(FeatureIdentity,MaskAux))
  noisy=Stage2Units(aug,sam_factories=(FeatureIdentity,MaskAux))
  changed=False
  for epoch in range(12):
   a=clean[(0,epoch,True,1)];b=noisy[(0,epoch,True,1)]
   self.assertTrue(torch.equal(a['roi'],b['roi']))
   self.assertTrue(torch.equal(a['geometry']['K'],b['geometry']['K']))
   self.assertTrue(torch.equal(a['gt_visible448'],b['gt_visible448']))
   self.assertEqual(a['identity'],b['identity'])
   self.assertEqual(a['sam_input']['training_gt_instance_id'],b['sam_input']['training_gt_instance_id'])
   self.assertIn('mask_augmentation',b['sam_input']['dependencies'])
   changed=changed or not torch.equal(a['mask_aux'].visible,b['mask_aux'].visible) or not torch.equal(a['mask_aux'].available,b['mask_aux'].available)
   if changed:break
  self.assertTrue(changed,'Actual predicted mask must consume the configured augmentation')
