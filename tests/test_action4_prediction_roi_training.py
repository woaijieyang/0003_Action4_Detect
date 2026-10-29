import copy,json,unittest
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_stage2_data import Stage2Units
from data.action4_dataset import prepare_rois
from data.action4_pose_rgb_batch import prepare_pose_rgb
from data import DataAdapter
from model.training.action4_roi_adaptation import lr_factor
class PredictionTraining(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  cls.cfg=json.loads((R/'outputs/action4_roi_adapt_1p5_v001/ENGINEERING_CONFIG.json').read_text())
  cls.data=Stage2Units(cls.cfg)
 def test_actual_RGB_K_and_physical_pose(self):
  unit=self.data[(0,0,True,2)];item=self.data.data[0];row=item['frozen_prediction_roi']
  rgb,A=prepare_rois([item['image']],[row['bbox_for_a0']],[0],roi_size=448,expansion=1.5)
  self.assertTrue(torch.equal(rgb[0],unit['roi']))
  down,_=DataAdapter.letterbox_affine((448,448),(112,112));K=down@A[0]@np.asarray(item['record']['camera']['K'])
  self.assertTrue(torch.equal(torch.tensor(K,dtype=torch.float32),unit['geometry']['K']))
  T=np.asarray(item['record']['instances'][item['instance']]['canonical8_geometry']['object_to_capture_cv_m'])
  self.assertTrue(torch.equal(torch.tensor(T[:3,3],dtype=torch.float32),unit['geometry']['translation']))
  self.assertEqual(unit['metadata']['roi_source'],'frozen_b0_prediction')
 def test_epoch_does_not_jitter_inputs(self):
  a=self.data[(0,0,True,2)];b=self.data[(0,5,True,2)]
  self.assertTrue(torch.equal(a['roi'],b['roi']));self.assertTrue(torch.equal(a['geometry']['K'],b['geometry']['K']))
 def test_GT_box_cannot_change_explicit_crop(self):
  item=self.data.data[0];box=item['frozen_prediction_roi']['bbox_for_a0']
  a,_,_=prepare_pose_rgb([item],boxes=[box],roi_expansion=1.5,roi_source='frozen_b0_prediction')
  altered=copy.deepcopy(item);altered['record']['instances'][item['instance']]['bbox_full_xyxy']=[0,0,5,5]
  b,_,_=prepare_pose_rgb([altered],boxes=[box],roi_expansion=1.5,roi_source='frozen_b0_prediction')
  self.assertTrue(torch.equal(a,b))
 def test_legacy_guard_and_new_jitter_rejection(self):
  item=self.data.data[0];box=item['frozen_prediction_roi']['bbox_for_a0']
  with self.assertRaises(ValueError):prepare_pose_rgb([item],roi_expansion=1.5)
  with self.assertRaises(ValueError):prepare_pose_rgb([item],boxes=[box],seeds=[1],roi_expansion=1.5,roi_source='frozen_b0_prediction')
  with self.assertRaises(ValueError):prepare_pose_rgb([item],seeds=[1],jitter_config={'center_fraction':1.,'scale_range':[.5,.7]})
class Schedule(unittest.TestCase):
 def test_exact_effective_update_endpoints(self):
  for peak in [2e-6,1e-5]:
   self.assertAlmostEqual(lr_factor(0,peak=peak,total=10850),.1)
   self.assertAlmostEqual(lr_factor(199,peak=peak,total=10850),1.)
   self.assertAlmostEqual(lr_factor(10849,peak=peak,total=10850)*peak,1e-7)
   self.assertLess(lr_factor(200,peak=peak,total=10850),1.)
 def test_short_budget_refused(self):
  with self.assertRaises(ValueError):lr_factor(0,peak=1e-5,total=200)
