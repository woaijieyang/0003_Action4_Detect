import copy,json,unittest
from unittest.mock import patch
import numpy as np
from data import action4_global_render as m
class GlobalPreviewTests(unittest.TestCase):
 def setUp(self):self.c=json.loads((m.ROOT/'configs/data/action4_global_preview_canonical8_v003.json').read_text())
 def test_native_frozen_input(self):m.validate_config(self.c)
 def test_reject_production_or_reduced_quality(self):
  for key,value in [('mode','production'),('dataset_generation_allowed',True),('resolution_wh',[812,616])]:
   c=copy.deepcopy(self.c);c[key]=value
   with self.assertRaises(ValueError):m.validate_config(c)
  for key,value in [('samples',16),('mask_samples',2),('amodal_margin_px',4)]:
   c=copy.deepcopy(self.c);c['render'][key]=value
   with self.assertRaises(ValueError):m.validate_config(c)
 def test_output_only_fresh_owned_diagnostics(self):
  for path in ('data/action4_synth/x','models/action4/x','outputs/action4_other/x','../escape'):
   with self.assertRaises(ValueError):m.output_path(path)
  with self.assertRaises(FileExistsError):m.output_path(m.OUTPUTS/'preflight_v001')
 def test_gpu_and_thread_validation_before_any_device_query(self):
  for gpu,threads,probe in [('GPU-invalid',2,False),(None,1,True),(m.ALLOWED_GPU_UUIDS[0],2,True)]:
   with patch.object(m,'worker_environment') as worker:
    with self.assertRaises(ValueError):m.launch('configs/data/action4_global_preview_canonical8_v003.json',m.OUTPUTS/'never_created',threads,gpu_uuid=gpu,probe_only=probe)
    worker.assert_not_called()
 def test_frozen_input_hash_rejected(self):
  c=copy.deepcopy(self.c);c['sources']['integrated_asset']['sha256']='0'*64
  with self.assertRaises(ValueError):m.validate_config(c)
 def test_mask_definitions_match_source_formula(self):
  visible=np.zeros((4,4),bool);visible[:2]=True
  full=np.zeros((12,12),bool);full[4:8,4:8]=True;full[0,:4]=True
  result=m.mask_measurements(visible,full,4,4)
  self.assertEqual(result['area_visible_px'],8);self.assertEqual(result['area_inframe_amodal_px'],16);self.assertEqual(result['area_full_amodal_px'],20)
  self.assertEqual(result['occlusion_inframe'],.5);self.assertAlmostEqual(result['truncation'],.2)
 def test_mask_exclusion_and_boundary_failures(self):
  visible=np.zeros((4,4),bool);full=np.zeros((12,12),bool);full[0,0]=True
  self.assertTrue(m.mask_measurements(visible,full,4,4)['excluded'])
  with self.assertRaises(ValueError):m.mask_measurements(visible,np.zeros_like(full),4,4)
  full[4,4]=True
  with self.assertRaises(ValueError):m.mask_measurements(np.ones_like(visible),full,4,4)
 def test_outside_image_checks_contradiction_before_exclusion(self):
  full=np.zeros((300,300),bool);full[0,0]=True
  with self.assertRaises(ValueError):m.mask_measurements(np.ones((100,100),bool),full,100,100)
  visible=np.zeros((100,100),bool)
  self.assertTrue(m.mask_measurements(visible,full,100,100)['excluded'])
  visible[0,:4]=True
  self.assertTrue(m.mask_measurements(visible,full,100,100)['excluded'])
  visible[0,4]=True
  with self.assertRaises(ValueError):m.mask_measurements(visible,full,100,100)
 def test_wrist_visibility_requirement_cannot_be_removed(self):
  c=copy.deepcopy(self.c);c['inspection_poses'][1]['required_visible_target_ids']=[]
  with self.assertRaises(ValueError):m.validate_config(c)
if __name__=='__main__':unittest.main()


