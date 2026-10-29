import unittest,copy,json
from pathlib import Path
from eval.action4_video_roi_contract import trial_ratio,config_ratio,ratio_tag
R=Path('/data/zy/vln/a-test')
class VideoROIContractTests(unittest.TestCase):
 def fixture(self):
  selection={'status':'SEALED_COMPLETED75_ALL8_RESEARCH_TRIAL','roi_expansion':1.15}
  seal={'status':'SEALED_COMPLETED75_ALL8_RESEARCH_NOT_DEPLOYMENT','formal_training_complete':True,'strict_load':True,'frozen_backbone_verified':True,'roi_expansion':1.15,'selected_route':'corner_candidate','deployment_admitted':False,'planned_epochs':75}
  return selection,seal
 def test_only_completed75_and_model_ratio_contract(self):
  a,b=self.fixture();self.assertEqual(trial_ratio(a,b),1.15);self.assertEqual(ratio_tag(1.15),'e1p15');self.assertEqual(ratio_tag(1.5),'e1p50')
  for key,value in [('formal_training_complete',False),('strict_load',False),('roi_expansion',1.5),('planned_epochs',74)]:
   bad=dict(b);bad[key]=value
   with self.assertRaises(ValueError):trial_ratio(a,bad)
 def test_unsealed_or_provisional_115_rejected(self):
  a,b=self.fixture()
  with self.assertRaises(ValueError):trial_ratio(a,None)
  with self.assertRaises(ValueError):trial_ratio({'status':'PRESERVED_ALL8_DEVELOPMENT_CANDIDATE_NOT_FINAL_SEAL','roi_expansion':1.15})
  with self.assertRaises(ValueError):config_ratio({'ratios':[1.15]})
  with self.assertRaises(ValueError):config_ratio({'ratios':[1.2],'corner_trial_binding':{}})
 def test_real_stopped7300_binding_and_saved_row(self):
  from eval.action4_corner_video import validate_corner_trial_row
  from eval.action4_roi_sweep import binding
  root=R/'outputs/action4_stage5_sealed7300_inputs_v001';selection=json.loads((root/'SELECTION.json').read_text());seal=json.loads((R/selection['sealed_research']['path']).read_text());self.assertEqual(trial_ratio(selection,seal),1.5)
  with (R/'outputs/action4_stage5_sealed7300_video_v001/e1p50/predictions.jsonl').open() as f:row=json.loads(next(f))
  lock=json.loads((root/'INPUT_LOCK.json').read_text());K=json.loads((R/lock['video']['K']['path']).read_text())['K_native'];cad=json.loads((R/lock['CAD']['path']).read_text())['corners_3d_mm']
  import numpy as np
  validate_corner_trial_row(row,np.array(K),np.array(cad)/1000,binding(root/'SELECTION.json'))
 def test_real_15_stabilization_receipt_is_still_readable(self):
  from eval.action4_video_stabilization import validate_render_receipt,bound
  root=R/'outputs/action4_stage5_coherent_stable_v005';receipt=json.loads((root/'STABILIZATION_RECEIPT.json').read_text())
  got=validate_render_receipt(str(root.relative_to(R)),receipt['input_lock'],bound(root/'STABILIZATION_RECEIPT.json'));self.assertEqual(got['frames'],2850)
