import copy,json,unittest
from pathlib import Path
from model.training.action4_roi_adaptation import validate_parent_input_contract,PROTOCOL
from data.action4_prediction_roi_data import validate_config
class ParentROI(unittest.TestCase):
 def cfg(self):
  return json.loads(Path("/data/zy/vln/a-test/outputs/action4_roi_adapt_1p5_v001/ENGINEERING_CONFIG.json").read_text())
 def test_same_prediction_input_registered(self):
  c=self.cfg();validate_parent_input_contract(c,c)
 def test_old_parent_unchanged(self):
  validate_parent_input_contract({},{"roi_jitter":{}})
 def test_no_silent_GT_or_old_expansion(self):
  p=self.cfg()
  for key,value in [("roi_source","GT"),("roi_expansion",2.25),("preprocess",{"roi_expansion":2.25})]:
   c=copy.deepcopy(p);c[key]=value
   with self.assertRaises(ValueError):validate_parent_input_contract(p,c)
 def test_declaration_without_actual_data_consumer_refused(self):
  p=self.cfg();c=copy.deepcopy(p);c["training_protocol"]="future_unimplemented"
  with self.assertRaises(ValueError):validate_parent_input_contract(p,c)
  with self.assertRaises(ValueError):validate_config(c,"train")
 def test_SAM_cannot_sneak_into_baseline_protocol(self):
  p=self.cfg();c=copy.deepcopy(p);c.update(sam_enabled=True,a0_variant="a0_sam_image")
  with self.assertRaises(ValueError):validate_parent_input_contract(p,c)
