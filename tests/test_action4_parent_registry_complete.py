import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from data.action4_paths import PROJECT_ROOT as R
from model.action4_best_registry import selected_entry_from_bundle,verify_loaded,verify_reference
class CompleteParentRegistry(unittest.TestCase):
 def setUp(self):
  b=R/".cache/registry_tests";b.mkdir(parents=True,exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=b);self.p=Path(self.tmp.name)/"A0.pt"
  self.state=dict(role="A0",selected_step=7,architecture="fixture",config_id="source-config",training_config_id="train-config",pose_method="fixture-pose")
  torch.save(self.state,self.p);self.sha=hashlib.sha256(self.p.read_bytes()).hexdigest()
 def tearDown(self):self.tmp.cleanup()
 def test_complete_entry_passes_existing_strict_consumers(self):
  e=selected_entry_from_bundle("A0",str(self.p),self.sha,7)
  reg=self.p.with_suffix(".json");reg.write_text(json.dumps(dict(schema="action4.selected_parent_registry.v1",entries={"A0":e})))
  self.assertEqual(verify_reference("A0",str(self.p),7,str(reg)),e)
  verify_loaded(e,self.state)
 def test_wrong_identity_fails_before_training(self):
  for role,sha,step in [("B0",self.sha,7),("A0","0"*64,7),("A0",self.sha,8)]:
   with self.assertRaises(ValueError):selected_entry_from_bundle(role,str(self.p),sha,step)
