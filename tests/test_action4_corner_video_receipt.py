import json,hashlib,tempfile,unittest
from pathlib import Path
from eval.action4_corner_video_receipt import _execution_binding
from data.action4_paths import PROJECT_ROOT as R
def bind(p):
 return {"path":str(p.relative_to(R)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
class CornerVideoReceiptTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=R/"outputs",prefix="corner_receipt_test_")
  self.root=Path(self.tmp.name);self.source=self.root/"source";self.source.mkdir()
  def put(name,value):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return bind(p)
  self.put=put
  (self.root/"A0.pt").write_bytes(b"fixture");(self.root/"B0.pt").write_bytes(b"fixture2")
  self.a=bind(self.root/"A0.pt");self.b=bind(self.root/"B0.pt")
  self.sel=put("selection.json",dict(status="PRESERVED_ALL8_DEVELOPMENT_CANDIDATE_NOT_FINAL_SEAL",roi_expansion=1.5,preserved_snapshot=self.a,selected={"step":2},B0=self.b["path"]))
  strict=put("reload.json",dict(status="STRICT_CORNER_BUNDLE_RELOAD_PASS_RESEARCH_TRIAL",checkpoint=self.a,selected_step=2))
  pair=put("pair.json",dict(models={"A0":dict(self.a,selected_step=2),"B0":self.b},trial_selection=self.sel,strict_reload=strict,quality_ready=False,deployment_admitted=False))
  self.lock=put("lock.json",{"formal_pair_gate":pair})
  self.cfg=dict(output=str(self.root.relative_to(R)),input_lock=self.lock,corner_trial_binding=self.sel,ratios=[1.5])
  cfgref=put("config.json",self.cfg);(self.root/"config.json").chmod(0o444)
  self.manifest=put("source/SOURCE_MANIFEST.json",dict(files={},configuration=cfgref))
 def tearDown(self):
  for p in self.root.rglob("*"):
   if p.is_file():p.chmod(0o644)
  self.tmp.cleanup()
 def test_correct_pair_config(self):
  _,_,_,pair=_execution_binding(self.root,self.lock,self.sel,self.manifest)
  self.assertFalse(pair["quality_ready"])
 def test_different_trial_rejected(self):
  other=self.put("other.json",{})
  with self.assertRaises(ValueError):_execution_binding(self.root,self.lock,other,self.manifest)
 def test_changed_configuration_rejected(self):
  p=self.root/"config.json";p.chmod(0o644);p.write_text(json.dumps(dict(self.cfg,ratios=[2.25])));p.chmod(0o444)
  with self.assertRaises(ValueError):_execution_binding(self.root,self.lock,self.sel,self.manifest)
 def test_source_cannot_belong_to_another_run(self):
  other=self.put("elsewhere/SOURCE_MANIFEST.json",dict(files={},configuration=bind(self.root/"config.json")))
  with self.assertRaises(ValueError):_execution_binding(self.root,self.lock,self.sel,other)
