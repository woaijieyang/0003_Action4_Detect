import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import eval.action4_sam_fullvideo_dispatch as module

class DispatchTests(unittest.TestCase):
 def setUp(self):
  parent=Path("/data/zy/vln/a-test/outputs/action4_fullvideo_contract_v001/test_tmp");parent.mkdir(exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.base=self.root/"bootstrap";self.old=self.base/"source";self.old.mkdir(parents=True)
  (self.base/"run").mkdir()
  self.cfg={"source_snapshot":"bootstrap/source","native":{"source_root":"bootstrap/source/upstream/sam3","source_receipt":"bootstrap/source/receipt.json","asset_receipt":"bootstrap/source/asset.json"},
   "sequence":{"frame_count":2850},"b0":{"sha256":"b"*64},"region":{"path":"region","sha256":"c"*64}}
  self.put(self.old/"config.json",self.cfg);(self.old/"config.json").chmod(0o444)
  self.manifest={"schema":"action4.sam.immutable_source.v1","files_sha256":{"config.json":module.sha(self.old/"config.json")}}
  self.manifest["identity"]=hashlib.sha256(json.dumps(self.manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
  self.put(self.old/"SNAPSHOT_MANIFEST.json",self.manifest)
  seeds=[{"track_id":"left"},{"track_id":"right"}]
  self.put(self.base/"run/SEEDS.json",{"seeds":seeds})
  self.put(self.base/"run/NATIVE_COMPLETE.json",{"status":"DISCOVERY_SEEDS_ONLY_COMPLETE_NOT_TRACKING","source_identity":self.manifest["identity"],"tracks":seeds})
  self.put(self.base/"run/B0_COMPLETE.json",{"frames":2850,"source_identity":self.manifest["identity"],"b0":self.cfg["b0"],"region":self.cfg["region"]})
  self.put(self.base/"run/NATIVE_LOAD.json",{"status":"LOADED_ZERO_FINAL_MISSING_UNEXPECTED"})
  for tid in ["left","right"]:
   folder=self.base/"run"/tid;folder.mkdir()
   self.put(folder/"SEED_BINDING.json",{"reason":"bound_unique_prediction_support"})
   self.put(folder/"CLOSE.json",{"released":True});self.put(folder/"RAW_RECEIPT.json",{})
  (self.base/"run_frozen.py").write_text("# engineering fixture, never executed\n")
  self.patch=patch.object(module,"ROOT",self.root);self.patch.start();self.addCleanup(self.patch.stop)
 def put(self,p,d):p.write_text(json.dumps(d))
 def test_prepare_preserves_bootstrap_and_seals_new_config(self):
  cfg,manifest,launcher=module.prepare(self.base,self.root/"tracking")
  self.assertEqual(cfg["seed_mode"],"instance_points");self.assertFalse(cfg["discovery_decode_prefix"])
  self.assertEqual(len(cfg["instance_seed_sources"]),2)
  self.assertFalse((self.root/"tracking/source/config.json").stat().st_mode&0o222)
  self.assertEqual(json.loads((self.old/"config.json").read_text()),self.cfg)
 def test_missing_seed_does_not_launch_or_create_run(self):
  self.put(self.base/"run/SEEDS.json",{"seeds":[{"track_id":"left"}]})
  with self.assertRaises(ValueError):module.prepare(self.base,self.root/"tracking")
  self.assertFalse((self.root/"tracking").exists())
 def test_unreleased_session_blocks(self):
  self.put(self.base/"run/left/CLOSE.json",{"released":False})
  with self.assertRaises(ValueError):module.prepare(self.base,self.root/"tracking")
 def test_model_mismatch_blocks(self):
  d=json.loads((self.base/"run/B0_COMPLETE.json").read_text());d["b0"]={"sha256":"d"*64}
  self.put(self.base/"run/B0_COMPLETE.json",d)
  with self.assertRaises(ValueError):module.prepare(self.base,self.root/"tracking")
 def test_existing_output_never_overwritten(self):
  (self.root/"tracking").mkdir()
  with self.assertRaises(FileExistsError):module.prepare(self.base,self.root/"tracking")
