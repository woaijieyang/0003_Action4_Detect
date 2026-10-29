import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
R=Path("/data/zy/vln/a-test")
spec=importlib.util.spec_from_file_location("launcher",R/"scripts/action4_run_frozen_sequence.py")
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
class Launcher(unittest.TestCase):
 def fixture(self):
  base=R/"outputs/action4_clip_replay_preparation_v001";base.mkdir(exist_ok=True)
  temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temp.cleanup)
  source=Path(temp.name)/"source";(source/"src").mkdir(parents=True)
  code=source/"src/tiny.py";code.write_text("x=1")
  cfg=source/"config.json";cfg.write_text(json.dumps({"source_snapshot":str(source.relative_to(R))}))
  files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (code,cfg)}
  manifest={"schema":"action4.sam.immutable_source.v1","config_file":"config.json","files_sha256":files}
  manifest["identity"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
  (source/"SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest))
  for p in (code,cfg):p.chmod(0o444)
  return source,cfg,code
 def test_verified_source(self):
  source,cfg,_=self.fixture();self.assertEqual(launcher.verify(cfg)[1],source)
 def test_changed_or_writable_source(self):
  source,cfg,code=self.fixture();code.chmod(0o644)
  with self.assertRaises(ValueError):launcher.verify(cfg)
  code.write_text("x=2");code.chmod(0o444)
  with self.assertRaises(ValueError):launcher.verify(cfg)
 def test_unlisted_import(self):
  source,cfg,_=self.fixture();(source/"src/other.py").write_text("y=2")
  with self.assertRaises(ValueError):launcher.verify(cfg)
