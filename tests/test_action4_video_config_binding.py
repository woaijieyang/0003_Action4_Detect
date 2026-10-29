import unittest,json,hashlib,uuid
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_roi_video_job import verify_source

class VideoConfigBinding(unittest.TestCase):
 def test_readonly_config_bytes_bound(self):
  root=R/"outputs/action4_video_config_binding_tests"/uuid.uuid4().hex
  source=root/"source";source.mkdir(parents=True)
  config=root/"CONFIG.json";config.write_text('{"mode":"fixture"}');config.chmod(0o444)
  ref={"path":str(config.relative_to(R)),"sha256":hashlib.sha256(config.read_bytes()).hexdigest()}
  (source/"SOURCE_MANIFEST.json").write_text(json.dumps({"files":{},"configuration":ref}))
  self.assertEqual(verify_source(source)["configuration"],ref)
  config.chmod(0o644)
  with self.assertRaisesRegex(ValueError,"configuration"):verify_source(source)
  config.write_text('{"mode":"changed"}');config.chmod(0o444)
  with self.assertRaisesRegex(ValueError,"configuration"):verify_source(source)
