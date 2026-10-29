import hashlib,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
from data import action4_sam_native_export as m
class DecodeReuse(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir='/data/zy/vln/a-test/outputs')
  self.p=Path(self.temp.name)/'rgb.png'
  Image.fromarray(np.arange(12*13*3,dtype=np.uint8).reshape(12,13,3)).save(self.p)
  self.sha=hashlib.sha256(self.p.read_bytes()).hexdigest();m._RGB_DECODE_CACHE.clear()
 def tearDown(self):self.temp.cleanup()
 def test_reuse_and_no_mutation_leak(self):
  with mock.patch.object(m.Image,'open',wraps=m.Image.open) as opened:
   a=m._v2_rgb(self.p,self.sha);expected=a.copy();a[:]=0
   b=m._v2_rgb(self.p,self.sha)
   np.testing.assert_array_equal(b,expected);self.assertEqual(opened.call_count,1)
 def test_changed_file_and_wrong_digest_rejected(self):
  m._v2_rgb(self.p,self.sha)
  with self.assertRaises(ValueError):m._v2_rgb(self.p,'0'*64)
  Image.fromarray(np.zeros((12,13,3),dtype=np.uint8)).save(self.p)
  with self.assertRaises(ValueError):m._v2_rgb(self.p,self.sha)
 def test_memory_bound_and_pid_boundary(self):
  with mock.patch.object(m,'_RGB_DECODE_MAX_BYTES',1):
   m._v2_rgb(self.p,self.sha);self.assertFalse(m._RGB_DECODE_CACHE)
  m._v2_rgb(self.p,self.sha);m._RGB_DECODE_PID=-1
  with mock.patch.object(m.Image,'open',wraps=m.Image.open) as opened:
   m._v2_rgb(self.p,self.sha);self.assertEqual(opened.call_count,1)
