"""Reader failure-contract tests; synthetic tensors, not model/cache accuracy."""
import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from data import action4_sam_input_cache as cache

class InputCacheTests(unittest.TestCase):
 def setUp(self):
  base=cache.PROJECT_ROOT/'outputs/action4_sam_input_cache_v002';base.mkdir(exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(dir=base,prefix='test_');self.d=Path(self.temp.name)
  self.raw=self.d/'original.bin';self.raw.write_bytes(b'synthetic-original')
  self.frame={'global_frame_id':4,'local_index':4,'pts':1600,'image':str(self.raw),'sha256':'fixture'}
  self.contract={'schema':cache.SCHEMA,'frames':[self.frame],'native_hw':[2464,3248]}
  self.array=np.zeros((3,1008,1008),np.float16)
  self.path=self.d/'frame_000000.npy'
  with self.path.open('wb') as f:np.save(f,self.array,allow_pickle=False)
  payload=self.path.read_bytes()
  self.manifest={'schema':cache.SCHEMA,'contract':self.contract,'complete':True,
   'entries':[{'file':'frame_000000.npy','sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'source_stat':cache._stat(self.raw),'frame':self.frame}]}
  self.mp=self.d/'CACHE.json';self.write_manifest()
  self.check=patch.object(cache,'_check_contract');self.check.start()
 def tearDown(self):self.check.stop();self.temp.cleanup()
 def write_manifest(self):
  content=cache._json(self.manifest);self.mp.write_bytes(content);self.sha=hashlib.sha256(content).hexdigest()
 def reader(self,**kwargs):return cache.SamInputTensorCache(self.mp,expected_sha256=kwargs.get('sha',self.sha),expected_contract=kwargs.get('contract',self.contract))
 def test_exact_shape_dtype_native_geometry_and_private_tensor(self):
  r=self.reader();self.assertEqual(r.as_video_frames()[1:],(2464,3248));self.assertEqual(len(r),1)
  x=r[0];self.assertEqual(tuple(x.shape),(3,1008,1008));self.assertEqual(str(x.dtype),'torch.float16')
  x[0,0,0]=1;self.assertEqual(float(r[0][0,0,0]),0)
 def test_foreign_manifest_and_changed_order_contract(self):
  with self.assertRaises(ValueError):self.reader(sha='0'*64)
  c=dict(self.contract,frames=[dict(self.frame,global_frame_id=5)])
  with self.assertRaises(ValueError):self.reader(contract=c)
 def test_incomplete_and_truncated_entries(self):
  self.manifest['complete']=False;self.write_manifest()
  with self.assertRaises(ValueError):self.reader()
  self.manifest['complete']=True;self.manifest['entries']=[];self.write_manifest()
  with self.assertRaises(ValueError):self.reader()
 def test_payload_corruption_same_length(self):
  r=self.reader();b=bytearray(self.path.read_bytes());b[-1]^=1;self.path.write_bytes(b)
  with self.assertRaises(ValueError):r[0]
 def test_truncation_before_and_after_admission(self):
  r=self.reader();self.path.write_bytes(b'truncated')
  with self.assertRaises(ValueError):self.reader()
  with self.assertRaises(ValueError):r[0]
 def test_original_stat_changes_fail_closed(self):
  r=self.reader();self.raw.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.reader()
  with self.assertRaises(ValueError):r[0]
 def test_manifest_mutation_after_admission(self):
  r=self.reader();self.mp.write_bytes(self.mp.read_bytes()+b' ')
  with self.assertRaises(ValueError):r[0]
 def test_foreign_cache_path_rejected(self):
  self.manifest['entries'][0]['file']='../foreign.npy';self.write_manifest()
  with self.assertRaises(ValueError):self.reader()
 def test_length_and_index_do_not_parse_entire_entry_list(self):
  r=self.reader();loads=json.loads
  self.assertIsInstance(r._entry_json,tuple)
  self.assertTrue(all(isinstance(x,bytes) for x in r._entry_json))
  with patch.object(cache.json,'loads',wraps=loads) as mocked:
   self.assertEqual(len(r),1);mocked.assert_not_called()
   tensor=r[0]
   self.assertEqual(tuple(tensor.shape),(3,1008,1008))
   mocked.assert_called_once_with(r._entry_json[0])
   self.assertTrue(mocked.call_args.args[0].startswith(b'{'))
 def test_bounds_and_caller_metadata_mutation(self):
  r=self.reader();self.contract['frames'][0]['image']='foreign'
  self.assertEqual(tuple(r[0].shape),(3,1008,1008))
  for i in (-1,1,True):
   with self.assertRaises(IndexError):r[i]

if __name__=='__main__':unittest.main()
