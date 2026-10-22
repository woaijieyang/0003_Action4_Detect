import unittest,sys,json,tempfile
from unittest.mock import patch
from pathlib import Path
from copy import deepcopy
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data.action4_template_batch import validate_config
from data.action4_devices import ALLOWED_GPU_UUIDS
class WorkerConfigTests(unittest.TestCase):
 def setUp(self):
  self.plan={'schema':'action4.fixed_template_batch_plan.v1','preview_count':2,'entries':[{'template':'wall','entry':{'target_presence':[True,True],'target_count':2,'split':'train'}},{'template':'table_full_garment','entry':{'target_presence':[True,True],'target_count':2,'split':'test'}}]}
  self.config={'schema':'action4.template_worker.v1','indices':[0],'template':'wall','gpu_uuid':ALLOWED_GPU_UUIDS[0],'threads':2,'max_attempts':32,'data_out':'data/action4_synth/unit_worker','runtime_out':'outputs/unit_worker'}
 def test_valid_one_fixed_scene(self):validate_config(self.config,self.plan)
 def test_scene_mix_duplicate_out_of_preview_rejected(self):
  for ids in ([0,1],[0,0],[2],[-1],[True],[]):
   c=deepcopy(self.config);c['indices']=ids
   with self.assertRaises(ValueError):validate_config(c,self.plan)
 def test_devices_threads_paths_retry_bounds(self):
  for key,value in [('gpu_uuid','GPU3'),('threads',72),('max_attempts',0),('max_attempts',100),('data_out','outputs/wrong_role'),('runtime_out','/tmp/escape')]:
   c=deepcopy(self.config);c[key]=value
   with self.assertRaises(ValueError):validate_config(c,self.plan)
 def test_test_requires_two_real_targets(self):
  c=deepcopy(self.config);c.update(indices=[1],template='table_full_garment');p=deepcopy(self.plan);p['entries'][1]['entry'].update(target_presence=[True,False],target_count=1)
  with self.assertRaises(ValueError):validate_config(c,p)
 def test_existing_run_never_receives_failed_marker(self):
  from data import action4_template_batch as batch
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='template_ownership_test_') as tmp:
   base=Path(tmp);(base/'configs').mkdir();config=deepcopy(self.config);config['plan_path']='configs/plan.json'
   (base/'configs/worker.json').write_text(json.dumps(config));(base/'configs/plan.json').write_text(json.dumps(self.plan));old=base/config['runtime_out'];old.mkdir(parents=True);(old/'COMPLETE.json').write_text('original')
   with patch.object(batch,'ROOT',base),patch.object(Path,'cwd',return_value=base):
    with self.assertRaises(FileExistsError):batch.run_blender('configs/worker.json')
   self.assertEqual({p.name for p in old.iterdir()},{'COMPLETE.json'});self.assertEqual((old/'COMPLETE.json').read_text(),'original');self.assertFalse((base/config['data_out']).exists())
 def test_failure_record_only_in_new_owned_directory(self):
  from data import action4_template_batch as batch
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='template_owned_failure_test_') as tmp:
   base=Path(tmp);(base/'configs').mkdir();config=deepcopy(self.config);config['plan_path']='configs/plan.json'
   (base/'configs/worker.json').write_text(json.dumps(config));(base/'configs/plan.json').write_text(json.dumps(self.plan))
   with patch.object(batch,'ROOT',base),patch.object(Path,'cwd',return_value=base),patch.object(batch,'_run_blender_owned',side_effect=RuntimeError('fixture_failure')):
    with self.assertRaises(RuntimeError):batch.run_blender('configs/worker.json')
   failure=json.loads((base/config['runtime_out']/'FAILED.json').read_text());self.assertEqual(failure['status'],'FAILED_OWNED_INVOCATION');self.assertIn('fixture_failure',failure['traceback'])
if __name__=='__main__':unittest.main()
