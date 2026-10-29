import unittest,copy,json,torch
from pathlib import Path
from model.training.action4_scoring_repair_continuation import validate_numerical_recipe,validate_state,canonical
R=Path('/data/zy/vln/a-test')
OLD='p2_a0_corner_roi115_finetune75_v001';NEW='p2_a0_corner_roi115_finetune75_v002'
def fixture():
 spec=json.loads((R/f'runtime/action4_runs/{OLD}/RUN_SPEC.json').read_text());cfg=json.loads((R/spec['config']).read_text());new=json.loads(json.dumps(cfg).replace(OLD,NEW));new['scoring_repair']={};ns=json.loads(json.dumps(spec).replace(OLD,NEW));ns['files']['src/eval/action4_corner_scoring_inputs.py']='d'*64
 return cfg,new,spec,ns
class RepairContractTests(unittest.TestCase):
 def test_unchanged_numeric_recipe_and_only_scoring_source(self):
  old,new,a,b=fixture();delta=validate_numerical_recipe(old,new,a,b);self.assertEqual(set(delta),{'src/eval/action4_corner_scoring_inputs.py'})
 def test_hyperparameter_data_and_topology_changes_rejected(self):
  old,new,a,b=fixture()
  for key,value in [('lr',.001),('epochs',76),('roi_expansion',1.2),('world_size',2),('dataset_root','data/phrase1'),('seed',7)]:
   bad=copy.deepcopy(new);bad[key]=value
   with self.assertRaises(ValueError):validate_numerical_recipe(old,bad,a,b)
 def test_training_math_source_changed_rejected(self):
  old,new,a,b=fixture();b['files']['src/model/action4_gdr_pose.py']='0'*64
  with self.assertRaises(ValueError):validate_numerical_recipe(old,new,a,b)
 def test_actual_full_checkpoint_counters_and_state_valid(self):
  old,new,a,b=fixture();state=torch.load(R/f'outputs/action4_improvement/stage4_corner/{OLD}/last.pt',map_location='cpu',weights_only=True)
  result=validate_state(state,old);self.assertEqual(result['step'],260);self.assertEqual(result['epoch'],1);self.assertEqual(result['cursor'],2064);self.assertEqual(state['config_id'],canonical(old))
 def test_wrong_config_cursor_scheduler_rng_rejected(self):
  old,new,a,b=fixture();state=torch.load(R/f'outputs/action4_improvement/stage4_corner/{OLD}/last.pt',map_location='cpu',weights_only=True)
  for key,value in [('config_id','wrong'),('cursor',2065),('rng_by_rank',[])]:
   bad=dict(state);bad[key]=value
   with self.assertRaises(ValueError):validate_state(bad,old)
  bad=dict(state);bad['scheduler']=dict(state['scheduler'],last_epoch=259)
  with self.assertRaises(ValueError):validate_state(bad,old)
 def test_nonfinite_moment_is_rejected(self):
  old,new,a,b=fixture();state=torch.load(R/f'outputs/action4_improvement/stage4_corner/{OLD}/last.pt',map_location='cpu',weights_only=True)
  first=next(iter(state['optimizer']['state'].values()));first['exp_avg'].reshape(-1)[0]=float('nan')
  with self.assertRaises(ValueError):validate_state(state,old)

 def test_parallel_readers_allowed_writer_rejected(self):
  import fcntl,tempfile
  from model.training.action4_scoring_repair_continuation import predecessor_reader_locks
  with tempfile.TemporaryDirectory(dir=R/'outputs/action4_roi115_recovery_v001') as d:
   p=Path(d)
   for name in ['controller.lock','train.lock']:(p/name).touch()
   with (p/'controller.lock').open('r') as owner:
    fcntl.flock(owner,fcntl.LOCK_SH|fcntl.LOCK_NB)
    with predecessor_reader_locks(p):pass
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):
     with predecessor_reader_locks(p):pass
