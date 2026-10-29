import json,threading,unittest,uuid
from pathlib import Path
from model.training.action4_capacity_barrier import settings,wait_ready,R
from model.training.action4_runtime import atomic_json
class CapacityBarrier(unittest.TestCase):
 def cfg(self):
  d=R/"outputs/action4_capacity_barrier_tests"/uuid.uuid4().hex;d.mkdir(parents=True)
  return dict(mode="integration",joint_capacity_probe=True,world_size=4,batch=48,run_spec="fixture",
   capacity_barrier=dict(experiment_id=uuid.uuid4().hex,directory=str(d.relative_to(R)),timeout_seconds=1800)),d
 def test_wait_preserves_rng_and_requires_go(self):
  import torch,numpy as np
  cfg,d=self.cfg();before=torch.get_rng_state().clone();npbefore=np.random.get_state()
  def release():
   while not (d/"train_0_READY.json").exists():threading.Event().wait(.01)
   atomic_json(d/"GO.json",{"experiment_id":cfg["capacity_barrier"]["experiment_id"]})
  t=threading.Thread(target=release);t.start();wait_ready(cfg,"train",0,first_batch_units=12);t.join()
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
  after=np.random.get_state();self.assertEqual(npbefore[0],after[0]);np.testing.assert_array_equal(npbefore[1],after[1])
 def test_formal_or_nonfull_first_batch_refused(self):
  c,d=self.cfg();c["mode"]="full_training"
  with self.assertRaises(ValueError):settings(c,"train")
  c,d=self.cfg()
  with self.assertRaises(ValueError):wait_ready(c,"train",0,first_batch_units=6)
 def test_abort_and_cross_experiment_go(self):
  c,d=self.cfg();atomic_json(d/"ABORT.json",{"reason":"fixture"})
  with self.assertRaises(RuntimeError):wait_ready(c,"train",0,first_batch_units=12)
  c,d=self.cfg();atomic_json(d/"GO.json",{"experiment_id":"wrong"})
  with self.assertRaises(ValueError):wait_ready(c,"train",0,first_batch_units=12)
 def test_no_barrier_is_noop(self):
  self.assertIsNone(settings({}, "train"));wait_ready({},"train",0)
