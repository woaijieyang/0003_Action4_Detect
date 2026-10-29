import json,multiprocessing,os,tempfile,time,unittest
from pathlib import Path
from model.training.action4_startup_readiness import wait_data_ready,wait_training_ready
from data.action4_paths import PROJECT_ROOT
def worker(out,rank):
 if rank==1:time.sleep(.4)
 wait_data_ready(out,"attempt",rank,2,10400,timeout=5)
class ReadinessTests(unittest.TestCase):
 def setUp(self):
  base=PROJECT_ROOT/".cache/startup_tests";base.mkdir(parents=True,exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=base);self.out=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_two_actual_processes_wait_for_late_data_rank(self):
  ctx=multiprocessing.get_context("spawn")
  processes=[ctx.Process(target=worker,args=(str(self.out),i)) for i in range(2)]
  for p in processes:p.start()
  for p in processes:p.join(10);self.assertEqual(p.exitcode,0)
  d=self.out/"startup/attempt"
  a=json.loads((d/"rank_0_released.json").read_text());b=json.loads((d/"rank_1_data_ready.json").read_text())
  self.assertGreaterEqual(a["all_data_ready_unix"],b["ready_unix"])
  self.assertEqual(len(set(a["ready_pids"])),2)
 def test_missing_rank_timeout_before_nccl(self):
  with self.assertRaises(TimeoutError):wait_data_ready(self.out,"missing",0,2,10,.1)
 def test_wrong_domain_rejected(self):
  d=self.out/"startup/bad";d.mkdir(parents=True)
  (d/"rank_1_data_ready.json").write_text(json.dumps(dict(attempt_id="bad",rank=1,world=2,units=11)))
  with self.assertRaises(ValueError):wait_data_ready(self.out,"bad",0,2,10,1)
 def test_evaluator_wait_rejects_failed_or_stale_trainer(self):
  class Failed:
   def poll(self):return 1
  with self.assertRaises(RuntimeError):wait_training_ready(self.out,"a",Failed(),1)
  class Alive:
   def poll(self):return None
  (self.out/"TRAIN_PROCESS.json").write_text(json.dumps(dict(attempt_id="old",pids=[1,2])))
  with self.assertRaises(ValueError):wait_training_ready(self.out,"a",Alive(),1)
 def test_correct_trainer_receipt_releases_evaluators(self):
  class Alive:
   def poll(self):return None
  (self.out/"TRAIN_PROCESS.json").write_text(json.dumps(dict(attempt_id="a",pids=[1,2])))
  self.assertEqual(wait_training_ready(self.out,"a",Alive(),1)["pids"],[1,2])
