import unittest
from eval.action4_joint_capacity_probe import summarize
class JointTiming(unittest.TestCase):
 def samples(self):
  return [dict(step=i,epoch=(i-1)//5+1,global_actual_units=24 if i%5==0 else 48,
      did_update=True,status_mtime_ns=i*2000000000,live_eval_workers=32,prediction_eval_workers=32)
      for i in range(1,21)]
 def test_full_batches_and_epoch_checkpoints_separated(self):
  r=summarize(self.samples())
  self.assertEqual(r["full_batch_intervals"],12)
  self.assertEqual(r["qualified_joint_windows"],4)
  self.assertEqual(r["fastest_observed_window_mean_seconds"],2)
  self.assertAlmostEqual(r["required_eval_calls_per_second_with20percent_margin"],.024)
  self.assertFalse(r["formal_admission"])
 def test_no_overlap_does_not_claim_joint_measurement(self):
  s=self.samples()
  for r in s:r["prediction_eval_workers"]=0
  r=summarize(s);self.assertFalse(r["joint_timing_usable"])
  self.assertIsNone(r["required_eval_calls_per_second_with20percent_margin"])
 def test_missed_or_nonmonotonic_updates_rejected(self):
  with self.assertRaises(ValueError):summarize(self.samples()[1:])
  s=self.samples();s[1]["status_mtime_ns"]=0
  with self.assertRaises(ValueError):summarize(s)


class ProcessTemporaryPath(unittest.TestCase):
 def test_short_project_path_supports_actual_tensor_ipc(self):
  import os,subprocess,json
  from eval.action4_joint_capacity_probe import process_tmp_path,R
  tmp=process_tmp_path("runtime/action4_runs/"+"long_run_name_"*15+"/RUN_SPEC.json")
  self.assertTrue(tmp.is_relative_to(R));self.assertLessEqual(len(os.fsencode(str(tmp))),70)
  tmp.mkdir(parents=True,exist_ok=True)
  code="import torch,tempfile,json; torch.set_num_threads(1); data=torch.utils.data.TensorDataset(torch.arange(20).reshape(10,2)); loader=torch.utils.data.DataLoader(data,batch_size=2,num_workers=2); rows=list(loader); assert sum(int(x[0].sum()) for x in rows)==190; print(json.dumps({'tmp':tempfile.gettempdir(),'batches':len(rows)}))"
  env=dict(os.environ,TMPDIR=str(tmp),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
  result=subprocess.run([str(R/".venv/bin/python"),"-B","-c",code],env=env,cwd=R,capture_output=True,text=True,timeout=30)
  self.assertEqual(result.returncode,0,result.stderr)
  self.assertEqual(json.loads(result.stdout)["tmp"],str(tmp))
  self.assertEqual(json.loads(result.stdout)["batches"],5)
