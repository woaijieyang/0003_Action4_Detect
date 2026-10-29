import json,os,subprocess,sys,unittest
from eval.action4_video_contract import gpu_admission,validate_cpu_affinity

class VideoResourcePolicyTests(unittest.TestCase):
    def policy(self):
        return dict(schema="action4.video_resource_admission.v1",allocated_gpu_ids=[7],allocation_receipt="explicit_receipt.json",allow_shared_gpu=True,uncommitted_margin_mib=8192,worker_memory_budget_mib=4096)
    def test_shared_high_util_not_automatic_rejection(self):
        self.assertTrue(gpu_admission(7,1,24000,24000,95,self.policy())["sharing_explicitly_allowed"])
    def test_explicit_idle_policy_and_optional_cap(self):
        p=self.policy();p["allow_shared_gpu"]=False
        with self.assertRaises(ValueError):gpu_admission(7,1,24000,24000,0,p)
        p=self.policy();p["max_instant_utilization_percent"]=10
        with self.assertRaises(ValueError):gpu_admission(7,1,24000,24000,95,p)
    def test_memory_and_allocation_remain_required(self):
        with self.assertRaises(ValueError):gpu_admission(7,2,10000,20000,0,self.policy())
        with self.assertRaises(ValueError):gpu_admission(6,1,24000,0,0,self.policy())
    def test_actual_child_inherits_affinity(self):
        parent=sorted(os.sched_getaffinity(0))
        authority={"cpu_policy":{"allowed_logical_cpu_ids":list(range(24,96))}}
        self.assertEqual(validate_cpu_affinity(parent,authority),parent)
        child=json.loads(subprocess.check_output([sys.executable,"-B","-c","import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))"],text=True))
        self.assertEqual(child,parent)
        with self.assertRaises(ValueError):validate_cpu_affinity(range(24,31),authority)
    def test_ffmpeg_all_thread_pools_bounded(self):
        from pathlib import Path
        source=Path("src/eval/action4_staged_video.py").read_text()
        self.assertEqual(source.count('"-filter_threads","2","-filter_complex_threads","2"'),2)
        self.assertIn('"-threads","2"',source)
if __name__=="__main__":unittest.main()
