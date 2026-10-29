import copy,unittest,uuid
from pathlib import Path
from eval.action4_sam_image_eval import registered_sam_allocation,acquire_sam_worker_lease
R=Path('/data/zy/vln/a-test')
class RegisteredResources(unittest.TestCase):
 def fixture(self):
  r=dict(worker_id='worker0',physical_gpu=4,gpu_uuid='GPU-test',cpu_affinity=[90,91],allocator_gib=8,fresh_free_mib=16384)
  s=dict(sam_worker_allocations=[copy.deepcopy(r)],sam_max_workers=1,sam_per_gpu_worker_limits={'4':1},sam_excluded_cpus=[92,93])
  return r,s
 def test_exact_allocation(self):
  r,s=self.fixture();self.assertTrue(registered_sam_allocation(r,s,range(24,96)))
 def test_mismatch_or_missing_worker_rejected(self):
  r,s=self.fixture()
  for key,value in [('worker_id','wrong'),('cpu_affinity',[88,89]),('allocator_gib',9),('physical_gpu',6)]:
   bad=dict(r);bad[key]=value
   with self.assertRaises(ValueError):registered_sam_allocation(bad,s,range(24,96))
 def test_duplicate_and_reserved_CPUs_rejected(self):
  r,s=self.fixture();s['sam_worker_allocations'].append(dict(r,worker_id='worker1'))
  with self.assertRaises(ValueError):registered_sam_allocation(r,s,range(24,96))
  r,s=self.fixture();s['sam_excluded_cpus']=[90]
  with self.assertRaises(ValueError):registered_sam_allocation(r,s,range(24,96))
 def test_worker_budget(self):
  r,s=self.fixture();s['sam_max_workers']=0
  with self.assertRaises(ValueError):registered_sam_allocation(r,s,range(24,96))
 def test_lease_excludes_duplicate_live_worker(self):
  name='test_slot_'+uuid.uuid4().hex
  first=acquire_sam_worker_lease(name,{'test':True})
  try:
   with self.assertRaises(BlockingIOError):acquire_sam_worker_lease(name,{'test':True})
  finally:first.close()
  second=acquire_sam_worker_lease(name,{'test':True});second.close()
  (R/'runtime/action4_resource_leases'/(name+'.lock')).unlink()

 def test_32gib_sequence_needs_explicit_tier(self):
  r,s=self.fixture();r.update(allocator_gib=32,fresh_free_mib=40960);s["sam_worker_allocations"]=[copy.deepcopy(r)]
  with self.assertRaises(ValueError):registered_sam_allocation(r,s,range(24,96))
  s["sam_allowed_allocator_gib"]=[8,32]
  self.assertTrue(registered_sam_allocation(r,s,range(24,96)))
 def test_sequence_headroom_and_aggregate_are_enforced(self):
  r,s=self.fixture();r.update(allocator_gib=32,fresh_free_mib=32768)
  s.update(sam_allowed_allocator_gib=[8,32],sam_worker_allocations=[copy.deepcopy(r)])
  with self.assertRaises(ValueError):registered_sam_allocation(r,s,range(24,96))
  r["fresh_free_mib"]=40960;s["sam_worker_allocations"]=[copy.deepcopy(r),dict(r,worker_id="worker1",cpu_affinity=[88,89])]
  s["sam_max_workers"]=2;s["sam_per_gpu_worker_limits"]={"4":2}
  with self.assertRaises(ValueError):registered_sam_allocation(r,s,range(24,96))
