import unittest
from unittest.mock import patch
from model.action4_cotracker_native import load_gpu_core
class GPUAdmission(unittest.TestCase):
 def test_forbidden_GPU_rejected_before_CUDA(self):
  r=dict(physical_gpu=0,gpu_uuid='GPU-fake',cpu_affinity=[28],allocator_gib=8,minimum_free_mib=16384,project_total_cpu=72)
  with patch('torch.cuda.device_count') as cuda:
   with self.assertRaisesRegex(ValueError,'GPU4-7'):load_gpu_core('not_read',r)
   cuda.assert_not_called()
 def test_missing_explicit_allocation_rejected(self):
  with self.assertRaises(ValueError):load_gpu_core('not_read',{})
