import os,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from eval.action4_sam_throughput import configure_compute

class ComputeSetupTests(unittest.TestCase):
    def test_cublas_precedes_cuda_initialization_and_budget(self):
        def props(device):
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"],":4096:8")
            self.assertEqual(device,0)
            return SimpleNamespace(total_memory=48*1024**3)
        cuda=SimpleNamespace(get_device_properties=Mock(side_effect=props),set_per_process_memory_fraction=Mock())
        torch=SimpleNamespace(cuda=cuda,set_num_threads=Mock(),set_num_interop_threads=Mock(),use_deterministic_algorithms=Mock())
        cv2=SimpleNamespace(setNumThreads=Mock())
        with patch.dict(os.environ,{},clear=True):configure_compute(torch,cv2,2)
        cuda.set_per_process_memory_fraction.assert_called_once_with(2/48,0)
        torch.set_num_threads.assert_called_once_with(1)
        torch.set_num_interop_threads.assert_called_once_with(1)
        cv2.setNumThreads.assert_called_once_with(1)

if __name__=="__main__":unittest.main()
