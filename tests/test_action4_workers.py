"""Shard selection and device assignment must not depend on worker timing."""
import os
import unittest
from unittest.mock import patch
from data.action4_synthesis import select_shard_entries
from data.action4_devices import ALLOWED_GPU_UUIDS,gpu_identity,worker_environment


class WorkerContracts(unittest.TestCase):
    def test_chunk_union_covers_plan_once_without_changing_entries(self):
        plan={"entries":[{"index":i,"sample_id":str(i)} for i in range(10000)]}
        selected=[]
        pilot=[]
        for shard in range(4):
            first=select_shard_entries(plan,shard,4,entry_count=32)
            pilot.extend(first);selected.extend(first)
            for start in range(32,2500,64):
                selected.extend(select_shard_entries(plan,shard,4,entry_start=start,entry_count=64))
        self.assertEqual(sorted(e["index"] for e in pilot),list(range(128)))
        self.assertEqual(sorted(e["index"] for e in selected),list(range(10000)))
        self.assertTrue(all(e is plan["entries"][e["index"]] for e in selected))

    def test_ranges_reject_conflicts_and_preserve_legacy_limit(self):
        plan={"entries":[{"index":i} for i in range(8)]}
        self.assertEqual(select_shard_entries(plan,1,2,limit=2),[{"index":1},{"index":3}])
        for kw in ({"entry_start":-1},{"entry_count":0},{"entry_start":99},{"limit":1,"entry_count":1}):
            with self.assertRaises(ValueError):select_shard_entries(plan,**kw)

    def test_unauthorized_gpu_rejected_before_query(self):
        with patch("data.action4_devices.subprocess.check_output",side_effect=AssertionError("not queried")):
            with self.assertRaises(ValueError):gpu_identity("GPU-unauthorized")

    def test_actual_mapping_and_worker_environment_contract(self):
        uuid=ALLOWED_GPU_UUIDS[1]
        reply=f"5, {uuid}, 00000000:D1:00.0, 40000, 4000, 49140\n"
        with patch("data.action4_devices.subprocess.check_output",return_value=reply):
            self.assertEqual(gpu_identity(uuid)["index"],5)
            with patch.dict(os.environ,{"RANK":"1","WORLD_SIZE":"4","LOCAL_RANK":"1"}):
                env=worker_environment(uuid)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"],uuid)
                self.assertNotIn("RANK",env);self.assertNotIn("WORLD_SIZE",env)
        with patch("data.action4_devices.subprocess.check_output",return_value=reply.replace("5,","0,",1)):
            with self.assertRaises(RuntimeError):gpu_identity(uuid)
