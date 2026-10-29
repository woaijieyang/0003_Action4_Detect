"""CPU counterexamples for corner async admission; no CUDA calls allowed."""
import json,unittest,uuid
from unittest.mock import patch
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_stage2_eval import worker

class CornerWorkerAdmissionTests(unittest.TestCase):
    def test_reject_before_allocator_and_dataset(self):
        out=R/"outputs/action4_corner_async_preparation_v001"/("admission_"+uuid.uuid4().hex)
        out.mkdir(parents=True)
        cfg={"role":"A0","mode":"integration","a0_variant":"a0_corner_image",
             "corner_enabled":True,"sam_enabled":True,"output":str(out.relative_to(R))}
        p=out/"config.json";p.write_text(json.dumps(cfg))
        with patch("eval.action4_stage2_eval.current_attempt",return_value="fixture-attempt"), \
             patch("model.training.action4_run_snapshot.verify_running_snapshot"), \
             patch("eval.action4_image_guided_eval.validate_corner_eval_environment",
                   side_effect=ValueError("resource admission denied")) as gate, \
             patch("model.training.action4_resource_policy.apply_allocator_budget",
                   side_effect=AssertionError("allocator reached before admission")) as allocator, \
             patch("eval.action4_stage2_eval.ComponentData",
                   side_effect=AssertionError("dataset reached before admission")) as dataset:
            with self.assertRaisesRegex(ValueError,"resource admission denied"):
                worker(str(p.relative_to(R)),"fixture")
            gate.assert_called_once_with(cfg,"cuda:0")
            allocator.assert_not_called();dataset.assert_not_called()

    def test_worker_keeps_committed_report_and_timing_separate(self):
        import copy,hashlib
        from eval.action4_stage2_eval import bind_evaluation_result
        out=R/"outputs/action4_corner_async_preparation_v001"/("timing_"+uuid.uuid4().hex)
        out.mkdir()
        cfg={"role":"A0","mode":"integration","a0_variant":"a0_corner_image"}
        report={"evaluation_images":650,"evaluation_wall_seconds":23.5,"pose_all8_at_002_success":7}
        (out/"REPORT.json").write_text(json.dumps(report))
        before=copy.deepcopy(report)
        value=bind_evaluation_result(cfg,{"step":10},report,out,650,25.2)
        self.assertEqual(report,before)
        self.assertEqual(value["report"],json.loads((out/"REPORT.json").read_text()))
        self.assertEqual(value["report_artifact"]["sha256"],hashlib.sha256((out/"REPORT.json").read_bytes()).hexdigest())
        self.assertEqual(value["worker_evaluation_wall_seconds"],25.2)
        with self.assertRaisesRegex(ValueError,"image count"):
            bind_evaluation_result(cfg,{"step":10},report,out,649,25.2)
        (out/"REPORT.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"immutable report"):
            bind_evaluation_result(cfg,{"step":10},report,out,650,25.2)

if __name__=="__main__":unittest.main(verbosity=2)
