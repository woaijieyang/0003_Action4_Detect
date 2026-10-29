"""Cross-module CPU analytic fixture; no real inference or effect claims."""
import copy,json,unittest
from pathlib import Path
import numpy as np
import tests.test_action4_corner_scoring_inputs as fixtures
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_image_guided_eval import run_engineering_evaluation_fixture
from eval.action4_corner_scoring_inputs import build_scoring_inputs
from eval.action4_corner_calibration import calibrate
from eval.action4_stage2_eval import prepare_corner_selection,bind_evaluation_result,rank_result
from model.training.action4_async_recovery import commit_result
from model.training.action4_runtime import canonical_hash

class CornerEvalChainTests(unittest.TestCase):
    def test_actual_scoring_calibration_and_async_reject_fixture(self):
        fixture=fixtures.ScoringInputTests();fixture.setUp()
        _,annotation=fixture.bindings()
        payload=copy.deepcopy(fixture.payload)
        class Data:
            records=[fixture.annotation]
            def __len__(self):return 1
            def __getitem__(self,index):return {"record":self.records[index],"image":np.zeros((64,64,3),np.uint8)}
        class Pipeline:
            def corner_context(self,image,rgb):return image
            def predict_corner(self,rgb,K,canonical,*,sam_context,diagnostic_candidate):
                assert sam_context=="img" and diagnostic_candidate is True
                np.testing.assert_array_equal(K,fixture.K)
                np.testing.assert_allclose(canonical,fixture.cad,atol=0,rtol=1e-15)
                return copy.deepcopy(payload)
        cfg={"mode":"integration","role":"A0","a0_variant":"a0_corner_image","corner_enabled":True,"sam_enabled":True,
             "corner_eval_engineering_fixture":True,"eval_loader_workers":0,"corner_eval_source":fixture.source,
             "fixture_annotations":annotation,"parent_bundle":fixture.models["A0"]["path"],
             "parent_bundle_sha256":fixture.models["A0"]["sha256"],"parent_best_step":0}
        snapshot=R/fixture.models["A0"]["path"];destination=fixture.out/"actual_chain"
        report=run_engineering_evaluation_fixture(cfg,snapshot,destination,Data(),
             fixture_factory=lambda *args:Pipeline(),fixture_scorer=build_scoring_inputs,fixture_calibrator=calibrate)
        progress=json.loads((destination/"PROGRESS.json").read_text())
        self.assertEqual(progress["completed_images"],1)
        self.assertEqual(progress["expected_images"],1)
        self.assertEqual(progress["stage"],"commit")
        self.assertFalse(progress["complete"])
        self.assertTrue(all(v>=0 for v in progress["timing_seconds"].values()))
        self.assertEqual(report["pose_gt_count"],2)
        self.assertEqual(report["pose_all8_at_002_success"],2)
        self.assertFalse(report["quality_gate_passed_for_selection"])
        self.assertEqual(report["corner_final_path"],"raw")
        fields,reference=prepare_corner_selection(cfg,0,snapshot,report)
        self.assertFalse(fields["deployment_eligible"])
        result=bind_evaluation_result(cfg,{"step":0},report,destination,1,1.25)
        result.update(fields)
        out=fixture.out/"async";(out/"eval").mkdir(parents=True)
        request=out/"eval/step_00000000.json"
        request.write_text(json.dumps({"step":0,"role":"A0","config_id":canonical_hash(cfg),
                                     "snapshot":str(snapshot.relative_to(R))}))
        commit_result(out,request,snapshot,result,rank_result)
        done=json.loads(request.with_suffix(".done.json").read_text())
        self.assertEqual(done["report"],json.loads((destination/"REPORT.json").read_text()))
        self.assertFalse((out/"BEST.json").exists())
        self.assertFalse((out/"best.pt").exists())
        self.assertTrue(snapshot.exists())
        class Broken(Pipeline):
            def predict_corner(self,*args,**kwargs):
                raise RuntimeError("injected prediction failure")
        failed=fixture.out/"failed_chain"
        with self.assertRaisesRegex(RuntimeError,"injected prediction failure"):
            run_engineering_evaluation_fixture(cfg,snapshot,failed,Data(),
                fixture_factory=lambda *args:Broken(),fixture_scorer=build_scoring_inputs,
                fixture_calibrator=calibrate)
        progress=json.loads((failed/"PROGRESS.json").read_text())
        self.assertEqual(progress["stage"],"failed:prediction")
        self.assertEqual(progress["completed_images"],0)
        self.assertFalse(progress["complete"])
        self.assertFalse((failed/"COMPLETE.json").exists())
        self.assertTrue((failed/"FAILED.json").exists())


if __name__=="__main__":unittest.main(verbosity=2)
