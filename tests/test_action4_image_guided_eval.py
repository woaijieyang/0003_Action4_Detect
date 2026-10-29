"""Explicit orchestration fixtures and mocked resource guards; no model/GPU work."""
import copy,json,hashlib,uuid,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
import eval.action4_image_guided_eval as runner
import model.training.action4_run_snapshot as snapshot_module

class DataFixture:
    def __init__(self,n=650):
        self.records=[{"sample_id":f"image{i:04d}","camera":{"K":np.eye(3).tolist()},
            "corners_3d_mm":[[0,0,0]]*8,"instances":[{"GT_secret":"never pass to inference"}]} for i in range(n)]
    def __len__(self):return len(self.records)
    def __getitem__(self,i):return {"record":self.records[i],"image":np.zeros((2,2,3),np.uint8)}

class PipelineFixture:
    def __init__(self,fail=None):self.calls=0;self.fail=fail
    def corner_context(self,image,rgb):return image
    def predict_corner(self,rgb,K,canonical,*,sam_context,diagnostic_candidate):
        if self.calls==self.fail:raise RuntimeError("fixture interrupted")
        self.calls+=1
        assert diagnostic_candidate is True and rgb.shape==(2,2,3) and K.shape==(3,3) and canonical.shape==(8,3)
        raw=[{"prediction_id":sam_context+":p","marker":"raw"}];candidate=[{"prediction_id":sam_context+":p","marker":"candidate"}]
        return {"image_id":sam_context,"identity":{"engineering_fixture":True,"fixture":"same-effective-config"},
            "quality_ready":False,"instances":copy.deepcopy(raw),"raw_instances":raw,"candidate_instances":candidate}

class ExecutorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.out=R/"outputs/action4_image_guided_eval_preparation_v001"/uuid.uuid4().hex;self.out.mkdir(parents=True)
        self.cfg={"mode":"integration","corner_eval_engineering_fixture":True,"eval_loader_workers":0,
            "corner_eval_source":{"path":"explicit-fixture-source","sha256":"a"*64},"fixture_annotations":{"path":"explicit-fixture-GT","sha256":"b"*64}}
    def scorer(self,predictions,annotations,source,output,*,diagnostic):
        self.assertTrue(diagnostic);p=R/predictions["path"]
        self.assertFalse(p.stat().st_mode & 0o222)
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),predictions["sha256"])
        envelope=json.loads(p.read_text());n=len(envelope["results"])
        return {"domain":{},"run_binding":{},"records":[],
            "raw_report":{"pose_report":{"score_schema":"action4.pose_projection_all8_002.v1","pose_gt_count":n,"marker":"raw"}},
            "candidate_report":{"pose_report":{"score_schema":"action4.pose_projection_all8_002.v1","pose_gt_count":n,"marker":"candidate"}}}
    def calibration(self,domain,run,records,raw,candidate):
        return {"quality_ready":True,"quality_gate_passed_for_selection":True,"raw_report":raw,"candidate_report":candidate,
            "final_report":candidate,"final_path":"candidate"}
    def test_full650_fixture_is_global_raw_and_never_admitted(self):
        pipeline=PipelineFixture();destination=self.out/"eval";destination.mkdir()
        report=runner.run_engineering_evaluation_fixture(self.cfg,"fixture_snapshot",destination,DataFixture(),
            fixture_factory=lambda *args:pipeline,fixture_scorer=self.scorer,fixture_calibrator=self.calibration)
        self.assertEqual(pipeline.calls,650);self.assertFalse(report["corner_quality_ready"])
        self.assertFalse(report["quality_gate_passed_for_selection"]);self.assertEqual(report["marker"],"raw")
        final=json.loads((R/report["final_predictions"]["path"]).read_text())
        self.assertEqual(len(final),650);self.assertTrue(all(p["instances"][0]["marker"]=="raw" for p in final))
        payload=json.loads((R/report["prediction_artifact"]["path"]).read_text())["results"]
        candidates=runner._global_predictions(payload,True)
        self.assertTrue(all(p["instances"][0]["marker"]=="candidate" for p in candidates))
        self.assertEqual(report["raw_pose_report"],json.loads((R/report["raw_pose_report_artifact"]["path"]).read_text()))
        self.assertIsNone(report["inference_execution_receipt"])
        self.assertTrue((destination/"COMPLETE.json").exists())
        with self.assertRaises(FileExistsError):
            runner.run_engineering_evaluation_fixture(self.cfg,"fixture_snapshot",destination,DataFixture(),
                fixture_factory=lambda *args:pipeline,fixture_scorer=self.scorer,fixture_calibrator=self.calibration)
    def test_interruption_has_no_success_report(self):
        pipeline=PipelineFixture(fail=3);destination=self.out/"failed"
        with self.assertRaisesRegex(RuntimeError,"interrupted"):
            runner.run_engineering_evaluation_fixture(self.cfg,"fixture_snapshot",destination,DataFixture(),
                fixture_factory=lambda *args:pipeline,fixture_scorer=self.scorer,fixture_calibrator=self.calibration)
        self.assertFalse((destination/"COMPLETE.json").exists());self.assertFalse((destination/"REPORT.json").exists())
        self.assertFalse((destination/"PREDICTION_ARTIFACT.json").exists())
        self.assertEqual(json.loads((destination/"FAILED.json").read_text())["stage"],"prediction")
    def test_production_and_source_guards_before_factory(self):
        cfg={"role":"A0","a0_variant":"a0_corner_image","corner_enabled":True,"sam_enabled":True,
            "require_immutable_source":True,"run_spec":"unused"}
        with patch("eval.action4_staged_pipeline.StagedPipeline.from_corner_bundles",side_effect=AssertionError("no model")):
            with self.assertRaisesRegex(ValueError,"frozen-source"):runner.run_corner_evaluation(cfg,"unused",self.out/"prod",None)
        with patch.object(runner,"CODE_ROOT",self.out),patch.object(runner,"__file__",str(self.out/"entry.py")):
            with self.assertRaisesRegex(ValueError,"Stale/mixed"):runner._origins(self.out)
        with self.assertRaises(ValueError):
            runner.run_engineering_evaluation_fixture({"mode":"integration"},"unused",self.out/"no",DataFixture(),
                fixture_factory=None,fixture_scorer=None,fixture_calibrator=None)
    def test_resource_guards_without_actual_GPU_queries(self):
        from contextlib import ExitStack
        good="GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";wrong="GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        cfg={"eval_gpus":[4],"evaluation_allocator_budget_mib":1024,"project_total_reserved_cpu":3}
        policy={"authority":{"path":"fixture","sha256":"a"*64},"forbidden_gpu_ids":[0,1,2,3],
                "cpu_policy":{"allowed_logical_cpu_ids":list(range(24,96)),"max_logical_cpus":72}}
        with ExitStack() as stack:
            stack.enter_context(patch("model.training.action4_resource_policy.validate_device_policy",return_value=policy))
            stack.enter_context(patch("model.training.action4_resource_policy.process_cpu_affinity",return_value=[38,40]))
            stack.enter_context(patch("os.sched_getaffinity",return_value={38,40}))
            query=stack.enter_context(patch("subprocess.check_output",return_value=f"4, {good}, 50000"))
            init=stack.enter_context(patch("torch.cuda.is_initialized",return_value=False))
            for env in ("",good+","+wrong,"0"):
                with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":env}):
                    with self.assertRaises(ValueError):runner._resource_guard(cfg,"cuda:0")
            self.assertEqual(query.call_count,0)
            with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":good}):
                proof=runner._resource_guard(cfg,"cuda:0");self.assertEqual(proof["physical_gpu"],4)
                with self.assertRaises(ValueError):runner._resource_guard(dict(cfg,eval_gpus=[0]),"cuda:0")
            with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":wrong}):
                with self.assertRaisesRegex(ValueError,"not an authorized"):runner._resource_guard(cfg,"cuda:0")
            init.return_value=True
            with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":good}),patch("torch.cuda.get_device_properties",return_value=SimpleNamespace(uuid=wrong)):
                with self.assertRaisesRegex(ValueError,"logical0"):runner._resource_guard(cfg,"cuda:0")
            with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":good}),patch("torch.cuda.current_device",return_value=0),patch("torch.cuda.memory_reserved",return_value=100*2**20),patch("torch.cuda.memory_allocated",return_value=10*2**20):
                import uuid as uuid_module
                for actual_uuid in (good,good[4:],uuid_module.UUID(good[4:]),uuid_module.UUID(good[4:]).bytes):
                    with patch("torch.cuda.get_device_properties",return_value=SimpleNamespace(uuid=actual_uuid)):
                        self.assertTrue(runner._resource_guard(cfg,"cuda:0")["initialized_mapping_checked"])
            init.return_value=False
            with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":good}),patch("os.sched_getaffinity",return_value={38}):
                with self.assertRaisesRegex(ValueError,"affinity"):runner._resource_guard(cfg,"cuda:0")
            query.return_value=f"4, {good}, 100"
            with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":good}):
                with self.assertRaisesRegex(ValueError,"headroom"):runner._resource_guard(cfg,"cuda:0")
            query.side_effect=AssertionError("CPU must not query GPU");init.side_effect=AssertionError("CPU must not inspect CUDA")
            self.assertFalse(runner._resource_guard({},"cpu")["GPU_queried"])
    def test_corner_freeze_code_only_and_baseline_compatibility(self):
        root=self.out/"fixture_project";root.mkdir()
        for relative,text in (("src/eval/action4_image_guided_eval.py","# fixture code"),("src/helper.py","# helper"),
            ("scripts/tool.py","# script"),("runtime/surface_deps/stub.py","# dep"),("configs/policy.json","{}"),("validation.json","{}")):
            p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        parent=root/"models/A0.pt";parent.parent.mkdir();parent.write_bytes(b"fixture parent bytes")
        b0=root/"models/B0.pt";b0.write_bytes(b"fixture counterpart")
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        registry=root/"configs/model/action4_best_registry_v003.json";registry.parent.mkdir()
        registry.write_text(json.dumps({"entries":{"A0":{"sha256":sha(parent),"selected_step":3}}}))
        base={"role":"A0","parent_bundle":"models/A0.pt","parent_best_step":3,"parent_bundle_sha256":sha(parent),
            "validation_receipt":"validation.json","evaluation_counterpart_bundle":"models/B0.pt","evaluation_counterpart_sha256":sha(b0)}
        def freeze(name,cfg):
            path=root/(name+".json");path.write_text(json.dumps(cfg))
            with patch.object(snapshot_module,"R",root),patch.object(snapshot_module.subprocess,"check_output",return_value="fixture-context-not-real-git\n"):
                return snapshot_module.freeze(path,name)
        baseline=freeze("baseline",base)
        old=json.loads((root/baseline["config"]).read_text());self.assertNotIn("corner_eval_source",old)
        cfg=dict(base,a0_variant="a0_corner_image",corner_enabled=True,sam_enabled=True,
            corner_counterpart={"B0":{"path":"models/B0.pt","sha256":sha(b0)}})
        corner=freeze("corner",cfg);frozen=json.loads((root/corner["config"]).read_text())
        manifest=root/frozen["corner_eval_source"]["path"];source=json.loads(manifest.read_text())
        self.assertTrue(all(p.endswith(".py") for p in source["files_sha256"]))
        self.assertNotIn("resolved_config.json",source["files_sha256"])
        self.assertEqual(frozen["corner_counterpart"]["B0"]["path"],frozen["evaluation_counterpart_bundle"])
        self.assertEqual(frozen["parent_bundle_sha256"],base["parent_bundle_sha256"])
        self.assertEqual(frozen["corner_eval_freeze_sources"]["parent"]["path"],"models/A0.pt")
        spec=json.loads((root/corner["run_spec"]).read_text())
        code_root=root/spec["code_root"]
        self.assertEqual(spec["files"]["CORNER_EVAL_SOURCE.json"],sha(manifest))
        self.assertFalse(manifest.stat().st_mode & 0o222)
        launcher=(root/corner["launcher"]).read_text()
        self.assertIn("eval.action4_image_guided_eval",launcher)
        self.assertIn("else 'model.training.action4_stage2_dispatch'",launcher)

if __name__=="__main__":unittest.main()
