"""Official-schema-shaped CPU fixtures; no SAM import, GPU, GT or real inference."""
import copy
import json
import os
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch
import numpy as np
from PIL import Image
from eval.action4_sam_image_eval import (
    PROJECT_ROOT, read_json, file_sha, load_manifest, prepare_run, run_jobs,
    NativeJobFailure, NativeBackend, ResourceWait, require_free_margin, bind_native_output, validate_config, validate_source_snapshot, validate_live_authority, digest,
)

class FakeBackend:
    kind = "cpu_fixture_not_native_inference"
    def __init__(self, mode="normal"):
        self.mode=mode;self.calls=[];self.active=set();self.count=0;self.created=0
    def call(self, request):
        self.calls.append(request.copy())
        kind=request["type"]
        if kind=="start_session":
            self.count+=1;sid="fixture-session-"+str(self.count);self.active.add(sid);self.created+=1
            return {"session_id":sid}
        if kind=="close_session":
            if self.mode=="close_fail":return {"is_success":False}
            self.active.remove(request["session_id"]);return {"is_success":True}
        assert kind=="add_prompt" and request["text"]=="camera"
        assert "points" not in request and request["frame_index"]==0
        if self.mode=="oom":raise RuntimeError("CUDA out of memory engineering fixture")
        if self.mode=="none":return {"frame_index":0,"outputs":None}
        box=np.array(request["bounding_boxes"][0],np.float32)
        x,y,w,h=box
        mask=np.zeros((8,10),bool)
        mask[round(y*8):round((y+h)*8),round(x*10):round((x+w)*10)]=True
        n=0 if self.mode=="empty_public" else 2 if self.mode=="ambiguous" else 1
        if self.mode=="empty_pixels":mask[:]=False
        if self.mode=="merged":
            mask[1:5,1:9]=True;box=np.array([.1,.125,.8,.5],np.float32)
        return {"frame_index":1 if self.mode=="wrong_frame" else 0,"outputs":{
            "out_obj_ids":np.arange(n,dtype=np.int64),
            "out_probs":np.array([.99,.3],np.float32) if self.mode=="ambiguous" else np.full(n,.99,np.float32),
            "out_boxes_xywh":np.repeat(box[None],n,axis=0).reshape(n,4),
            "out_binary_masks":np.repeat(mask[None],n,axis=0).reshape(n,8,10)}}
    def assert_idle(self):
        if self.active:raise RuntimeError("Fixture leaked session")
    def stats(self):return {"fixture":True,"active_sessions":len(self.active)}

class SamImageEvalTest(unittest.TestCase):
    def setUp(self):
        os.sched_setaffinity(0,{45,46})
        base=PROJECT_ROOT/"outputs/action4_sam_full_dev_eval_v001/r096_v005/cpu_fixtures"/uuid.uuid4().hex
        base.mkdir(parents=True)
        self.base=base
        # Job-state fixtures replace only sealed-source verification; actual
        # source/authority guard tests below call the original verifier directly.
        self.snapshot_patch=patch("eval.action4_sam_image_eval.validate_source_snapshot",return_value={"identity":"c"*64})
        self.snapshot_patch.start();self.addCleanup(self.snapshot_patch.stop)
        self.config=copy.deepcopy(read_json("configs/eval/action4_sam_full_dev_r096_v005.json"))
        self.config.update(expected_images=3,expected_saved_candidates=6,expected_prompts=6,
                           output=str((base/"run").relative_to(PROJECT_ROOT)))
        self.rows=[]
        for i in range(3):
            image=base/f"image{i}.png";Image.fromarray(np.full((8,10,3),i*10,np.uint8)).save(image)
            candidates=[]
            for j,b in enumerate(([1,1,4,5],[6,1,9,5])):
                cid=f"fixture:{i}:{j}"
                candidates.append({"candidate_id":cid,"identity_binding":{"sample_id":f"fixture-image{i}"},
                    "source":"frozen_B0_parent_step0_saved_prediction","published":True,"score":.8,
                    "eligible_for_SAM_diagnostic_prompt":True,"bbox_full_native_xyxy":b,
                    "bbox_sam_native_xyxy":b,"bbox_sam_normalized_xywh":[b[0]/10,b[1]/8,(b[2]-b[0])/10,(b[3]-b[1])/8],
                    "GT_instance_id":None})
            self.rows.append({"schema":"action4.sam.prediction_prompt_image.v1","image_id":f"fixture-image{i}",
                "image_path":str(image.relative_to(PROJECT_ROOT)),"image_sha256":file_sha(image),
                "native_shape_hw":[8,10],"pixel_space_id":"fixture-native","pixel_convention":"integer_centers",
                "source_receipt":{"parent_sha256":"b"*64,"fixture_only":True},
                "raw_saved_candidate_count":2,"SAM_prompt_count":2,"candidates":candidates,
                "GT_used_for_prompt_selection":False})
        self.write_manifest()
    def write_manifest(self):
        p=self.base/"prompts.jsonl";p.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        self.config["manifest"]=str(p.relative_to(PROJECT_ROOT));self.config["manifest_sha256"]=file_sha(p)
    def results(self):
        return [read_json(p) for p in (self.base/"run/jobs").glob("*/attempt_*/RESULT.json")]
    def test_prepare_no_native_load_and_two_distinct_image_first_batch(self):
        _,jobs,lock,first=prepare_run(self.config)
        self.assertEqual(len(jobs),6);self.assertEqual(len(first),2)
        self.assertEqual(len({j["image"]["image_id"] for j in jobs if j["task_key"] in first}),2)
        backend=NativeBackend(self.config)
        self.assertIsNone(backend.predictor);self.assertIsNone(backend.torch)
    def test_first_batch_then_resume_all_and_repeated_resume_no_calls(self):
        backend=FakeBackend()
        first=run_jobs(self.config,backend,first_batch=True)
        self.assertEqual(first["completed_jobs"],2);self.assertEqual(backend.created,2)
        full=run_jobs(self.config,backend)
        self.assertEqual(full["completed_jobs"],6);self.assertEqual(backend.created,6)
        self.assertEqual(full["available_masks"],6);self.assertFalse(backend.active)
        calls=len(backend.calls);run_jobs(self.config,backend)
        self.assertEqual(len(backend.calls),calls)
        for r in self.results():
            self.assertTrue(r["close_success"]);self.assertEqual(r["active_sessions_after_close"],0)
            self.assertIn("raw__outputs__out_binary_masks",r["mask_registration"]["storage_reference"])
            self.assertFalse(r["class_identity_confirmation"])
    def test_all_native_objects_retained_ambiguity_not_probability_pick(self):
        result=run_jobs(self.config,FakeBackend("ambiguous"),first_batch=True)
        self.assertEqual(result["available_masks"],0)
        for r in self.results():
            self.assertEqual(r["status"],"ambiguous_identity");self.assertEqual(len(r["native_candidates"]),2)
            self.assertGreater(r["native_candidates"][0]["probability"],r["native_candidates"][1]["probability"])
            self.assertIn("multiple_native_masks_compete_for_same_prompt",r["binding_reasons"])
            self.assertIsNone(r["bound_obj_id"]);self.assertFalse(r["use_as_training_negative"])
    def test_empty_public_and_empty_pixels_are_unknown_not_absent(self):
        for mode in ("empty_public","empty_pixels"):
            cfg=copy.deepcopy(self.config);cfg["output"]=str((self.base/mode).relative_to(PROJECT_ROOT))
            progress=run_jobs(cfg,FakeBackend(mode),first_batch=True)
            self.assertEqual(progress["available_masks"],0)
            self.assertEqual(progress["statuses"],{"target_visibility_unknown":2})
    def test_merged_support_not_reused_for_two_prompt_targets(self):
        progress=run_jobs(self.config,FakeBackend("merged"),first_batch=True)
        self.assertEqual(progress["available_masks"],0)
        self.assertEqual(progress["statuses"],{"ambiguous_identity":2})
    def test_none_failstop_preserves_failure_and_releases_session(self):
        backend=FakeBackend("none")
        with self.assertRaises(NativeJobFailure):run_jobs(self.config,backend)
        self.assertEqual(backend.created,1);self.assertFalse(backend.active)
        result=self.results()[0]
        self.assertEqual(result["status"],"missing_response");self.assertFalse(result["available"])
        self.assertFalse(list((self.base/"run/jobs").glob("*/COMPLETED.json")))
        with self.assertRaises(NativeJobFailure):run_jobs(self.config,FakeBackend())
        final=run_jobs(self.config,FakeBackend(),retry_failed=True)
        self.assertEqual(final["completed_jobs"],6)
        self.assertEqual(len(list((self.base/"run/jobs").glob("*/attempt_01/FAILED.json"))),1)
    def test_oom_or_schema_error_failstop_not_target_absent(self):
        for mode in ("oom","wrong_frame"):
            cfg=copy.deepcopy(self.config);cfg["output"]=str((self.base/mode).relative_to(PROJECT_ROOT))
            backend=FakeBackend(mode)
            with self.assertRaises(NativeJobFailure):run_jobs(cfg,backend)
            self.assertEqual(backend.created,1);self.assertFalse(backend.active)
            r=read_json(next((self.base/mode/"jobs").glob("*/attempt_*/RESULT.json")))
            self.assertEqual(r["status"],"inference_error");self.assertFalse(r["available"])
    def test_close_failure_cannot_publish_success(self):
        backend=FakeBackend("close_fail")
        with self.assertRaises(NativeJobFailure):run_jobs(self.config,backend)
        r=self.results()[0];self.assertFalse(r["close_success"]);self.assertFalse(r["available"])
        self.assertEqual(r["status"],"session_lost")
        self.assertFalse(list((self.base/"run/jobs").glob("*/COMPLETED.json")))
    def test_completed_raw_corruption_rejected_without_new_calls(self):
        backend=FakeBackend();run_jobs(self.config,backend,first_batch=True)
        raw=next((self.base/"run/jobs").glob("*/attempt_*/raw.npz"));raw.write_bytes(b"corrupted")
        created=backend.created
        with self.assertRaisesRegex(ValueError,"missing or changed"):run_jobs(self.config,backend)
        self.assertEqual(backend.created,created)
    def test_input_coordinate_gt_and_duplicate_identity_rejected(self):
        self.rows[0]["candidates"][0]["bbox_sam_normalized_xywh"][0]=.9;self.write_manifest()
        with self.assertRaisesRegex(ValueError,"prompt mismatch"):load_manifest(self.config)
        self.rows[0]["candidates"][0]["bbox_sam_normalized_xywh"][0]=.1
        self.rows[0]["candidates"][0]["GT_instance_id"]=1;self.write_manifest()
        with self.assertRaisesRegex(ValueError,"GT field"):load_manifest(self.config)
        self.rows[0]["candidates"][0]["GT_instance_id"]=None
        self.rows[1]["candidates"][0]["candidate_id"]=self.rows[0]["candidates"][0]["candidate_id"];self.write_manifest()
        with self.assertRaisesRegex(ValueError,"identity collision"):load_manifest(self.config)
    def test_unpublished_candidates_preserved_not_invented(self):
        self.rows[1]["candidates"][1]["eligible_for_SAM_diagnostic_prompt"]=False
        self.rows[1]["candidates"][1]["published"]=False;self.rows[1]["candidates"][1]["score"]=.2
        self.rows[1]["SAM_prompt_count"]=1;self.config["expected_prompts"]=5;self.write_manifest()
        p=run_jobs(self.config,FakeBackend())
        self.assertEqual(p["image_domain_count"],3);self.assertEqual(p["upstream_unprompted_candidates"],1)
        self.assertEqual(p["completed_jobs"],5)
    def test_small_visible_support_binds_without_full_box_iou_gate(self):
        from model.action4_sam_response_adapter import ValidatedOutput
        row=self.rows[0];target=row["candidates"][0]
        mask=np.zeros((1,8,10),bool);mask[0,2,2]=True
        native=ValidatedOutput(0,(7,),np.array([.2],np.float32),
            np.array([[.2,.25,.1,.125]],np.float32),mask)
        obj,status,relations=bind_native_output(native,row,target,self.config["binding"])
        self.assertEqual(obj,7);self.assertEqual(status,"bound_unique_prediction_support")
        d=relations[0]
        self.assertLess(d["bbox_iou_prompt"],.5)
        self.assertEqual(d["target_mask_coverage"],1)
        self.assertEqual(d["best_other_candidate_mask_coverage"],0)
        self.assertEqual(d["coverage_margin"],1)
        self.assertEqual(d["bbox_iou_role"],"diagnostic_only")
    def test_overlap_fragment_supports_two_fullboxes_remains_ambiguous(self):
        from model.action4_sam_response_adapter import ValidatedOutput
        row=copy.deepcopy(self.rows[0]);row["candidates"][1]["bbox_sam_native_xyxy"]=[2,1,5,5]
        mask=np.zeros((1,8,10),bool);mask[0,2,2]=True
        native=ValidatedOutput(0,(9,),np.array([1.],np.float32),
            np.array([[.2,.25,.1,.125]],np.float32),mask)
        for target in row["candidates"]:
            obj,status,relations=bind_native_output(native,row,target,self.config["binding"])
            self.assertIsNone(obj);self.assertEqual(status,"ambiguous_identity")
            d=relations[0]
            self.assertEqual(d["target_mask_coverage"],1)
            self.assertEqual(d["best_other_candidate_mask_coverage"],1)
            self.assertEqual(d["coverage_margin"],0)
            self.assertIn("one_mask_strongly_supports_multiple_candidates",d["binding_rejection_reasons"])

    def test_session_margin_8gib_boundary(self):
        require_free_margin(8192,8192)
        with self.assertRaises(ResourceWait):require_free_margin(8191,8192)
        with self.assertRaises(ValueError):require_free_margin(float("nan"),8192)
        # Exercise the real admission/release code with a CPU placeholder and
        # mocked measurement; never initialize torch/CUDA or query any GPU.
        config=copy.deepcopy(self.config)
        config["output"]=str((self.base/"native_wait_fixture").relative_to(PROJECT_ROOT))
        backend=NativeBackend(config);backend.predictor=object()
        raw=config["resources"]["gpu_uuid"]+", 8191, 40960, 77"
        with patch("subprocess.check_output",return_value=raw):
            with self.assertRaises(ResourceWait):backend.admit_session()
        self.assertIsNone(backend.predictor);self.assertIsNone(backend.torch)
        receipt=read_json(self.base/"native_wait_fixture/RESOURCE_WAIT.json")
        self.assertTrue(receipt["model_released"])
        self.assertFalse(receipt["new_session_started"])
        self.assertIsNone(receipt["target_outcome"])
    def test_resource_wait_preserves_nonprefix_first_batch_and_resumes(self):
        first=run_jobs(self.config,FakeBackend(),first_batch=True)
        self.assertEqual(first["completed_jobs"],2)
        class WaitingBackend(FakeBackend):
            def admit_session(self):raise ResourceWait("global free under8GiB fixture")
        waiting=WaitingBackend()
        progress=run_jobs(self.config,waiting)
        self.assertEqual(progress["status"],"RESOURCE_WAIT_RELEASED_RESTART_SAFE")
        self.assertEqual(progress["completed_jobs"],2)
        self.assertEqual(progress["pending_jobs"],4)
        self.assertEqual(waiting.created,0)
        self.assertIsNone(progress["target_outcome"])
        self.assertFalse(list((self.base/"run/jobs").glob("*/attempt_*/FAILED.json")))
        resumed=FakeBackend();full=run_jobs(self.config,resumed)
        self.assertEqual(full["completed_jobs"],6)
        self.assertEqual(resumed.created,4)

    def test_actual_snapshot_closure_and_guard_execute_from_copy(self):
        config=read_json("configs/eval/action4_sam_full_dev_r096_v005.json")
        seal=validate_source_snapshot(config,require_execution=True)
        for required in ("src/data/action4_adapter.py","src/model/training/action4_resource_policy.py",
                         "src/model/training/__init__.py","upstream/sam3/sam3/model_builder.py"):
            self.assertIn(required,seal["files_sha256"])
        from model.training import action4_resource_policy
        self.assertTrue(str(Path(action4_resource_policy.__file__).resolve()).startswith(str(PROJECT_ROOT/config["source_snapshot"]/"src")))
        self.assertEqual(validate_live_authority(config)["authority"],config["authority_ref"])
    def test_forbidden_or_stale_authority_rejects_before_native_load(self):
        config=read_json("configs/eval/action4_sam_full_dev_r096_v005.json")
        forbidden=copy.deepcopy(config);forbidden["resources"]["physical_gpu"]=1
        with self.assertRaisesRegex(ValueError,"permission revoked"):
            validate_live_authority(forbidden)
        stale=copy.deepcopy(config);stale["authority_ref"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"Latest active authority"):
            validate_live_authority(stale)
        backend=NativeBackend(config)
        with patch("eval.action4_sam_image_eval.validate_live_authority",side_effect=ValueError("revoked")):
            with self.assertRaisesRegex(ValueError,"revoked"):backend.call({"type":"start_session"})
        self.assertIsNone(backend.torch);self.assertIsNone(backend.predictor)
    def test_live_module_execution_rejected_and_old_live_entry_revoked(self):
        import importlib.util,subprocess,sys
        config=read_json("configs/eval/action4_sam_full_dev_r096_v005.json")
        spec=importlib.util.spec_from_file_location("_live_sam_review",PROJECT_ROOT/"src/eval/action4_sam_image_eval.py")
        live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)
        with self.assertRaisesRegex(ValueError,"must execute project code"):
            live.validate_source_snapshot(config,require_execution=True)
        result=subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts/run_action4_sam_full_dev.py"),
            "--config","configs/eval/action4_sam_full_dev_v001.json","--mode","run","--execute-native"],
            cwd=PROJECT_ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Live native SAM entry revoked",result.stderr)
    def test_snapshot_file_tampering_rejects(self):
        source=self.base/"tampered_source";source.mkdir()
        f=source/"unit.py";f.write_text("original")
        expected=file_sha(f);f.write_text("changed");f.chmod(0o444)
        seal={"schema":"action4.sam.immutable_source.v1","files_sha256":{"unit.py":expected},
              "config_file":"config.json","official_commit":"fixture","official_source_relative":"upstream/sam3"}
        seal["identity"]=digest(seal)
        (source/"SNAPSHOT_MANIFEST.json").write_text(json.dumps(seal))
        config=copy.deepcopy(self.config);config["source_snapshot"]=str(source.relative_to(PROJECT_ROOT))
        with self.assertRaisesRegex(ValueError,"missing/changed/escaped"):
            validate_source_snapshot(config)

    def test_resource_policy_gpu0_and_changed_lock_reject(self):
        bad=copy.deepcopy(self.config);bad["resources"]["gpu_uuid"]="GPU0"
        with self.assertRaises(ValueError):validate_config(bad)
        prepare_run(self.config)
        changed=copy.deepcopy(self.config);changed["resources"]["torch_threads"]=1
        with self.assertRaisesRegex(ValueError,"lock changed"):prepare_run(changed)

if __name__=="__main__":unittest.main()
