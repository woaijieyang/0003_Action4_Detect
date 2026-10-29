"""Actual backbone/owned-head CPU bridge with explicit synthetic SAM-mask cache.

Uses one existing eval RGB image as engineering input, never its GT instances.
No calibration or mask-quality claim; no model parameter update.
"""
import copy,json,hashlib,gc
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from PIL import Image
from data.action4_sam_roi_inputs import ImageFrame,NativeMaskIdentity,prediction_metadata,array_digest,SOURCE_KEYS
from eval.action4_sam_pipeline_inputs import SamInferenceContext
from eval.action4_staged_pipeline import StagedPipeline
from eval.action4_corner_prediction import EngineeringFullBoxGate
from eval.action4_b0_scoring import predict,REFINED_QUALITY_PROTOCOL
from model.action4_release_bundle import load_bundle
from model.training.action4_stage2_core import construct,bundle_snapshot
from tests import test_action4_corner_model_wiring as fixtures

R=Path("/data/zy/vln/a-test");OUT=R/"outputs/action4_corner_prediction_preparation_v001"
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write_json(p,v):
    p.write_text(json.dumps(v));return {"path":str(p.relative_to(R)),"sha256":sha(p)}


class CornerPredictionBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(732);OUT.mkdir(exist_ok=True)
        # Select only camera/image metadata; no GT instance geometry is consumed.
        with (R/"data/phrase2/eval/annotations.jsonl").open() as f:record=json.loads(f.readline())
        cls.image_id=record["sample_id"];cls.image_path=R/record["image"]
        cls.rgb=np.array(Image.open(cls.image_path).convert("RGB"))
        cls.K=np.asarray(record["camera"]["K"]);cls.cad=np.asarray(record["corners_3d_mm"])/1000
        del record
        cls.b0_path=R/"models/candidates/action4_improvement/stage1_B0_completed_20261024/B0.pt"
        b0,_,state=load_bundle(cls.b0_path,"B0")
        policy=state.get("inference_scoring_policy",state["config"].get("evaluation_scoring_policy","learned"))
        cls.saved=predict(b0,cls.rgb,state["config"],"cpu",policy)
        enriched=predict(b0,cls.rgb,state["config"],"cpu",policy,include_refined_quality=True)
        cls.original_fields_equal=([{k:v for k,v in row.items() if k not in ("refined_quality_score","refined_quality_protocol")} for row in enriched]==cls.saved)
        if len([r for r in cls.saved if r["score"]>=.3])<2:raise AssertionError("Engineering input requires actual2 B0 detections")
        del b0,state;gc.collect()
        frame=ImageFrame(cls.image_id,cls.rgb,"engineering_native_original",sha(cls.image_path),"native_original",array_digest(cls.rgb))
        sources={k:"a"*64 for k in SOURCE_KEYS};sources["b0_model_sha256"]=sha(cls.b0_path)
        entries=[]
        for i,row in enumerate(cls.saved):
            pid=f"b0:{cls.image_id}:grid:{row['grid_index']}"
            mask=None
            if i==0:
                mask=np.zeros(cls.rgb.shape[:2],bool)
                x1,y1,x2,y2=np.clip(np.array(row["bbox_full_xyxy"]).astype(int),[0,0,0,0],[mask.shape[1],mask.shape[0],mask.shape[1],mask.shape[0]])
                mask[y1:y2,x1:x2]=True
            metadata=prediction_metadata(frame,NativeMaskIdentity(cls.image_id,pid,frame.pixel_space_id),mask,sources,
                status="available" if mask is not None else "missing",assigned_instance_id=pid if mask is not None else None,
                reason=None if mask is not None else "engineering_missing_mask_fixture",origin="synthetic_fixture",
                sam_binding={"session_id":"engineering","obj_id":i,"generation":0,"chosen_mask_id":"engineering-mask","match_basis":"fixture-only"} if mask is not None else None)
            dest=OUT/f"fixture_mask_{i}.npz";values={"metadata":np.array(json.dumps(metadata))}
            if mask is not None:values["mask"]=mask
            np.savez_compressed(dest,**values);entries.append({"prediction_id":pid,"path":str(dest.relative_to(R)),"sha256":sha(dest)})
        index={"schema":"action4.sam_native_v2_index.v1","split":"eval","origin":"synthetic_fixture","prompt_origin":"frozen_prediction_only",
            "sources":sources,"images":[{"image_id":cls.image_id,"image_path":str(cls.image_path.relative_to(R)),"source_sha256":frame.source_sha256,
                "pixel_space_id":frame.pixel_space_id,"variant_id":frame.variant_id,"variant_sha256":frame.variant_sha256,
                "predictions":entries,"engineering_saved_predictions":cls.saved}]}
        cls.spec={"manifest":write_json(OUT/"FIXTURE_INDEX.json",index),"expected_sources":sources}
        fixtures.CornerModelWiringTests.setUpClass();f=fixtures.CornerModelWiringTests
        cfg=copy.deepcopy(f.cfg);cfg["corner_counterpart"]["eval_sam_cache"]=cls.spec["manifest"]
        cls.sam_path=f.sam_path
        del f.model;gc.collect()
        model,parent=construct(cfg,"cpu")
        cls.candidate_path=OUT/"engineering_corner_candidate.pt"
        torch.save(bundle_snapshot(model,parent,cfg,0),cls.candidate_path)
        del model,parent;gc.collect()
        cls.pipeline=StagedPipeline.from_corner_bundles(cls.candidate_path,cls.b0_path,cls.spec,"cpu",engineering_fixture=True)

    def test_actual_single_forward_owned_head_and_raw_formal_outputs(self):
        p=self.pipeline;context=p.corner_context(self.image_id,self.rgb)
        calls={"B0":0,"A0":0};packets=[];features=[]
        def inc(name):
            def hook(*args):calls[name]+=1
            return hook
        handles=[p.b0.detector.backbone.register_forward_hook(inc("B0")),
            p.base.backbone.register_forward_hook(inc("A0")),
            p.a0.register_forward_hook(lambda mod,args,out:packets.append(out)),
            p.base.backbone.register_forward_hook(lambda mod,args,out:features.append(out.clone()))]
        gates={pid:EngineeringFullBoxGate(self.image_id,pid,tuple(row["bbox_full_xyxy"]),True,"explicit_engineering_gate_not_calibrated")
               for row in self.saved for pid in [f"b0:{self.image_id}:grid:{row['grid_index']}"]}
        from model.action4_single_frame_pose import decode_single_frame_pose
        try:
            from eval.action4_corner_prediction import _checked_context as real_check
            def checked_then_caller_changes(*args,**kwargs):
                owned=real_check(*args,**kwargs)
                context.predictions.clear()
                context.expected_sources.clear()
                return owned
            with patch("eval.action4_corner_prediction.decode_single_frame_pose",wraps=decode_single_frame_pose) as decoder,patch("eval.action4_corner_prediction._checked_context",side_effect=checked_then_caller_changes):
                result=p.predict_corner(self.rgb,self.K,self.cad,sam_context=context,diagnostic_candidate=True,engineering_route_gates=gates)
                self.assertEqual(decoder.call_count,len(result["detections"]))
                self.assertTrue(all(call.kwargs["quality_ready"] is False for call in decoder.call_args_list))
        finally:
            for h in handles:h.remove()
        self.assertEqual(calls,{"B0":1,"A0":1})
        self.assertFalse(result["quality_ready"]);self.assertFalse(result["deployment_admitted"])
        self.assertEqual(result["instances"],result["raw_instances"])
        self.assertEqual(len(result["candidate_instances"]),len(result["detections"]))
        self.assertTrue(self.original_fields_equal)
        for i,row in enumerate(result["corner_observations"]):
            obs=row["observations"]
            np.testing.assert_array_equal(obs["uv224"],packets[0].corners.uv224[i].cpu().numpy())
            np.testing.assert_array_equal(obs["p_valid"],packets[0].corners.p_valid[i].cpu().numpy())
            self.assertEqual(obs["corner_ids"],list(range(8)))
            self.assertEqual(row["prediction_id"],row["roi_identity"][1])
            self.assertFalse(row["deployed"]);self.assertFalse(row["decoder_diagnostics"]["accepted"])
            self.assertNotIn("logits",obs)
            self.assertEqual(row["route"]["quality_gate_scope"],"engineering_fixture_only")
        missing=next(i for i,row in enumerate(result["raw_instances"]) if not row["sam_input"]["available"])
        torch.testing.assert_close(packets[0].sam_aux.c5[missing],features[0][missing],atol=0,rtol=0)
        json.dumps(result,allow_nan=False)
        (OUT/"ENGINEERING_PREDICTION.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        (OUT/"ENGINEERING_FORWARD_COUNTS.json").write_text(json.dumps({
            "scope":"actual_B0_and_A0_backbone_forward_with_synthetic_SAM_masks_and_untrained_corner_fixture",
            "B0_backbone_calls":calls["B0"],"A0_backbone_calls":calls["A0"],
            "instances":len(result["raw_instances"]),"geometric_candidate_accepted":sum(
                r["decoder_diagnostics"]["geometric_candidate_accepted"] for r in result["corner_observations"]),
            "formal_output_is_raw":result["instances"]==result["raw_instances"],
            "quality_ready":False,"deployment_admitted":False,"GT_used_for_selection":False,
            "refined_quality_optin_preserves_original_B0_fields":self.original_fields_equal},indent=2)+"\n")

    def test_default_raw_path_and_legacy_SAM_unchanged(self):
        p=self.pipeline;context=p.corner_context(self.image_id,self.rgb)
        with self.assertRaisesRegex(ValueError,"deployment predict refused"):p.predict(self.rgb,self.K,self.cad)
        with self.assertRaisesRegex(ValueError,"Unadmitted corner"):
            StagedPipeline.from_bundles(self.candidate_path,self.b0_path,"cpu")
        result=p.predict_corner(self.rgb,self.K,self.cad,sam_context=context)
        self.assertEqual(result["instances"],result["raw_instances"])
        self.assertTrue(all(not r["route"]["full_box_quality_passed"] for r in result["corner_observations"]))
        self.assertTrue(all(r["route"]["full_box_quality_reason"]=="full_box_quality_not_calibrated" for r in result["corner_observations"]))
        old=StagedPipeline.from_bundles(self.sam_path,self.b0_path,"cpu")
        baseline=old.predict(self.rgb,self.K,self.cad,sam_context=context)
        for a,b in zip(result["raw_instances"],baseline["instances"]):
            self.assertEqual(a["prediction_id"],b["prediction_id"])
            self.assertEqual(a["pose_valid"],b["pose_valid"])
            np.testing.assert_array_equal(a["corners_px"],b["corners_px"])

    def test_counterpart_context_duplicate_and_swapped_candidates_rejected(self):
        p=self.pipeline;context=p.corner_context(self.image_id,self.rgb)
        bad=copy.deepcopy(self.spec);bad["manifest"]["sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"SAM counterpart"):
            StagedPipeline.from_corner_bundles(self.candidate_path,self.b0_path,bad,"cpu",engineering_fixture=True)
        with self.assertRaisesRegex(ValueError,"B0 counterpart"):
            StagedPipeline.from_corner_bundles(self.candidate_path,R/"models/default/B0.pt",self.spec,"cpu",engineering_fixture=True)
        badsources=dict(context.expected_sources,b0_model_sha256="b"*64)
        with patch.object(p,"_detect_corner",side_effect=AssertionError("context mismatch must fail before B0")):
            with self.assertRaises(ValueError):p.predict_corner(self.rgb,self.K,self.cad,sam_context=replace(context,expected_sources=badsources))
            swapped=dict(context.predictions);keys=list(swapped);swapped[keys[0]],swapped[keys[1]]=swapped[keys[1]],swapped[keys[0]]
            with self.assertRaises(ValueError):p.predict_corner(self.rgb,self.K,self.cad,sam_context=replace(context,predictions=swapped))
        wrongs=[self.saved+[self.saved[0]],copy.deepcopy(self.saved),copy.deepcopy(self.saved)]
        wrongs[1][0]["prediction_id"]=f"b0:{self.image_id}:grid:{self.saved[1]['grid_index']}"
        wrongs[2][0]["bbox_full_xyxy"][0]+=.01
        for wrong in wrongs:
            with patch.object(p,"_detect_corner",return_value=wrong):
                with self.assertRaises(ValueError):p.predict_corner(self.rgb,self.K,self.cad,sam_context=context)

    def test_refined_quality_is_diagnostic_and_gate_receipt_bound(self):
        ref=write_json(OUT/"wrong_gate.json",{"schema":"action4.b0_refined_full_box_quality_gate.v1",
            "status":"development_calibrated","B0_bundle_sha256":sha(self.b0_path),"score_field":"old_quality_score",
            "score_protocol":REFINED_QUALITY_PROTOCOL,"threshold":.5})
        with self.assertRaisesRegex(ValueError,"score|protocol|field"):
            StagedPipeline.from_corner_bundles(self.candidate_path,self.b0_path,self.spec,"cpu",engineering_fixture=True,full_box_quality_receipt=ref)
        # No numeric threshold is invented or accepted without explicit bound receipt.
        self.assertIsNone(self.pipeline._corner_box_gate)


class ContextOwnershipTests(unittest.TestCase):
    """Small fabricated cache-format fixture; production flag false, no SAM/model calls."""
    def setUp(self):
        import uuid
        from types import SimpleNamespace
        from data.action4_sam_cache_provider import SamCacheProvider
        self.folder=OUT/("context_ownership_"+uuid.uuid4().hex);self.folder.mkdir(parents=True)
        self.rgb=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
        image=self.folder/"rgb.png";Image.fromarray(self.rgb).save(image)
        self.frame=ImageFrame("ownership-fixture",self.rgb,"native",sha(image),"native_original",array_digest(self.rgb))
        self.nid=NativeMaskIdentity("ownership-fixture","b0:ownership-fixture:grid:1","native")
        self.sources={k:"a"*64 for k in SOURCE_KEYS}
        self.mask=np.zeros((32,32),bool);self.mask[8:24,10:22]=True
        self.meta=prediction_metadata(self.frame,self.nid,self.mask,self.sources,
            assigned_instance_id=self.nid.instance_id,origin="predicted_sam",
            sam_binding={"session_id":"fabricated-native-format-fixture","obj_id":1,"generation":0,
                "chosen_mask_id":"fixture-not-SAM-inference","match_basis":"fixture"})
        self.cache=self.folder/"cache.npz";np.savez(self.cache,metadata=np.array(json.dumps(self.meta)),mask=self.mask)
        self.cache_sha=sha(self.cache)
        index={"schema":"action4.sam_native_v2_index.v1","split":"eval","origin":"predicted_sam",
            "prompt_origin":"frozen_prediction_only","sources":self.sources,"images":[{
                "image_id":self.frame.image_id,"image_path":str(image.relative_to(R)),
                "source_sha256":self.frame.source_sha256,"pixel_space_id":"native","variant_id":"native_original",
                "variant_sha256":self.frame.variant_sha256,"predictions":[{"prediction_id":self.nid.instance_id,
                    "path":str(self.cache.relative_to(R)),"sha256":self.cache_sha}]}]}
        spec={"manifest":write_json(self.folder/"INDEX.json",index),"expected_sources":self.sources}
        provider=SamCacheProvider(spec,split="eval",allow_synthetic=False)
        self.pipeline=SimpleNamespace(corner_cache=provider,_corner_engineering_fixture=False)

    def test_returned_map_metadata_and_frame_are_independent(self):
        from eval.action4_corner_prediction import cache_context,_checked_context
        context=cache_context(self.pipeline,self.frame.image_id,self.rgb)
        internal=self.pipeline.corner_cache.frames[self.frame.image_id]
        key=next(iter(context.predictions));public=context.predictions[key]
        self.assertIsNot(context.predictions,internal[1]);self.assertIsNot(public,internal[1][key])
        self.assertIsNot(context.frame,internal[0])
        metadata=public.metadata;metadata["sam_binding"]["chosen_mask_id"]="tampered"
        context.predictions.clear();context.expected_sources.clear()
        again=cache_context(self.pipeline,self.frame.image_id,self.rgb)
        self.assertEqual(again.predictions[key].metadata,self.meta)
        np.testing.assert_array_equal(again.predictions[key].mask,self.mask)
        with self.assertRaises(ValueError):again.frame.pixels[0,0,0]=0
        with self.assertRaises(ValueError):again.predictions[key].mask[0,0]=True
        owned=_checked_context(self.pipeline,self.rgb,again)
        again.predictions.clear()
        self.assertEqual(set(owned.predictions),{key})
        np.testing.assert_array_equal(owned.predictions[key].mask,self.mask)

    def test_recomputed_fake_or_GT_payload_cannot_replace_owned_mask(self):
        from eval.action4_corner_prediction import cache_context,_checked_context
        from data.action4_sam_roi_inputs import NativeSamPrediction,cache_key
        ones=np.ones_like(self.mask)
        for origin in ("predicted_sam","ground_truth"):
            context=cache_context(self.pipeline,self.frame.image_id,self.rgb)
            key=next(iter(context.predictions));fake=copy.deepcopy(self.meta)
            fake.update(mask_sha256=array_digest(ones),origin=origin);fake["cache_key"]=cache_key(fake)
            context.predictions[key]=NativeSamPrediction(fake,ones)
            with self.subTest(origin=origin),self.assertRaises(ValueError):
                _checked_context(self.pipeline,self.rgb,context)
        self.assertEqual(sha(self.cache),self.cache_sha)
        fresh=cache_context(self.pipeline,self.frame.image_id,self.rgb)
        np.testing.assert_array_equal(next(iter(fresh.predictions.values())).mask,self.mask)

    def test_actual_payload_compared_even_with_unchanged_verified_metadata(self):
        from eval.action4_corner_prediction import cache_context,_checked_context
        context=cache_context(self.pipeline,self.frame.image_id,self.rgb)
        key=next(iter(context.predictions));public=context.predictions[key]
        fake=np.frombuffer(np.ones_like(self.mask).tobytes(),dtype=np.bool_).reshape(self.mask.shape)
        # Deliberately bypass the frozen dataclass only in this adversarial fixture.
        # Metadata/cache key remain unchanged, so metadata-only comparison cannot help.
        object.__setattr__(public,"mask",fake)
        with self.assertRaisesRegex(ValueError,"payload"):
            _checked_context(self.pipeline,self.rgb,context)
        self.assertEqual(sha(self.cache),self.cache_sha)
        np.testing.assert_array_equal(self.pipeline.corner_cache.frames[self.frame.image_id][1][key].mask,self.mask)


class GateConsumerTests(unittest.TestCase):
    """Validate the published dev receipt and corruption fixtures; no model loading."""
    def setUp(self):
        import uuid
        self.folder=OUT/("gate_consumer_"+uuid.uuid4().hex);self.folder.mkdir(parents=True)
        self.gate_path=R/"outputs/action4_refined_box_quality_calibration_v001/FULL_BOX_QUALITY_GATE.json"
        self.gate=json.loads(self.gate_path.read_text())
        self.report=json.loads((R/self.gate["calibration_report"]["path"]).read_text())
        self.binding={"path":str(self.gate_path.relative_to(R)),"sha256":"e000e7db8188ad788bd6a3f39dc49f0eb9cc60ddc5e346bc729989a5a0cfba73"}
        self.model_sha=self.gate["B0_bundle_sha256"]

    def validate(self,binding):
        from eval.action4_corner_prediction import validate_full_box_quality_gate
        return validate_full_box_quality_gate(binding,b0_sha256=self.model_sha)

    def rebound(self,gate,report=None,name="bad"):
        if report is not None:gate["calibration_report"]=write_json(self.folder/(name+"_report.json"),report)
        return write_json(self.folder/(name+"_gate.json"),gate)

    def test_actual_published_gate_and_bound_report_pass_without_model(self):
        before=sha(self.gate_path)
        actual=self.validate(self.binding)
        self.assertEqual(actual,self.gate);self.assertEqual(actual["threshold"],.5)
        self.assertEqual(self.report["selected"]["selected"],1300)
        self.assertFalse(actual["real_world_quality_calibrated"])
        self.assertEqual(before,sha(self.gate_path))
        from eval.action4_corner_prediction import validate_full_box_quality_gate
        complete=json.loads((R/self.report["capture_complete"]["path"]).read_text())
        identity=complete["identity"]
        validate_full_box_quality_gate(self.binding,b0_sha256=self.model_sha,
            b0_config_id=identity["config_id"],expected_reference_predictions=identity["reference_predictions"])
        with self.assertRaises(ValueError):
            validate_full_box_quality_gate(self.binding,b0_sha256=self.model_sha,b0_config_id="wrong-config")
        wrong_ref=dict(identity["reference_predictions"],sha256="b"*64)
        with self.assertRaisesRegex(ValueError,"actual eval SAM"):
            validate_full_box_quality_gate(self.binding,b0_sha256=self.model_sha,expected_reference_predictions=wrong_ref)
        actual["calibration_report"]["sha256"]="0"*64
        self.assertEqual(self.validate(self.binding),self.gate)

    def test_wrong_report_threshold_model_and_selected_counts_rejected(self):
        for kind in ("report_model","threshold","missing_report","counts","capture_model","policy","report_sha"):
            gate=copy.deepcopy(self.gate);report=copy.deepcopy(self.report)
            if kind=="report_model":report["B0_bundle_sha256"]="b"*64
            elif kind=="threshold":gate["threshold"]=.6
            elif kind=="missing_report":
                gate.pop("calibration_report")
                with self.assertRaises(ValueError):self.validate(self.rebound(gate,name=kind))
                continue
            elif kind=="counts":report["threshold_results"][0]["selected"]=1299;report["selected"]["selected"]=1299
            elif kind=="capture_model":
                complete=json.loads((R/report["capture_complete"]["path"]).read_text())
                complete["identity"]["B0_bundle"]["sha256"]="b"*64
                report["capture_complete"]=write_json(self.folder/"bad_complete.json",complete)
            elif kind=="policy":
                policy=json.loads((R/report["preregistration"]["path"]).read_text())
                policy["minimum_selected_precision"]=.90
                report["preregistration"]=write_json(self.folder/"bad_policy.json",policy)
            else:
                gate["calibration_report"]["sha256"]="b"*64
                with self.assertRaises(ValueError):self.validate(self.rebound(gate,name=kind))
                continue
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.validate(self.rebound(gate,report,kind))


if __name__=="__main__":unittest.main(verbosity=2)
