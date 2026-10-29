"""Analytic diagnostic bridge payloads; no inference or real calibration claim."""
import copy,json,hashlib,unittest,uuid
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data import DataAdapter
from data.action4_dataset import virtual_roi
from eval.action4_corner_scoring_inputs import build_scoring_inputs,SOURCE_SCHEMA,ARTIFACT_SCHEMA
from eval.action4_corner_calibration import calibrate
from model.action4_single_frame_pose import POLICY

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p.write_text(json.dumps(value));return {"path":str(p.relative_to(R)),"sha256":sha(p)}

class ScoringInputTests(unittest.TestCase):
    def setUp(self):
        self.out=R/"outputs/action4_corner_scoring_preparation_v001"/uuid.uuid4().hex;self.out.mkdir(parents=True)
        src=self.out/"source";src.mkdir();code=src/"fixture.py";code.write_text("# analytic fixture only\n");code.chmod(0o444)
        source={"schema":SOURCE_SCHEMA,"code_root":str(src.relative_to(R)),"files_sha256":{"fixture.py":sha(code)}}
        from data.action4_sam_native_export import canonical
        source["identity"]=canonical(source);self.source=write(self.out/"source.json",source);(R/self.source["path"]).chmod(0o444)
        self.code=code
        rgb=self.out/"rgb.png";Image.fromarray(np.zeros((64,64,3),np.uint8)).save(rgb)
        self.cad=np.array([[-.03,-.02,-.01],[.03,-.02,-.01],[.03,.02,-.01],[-.03,.02,-.01],
            [-.03,-.02,.01],[.03,-.02,.01],[.03,.02,.01],[-.03,.02,.01]])
        self.K=np.array([[100.,0,31.5],[0,100.,31.5],[0,0,1.]])
        objects=[];boxes=[];raws=[];candidates=[];observations=[]
        for i,x in enumerate((-.15,.15)):
            t=np.array([x,0.,1.]);T=np.eye(4);T[:3,3]=t
            camera=self.cad+t;h=camera@self.K.T;uv=h[:,:2]/h[:,2:];bbox=np.r_[uv.min(0),uv.max(0)].tolist()
            mask=np.zeros((64,64),np.uint8);mask[int(bbox[1]):int(bbox[3])+1,int(bbox[0]):int(bbox[2])+1]=255
            mp=self.out/f"GT{i}.png";Image.fromarray(mask).save(mp)
            objects.append({"instance_id":i+1,"bbox_full_xyxy":bbox,"corners_px":uv.tolist(),"corner_visible":[True]*4+[False]*4,
                "canonical8_geometry":{"object_to_capture_cv_m":T.tolist()},"ignore":False,"occlusion_inframe":.5,
                "mask_visible":str(mp.relative_to(R)),"mask_visible_sha256":sha(mp)})
            pid=f"b0:img:grid:{i+1}";box={"prediction_id":pid,"grid_index":i+1,"score":.9,"bbox_full_xyxy":bbox};boxes.append(box)
            raw=dict(box,corner_ids=list(range(8)),published=True,pose_valid=True,positive_depth_verified=True,
                corners_px=uv.tolist(),rotation=np.eye(3).tolist(),translation_m=t.tolist(),pose_diagnostic={"method":"raw_network","valid":True})
            raws.append(raw);candidate=copy.deepcopy(raw)
            candidate["pose_diagnostic"]={"method":"geometric_candidate_or_raw","reason":"insufficient_corner_objective_improvement","geometric_candidate_accepted":False,"deployed":False}
            candidates.append(candidate)
            bounds=virtual_roi(bbox,getattr(self,"roi_expansion",2.25))
            rid=hashlib.sha256(json.dumps({"image":"img","prediction":pid,"bounds":tuple(bounds),"size":448},sort_keys=True).encode()).hexdigest()
            identity=["img",pid,rid,"action4.roi448.v1"]
            down,_=DataAdapter.letterbox_affine((448,448),(224,224));A=down@DataAdapter.crop_affine(bounds,(448,448))
            obs={"identity":identity,"pixel_space_id":"native","corner_ids":list(range(8)),
                "uv224":DataAdapter.transform_points(uv,A).tolist(),"q_corner":[.9]*8,"p_valid":[.9]*8,"b224":[1.]*8,"multimodal":[False]*8}
            route={"kind":"full","identity":identity,"pixel_space_id":"native","full_box_native":bbox,
                "full_box_quality_passed":False,"full_box_quality_reason":"not_calibrated","quality_gate_scope":"unadmitted"}
            diag={"identity":identity,"pixel_space_id":"native","quality_ready":False,"accepted":False,"geometric_candidate_accepted":False,
                "config":POLICY,"rotation":raw["rotation"],"translation_m":raw["translation_m"],"reason":"insufficient_corner_objective_improvement"}
            from model.action4_single_frame_pose import decode_single_frame_pose
            diag=decode_single_frame_pose(canonical_m=self.cad,K_native=self.K,A224_from_native=A,
                raw_rotation=np.eye(3),raw_translation_m=t,observations=dict(obs,multimodal=np.asarray(obs["multimodal"],bool)),
                route=route,pixel_space_id="native",raw_identity=identity,roi_identity=identity,
                quality_ready=False,config=POLICY,diagnostic_candidate=True)
            candidate["pose_diagnostic"]["reason"]=diag["reason"]
            observations.append({"image_id":"img","prediction_id":pid,"roi_identity":identity,"observations":obs,
                "A224_from_native":A.tolist(),"K_native":self.K.tolist(),"canonical_m":self.cad.tolist(),
                "predicted_full_box":bbox,"raw_rotation":raw["rotation"],"raw_translation_m":raw["translation_m"],
                "raw_corners_native":raw["corners_px"],"route":route,"decoder_diagnostics":diag,"quality_ready":False,"deployed":False})
        self.annotation={"sample_id":"img","split":"eval","status":"training_accepted","image":str(rgb.relative_to(R)),
            "rgb_sha256":sha(rgb),"width":64,"height":64,"camera":{"K":self.K.tolist()},"corners_3d_mm":(self.cad*1000).tolist(),
            "instances":objects,"scene_id":"analytic"}
        self.models={}
        for role in ("A0","B0"):
            f=self.out/(role+".bin");f.write_bytes(("diagnostic "+role).encode());self.models[role]={"path":str(f.relative_to(R)),"sha256":sha(f)}
        self.cache={"schema":"action4.sam_native_v2_index.v1","split":"eval","origin":"synthetic_fixture",
            "sources":{"b0_model_sha256":self.models["B0"]["sha256"],"sam_model_sha256":"c"*64},
            "images":[{"image_id":"img","image_path":str(rgb.relative_to(R)),"source_sha256":sha(rgb),"pixel_space_id":"native",
                "predictions":[{"prediction_id":b["prediction_id"]} for b in boxes],
                "engineering_saved_predictions":[{k:v for k,v in b.items() if k!="prediction_id"} for b in boxes]}]}
        cb=write(self.out/"cache.json",self.cache)
        identity={"schema":"action4.corner_prediction_bridge.v1","models":self.models,"A0_kind":"joint_pose_sam_corner_candidate",
            "B0_bundle_sha256":self.models["B0"]["sha256"],"A0_config_id":"a"*64,"B0_config_id":"b"*64,"B0_scoring_policy":"center",
            "counterpart":{"B0":self.models["B0"],"eval_sam_cache":cb,"train_sam_cache":cb},
            "eval_SAM_cache":cb,"decoder_policy":POLICY,"engineering_fixture":True,"quality_ready":False,"deployment_admitted":False,
            "full_box_quality_receipt":None,"full_box_quality_policy":"not_calibrated"}
        self.payload={"schema":"action4.corner_prediction_bridge.v1","image_id":"img","identity":identity,"box_candidates":boxes,
            "detections":copy.deepcopy(boxes),"raw_instances":raws,"instances":copy.deepcopy(raws),"candidate_instances":candidates,
            "corner_observations":observations,"quality_ready":False,"deployment_admitted":False,"diagnostic_candidate":True,"GT_used":False}
    def bindings(self):
        prediction=write(self.out/(uuid.uuid4().hex+".json"),{"schema":ARTIFACT_SCHEMA,"source_manifest":self.source,
            "effective_identity":self.payload["identity"],"results":[self.payload]})
        path=self.out/(uuid.uuid4().hex+".jsonl");path.write_text(json.dumps(self.annotation)+"\n")
        return prediction,{"path":str(path.relative_to(R)),"sha256":sha(path)}
    def build(self,diagnostic=True):
        p,a=self.bindings()
        return build_scoring_inputs(p,a,self.source,self.out/uuid.uuid4().hex,diagnostic=diagnostic)
    def update_cache(self):
        ref=write(self.out/(uuid.uuid4().hex+".json"),self.cache)
        self.payload["identity"]["eval_SAM_cache"]=ref
        self.payload["identity"]["counterpart"]["eval_sam_cache"]=ref
    def test_verified_sidecar_gate_does_not_require_mutating_raw_B0(self):
        from eval.action4_corner_scoring_inputs import _validate_route_quality
        identity={"refined_quality_bridge":{"path":"bound","sha256":"a"*64},
            "full_box_quality_policy":{"threshold":.5,"scope":"phrase2_synthetic_development_full650"}}
        box={"prediction_id":"p1"}
        route={"quality_gate_scope":"phrase2_synthetic_development_full650",
            "full_box_quality_passed":True,"full_box_quality_reason":"exact_capture_bound_development_quality"}
        q={"p1":{"score":.8,"threshold":.5,"passed":True,
            "scope":"phrase2_synthetic_development_full650","real_world_quality_calibrated":False}}
        _validate_route_quality(route,box,identity,False,q)
        self.assertEqual(box,{"prediction_id":"p1"})
        for bad in (None,{},{"p2":q["p1"]}):
            with self.assertRaisesRegex(ValueError,"sidecar quality absent"):
                _validate_route_quality(route,box,identity,False,bad)
        for key,value in (("full_box_quality_passed",False),("quality_gate_scope","declared_calibration_receipt"),
                          ("full_box_quality_reason","guessed")):
            altered=dict(route);altered[key]=value
            with self.assertRaisesRegex(ValueError,"operating point"):
                _validate_route_quality(altered,box,identity,False,q)

    def test_native_roi15_scoring_and_wrong_contract_rejected(self):
        from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL
        fixture=ScoringInputTests();fixture.roi_expansion=1.5;fixture.setUp()
        fixture.payload["identity"]["input_contract"]={
            "schema":CORNER_NATIVE_PROTOCOL,"roi_source":"frozen_b0_prediction","roi_expansion":1.5}
        result=fixture.build()
        self.assertEqual(result["raw_report"]["pose_all8_at_002_success"],2)
        self.assertTrue(all(all(r["gt_valid"]) for r in result["records"]))
        fixture.payload["identity"]["input_contract"]={
            "schema":"action4.corner_legacy_live_B0_roi225.v1",
            "roi_source":"legacy_live_B0_prediction","roi_expansion":2.25}
        with self.assertRaisesRegex(ValueError,"ROI observation"):fixture.build()

    def test_diagnostic_full_chain_and_hidden_valid(self):
        result=self.build();self.assertFalse(result["production_eligible"]);self.assertFalse(result["source_execution_verified"])
        self.assertFalse(result["raw_report"]["complete"]);self.assertFalse(result["candidate_report"]["complete"])
        self.assertEqual(result["raw_report"]["pose_gt_count"],2);self.assertEqual(result["raw_report"]["pose_all8_at_002_success"],2)
        self.assertTrue(all(all(r["gt_valid"]) for r in result["records"]))
        self.assertTrue(all(max(r["error224"])<1e-10 for r in result["records"]))
        self.assertEqual(result["run_binding"]["config_sha256"],__import__("eval.action4_corner_calibration",fromlist=["digest"]).digest(self.payload["identity"]))
        gate=calibrate(result["domain"],result["run_binding"],result["records"],result["raw_report"],result["candidate_report"])
        self.assertFalse(gate["quality_ready"])
        with self.assertRaisesRegex(ValueError,"Engineering"):self.build(False)
    def test_missing_GT_and_unmatched_prediction_retained(self):
        self.payload["box_candidates"][1]["score"]=.2;self.cache["images"][0]["engineering_saved_predictions"][1]["score"]=.2
        for field in ("detections","raw_instances","instances","candidate_instances","corner_observations"):self.payload[field]=self.payload[field][:1]
        self.update_cache();r=self.build()
        self.assertEqual(len(r["domain"]["gt_units"]),2);self.assertEqual(len(r["records"]),2)
        self.assertIsNone(r["records"][1]["prediction_id"])
        self.assertEqual(r["raw_report"]["pose_report"]["pose_fn"],1)
    def test_IoU_unmatched_is_unknown_not_background(self):
        self.annotation["instances"][0]["bbox_full_xyxy"]=[0.,0.,1.,1.]
        r=self.build();extra=[x for x in r["records"] if x["gt_instance_id"] is None]
        self.assertEqual(len(extra),1);self.assertFalse(any(extra[0]["label_known"]))
        self.assertEqual(len(r["domain"]["prediction_units"]),2)
    def test_unknown_and_out_of_domain_targets(self):
        self.annotation["instances"][0]["corner_annotation_known"]=[False]*8
        self.annotation["instances"][1]["corners_px"][0]=[1000.,1000.]
        r=self.build();self.assertFalse(any(r["records"][0]["label_known"]))
        self.assertFalse(r["records"][1]["gt_valid"][0]);self.assertTrue(r["records"][1]["label_known"][0])
    def test_cross_identity_affine_K_and_candidate_evidence_reject(self):
        original=copy.deepcopy(self.payload)
        for field in ("identity","affine","K","candidate"):
            self.payload=copy.deepcopy(original)
            if field=="identity":self.payload["corner_observations"][0]["observations"]["identity"][0]="other"
            elif field=="affine":self.payload["corner_observations"][0]["A224_from_native"][0][2]+=.01
            elif field=="K":self.payload["corner_observations"][0]["K_native"][0][0]+=1
            else:self.payload["candidate_instances"][0]["corners_px"][0][0]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):self.build()
    def test_byte_bindings_mask_and_source_reject(self):
        p,a=self.bindings();(R/p["path"]).write_text("[]")
        with self.assertRaises(ValueError):build_scoring_inputs(p,a,self.source,self.out/"bad",diagnostic=True)
        self.annotation["instances"][0]["mask_visible_sha256"]="d"*64
        with self.assertRaises(ValueError):self.build()
        self.annotation["instances"][0]["mask_visible_sha256"]=sha(R/self.annotation["instances"][0]["mask_visible"])
        self.code.chmod(0o644);self.code.write_text("changed");self.code.chmod(0o444)
        with self.assertRaises(ValueError):self.build()
    def test_order_and_duplicate_predicates(self):
        a=self.build()
        for field in ("box_candidates","detections","raw_instances","instances","candidate_instances","corner_observations"):
            self.payload[field].reverse()
        b=self.build()
        self.assertEqual(a["raw_report"]["pose_report"],b["raw_report"]["pose_report"])
        self.assertEqual([{k:v for k,v in r.items() if k!="run_identity"} for r in a["records"]],
                         [{k:v for k,v in r.items() if k!="run_identity"} for r in b["records"]])
        self.assertNotEqual(a["run_binding"]["predictions_sha256"],b["run_binding"]["predictions_sha256"])
        self.payload["raw_instances"].append(copy.deepcopy(self.payload["raw_instances"][0]))
        with self.assertRaises(ValueError):self.build()
    def test_near_duplicate_ownership_not_corner_error_matching(self):
        self.annotation["instances"][1]["bbox_full_xyxy"]=copy.deepcopy(self.annotation["instances"][0]["bbox_full_xyxy"])
        result=self.build()
        gtrows=[r for r in result["records"] if r["gt_instance_id"] is not None]
        self.assertEqual(gtrows[0]["prediction_id"],"b0:img:grid:1")
        self.assertIsNone(gtrows[1]["prediction_id"])
        extras=[r for r in result["records"] if r["gt_instance_id"] is None]
        self.assertEqual(extras[0]["prediction_id"],"b0:img:grid:2")
        # Its raw pose matches second GT perfectly; ownership still follows B0 box matching.
        self.assertEqual(result["raw_report"]["pose_all8_at_002_success"],2)
    def test_nonfinite_GT_is_masked_before_error_arithmetic(self):
        self.annotation["instances"][0]["corners_px"][0]=[float("nan"),float("nan")]
        result=self.build();row=result["records"][0]
        self.assertFalse(row["label_known"][0]);self.assertIsNone(row["error224"][0])
        self.assertTrue(all(row["label_known"][1:]))
        self.assertIsNone(result["raw_report"]["pose_report"]);self.assertFalse(result["raw_report"]["complete"])

    def test_actual_weight_binding_tamper_refused(self):
        (R/self.models["A0"]["path"]).write_bytes(b"changed declared weights")
        with self.assertRaisesRegex(ValueError,"artifact bytes"):self.build()
    def test_production_source_closure_refuses_missing_project_import(self):
        from eval.action4_corner_scoring_inputs import REQUIRED_CODE,_source
        from data.action4_sam_native_export import canonical
        root=self.out/"closure";root.mkdir();files={}
        for relative in REQUIRED_CODE:
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            text="from model.not_in_snapshot import missing\n" if relative.endswith("action4_corner_prediction.py") else "# source-format fixture\n"
            path.write_text(text);path.chmod(0o444);files[relative]=sha(path)
        manifest={"schema":SOURCE_SCHEMA,"code_root":str(root.relative_to(R)),"files_sha256":files}
        manifest["identity"]=canonical(manifest);binding=write(self.out/"closure.json",manifest);(R/binding["path"]).chmod(0o444)
        with self.assertRaisesRegex(ValueError,"import closure"):_source(binding,False)

    def test_fabricated_candidate_and_raw_projection_refused(self):
        original=copy.deepcopy(self.payload)
        o=self.payload["corner_observations"][0];c=self.payload["candidate_instances"][0];d=o["decoder_diagnostics"]
        forged=(np.asarray(c["corners_px"])+123).tolist()
        c["corners_px"]=forged;c["pose_diagnostic"]["geometric_candidate_accepted"]=True
        d.update(geometric_candidate_accepted=True,candidate_rotation=c["rotation"],candidate_translation_m=c["translation_m"],
            candidate_corners_native=forged,used_corner_ids=[])
        with self.assertRaisesRegex(ValueError,"replay"):self.build()
        self.payload=copy.deepcopy(original)
        forged=(np.asarray(self.payload["raw_instances"][0]["corners_px"])+123).tolist()
        for field in ("raw_instances","instances","candidate_instances"):self.payload[field][0]["corners_px"]=copy.deepcopy(forged)
        self.payload["corner_observations"][0]["raw_corners_native"]=copy.deepcopy(forged)
        with self.assertRaisesRegex(ValueError,"projection"):self.build()
    def test_wrong_used_ids_and_insufficient_evidence_refused(self):
        self.payload["corner_observations"][0]["decoder_diagnostics"]["used_corner_ids"]=[]
        with self.assertRaisesRegex(ValueError,"used_corner_ids"):self.build()
    def test_status_only_fullbox_gate_is_not_accepted(self):
        from eval.action4_b0_scoring import REFINED_QUALITY_PROTOCOL
        gate={"schema":"action4.b0_refined_full_box_quality_gate.v1","status":"development_calibrated",
            "B0_bundle_sha256":self.models["B0"]["sha256"],"score_field":"refined_quality_score",
            "score_protocol":REFINED_QUALITY_PROTOCOL,"threshold":.5,"scope":"phrase2_synthetic_development_full650",
            "real_world_quality_calibrated":False}
        self.payload["identity"]["full_box_quality_receipt"]=write(self.out/"gate.json",gate)
        self.payload["identity"]["full_box_quality_policy"]=gate
        with self.assertRaisesRegex(ValueError,"calibration report"):self.build()
    def test_scripts_project_import_closure_checked(self):
        from eval.action4_corner_scoring_inputs import REQUIRED_CODE,_source
        from data.action4_sam_native_export import canonical
        root=self.out/"script_closure";root.mkdir();files={}
        for relative in REQUIRED_CODE|{"scripts/runner.py"}:
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            text="from model.unbound_live import X" if relative=="scripts/runner.py" else "# fixture"
            path.write_text(text);path.chmod(0o444);files[relative]=sha(path)
        manifest={"schema":SOURCE_SCHEMA,"code_root":str(root.relative_to(R)),"files_sha256":files}
        manifest["identity"]=canonical(manifest);binding=write(self.out/"script_source.json",manifest);(R/binding["path"]).chmod(0o444)
        with self.assertRaisesRegex(ValueError,"import closure"):_source(binding,False)
    def test_raw_validity_flags_cannot_be_downgraded(self):
        original=copy.deepcopy(self.payload)
        for flags in (("pose_valid",),("positive_depth_verified",),("pose_valid","positive_depth_verified")):
            self.payload=copy.deepcopy(original)
            for field in ("raw_instances","instances","candidate_instances"):
                for flag in flags:self.payload[field][0][flag]=False
                if "pose_valid" in flags:self.payload[field][0]["corners_px"]=None
            if "pose_valid" in flags:self.payload["corner_observations"][0]["raw_corners_native"]=None
            with self.subTest(flags=flags),self.assertRaisesRegex(ValueError,"validity flags"):
                self.build()
    def test_raw_FP32_clamp_contract_and_negative_depth(self):
        import torch
        from model.action4_single_frame_pose import decode_single_frame_pose
        original=copy.deepcopy(self.payload)
        for z in (.01001,-1.):
            self.payload=copy.deepcopy(original)
            o=self.payload["corner_observations"][0]
            R32=torch.eye(3);t32=torch.tensor([-.15,0.,z],dtype=torch.float32)
            c32=torch.tensor(self.cad,dtype=torch.float32)[None]
            A448=DataAdapter.crop_affine(virtual_roi(o["predicted_full_box"],2.25),(448,448))
            down,_=DataAdapter.letterbox_affine((448,448),(112,112));A112=down@A448
            K32=torch.tensor(A112@self.K,dtype=torch.float32)[None]
            camera=torch.bmm(c32,R32[None].transpose(1,2))+t32[None,None]
            hom=torch.bmm(camera,K32.transpose(1,2))
            uv112=hom[:,:,:2]/camera[:,:,2:].clamp_min(1e-4)
            valid=bool(torch.isfinite(camera).all() and torch.isfinite(uv112).all() and (camera[:,:,2]>0).all())
            corners=DataAdapter.transform_points(uv112[0].numpy(),np.linalg.inv(A112)).tolist() if valid else None
            for field in ("raw_instances","instances","candidate_instances"):
                self.payload[field][0].update(rotation=R32.tolist(),translation_m=t32.tolist(),
                    corners_px=copy.deepcopy(corners),pose_valid=valid,positive_depth_verified=valid)
            o.update(raw_rotation=R32.tolist(),raw_translation_m=t32.tolist(),raw_corners_native=copy.deepcopy(corners))
            obs=o["observations"]
            diag=decode_single_frame_pose(canonical_m=o["canonical_m"],K_native=o["K_native"],A224_from_native=o["A224_from_native"],
                raw_rotation=o["raw_rotation"],raw_translation_m=o["raw_translation_m"],observations=dict(obs,multimodal=np.asarray(obs["multimodal"],bool)),
                route=o["route"],pixel_space_id=obs["pixel_space_id"],raw_identity=obs["identity"],roi_identity=obs["identity"],
                quality_ready=False,config=POLICY,diagnostic_candidate=True)
            o["decoder_diagnostics"]=diag;self.payload["candidate_instances"][0]["pose_diagnostic"].update(reason=diag["reason"],geometric_candidate_accepted=False)
            result=self.build()
            self.assertFalse(diag["pose_valid"]) # Decoder has stricter1e-4 gate.
            self.assertEqual(valid,z>0)
            self.assertEqual(result["raw_report"]["pose_report"]["invalid_pose_count"],0 if z>0 else 1)
    def test_unreliable_FP32_range_refused(self):
        from eval.action4_corner_scoring_inputs import _raw_network_projection
        o=copy.deepcopy(self.payload["corner_observations"][0]);o["raw_translation_m"]=[0.,0.,1e40]
        with self.assertRaisesRegex(ValueError,"FP32"):_raw_network_projection(o)
        o=copy.deepcopy(self.payload["corner_observations"][0]);o["raw_translation_m"]=[0.,0.,.01]
        with self.assertRaisesRegex(ValueError,"ambiguous"):_raw_network_projection(o)

    def test_offline_production_flags_never_claim_execution(self):
        from eval.action4_corner_scoring_inputs import _provenance_flags
        flags=_provenance_flags(False,True)
        self.assertTrue(flags["source_binding_verified"]);self.assertTrue(flags["production_eligible"])
        self.assertFalse(flags["source_execution_verified"])
        self.assertIn("not_training_effect_or_deployment",flags["production_eligibility_scope"])
        self.assertFalse(_provenance_flags(True,False)["production_eligible"])
    def test_checkpoint_metadata_uses_strict_corner_contract(self):
        from eval.action4_corner_scoring_inputs import _check_checkpoint_metadata
        from model.action4_sam_joint import CORNER_PROTOCOL,BUNDLE_PROTOCOL,PROTOCOL_ID
        from model.action4_corner_supervision import LOSS_WEIGHTS,parameter_loss_dependencies
        from model.training.action4_supervision import A0_KEYS
        from data.action4_sam_native_export import canonical
        cfg={"role":"A0","mode":"integration","a0_variant":"a0_corner_image","sam_enabled":True,"corner_enabled":True,
            "train_backbone":False,"sam_protocol_id":PROTOCOL_ID,"corner_protocol":copy.deepcopy(CORNER_PROTOCOL),
            "corner_lr":1e-4,"corner_quality_lr":1e-4,"corner_counterpart":copy.deepcopy(self.payload["identity"]["counterpart"]),
            "parent_bundle":str((self.out/"parent.pt").relative_to(R)),"parent_bundle_sha256":"a"*64,"parent_best_step":1,
            "loss_weights":dict({k:1. for k in A0_KEYS},sam_mask_quality=.1,**LOSS_WEIGHTS),
            "independent_head_dependencies":[{"parameter_prefixes":["head.surface"],"losses":list(A0_KEYS)},
                {"parameter_prefixes":["head.pose"],"losses":[k for k in A0_KEYS if k not in ("xyz","region","visible_mask","amodal_mask")]},
                {"parameter_prefixes":["sam_adapter.encoder","sam_adapter.projection"],"losses":list(A0_KEYS)+[k for k in LOSS_WEIGHTS if k!="corner_quality_bce"]}]+parameter_loss_dependencies("corner_head")}
        state={"role":"A0","schema":"action4.stage_bundle.corner_candidate.v1","config":cfg,"config_id":canonical(cfg),
            "selected_step":2,"architecture":"joint_pose_sam_corner_candidate","corner_counterpart":cfg["corner_counterpart"],
            "corner_protocol":copy.deepcopy(CORNER_PROTOCOL),"sam_protocol":copy.deepcopy(BUNDLE_PROTOCOL),
            "quality_ready":False,"deployment_admitted":False,"pose_method":"network_raw_corner_research_only",
            "corner_parent":{"path":cfg["parent_bundle"],"sha256":cfg["parent_bundle_sha256"],"selected_step":1},
            "parent_bundle":cfg["parent_bundle"],"parent_selected_step":1,
            "model":{"fixture":1},"backbone":{"fixture":1},"sam_adapter":{"fixture":1},"corner_head":{"fixture":1}}
        identity=dict(self.payload["identity"],A0_config_id=state["config_id"],A0_step=2,corner_protocol=copy.deepcopy(CORNER_PROTOCOL))
        _check_checkpoint_metadata(state,"A0",identity,cfg["corner_counterpart"])
        for key,value in (("quality_ready",True),("deployment_admitted",True),("parent_selected_step",99),("sam_protocol",{})):
            bad=copy.deepcopy(state);bad[key]=value
            with self.assertRaises(ValueError):_check_checkpoint_metadata(bad,"A0",identity,cfg["corner_counterpart"])
        bad=copy.deepcopy(state);bad["config"]["sam_enabled"]=False;bad["config_id"]=canonical(bad["config"])
        altered=dict(identity,A0_config_id=bad["config_id"])
        with self.assertRaises(ValueError):_check_checkpoint_metadata(bad,"A0",altered,cfg["corner_counterpart"])

    def test_envelope_required_and_source_binding_matches(self):
        p,a=self.bindings()
        bare=write(self.out/"bare.json",[self.payload])
        with self.assertRaisesRegex(ValueError,"envelope"):build_scoring_inputs(bare,a,self.source,self.out/"bad",diagnostic=False)
        artifact=json.loads((R/p["path"]).read_text());artifact["source_manifest"]["sha256"]="a"*64
        forged=write(self.out/"wrong_source.json",artifact)
        with self.assertRaises(ValueError):build_scoring_inputs(forged,a,self.source,self.out/"bad2",diagnostic=True)

if __name__=="__main__":unittest.main()
