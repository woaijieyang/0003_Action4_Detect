"""GT scoring after frozen corner predictions. Never calls inference or selects poses by GT."""
import ast,copy,hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from data.action4_dataset import is_ignored,virtual_roi
from data.action4_sam_native_export import _v2_checked,_v2_read,_v2_sha,canonical,verify_completed_export_receipt
from data.action4_sam_roi_inputs import sample_native_nearest
from eval.action4_pose_protocol import score_pose,match_geometry,gt_groups,SCHEMA as POSE_SCHEMA
from eval.action4_corner_calibration import digest
from model.action4_single_frame_pose import POLICY as DECODER_POLICY,decode_single_frame_pose

SOURCE_SCHEMA="action4.corner_inference_source.v1"
ARTIFACT_SCHEMA="action4.corner_prediction_artifact.v1"
REQUIRED_CODE={"src/eval/action4_corner_prediction.py","src/eval/action4_staged_pipeline.py",
    "src/eval/action4_b0_scoring.py","src/model/action4_single_frame_pose.py",
    "src/model/action4_release_bundle.py","src/model/action4_joint_surface_rgb.py",
    "src/model/action4_corner_supervision.py","src/model/action4_corner_observation_head.py",
    "src/data/action4_sam_cache_provider.py","src/data/action4_sam_roi_inputs.py",
    "src/data/action4_sam_native_export.py","src/data/action4_dataset.py","src/data/action4_adapter.py"}

def _same_binding(a,b):
    return set(a)==set(b)=={"path","sha256"} and a["sha256"]==b["sha256"] and contained_path(ROOT,a["path"],exists=True)==contained_path(ROOT,b["path"],exists=True)

def _file(binding):
    if not isinstance(binding,dict) or set(binding)!={"path","sha256"} or _v2_sha(binding["path"])!=binding["sha256"]:
        raise ValueError("Bound artifact bytes changed")
    return contained_path(ROOT,binding["path"],exists=True)

def _source(binding,diagnostic):
    path=_file(binding);source=_v2_checked(binding)
    if source.get("schema")!=SOURCE_SCHEMA or source.get("identity")!=canonical({k:v for k,v in source.items() if k!="identity"}):
        raise ValueError("Immutable source manifest identity mismatch")
    root=contained_path(ROOT,source["code_root"],exists=True)
    files=source["files_sha256"]
    if not isinstance(files,dict) or not files:raise ValueError("Explicit inference source files required")
    if not diagnostic and not REQUIRED_CODE<=set(files):raise ValueError("Incomplete project inference import closure")
    if path.stat().st_mode & 0o222:raise ValueError("Source manifest must be readonly")
    for relative,sha in files.items():
        p=contained_path(ROOT,root/relative,exists=True)
        if Path(relative).is_absolute() or not p.is_relative_to(root) or p.is_symlink() or p.stat().st_mode & 0o222 or _v2_sha(p)!=sha:
            raise ValueError("Source code changed/writable/escaped")
    if not diagnostic:
        for folder in ("src","scripts"):
            current=root/folder
            if current.exists():
                for directory,dirs,names in os.walk(current,followlinks=False):
                    if any((Path(directory)/n).is_symlink() for n in dirs+names):raise ValueError("Source symlink forbidden")
                    for name in names:
                        path=Path(directory)/name
                        if path.suffix in (".py",".pyc",".so",".pyd") and str(path.relative_to(root)) not in files:raise ValueError("Unlisted project Python import source")
        for relative in files:
            if not relative.startswith(("src/","scripts/")) or not relative.endswith(".py"):continue
            package=list(Path(relative).parts[1:-1]) if relative.startswith("src/") else list(Path(relative).parts[:-1])
            text=_v2_read({"path":str(root/relative),"sha256":files[relative]},lambda f:f.read().decode("utf-8"))
            tree=ast.parse(text)
            modules=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
                elif isinstance(node,ast.ImportFrom):
                    prefix=package[:len(package)-node.level+1] if node.level else []
                    name=".".join(prefix+([] if node.module is None else node.module.split(".")))
                    modules.append(name)
            for module in modules:
                if module.split(".")[0] not in ("data","model","eval","scripts"):continue
                prefix="" if module.split(".")[0]=="scripts" else "src/"
                options=(prefix+module.replace(".","/")+".py",prefix+module.replace(".","/")+"/__init__.py")
                if not any(name in files for name in options):raise ValueError("Missing project import closure: "+module)
    return source

def _scoring_roi_expansion(identity):
    """Consume the registered inference crop, never infer it from GT or observed points."""
    from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL
    contract=identity.get("input_contract")
    if contract is None:
        # Historical artifacts predate the explicit contract and used only2.25.
        return 2.25
    if (contract.get("schema")==CORNER_NATIVE_PROTOCOL and
        contract.get("roi_source")=="frozen_b0_prediction" and
        contract.get("roi_expansion") in (1.5,1.15)):
        # _check_checkpoint_metadata authenticates this against the actual bundle;
        # ROI1.15 additionally requires its registered roi_finetune config there.
        return float(contract["roi_expansion"])
    if contract=={"schema":"action4.corner_legacy_live_B0_roi225.v1",
                 "roi_expansion":2.25,"roi_source":"legacy_live_B0_prediction"}:
        return 2.25
    raise ValueError("Unregistered scoring ROI input contract")

def _check_checkpoint_metadata(state,role,identity,counterpart):
    if (state.get("role")!=role or state.get("config_id")!=identity[role+"_config_id"]
            or state.get("config_id")!=canonical(state["config"]) or state.get("selected_step")!=identity[role+"_step"]):
        raise ValueError("Actual checkpoint role/config/step differs from bridge identity")
    if role=="A0":
        from model.action4_sam_joint import validate_corner_config,sam_variant,corner_variant,CORNER_PROTOCOL,BUNDLE_PROTOCOL
        cfg=state["config"];validate_corner_config(cfg)
        from eval.action4_corner_prediction import _corner_input_contract
        expected_input=_corner_input_contract(cfg)
        declared_input=identity.get("input_contract")
        if declared_input is None and expected_input["roi_source"]=="legacy_live_B0_prediction":
            declared_input=expected_input
        if identity.get("refined_quality_bridge")!=cfg.get("corner_refined_quality_bridge"):
            raise ValueError("Scoring quality bridge differs from actual checkpoint")
        if declared_input!=expected_input:
            raise ValueError("Scoring ROI contract differs from actual checkpoint")

        parent={"path":cfg["parent_bundle"],"sha256":cfg["parent_bundle_sha256"],"selected_step":cfg["parent_best_step"]}
        if (not sam_variant(cfg) or not corner_variant(cfg)
                or state.get("schema")!="action4.stage_bundle.corner_candidate.v1"
                or state.get("architecture")!="joint_pose_sam_corner_candidate"
                or identity.get("A0_kind")!=state["architecture"]
                or state.get("corner_counterpart")!=counterpart or state.get("corner_counterpart")!=cfg["corner_counterpart"]
                or state.get("corner_protocol")!=CORNER_PROTOCOL or state.get("corner_protocol")!=identity["corner_protocol"]
                or state.get("sam_protocol")!=BUNDLE_PROTOCOL
                or state.get("quality_ready") is not False or state.get("deployment_admitted") is not False
                or state.get("pose_method")!="network_raw_corner_research_only"
                or state.get("corner_parent")!=parent or state.get("parent_bundle")!=parent["path"]
                or state.get("parent_selected_step")!=parent["selected_step"]
                or any(not isinstance(state.get(key),dict) or not state[key] for key in ("model","backbone","sam_adapter","corner_head"))):
            raise ValueError("Actual A0 checkpoint header differs from strict corner candidate contract")
    elif (state.get("schema")!="action4.stage_bundle.v1" or state.get("architecture")!="online_b0"
            or not isinstance(state.get("model"),dict) or not state["model"]
            or state.get("inference_scoring_policy",state["config"].get("evaluation_scoring_policy","learned"))!=identity["B0_scoring_policy"]):
        raise ValueError("Actual B0 architecture/scoring policy differs")

def _deps(identity,diagnostic):
    if identity.get("schema")!="action4.corner_prediction_bridge.v1" or identity.get("quality_ready") is not False or identity.get("deployment_admitted") is not False:
        raise ValueError("Unadmitted raw corner bridge identity required")
    if type(identity.get("engineering_fixture")) is not bool or identity["engineering_fixture"] and not diagnostic:
        raise ValueError("Engineering fixture prediction refused for production scoring")
    if identity.get("decoder_policy")!=DECODER_POLICY:raise ValueError("Frozen decoder policy differs")
    models=identity["models"]
    if set(models)!={"A0","B0"}:raise ValueError("Actual A0/B0 model bindings required")
    for b in models.values():_file(b)
    counterpart=identity["counterpart"]
    if not _same_binding(counterpart["B0"],models["B0"]) or identity["B0_bundle_sha256"]!=models["B0"]["sha256"]:
        raise ValueError("B0 model/counterpart mismatch")
    cache_binding=identity["eval_SAM_cache"]
    if not _same_binding(counterpart["eval_sam_cache"],cache_binding):raise ValueError("SAM counterpart mismatch")
    _file(counterpart["train_sam_cache"])
    cache=_v2_checked(cache_binding)
    if cache["split"]!="eval" or cache["sources"]["b0_model_sha256"]!=models["B0"]["sha256"]:
        raise ValueError("Eval cache B0/split mismatch")
    saved=None;reference_predictions=None
    if not diagnostic:
        receipt=verify_completed_export_receipt(cache)
        reference_predictions=receipt["source_contract"]["predictions"]
        saved=_v2_checked(reference_predictions)
        lock=_v2_checked(receipt["native_lock"]);weight=lock["config"]["native"]["weight"]
        if _v2_sha(weight)!=cache["sources"]["sam_model_sha256"]:raise ValueError("Actual SAM weight bytes differ")
    elif cache.get("origin")!="synthetic_fixture" and cache["sources"].get("upstream_mode")=="b0_completed_native_receipt_v2":
        receipt=verify_completed_export_receipt(cache)
        reference_predictions=receipt["source_contract"]["predictions"]
        saved=_v2_checked(reference_predictions)
    if saved is None:
        if not diagnostic or cache.get("origin")!="synthetic_fixture":raise ValueError("Unregistered cache source")
        saved=[{"sample_id":r["image_id"],"instances":r["engineering_saved_predictions"]} for r in cache["images"]]
    gate=identity.get("full_box_quality_receipt")
    if gate is not None:
        from eval.action4_corner_prediction import validate_full_box_quality_gate
        measured=validate_full_box_quality_gate(gate,b0_sha256=models["B0"]["sha256"],
            b0_config_id=identity["B0_config_id"],expected_reference_predictions=reference_predictions)
        if measured!=identity["full_box_quality_policy"]:raise ValueError("Full-box gate receipt/policy mismatch")
    if not diagnostic:
        import torch
        for role,model_binding in models.items():
            state=_v2_read(model_binding,lambda stream:torch.load(stream,map_location="cpu",weights_only=True))
            _check_checkpoint_metadata(state,role,identity,counterpart)
            del state
    return cache,{r["sample_id"]:r["instances"] for r in saved}

def _array(value,shape,name):
    a=np.asarray(value,float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("Invalid finite "+name)
    return a

RAW_PROJECTION_ATOL_PX=1e-3
CANDIDATE_PROJECTION_ATOL_PX=1e-8

def _projection_check(corners,R,t,cad,K,valid,*,tolerance):
    if not valid:
        if corners is not None:raise ValueError("Invalid pose must not publish corner coordinates")
        return
    camera=cad@np.asarray(R,float).T+np.asarray(t,float)
    if not np.isfinite(camera).all() or (camera[:,2]<=0).any():raise ValueError("Published pose has nonpositive depth")
    hom=camera@K.T;projected=hom[:,:2]/hom[:,2:]
    if not np.isfinite(projected).all() or not np.allclose(_array(corners,(8,2),"projected corners"),projected,rtol=0,atol=tolerance):
        raise ValueError("Published corners disagree with stored physical R/t projection")

def _raw_network_projection(o):
    """Reconstruct the original FP32 network projection, not decoder depth admission."""
    import torch
    cad=_array(o["canonical_m"],(8,3),"raw CAD")
    R=_array(o["raw_rotation"],(3,3),"raw R");t=_array(o["raw_translation_m"],(3,),"raw t")
    A224=_array(o["A224_from_native"],(3,3),"raw A224")
    conversion,_=DataAdapter.letterbox_affine((112,112),(224,224))
    A112=np.linalg.inv(conversion)@A224
    K112=A112@_array(o["K_native"],(3,3),"raw K")
    tensors=[torch.as_tensor(v,dtype=torch.float32) for v in (cad,R,t,K112)]
    if any(not torch.isfinite(v).all() for v in tensors):
        raise ValueError("Raw FP32 projection range cannot be reliably reconstructed")
    c,r,tr,k=tensors
    camera=torch.bmm(c[None],r[None].transpose(1,2))+tr[None,None]
    hom=torch.bmm(camera,k[None].transpose(1,2))
    uv112=hom[:,:,:2]/camera[:,:,2:].clamp_min(1e-4)
    if not torch.isfinite(camera).all() or not torch.isfinite(uv112).all():
        raise ValueError("Nonfinite FP32 projection requires explicit numerical-failure handling")
    # A sign decision within float32 dot-product rounding uncertainty is not evidence.
    magnitude=np.abs(cad)@np.abs(R[2])+abs(t[2])
    bound=8*np.finfo(np.float32).eps*np.maximum(magnitude,np.finfo(np.float32).tiny)
    z=camera[0,:,2].numpy().astype(float)
    if np.any(np.abs(z)<=bound):
        raise ValueError("Raw FP32 depth sign is numerically ambiguous; refuse inferred validity")
    valid=bool((camera[:,:,2]>0).all())
    corners=DataAdapter.transform_points(uv112[0].numpy(),np.linalg.inv(A112)) if valid else None
    if corners is not None and not np.isfinite(corners).all():
        raise ValueError("Raw inverse-affine projection cannot be reliably reconstructed")
    return valid,corners

def _check_raw_network_contract(o,raw):
    valid,corners=_raw_network_projection(o)
    if raw["pose_valid"] is not valid or raw["positive_depth_verified"] is not valid:
        raise ValueError("Raw validity flags disagree with actual FP32 network projection contract")
    if valid:
        if not np.allclose(_array(raw["corners_px"],(8,2),"raw network corners"),corners,rtol=0,atol=RAW_PROJECTION_ATOL_PX):
            raise ValueError("Raw corners disagree with FP32 clamp/inverse-ROI projection")
    elif raw["corners_px"] is not None:
        raise ValueError("Invalid raw network projection must have no published corners")

def _decoder_replay(payload,o,raw,candidate):
    obs=o["observations"];identity=obs["identity"]
    # No GT argument enters this replay; all values are frozen prediction fields.
    replay=decode_single_frame_pose(canonical_m=o["canonical_m"],K_native=o["K_native"],
        A224_from_native=o["A224_from_native"],raw_rotation=o["raw_rotation"],raw_translation_m=o["raw_translation_m"],
        observations=dict(obs,multimodal=np.asarray(obs["multimodal"],dtype=bool)),route=o["route"],
        pixel_space_id=obs["pixel_space_id"],raw_identity=identity,roi_identity=identity,
        quality_ready=False,config=payload["identity"]["decoder_policy"],diagnostic_candidate=payload["diagnostic_candidate"])
    declared=o["decoder_diagnostics"]
    for name in ("identity","pixel_space_id","quality_ready","accepted","geometric_candidate_accepted","reason","pose_valid","used_corner_ids"):
        if declared.get(name)!=replay[name]:raise ValueError("Stored decoder evidence differs from prediction-only replay: "+name)
    if replay["geometric_candidate_accepted"]:
        for name,shape in (("candidate_rotation",(3,3)),("candidate_translation_m",(3,)),("candidate_corners_native",(8,2))):
            if not np.allclose(_array(declared.get(name),shape,name),np.asarray(replay[name]),rtol=0,atol=1e-8):
                raise ValueError("Stored accepted candidate differs from decoder replay")
    elif any(name in declared for name in ("candidate_rotation","candidate_translation_m","candidate_corners_native")):
        raise ValueError("Rejected decoder evidence contains a fabricated accepted candidate")
    cad=np.asarray(o["canonical_m"],float);K=np.asarray(o["K_native"],float)
    _check_raw_network_contract(o,raw)
    if replay["geometric_candidate_accepted"]:
        _projection_check(candidate["corners_px"],candidate["rotation"],candidate["translation_m"],cad,K,candidate["pose_valid"],
            tolerance=CANDIDATE_PROJECTION_ATOL_PX)
    else:
        _check_raw_network_contract(o,candidate)

def _validate_route_quality(route,box,identity,diagnostic,quality_evidence):
    scope=route.get("quality_gate_scope")
    if identity.get("refined_quality_bridge") is not None:
        if quality_evidence is None or box["prediction_id"] not in quality_evidence:
            raise ValueError("Verified current-instance sidecar quality absent")
        q=quality_evidence[box["prediction_id"]]
        gate=identity["full_box_quality_policy"]
        if (q["threshold"]!=gate["threshold"] or q["scope"]!=gate["scope"]
            or scope!=q["scope"] or q["real_world_quality_calibrated"] is not False
            or route["full_box_quality_passed"] is not q["passed"]
            or q["passed"] is not (q["score"]>=gate["threshold"])
            or route["full_box_quality_reason"]!="exact_capture_bound_development_quality"):
            raise ValueError("Route differs from verified sidecar operating point")
        return
    if identity.get("full_box_quality_receipt") is not None and scope!="engineering_fixture_only":
        gate=identity["full_box_quality_policy"]
        if box.get("refined_quality_protocol")!=gate["score_protocol"] or type(box.get("refined_quality_score")) not in (int,float) or not np.isfinite(box["refined_quality_score"]) or not 0<=box["refined_quality_score"]<=1:
            raise ValueError("Actual refined-quality source absent")
        if route["full_box_quality_passed"] is not (box["refined_quality_score"]>=gate["threshold"]):
            raise ValueError("Per-image gate differs from frozen global threshold")
    if scope not in ("unadmitted","engineering_fixture_only","declared_calibration_receipt"):
        raise ValueError("Unknown route gate scope")
    if scope=="unadmitted" and route["full_box_quality_passed"] is not False:
        raise ValueError("Unadmitted full-box gate cannot pass")
    if identity.get("full_box_quality_receipt") is None and not diagnostic and route["full_box_quality_passed"] is not False:
        raise ValueError("No calibrated full-box gate supplied")

def _validate_image(payload,annotation,cache,saved,diagnostic,quality_evidence=None):
    image=payload["image_id"];identity=payload["identity"]
    if (payload.get("schema")!="action4.corner_prediction_bridge.v1" or payload.get("GT_used") is not False
            or payload.get("quality_ready") is not False or payload.get("deployment_admitted") is not False
            or payload["instances"]!=payload["raw_instances"]):
        raise ValueError("Prediction-only raw formal output required")
    entries=[r for r in cache["images"] if r["image_id"]==image]
    if len(entries)!=1:raise ValueError("Image absent from exact SAM cache domain")
    entry=entries[0]
    if annotation["rgb_sha256"]!=entry["source_sha256"] or contained_path(ROOT,annotation["image"],exists=True)!=contained_path(ROOT,entry["image_path"],exists=True):
        raise ValueError("Annotation/cache RGB identity mismatch")
    if _v2_sha(annotation["image"])!=annotation["rgb_sha256"]:raise ValueError("Actual RGB changed")
    def keyed(rows):
        d={r["prediction_id"]:r for r in rows}
        if len(d)!=len(rows):raise ValueError("Duplicate prediction ID")
        return d
    boxes=keyed(payload["box_candidates"]);raw=keyed(payload["raw_instances"]);candidate=keyed(payload["candidate_instances"])
    observations=keyed(payload["corner_observations"]);detections=keyed(payload["detections"])
    published={k for k,v in boxes.items() if v["score"]>=.3}
    if set(raw)!=published or set(candidate)!=published or set(observations)!=published or set(detections)!=published:
        raise ValueError("Published raw/candidate/observation domain mismatch")
    if set(boxes)!={p["prediction_id"] for p in entry["predictions"]}:raise ValueError("B0/cache complete candidate domain mismatch")
    expected={f"b0:{image}:grid:{r['grid_index']}":r for r in saved[image]}
    if len(expected)!=len(saved[image]) or set(expected)!=set(boxes):raise ValueError("Frozen B0 source candidate domain changed")
    for pid,b in boxes.items():
        if any(k not in b or b[k]!=v for k,v in expected[pid].items()):raise ValueError("Frozen B0 candidate fields changed")
        if type(b["grid_index"]) is not int or pid!=f"b0:{image}:grid:{b['grid_index']}":raise ValueError("Stable B0 grid ID mismatch")
    for pid,p in raw.items():
        c=candidate[pid];o=observations[pid];obs=o["observations"];b=boxes[pid]
        if p.get("published") is not True or c.get("published") is not True or p["corner_ids"]!=list(range(8)) or c["corner_ids"]!=list(range(8)) or p["bbox_full_xyxy"]!=b["bbox_full_xyxy"] or o["predicted_full_box"]!=b["bbox_full_xyxy"]:
            raise ValueError("Frozen predicted box/published identity mismatch")
        box=_array(b["bbox_full_xyxy"],(4,),"predicted box");bounds=virtual_roi(box,_scoring_roi_expansion(identity))
        A448=DataAdapter.crop_affine(bounds,(448,448));down,_=DataAdapter.letterbox_affine((448,448),(224,224));A=down@A448
        roi=hashlib.sha256(json.dumps({"image":image,"prediction":pid,"bounds":tuple(bounds),"size":448},sort_keys=True).encode()).hexdigest()
        key=[image,pid,roi,"action4.roi448.v1"]
        if o["image_id"]!=image or o["roi_identity"]!=key or list(obs["identity"])!=key or o["route"]["identity"]!=key:
            raise ValueError("Cross-frame/instance/ROI observation")
        if not np.allclose(_array(o["A224_from_native"],(3,3),"A224"),A,atol=1e-10,rtol=0):
            raise ValueError("224 affine differs from registered full-box ROI crop")
        if not np.array_equal(_array(o["K_native"],(3,3),"K"),np.asarray(annotation["camera"]["K"],float)):
            raise ValueError("Native K source mismatch")
        if not np.allclose(_array(o["canonical_m"],(8,3),"CAD"),np.asarray(annotation["corners_3d_mm"],float)/1000,atol=1e-12,rtol=0):
            raise ValueError("Canonical8 source mismatch")
        if obs["pixel_space_id"]!=entry["pixel_space_id"] or obs["corner_ids"]!=list(range(8)):
            raise ValueError("Native pixel domain or corner IDs mismatch")
        uv=_array(obs["uv224"],(8,2),"UV")
        for name in ("p_valid","q_corner"):
            v=_array(obs[name],(8,),name)
            if ((v<0)|(v>1)).any():raise ValueError("Invalid predicted probabilities")
        scale=_array(obs["b224"],(8,),"b224")
        if ((scale<.5)|(scale>20)).any() or any(type(v) is not bool for v in obs["multimodal"]) or len(obs["multimodal"])!=8:
            raise ValueError("Invalid scale/multimodal output")
        d=o["decoder_diagnostics"]
        if (d["identity"]!=key or d["pixel_space_id"]!=entry["pixel_space_id"] or d["quality_ready"] is not False
                or d["accepted"] is not False or d["config"]!=identity["decoder_policy"] or o["quality_ready"] is not False or o["deployed"] is not False):
            raise ValueError("Decoder identity/config/admission mismatch")
        from model.action4_single_frame_pose import _rotation
        _rotation(o["raw_rotation"]);_array(o["raw_translation_m"],(3,),"raw translation")
        _rotation(c["rotation"]);_array(c["translation_m"],(3,),"candidate translation")
        if type(p["pose_valid"]) is not bool or type(p["positive_depth_verified"]) is not bool or type(c["pose_valid"]) is not bool or type(c["positive_depth_verified"]) is not bool:
            raise ValueError("Explicit pose validity flags required")
        if p["pose_valid"]:_array(p["corners_px"],(8,2),"published raw corners")
        if c["pose_valid"]:_array(c["corners_px"],(8,2),"published candidate corners")
        if p["rotation"]!=o["raw_rotation"] or p["translation_m"]!=o["raw_translation_m"] or p["corners_px"]!=o["raw_corners_native"]:
            raise ValueError("Raw pose evidence mismatch")
        if d["rotation"]!=o["raw_rotation"] or d["translation_m"]!=o["raw_translation_m"]:
            raise ValueError("Diagnostic decoder substituted formal raw pose")
        if d["geometric_candidate_accepted"]:
            if (c["corners_px"]!=d["candidate_corners_native"] or c["rotation"]!=d["candidate_rotation"] or c["translation_m"]!=d["candidate_translation_m"]
                    or c["pose_valid"] is not True or c["positive_depth_verified"] is not True):
                raise ValueError("Candidate does not match decoder evidence")
        elif any(c[k]!=p[k] for k in ("corners_px","rotation","translation_m","pose_valid","positive_depth_verified")):
            raise ValueError("Rejected candidate must preserve raw pose")
        if (type(d["geometric_candidate_accepted"]) is not bool or type(payload.get("diagnostic_candidate")) is not bool
                or d["geometric_candidate_accepted"] and not payload["diagnostic_candidate"]
                or c["pose_diagnostic"].get("geometric_candidate_accepted") is not d["geometric_candidate_accepted"]
                or c["pose_diagnostic"].get("reason")!=d["reason"]):
            raise ValueError("Candidate diagnostic evidence mismatch")
        route=o["route"]
        if route["pixel_space_id"]!=entry["pixel_space_id"] or route["kind"]!="full" or route["full_box_native"]!=b["bbox_full_xyxy"]:
            raise ValueError("Wrong decoder route source")
        if route.get("quality_gate_scope")=="engineering_fixture_only" and not (diagnostic and identity["engineering_fixture"]):
            raise ValueError("Engineering gate not production")
        _validate_route_quality(route,b,identity,diagnostic,quality_evidence)
        _decoder_replay(payload,o,p,c)
    return raw,candidate,observations

def _mask(g,record):
    def decode(f):
        with Image.open(f) as image:return np.asarray(image.convert("L"))>0
    mask=_v2_read({"path":g["mask_visible"],"sha256":g["mask_visible_sha256"]},decode)
    if mask.shape!=(record["height"],record["width"]):raise ValueError("GT visible mask native shape mismatch")
    return mask

def _provenance_flags(diagnostic,source_bound):
    if not diagnostic and not source_bound:raise ValueError("Production scoring requires source-bound envelope")
    return {"diagnostic":diagnostic,"production_eligible":not diagnostic,
        "source_execution_verified":False,"source_binding_verified":source_bound,"source_manifest_verified":True,
        "production_eligibility_scope":"scoring_input_completeness_and_provenance_only_not_training_effect_or_deployment"}

def build_scoring_inputs(predictions_binding,annotations_binding,source_manifest_binding,output,*,diagnostic=False):
    if type(diagnostic) is not bool:raise ValueError("Explicit diagnostic boolean required")
    frozen=_v2_checked(predictions_binding) # First freeze prediction evidence, before any GT read.
    source_bound=isinstance(frozen,dict) and frozen.get("schema")==ARTIFACT_SCHEMA
    if source_bound:
        if not _same_binding(frozen["source_manifest"],source_manifest_binding):raise ValueError("Prediction/source envelope mismatch")
        payloads=frozen["results"];effective=frozen["effective_identity"]
    elif diagnostic:
        payloads=frozen if isinstance(frozen,list) else [frozen];effective=payloads[0]["identity"]
    else:raise ValueError("Production requires source-bound prediction artifact envelope, not bare list")
    if not payloads or not isinstance(payloads,list):raise ValueError("Nonempty prediction results list required")
    if any(p["identity"]!=effective for p in payloads):raise ValueError("Mixed effective inference configurations")
    if effective.get("engineering_fixture") is True and not diagnostic:raise ValueError("Engineering fixture prediction refused for production scoring")
    if not diagnostic and _file(predictions_binding).stat().st_mode & 0o222:raise ValueError("Production prediction artifact must be frozen readonly")
    source=_source(source_manifest_binding,diagnostic)
    decoder_source=source["files_sha256"].get("src/model/action4_single_frame_pose.py")
    if decoder_source is not None:
        import model.action4_single_frame_pose as decoder_module
        if _v2_sha(decoder_module.__file__)!=decoder_source:raise ValueError("Scoring decoder replay differs from declared frozen source")

    cache,saved=_deps(effective,diagnostic)
    quality_bridge=None
    if effective.get("refined_quality_bridge") is not None:
        from eval.action4_corner_quality_bridge import CornerQualityBridge
        quality_bridge=CornerQualityBridge(effective["refined_quality_bridge"],
            b0_sha256=effective["models"]["B0"]["sha256"],b0_config_id=effective["B0_config_id"],
            b0_step=effective["B0_step"],base_predictions=effective["input_contract"]["frozen_predictions"],
            quality_gate=effective["full_box_quality_receipt"])

    mapping={p["image_id"]:p for p in payloads}
    if len(mapping)!=len(payloads):raise ValueError("Duplicate prediction image")
    annotations=_v2_read(annotations_binding,lambda f:[json.loads(line) for line in f if line.strip()])
    gt={r["sample_id"]:r for r in annotations}
    if len(gt)!=len(annotations) or not set(mapping)<=set(gt):raise ValueError("Annotation/prediction image mismatch")
    if not diagnostic and (len(mapping)!=650 or set(mapping)!=set(gt)):raise ValueError("Production full650 coverage required")
    records=[gt[k] for k in sorted(mapping)]
    for r in records:
        for g in r["instances"]:
            box=_array(g["bbox_full_xyxy"],(4,),"GT complete box")
            if (box[2:]<=box[:2]).any():raise ValueError("Invalid GT complete box")
        if r.get("split")!="eval" or r.get("status")!="training_accepted":raise ValueError("Published eval annotations required")
    # Caller-visible identity covers the complete effective bridge configuration.
    run={"checkpoint_sha256":effective["models"]["A0"]["sha256"],"config_sha256":digest(effective),
        "decoder_sha256":digest({"policy":effective["decoder_policy"],
            "source_sha256":source["files_sha256"].get("src/model/action4_single_frame_pose.py"),
            "scope":"verified_code_binding" if "src/model/action4_single_frame_pose.py" in source["files_sha256"] else "diagnostic_policy_only"}),
        "data_sha256":annotations_binding["sha256"],
        "source_sha256":source_manifest_binding["sha256"],"b0_model_sha256":effective["models"]["B0"]["sha256"],
        "sam_model_sha256":cache["sources"]["sam_model_sha256"],"sam_cache_sha256":effective["eval_SAM_cache"]["sha256"],
        "predictions_sha256":predictions_binding["sha256"]};run["identity"]=digest(run)
    units=[];prediction_units=[];scoring=[];raw_inputs=[];candidate_inputs=[]
    for record in records:
        image=record["sample_id"];payload=mapping[image]
        quality_evidence=None
        if quality_bridge is not None:
            from data.action4_sam_native_export import _v2_rgb
            arguments=dict(sample_id=image,K=np.asarray(record["camera"]["K"]),
                base_instances=saved[image],b0_sha256=effective["models"]["B0"]["sha256"])
            quality_evidence=quality_bridge.consume_previously_verified_original(**arguments)
            if quality_evidence is None:
                rgb=_v2_rgb(record["image"],record["rgb_sha256"])
                quality_evidence=quality_bridge.consume(rgb=rgb,**arguments)
        raw,candidate,observations=_validate_image(payload,record,cache,saved,diagnostic,quality_evidence)
        raw_inputs.append({"sample_id":image,"instances":list(raw.values())});candidate_inputs.append({"sample_id":image,"instances":list(candidate.values())})
        gs=[g for g in record["instances"] if not is_ignored(g)];pids=sorted(raw)
        if len({str(g["instance_id"]) for g in record["instances"]})!=len(record["instances"]):raise ValueError("Duplicate GT unit")
        matchable=[g for g in gs if np.asarray(g["bbox_full_xyxy"]).shape==(4,) and np.isfinite(g["bbox_full_xyxy"]).all()]
        pairs=match_geometry([np.asarray(observations[p]["predicted_full_box"],float) for p in pids],
            [np.asarray(g["bbox_full_xyxy"],float) for g in matchable],pids,[g["instance_id"] for g in matchable])
        owners={str(matchable[g]["instance_id"]):pids[p] for p,g in pairs};assigned=set(owners.values())
        for pid in pids:prediction_units.append({"image_id":image,"prediction_id":pid})
        for g in record["instances"]:
            gid=str(g["instance_id"]);nonignore=not is_ignored(g)
            native=np.asarray(g["corners_px"],float);cad=np.asarray(record["corners_3d_mm"],float)/1000
            T=np.asarray(g["canonical8_geometry"]["object_to_capture_cv_m"],float)
            geometry_known=native.shape==(8,2) and np.isfinite(native).all() and T.shape==(4,4) and np.isfinite(T).all() and cad.shape==(8,3) and np.isfinite(cad).all()
            groups=gt_groups(record,g,gs) if nonignore else ["ignored"]
            units.append({"image_id":image,"gt_instance_id":gid,"pose_nonignore":nonignore,
                "eligible":bool(nonignore and geometry_known),"pose_groups":groups if nonignore else [],
                "groups":groups})
            pid=owners.get(gid);base={"image_id":image,"gt_instance_id":gid,"prediction_id":pid,"run_identity":run["identity"]}
            if pid is None:scoring.append(base);continue
            o=observations[pid];obs=o["observations"];A=np.asarray(o["A224_from_native"],float)
            target=np.zeros((8,2));depth=np.zeros(8);known=np.zeros(8,bool)
            if native.shape==(8,2) and T.shape==(4,4) and np.isfinite(T).all() and cad.shape==(8,3) and np.isfinite(cad).all():
                finite=np.isfinite(native).all(1)
                target=DataAdapter.transform_points(np.where(np.isfinite(native),native,0.),A);depth=(cad@T[:3,:3].T+T[:3,3])[:,2]
                flags=g.get("corner_annotation_known",[True]*8)
                if len(flags)!=8 or any(type(v) is not bool for v in flags):raise ValueError("Explicit corner known flags required")
                known=np.asarray(flags,bool)&finite&np.isfinite(target).all(1)&np.isfinite(depth)
            support=sample_native_nearest(_mask(g,record),DataAdapter.crop_affine(virtual_roi(o["predicted_full_box"],_scoring_roi_expansion(effective)),(448,448)))
            corners=DataAdapter.transform_points(np.asarray(g["bbox_full_xyxy"],float).reshape(2,2),A)
            diagonal=float(np.linalg.norm(corners[1]-corners[0]))
            valid=known&(depth>0)&((target>=0)&(target<=223)).all(1)&bool(support.any())&nonignore
            errors=[float(np.linalg.norm(np.asarray(obs["uv224"][i])-target[i])) if known[i] else None for i in range(8)]
            scoring.append(dict(base,corner_ids=list(range(8)),q_corner=obs["q_corner"],p_valid=obs["p_valid"],uv224=obs["uv224"],
                multimodal=obs["multimodal"],label_known=known.tolist(),gt_valid=valid.tolist(),error224=errors,
                gt_diagonal224=diagonal if np.isfinite(diagonal) and diagonal>0 else None))
        for pid in sorted(set(pids)-assigned):
            obs=observations[pid]["observations"]
            scoring.append({"image_id":image,"gt_instance_id":None,"prediction_id":pid,"run_identity":run["identity"],
                "corner_ids":list(range(8)),"q_corner":obs["q_corner"],"p_valid":obs["p_valid"],"uv224":obs["uv224"],
                "multimodal":obs["multimodal"],"label_known":[False]*8,"gt_valid":[False]*8,"error224":[None]*8,"gt_diagonal224":None})
    domain={"schema":"action4.corner_calibration_domain.v1","split":"eval","image_ids":sorted(mapping),"gt_units":units,
        "prediction_units":prediction_units,"data_sha256":annotations_binding["sha256"],"predictions_sha256":predictions_binding["sha256"]}
    domain["domain_sha256"]=digest(domain)
    def wrap(inputs):
        measurable=all(np.asarray(g["corners_px"]).shape==(8,2) and np.isfinite(g["corners_px"]).all()
            for r in records for g in r["instances"] if not is_ignored(g))
        full=score_pose(inputs,records) if measurable else None
        outcomes=[{"image_id":r["sample_id"],"gt_instance_id":str(r["instance_id"]),"success":r["all8_at_002"]} for r in full["per_gt_failure_reason"]] if full is not None else [
            {"image_id":u["image_id"],"gt_instance_id":u["gt_instance_id"],"success":None} for u in units if u["pose_nonignore"]]
        return {"schema":"action4.corner_calibration_pose_outcomes.v1","run_identity":run["identity"],"data_sha256":run["data_sha256"],
            "score_schema":POSE_SCHEMA,"complete":not diagnostic and measurable,"image_ids":sorted(mapping),
            "pose_gt_count":sum(u["pose_nonignore"] for u in units),"pose_all8_at_002_success":full["pose_all8_at_002_success"] if full is not None else 0,
            "outcomes":outcomes,
            "pose_report":full}
    result={"schema":"action4.corner_calibration_inputs.v1",**_provenance_flags(diagnostic,source_bound),
        "source_manifest":copy.deepcopy(source_manifest_binding),
        "source_verification_scope":"offline producer envelope binding and readonly code/dependency bytes only; actual execution not established",
        "input_bindings":{"predictions":copy.deepcopy(predictions_binding),"annotations":copy.deepcopy(annotations_binding),
            "source_manifest":copy.deepcopy(source_manifest_binding)},"effective_identity":copy.deepcopy(effective),
        "domain":domain,"run_binding":run,"records":scoring,"raw_report":wrap(raw_inputs),"candidate_report":wrap(candidate_inputs),
        "GT_usage":"scoring_only_after_prediction_freeze; no inference/model calls"}
    dest=contained_path(ROOT,output,exists=False)
    if dest.exists():raise FileExistsError("Preserve existing scoring output")
    dest.mkdir(parents=True)
    (dest/"SCORING_INPUTS.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result
