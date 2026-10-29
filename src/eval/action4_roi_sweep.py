"""Identity gate for the user-authorized six-ratio development experiment."""
import hashlib,json,math
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
RATIOS=(1.0,1.1,1.2,1.25,1.5,2.25)
SCORE_SCHEMA="action4.pose_projection_all8_002.v1"

def binding(path):
    p=contained_path(R,path,exists=True)
    with p.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    return {"path":str(p.relative_to(R)),"sha256":sha}

def read_binding(ref):
    p=contained_path(R,ref["path"],exists=True)
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref["sha256"]:raise ValueError("Experiment dependency changed")
    return json.loads(raw)

def write_new(path,value):
    p=contained_path(R,path,exists=False);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2)

def freeze_formal_pair(a0_seal,b0_seal,output):
    from model.action4_release_bundle import load_bundle
    a_ref=binding(a0_seal);b_ref=binding(b0_seal)
    a=read_binding(a_ref);b=read_binding(b_ref)
    if (a.get("schema")!="action4.sealed_lineage_training_candidate.v1" or a.get("role")!="A0"
            or a.get("strict_load") is not True or a.get("completion",{}).get("formal_training_complete") is not True):
        raise ValueError("Completed independently reloaded formal A0 lineage required")
    spec=read_binding(a["run_spec"]);code=contained_path(R,spec["code_root"],exists=True)
    cfg=read_binding({"path":spec["config"],"sha256":spec["files"]["resolved_config.json"]})
    out=contained_path(R,cfg["output"],exists=True)
    refs={n:binding(out/n) for n in ("TRAIN_COMPLETE.json","EVAL_COMPLETE.json","ONELINEAGE_EVAL_COMPLETE.json","CONTROLLER_COMPLETE.json")}
    records={n:read_binding(ref) for n,ref in refs.items()}
    train=records["TRAIN_COMPLETE.json"];ev=records["EVAL_COMPLETE.json"];ctrl=records["CONTROLLER_COMPLETE.json"]
    if (train.get("mode")!="full_training" or train.get("formal_training_complete") is not True
            or train["epochs"]<80 or train["step"]!=a["completion"]["optimizer_steps"]
            or train["epochs"]!=a["completion"]["epochs"] or any(x["config_id"]!=a["completed_run_config_id"] for x in (train,ev,ctrl))):
        raise ValueError("Formal training/completion identity mismatch")
    expected={0,train["step"]}|set(range(cfg["eval_every"],train["step"]+1,cfg["eval_every"]))
    if set(ev.get("all_steps",[]))!=expected or ev["evaluations"]!=len(expected) or ev["evaluations"]!=a["completion"]["evaluations"]:
        raise ValueError("Prescribed async evaluation domain incomplete")
    selected=read_binding(a["selection_pointer"])
    if selected.get("sealed") is not True or selected.get("deployment_guard_veto_used") is not False:
        raise ValueError("Explicit sealed global training-best selection required")
    winner=selected["winner"]
    if (winner["record"]["step"]!=a["selected_step"] or winner["record"]["config_id"]!=a["config_id"]
            or winner["checkpoint_source"]["sha256"]!=a["sha256"]
            or a["metrics"]["score_schema"]!=SCORE_SCHEMA):
        raise ValueError("Sealed A0 differs from actual full-lineage winner")
    if b.get("role")!="B0" or b.get("strict_load") is not True or b["sha256"]!=a["dependencies"]["evaluation_counterpart_sha256"]:
        raise ValueError("B0 differs from completed A0 counterpart")
    models={}
    for role,seal in (("A0",a),("B0",b)):
        model,base,state=load_bundle(seal["bundle"],role,"cpu",expected_sha256=seal["sha256"])
        if (state["role"],state["selected_step"],state["config_id"])!=(role,seal["selected_step"],seal["config_id"]):
            raise ValueError("Independent reload differs from seal")
        model.eval().requires_grad_(False)
        if base is not None:base.eval().requires_grad_(False)
        models[role]={"path":seal["bundle"],"sha256":seal["sha256"],"selected_step":state["selected_step"],
            "architecture":state["architecture"],"config_id":state["config_id"],"pose_method":state.get("pose_method")}
        del model,base,state
    result={"schema":"action4.roi_sweep.formal_pair_gate.v1","status":"FORMAL_PAIR_COMPLETION_SELECTION_RELOAD_PASS",
        "purpose":"development_sweep_not_deployment_acceptance","models":models,"A0_seal":a_ref,"B0_seal":b_ref,
        "selection_pointer":a["selection_pointer"],"completion_refs":refs,"source_run_spec":a["run_spec"],
        "score_schema":SCORE_SCHEMA,"legacy_deployment_best_step_not_selected":a.get("legacy_deployment_best_step"),
        "defaults_changed":False,"test_consumed":False,"ratios":list(RATIOS),
        "validation_receipt":{"path":a["dependencies"]["validation_receipt"],"sha256":a["dependencies"]["validation_receipt_sha256"]}}
    read_binding(result["validation_receipt"])
    write_new(output,result)
    return result


def evaluate_fixed_ratio(input_lock,expansion,output,*,device="cuda:0"):
    """Full image eval through the existing pipeline, with no new B0 forward."""
    import copy,time,os
    import numpy as np
    import torch
    from data.action4_roi_contract import validate_expansion
    from data.action4_component_data import ComponentData
    from data.action4_sam_roi_inputs import ImageFrame,array_digest
    from data.action4_frozen_detection_context import FrozenDetectionContext
    from data.action4_dataset import is_ignored
    from eval.action4_staged_pipeline import load_pipeline
    from eval.action4_prefetch import ordered_items
    from eval.action4_pose_protocol import score_pose,match_geometry,bounds,ratio
    e=validate_expansion(expansion)
    if e not in RATIOS:raise ValueError("Only preregistered six coefficients allowed")
    lock=json.loads(contained_path(R,input_lock,exists=True).read_text())
    gate=read_binding(lock["formal_pair_gate"]);src=lock["image_eval"]
    root=contained_path(R,Path(input_lock).parent,exists=True)
    if e!=2.25:
        proof=json.loads((root/"BASELINE_225_PASS.json").read_text())
        baseline=read_binding(proof["report"])
        if proof["status"]!="FULL650_BASELINE_REPRODUCED" or baseline["input_lock"]!=binding(input_lock) or baseline["roi_expansion"]!=2.25:
            raise ValueError("Different experiment baseline cannot release the sweep")
    if src["track"] is not False or gate["models"]["A0"]["architecture"]!="joint_pose":
        raise ValueError("This admitted sweep preserves the selected baseline image architecture")
    for ref in gate["models"].values():
        if binding(ref["path"])["sha256"]!=ref["sha256"]:raise ValueError("Frozen weights changed")
    out=contained_path(R,output,exists=False);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cfg={"a0_bundle":gate["models"]["A0"]["path"],"b0_bundle":gate["models"]["B0"]["path"],
         "b0_scoring_policy":src["B0_scoring_policy"],"roi_expansion":e}
    pipeline=load_pipeline(cfg,device);pipeline.b0.eval().requires_grad_(False)
    cached=read_binding(src["B0_candidates"]);boxmap={r["sample_id"]:r["instances"] for r in cached}
    data=ComponentData(gate["validation_receipt"]["path"],"eval","B0")
    if len(data)!=src["images"] or set(boxmap)!={r["sample_id"] for r in data.records}:raise ValueError("Full evaluation domain mismatch")
    cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000.
    count=[0]
    hook=pipeline.b0.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
    predictions=[];conditional=[];b0_gt=0;b0_matched=0;start=time.monotonic()
    for index,item in ordered_items(data,workers=2):
        record=item["record"];sid=record["sample_id"]
        if binding(record["image"])["sha256"]!=record["rgb_sha256"]:raise ValueError("Actual image differs from frozen data")
        rgb=item["image"].permute(1,2,0).numpy()
        frame=ImageFrame(sid,rgb,"phrase2_native_integer_centers",record["rgb_sha256"],"native_original",array_digest(rgb))
        K=np.asarray(record["camera"]["K"])
        context=FrozenDetectionContext.from_frame(frame,K,boxmap[sid],b0_sha256=gate["models"]["B0"]["sha256"],
            scoring_policy=src["B0_scoring_policy"],source_cache_identity=src["B0_candidates"]["sha256"])
        result=pipeline.predict(item["image"],K,cad,frozen_detections=context)
        for pred in result["instances"]:
            pred["frozen_B0_candidate_id"]=sid+":grid:"+str(pred["grid_index"])
            pred["roi_expansion"]=e;pred["roi_source"]="frozen_b0_prediction"
        predictions.append({"sample_id":sid,"instances":result["instances"]})
        gs=[g for g in record["instances"] if not is_ignored(g)]
        bs=[b for b in boxmap[sid] if b["score"]>=src["score_threshold"]]
        pairs=match_geometry([np.asarray(b["bbox_full_xyxy"]) for b in bs],[np.asarray(g["bbox_full_xyxy"]) for g in gs],
            [b["grid_index"] for b in bs],[g["instance_id"] for g in gs])
        b0_gt+=len(gs);b0_matched+=len(pairs)
        bygrid={p["grid_index"]:p for p in result["instances"]}
        for bi,gi in pairs:
            pred=bygrid.get(bs[bi]["grid_index"]);gt=np.asarray(gs[gi]["corners_px"]);err=None
            if pred and pred.get("pose_valid") and pred.get("positive_depth_verified") and pred.get("corner_ids")==list(range(8)):
                uv=np.asarray(pred["corners_px"])
                if uv.shape==(8,2) and np.isfinite(uv).all():err=np.linalg.norm(uv-gt,axis=1)
            d=np.linalg.norm(bounds(gt)[2:]-bounds(gt)[:2])
            conditional.append({"sample_id":sid,"GT_instance_id":gs[gi]["instance_id"],"B0_grid":bs[bi]["grid_index"],
                "errors_native_px":None if err is None else err.tolist(),
                "all8_002":bool(err is not None and err.max()<=.02*d),"all8_005":bool(err is not None and err.max()<=.05*d)})
        if index%25==0:
            temp=out/"STATUS.tmp";temp.write_text(json.dumps({"images":index+1,"expected":len(data),"roi_expansion":e}));temp.replace(out/"STATUS.json")
    hook.remove()
    if count[0]!=0:raise ValueError("Frozen cache path executed B0")
    report=score_pose(predictions,data.records)
    errors=[x for row in report["per_gt_failure_reason"] if row["errors_px"] is not None for x in row["errors_px"]]
    tp5=sum(r["all8_at_005"] for r in report["per_gt_failure_reason"])
    cond_err=[x for row in conditional if row["errors_native_px"] is not None for x in row["errors_native_px"]]
    report.update(roi_expansion=e,purpose="development_sweep",tracking=False,B0_forward_calls=0,
        pose_corner_median_valid_matched_only=float(np.median(errors)) if errors else None,
        pose_all8_at_005_success=tp5,pose_all8_at_005_precision=ratio(tp5,report["pose_output_count"]),
        pose_fp005=report["pose_output_count"]-tp5,pose_fn005=report["pose_gt_count"]-tp5,
        B0_gt_count=b0_gt,B0_correctly_associated=b0_matched,B0_missed_gt=b0_gt-b0_matched,
        conditional_A0={"GT_denominator":b0_matched,"all8_002":ratio(sum(x["all8_002"] for x in conditional),b0_matched),
            "all8_005":ratio(sum(x["all8_005"] for x in conditional),b0_matched),
            "EPE_median_valid_only":float(np.median(cond_err)) if cond_err else None,"EPE_p95_valid_only":float(np.percentile(cond_err,95)) if cond_err else None},
        elapsed_seconds=time.monotonic()-start,identity=cfg,input_lock=binding(input_lock),source_code_identity=os.environ.get("ACTION4_ROI_SOURCE_ID"),
        additional_coverage_diagnostics="pending_separate_GT_only_analysis_not_used_for_inference")
    write_new(out/"PREDICTIONS.json",predictions);write_new(out/"CONDITIONAL_ROWS.json",conditional);write_new(out/"REPORT.json",report)
    if e==2.25:
        reference=read_binding(src["reference_A0_225_predictions"]);refmap={r["sample_id"]:r["instances"] for r in reference}
        maximum=0.
        for row in predictions:
            old=refmap[row["sample_id"]];new=row["instances"]
            if len(old)!=len(new):raise ValueError("Baseline instance count changed")
            for a,b in zip(old,new):
                for key in ("grid_index","score","bbox_full_xyxy","pose_valid","corner_ids"):
                    if a.get(key)!=b.get(key):raise ValueError("Baseline selection/validity changed: "+key)
                if a["corners_px"] is None:
                    if b["corners_px"] is not None:raise ValueError("Baseline invalid pose changed")
                else:maximum=max(maximum,float(np.abs(np.asarray(a["corners_px"])-np.asarray(b["corners_px"])).max()))
        refreport=read_binding(src["reference_report"])
        for key in ("pose_gt_count","pose_output_count","pose_all8_at_002_success","pose_fp","pose_fn","invalid_pose_count"):
            if report[key]!=refreport[key]:raise ValueError("Baseline score changed: "+key)
        if maximum>lock["baseline_coordinate_tolerance_native_px"]:raise ValueError("Baseline coordinate reproduction failed")
        write_new(root/"BASELINE_225_PASS.json",{"status":"FULL650_BASELINE_REPRODUCED","max_native_coordinate_delta":maximum,
            "report":binding(out/"REPORT.json"),"predictions":binding(out/"PREDICTIONS.json"),"B0_forward_calls":0,"images":len(data)})
    write_new(out/"COMPLETE.json",{"status":"FULL_EVAL_PREDICTIONS_AND_PRIMARY_METRICS_COMPLETE","report":binding(out/"REPORT.json"),"roi_expansion":e})
    return report
