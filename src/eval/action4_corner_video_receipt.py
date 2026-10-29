"""Bound full-video corner trial receipt; never upgrades model calibration."""
import json,hashlib
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_video_roi_contract import config_ratio,ratio_tag,trial_ratio

SCHEMA="action4.corner_video_development_receipt.v1"
def _execution_binding(video_root,input_lock,trial_binding,source_manifest):
    from eval.action4_roi_video_job import verify_source
    root=contained_path(ROOT,video_root,exists=True)
    source_path=contained_path(ROOT,source_manifest["path"],exists=True)
    source=read_binding(source_manifest)
    if source_path!=root/"source/SOURCE_MANIFEST.json":
        raise ValueError("Source manifest belongs to another video run")
    verify_source(source_path.parent)
    if source.get("configuration") is None:raise ValueError("Video source lacks frozen configuration binding")
    cfg=read_binding(source["configuration"])
    if (contained_path(ROOT,cfg["output"],exists=True)!=root or cfg["input_lock"]!=input_lock
        or cfg.get("corner_trial_binding")!=trial_binding):
        raise ValueError("Effective video configuration differs from trial")
    selection=read_binding(trial_binding);lock=read_binding(input_lock);pair=read_binding(lock["formal_pair_gate"])
    a=pair["models"]["A0"];b=pair["models"]["B0"]
    if ({k:a[k] for k in ("path","sha256")}!=selection["preserved_snapshot"]
        or a["selected_step"]!=selection["selected"]["step"]
        or contained_path(ROOT,b["path"],exists=True)!=contained_path(ROOT,selection["B0"],exists=True)
        or pair.get("trial_selection")!=trial_binding or pair.get("quality_ready") is not False
        or pair.get("deployment_admitted") is not False):
        raise ValueError("Video pair is not the selected uncalibrated research candidate")
    strict=read_binding(pair["strict_reload"])
    if strict.get("status") in ("SEALED_USER_STOPPED_ALL8_RESEARCH","SEALED_COMPLETED75_ALL8_RESEARCH_NOT_DEPLOYMENT"):
        if (strict.get("strict_load") is not True or strict.get("frozen_backbone_verified") is not True
            or strict.get("selected_step")!=selection["selected"]["step"]
            or strict.get("sha256")!=selection["preserved_snapshot"]["sha256"]
            or strict.get("bundle")!=selection["preserved_snapshot"]["path"]):
            raise ValueError("Sealed selected parent differs")
    elif (strict.get("checkpoint")!=selection["preserved_snapshot"]
        or strict.get("selected_step")!=selection["selected"]["step"]
        or strict.get("status")!="STRICT_CORNER_BUNDLE_RELOAD_PASS_RESEARCH_TRIAL"):
        raise ValueError("Missing selected candidate strict reload")
    selected_ratio=trial_ratio(selection,strict if selection["status"]!="PRESERVED_ALL8_DEVELOPMENT_CANDIDATE_NOT_FINAL_SEAL" else None)
    if config_ratio(cfg)!=selected_ratio:raise ValueError("Selected model ROI differs from execution")
    return cfg,lock,selection,pair

def seal_trial_video(video_root,input_lock,trial_binding,source_manifest):
    from eval.action4_corner_video import validate_corner_trial_row
    root=contained_path(ROOT,video_root,exists=True)
    cfg,lock,selection,pair=_execution_binding(video_root,input_lock,trial_binding,source_manifest)
    if selection["quality_calibrated"] is not False:raise ValueError("Explicit uncalibrated trial required")
    manifest=read_binding(lock["video"]["frames_manifest"]);source=read_binding(source_manifest)
    if not source.get("files"):raise ValueError("Frozen video source required")
    selected_ratio=config_ratio(cfg);frames=manifest["frames"];predictions=root/ratio_tag(selected_ratio)/"predictions.jsonl"
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000
    accepted=0;instances=0;count=0;reasons={}
    with predictions.open() as stream:
        for frame in frames:
            line=next(stream,None)
            if line is None:raise ValueError("Missing video prediction")
            row=json.loads(line)
            if (row["frame_id"]!=frame["frame_id"] or row["pts"]!=frame["pts"]
                or row["RGB_sha256"]!=frame["image_sha256"]):raise ValueError("Video identity mismatch")
            validate_corner_trial_row(row,K,cad,trial_binding)
            for item in row["prediction"]["instances"]:
                instances+=1
                diag=item.get("decoder_diagnostics",item.get("corner_decoder",{}))
                accepted+=int(diag.get("geometric_candidate_accepted",False))
                reason=diag.get("reason","see_instance_decoder_trace")
                reasons[reason]=reasons.get(reason,0)+1
            count+=1
        if next(stream,None) is not None:raise ValueError("Extra prediction frames")
    if count!=2850:raise ValueError("Full original video required")
    result=dict(schema=SCHEMA,status="FULL2850_CORNER_TRIAL_SAVED_NOT_ACCURACY_ACCEPTANCE",
        frames=count,instances=instances,accepted_candidates=accepted,decoder_reasons=reasons,roi_expansion=selected_ratio,
        input_lock=input_lock,trial_binding=trial_binding,source_manifest=source_manifest,
        configuration=source["configuration"],model_pair=lock["formal_pair_gate"],models=pair["models"],
        predictions=binding(predictions),quality_ready=False,deployment_admitted=False,
        geometry_changed_by_renderer=False,GT_used=False,test_used=False)
    path=root/"CORNER_TRIAL_RECEIPT.json"
    with path.open("x") as stream:json.dump(result,stream,indent=2)
    return binding(path)

def validate_trial_render(video_root,input_lock,receipt_binding):
    root=contained_path(ROOT,video_root,exists=True);receipt=read_binding(receipt_binding)
    if (receipt.get("schema")!=SCHEMA or receipt.get("frames")!=2850 or
        receipt.get("quality_ready") is not False or receipt.get("deployment_admitted") is not False or
        receipt.get("input_lock")!=input_lock or receipt.get("geometry_changed_by_renderer") is not False):
        raise ValueError("Wrong/unverified full corner trial receipt")
    cfg,lock,selection,pair=_execution_binding(video_root,input_lock,receipt["trial_binding"],receipt["source_manifest"])
    selected_ratio=config_ratio(cfg)
    if receipt.get("roi_expansion",1.5)!=selected_ratio:raise ValueError("Trial receipt ROI mismatch")
    expected=root/ratio_tag(selected_ratio)/"predictions.jsonl"
    if contained_path(ROOT,receipt["predictions"]["path"],exists=True)!=expected or binding(expected)!=receipt["predictions"]:
        raise ValueError("Corner trial predictions changed")
    source=read_binding(receipt["source_manifest"])
    if (receipt.get("configuration")!=source["configuration"] or receipt.get("model_pair")!=lock["formal_pair_gate"]
        or receipt.get("models")!=pair["models"]):
        raise ValueError("Rendered trial effective identity differs from saved receipt")
    return receipt
