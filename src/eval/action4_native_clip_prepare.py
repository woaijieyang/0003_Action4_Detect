"""Prepare registered native clip inputs from closed baseline predictions and SAM replay."""
import json,hashlib,shutil,re
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_native_texture_support import prepare_texture_support
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_roi_sweep import read_binding,binding


def prepare_clips(output_root,clips,*,input_lock=None,predictions=None,max_clip_frames=120,pose_input_kind="raw_A0"):
    if pose_input_kind not in ("raw_A0","selected_A0"):raise ValueError("Explicit native pose input kind required")
    if type(max_clip_frames) is not int or max_clip_frames not in (120,360):raise ValueError("Registered120/360 frame budget required")
    if (input_lock is None)!=(predictions is None):raise ValueError("Explicit lock and completed predictions must be paired")
    if any(not isinstance(name,str) or re.fullmatch(r"[a-z0-9_]+",name) is None for name in clips):raise ValueError("Simple registered clip names required")
    out=contained_path(R,output_root,exists=False)
    requested={f for lo,hi in clips.values() for f in range(lo,hi)}
    if not clips or any(type(lo) is not int or type(hi) is not int or not 4<=lo<hi<=2850 or hi-lo>max_clip_frames for lo,hi in clips.values()):
        raise ValueError("Registered contiguous development clips of within selected bounded budget required")
    out.mkdir(exist_ok=False)
    lock_path=contained_path(R,input_lock or "outputs/action4_fixed_best_video_v001/INPUT_LOCK.json",exists=True)
    lock=json.loads(lock_path.read_text());pair=read_binding(lock["formal_pair_gate"])
    from eval.action4_video_roi_contract import trial_ratio
    selected_ratio=1.5
    if pair.get("trial_selection") is not None:
        selection=read_binding(pair["trial_selection"])
        seal=read_binding(selection["sealed_research"]) if selection.get("sealed_research") is not None else None
        selected_ratio=trial_ratio(selection,seal)

    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=pair.get("video_feature_mode_required")=="current_native_v1")
    K=np.array(read_binding(lock["video"]["K"])["K_native"])
    canonical=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"],float)/1000.
    from model.action4_final_pose_record import resolve_final_pose
    baseline=contained_path(R,predictions or "outputs/action4_fixed_best_video_v001/e1p50/predictions.jsonl",exists=True)
    completion=json.loads(baseline.with_name("COMPLETE.json").read_text())
    if completion["frames"]!=2850 or completion["roi_expansion"]!=selected_ratio:raise ValueError("Closed baseline domain required")
    rows={};digest=hashlib.sha256();count=0
    with baseline.open("rb") as stream:
        for line in stream:
            digest.update(line);row=json.loads(line)
            if row["frame_id"]!=count:raise ValueError("Baseline frame order differs")
            if count in requested:rows[count]=row
            count+=1
    if count!=2850 or set(rows)!=requested:raise ValueError("Missing baseline frames")
    parent=R/"outputs/action4_m3t_region_gate_v001"
    for name,(start,end) in clips.items():
        target=out/name;target.mkdir();(target/"inputs").mkdir()
        for source in ["MODEL_COMPLETE.json","track_00_body.yaml","track_01_body.yaml"]:shutil.copyfile(parent/source,target/source)
        inputs=[];supports=[]
        for fid in range(start,end):
            old=rows[fid];source=provider.rows[fid]["frame"]
            if predictions is not None:
                identity=old["prediction"]["identity"]
                if any(identity[role+"_step"]!=pair["models"][role]["selected_step"] or
                       identity[role+"_config_id"]!=pair["models"][role]["config_id"] for role in ("A0","B0")):
                    raise ValueError("Clip predictions differ from explicit model pair")
            if old["pts"]!=source["pts"] or old["RGB_sha256"]!=source["sha256"] or old["roi_expansion"]!=selected_ratio:
                raise ValueError("Current baseline frame/ROI source differs")
            if pair.get("trial_selection") is not None and any(old["prediction"]["identity"]["models"][role]["sha256"]!=pair["models"][role]["sha256"] for role in ("A0","B0")):
                raise ValueError("Native clip model weights differ from selected pair")
            old_by_id={i["track_id"]:i for i in old["prediction"]["instances"]}
            tracks=["track_00","track_01"]
            if set(old_by_id)!=set(tracks):raise ValueError("Missing or duplicated baseline instances; no inferred replacement")
            masks,receipt=provider.current_native_masks(fid)
            support,audit=prepare_texture_support(masks,receipt,frame_id=fid,pts=source["pts"],
                time_base=provider.region["time_base"],rgb_sha256=source["sha256"],
                image_shape=(provider.region["height"],provider.region["width"]),tracks=tracks)
            arrays=dict(support)
            for tid in tracks:arrays["raw_"+tid]=masks.get(tid,np.zeros((provider.region["height"],provider.region["width"]),bool))
            path=target/"inputs"/f"support_{fid:08d}.npz"
            with path.open("xb") as f:np.savez_compressed(f,**arrays)
            supports.append(dict(frame_id=fid,support=binding(path),audit=audit,source_receipt=receipt))
            instances=[]
            for index,tid in enumerate(tracks,1):
                inst=old_by_id[tid]
                if inst["corner_ids"]!=list(range(8)) or (pose_input_kind=="raw_A0" and inst["corners_final"]!=inst["corners_raw_from_resolved_roi"]):
                    raise ValueError("Expected fixed-ID raw baseline before native geometry")
                ev={k:inst[k] for k in ["frame_id","pts","current_support_status","score","roi_decision_id","pose_valid","source_detection_ids"]}
                if inst["current_support_status"]["SAM"] and tid not in masks:raise ValueError("Declared current SAM support missing from source")
                selection_fields={}
                if pose_input_kind=="selected_A0":
                    selected_pose,selected_receipt=resolve_final_pose(inst,canonical,K)
                    selection_fields=dict(selected_A0_pose=selected_pose,selected_pose_input_receipt=selected_receipt,selected_corners=inst["corners_final"])
                instances.append(dict(track_id=tid,native_body_id=index,raw_A0_pose=inst["raw_network_pose"],raw_corners=inst["corners_raw_from_resolved_roi"],
                    **selection_fields,
                    raw_B0_box=inst["bbox_b0_raw"],SAM_status=receipt.get("per_track_status",{}).get(tid),SAM_mask_sha256=receipt.get("mask_sha256",{}).get(tid),
                    recovery_evidence=ev,bbox_for_a0=inst["bbox_for_a0"],roi_source=inst["roi_source"],
                    resolved_ROI_current_support=bool(inst["roi_reject_reason"] is None and any(inst["current_support_status"].values()))))
            inputs.append(dict(frame_id=fid,pts=source["pts"],time_base=provider.region["time_base"],
                source_png=dict(path=source["image"],sha256=source["sha256"]),shape=[provider.region["height"],provider.region["width"],3],K=K.tolist(),
                current_SAM_receipt=receipt,instances=instances))
        (target/"INPUT_MANIFEST.json").write_text(json.dumps(dict(frame_ids=list(range(start,end)),tracks=tracks,rows=inputs,
            roi_expansion=selected_ratio,native_pose_input_kind=pose_input_kind,recovery_mode="current_evidence_optional_reseed_and_SAM_only_continuation",baseline_predictions=dict(path=str(baseline.relative_to(R)),sha256=digest.hexdigest()),
            input_lock=binding(lock_path),model_pair=lock["formal_pair_gate"],max_clip_frames=max_clip_frames,GT_used=False),indent=2))
        (target/"SUPPORT_INPUTS.json").write_text(json.dumps(supports,indent=2))
    result=dict(status="CURRENT_NATIVE_INPUTS_PREPARED_NOT_TRACKED",clips=clips,baseline_sha256=digest.hexdigest(),new_RGB_NPY_files=0,new_model_inference=0)
    (out/"INPUTS_COMPLETE.json").write_text(json.dumps(result,indent=2));return result
