"""Native return-to-start tracking check; no query overwrite or pose update."""
import json,os,time
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding,binding
from model.action4_cotracker_native import load_gpu_core,TrackWindowRequest,WindowImageReader,track_window_raw
from model.action4_phase2_tracking_contracts import AnchorKey,WindowIdentity,CropTransform
def cycle_prefix_index(frame_ids, requested_frame=None):
    frame=frame_ids[-1] if requested_frame is None else requested_frame
    if type(frame) is not int or frame not in frame_ids or frame==frame_ids[0]:
        raise ValueError("Registered nonquery return-seed frame required")
    return frame_ids.index(frame)


def run(config_path):
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=contained_path(R,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    lock=read_binding(cfg["input_lock"]);frames=read_binding(lock["video"]["frames_manifest"])["frames"]
    core,assets=load_gpu_core(cfg["assets"],cfg["allocation"]);reader=WindowImageReader();summaries=[]
    ray=read_binding(cfg["ray_seeds"])
    for ref in cfg["tracks"]:
        source=read_binding(ref);identity=source["identity"];tid=source["track_id"]
        path=contained_path(R,source["arrays"]["path"],exists=True)
        if binding(path)!=source["arrays"]:raise ValueError("Source track changed")
        with np.load(path,allow_pickle=False) as z:uv=z["uv_native"].copy()
        # Use only genuinely supported terminal predictions as return queries.
        endpoint=cycle_prefix_index(identity["frame_ids"],cfg.get("return_query_frame_id"))
        last=source["frame_observations"][endpoint]
        eligible=np.asarray(last["partial_supported_candidate"],bool)&np.isfinite(uv[endpoint]).all(1)
        indices=np.flatnonzero(eligible)
        row=next(x for x in ray["rows"] if x["track_id"]==tid)
        box=np.asarray(row["pose"]["bbox_for_a0"]);diag=float(np.linalg.norm(box[2:]-box[:2]));threshold=max(2.,.005*diag)
        result=dict(track_id=tid,original_anchor_count=len(uv[0]),terminal_query_count=len(indices),
            scale_ref_kind="frozen_upstream_full_bbox",scale_source_frame=row["pose"]["frame_id"],d_gate=diag,
            cycle_threshold_native_px=threshold,source_track=ref,geometry_accepted=False,records=[],
            validated_frame_interval=[identity["frame_ids"][0],identity["frame_ids"][endpoint]+1],
            query_frame_id=identity["frame_ids"][endpoint],source_forward_window_unchanged=True)
        if len(indices):
            ids=tuple(AnchorKey(**identity["anchor_ids"][int(i)]) for i in indices)
            fids=identity["frame_ids"][:endpoint+1];window=WindowIdentity(tid,identity["window"]["track_generation"],"return_check_"+tid,fids[0],fids[-1]+1)
            request=TrackWindowRequest(window=window,frames=tuple(frames[i] for i in fids),crop=CropTransform(**identity["crop"]),
                anchor_ids=ids,query_frame_ids=tuple(fids[-1] for _ in ids),query_uv_native=uv[endpoint,indices],
                pixel_space_id=identity["pixel_space_id"],query_role="material_candidates",time_base=identity["time_base"])
            back=track_window_raw(core,request,reader=reader,iters=assets["iters"],model_identity=core._action4_native_identity)
            raw=back["merged"]["raw"]
            with (out/(tid+"_RETURN.npz")).open("xb") as stream:np.savez_compressed(stream,
                raw_uv_native=raw[0],visibility=raw[1],confidence=raw[2],is_query=back["merged"]["is_query"],
                original_anchor_indices=indices)
            if back["merged"]["is_query"][0].any():raise ValueError("Return frame cannot be forced query")
            returned={int(j):(raw[0][0,k],float(raw[1][0,k]),float(raw[2][0,k])) for k,j in enumerate(indices)}
        else:returned={}
        for j,key in enumerate(identity["anchor_ids"]):
            r=dict(anchor_id=key["anchor_id"],generation=key["generation"],index=j,cycle_available=False,cycle_passed=False,
                   error_native_px=None,reason="terminal_observation_unavailable",query_corrected_used=False)
            if j in returned:
                point,v,c=returned[j];finite=bool(np.isfinite(point).all())
                error=float(np.linalg.norm(point-np.asarray(identity["query_uv_native"][j]))) if finite else None
                valid=finite and v>=.9 and c>=.6
                r.update(cycle_available=valid,cycle_passed=bool(valid and error<=threshold),error_native_px=error if finite else None,
                    raw_return_uv=point.tolist(),visibility=v,confidence=c,
                    reason=("cycle_pass" if valid and error<=threshold else "cycle_error_above_gate" if valid else "return_observation_unavailable"))
            result["records"].append(r)
        result["passed"]=sum(x["cycle_passed"] for x in result["records"])
        (out/(tid+"_CYCLE.json")).write_text(json.dumps(result,indent=2,allow_nan=False));summaries.append(dict(track_id=tid,passed=result["passed"],total=result["original_anchor_count"]))
    done=dict(status="NATIVE_CYCLE_CHECK_COMPLETE_NOT_POSE_ACCURACY",results=summaries,GT_used=False,geometry_accepted=False,pose_modified=False)
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2));return done
