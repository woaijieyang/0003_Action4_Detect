"""Bounded native CoTracker on actual provisional seeds; no pose publication."""
import json,hashlib,time,os
from pathlib import Path
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from eval.action4_roi_sweep import read_binding,binding
from model.action4_phase2_tracking_contracts import AnchorKey,WindowIdentity,CropTransform
from model.action4_cotracker_native import load_gpu_core,TrackWindowRequest,WindowImageReader,track_window_raw
from eval.action4_reanchor_roi_provider import ReanchorROIProvider


def prediction_crop(boxes):
    """Offline window crop from frozen upstream full-body predictions only.

    Union of each prediction expanded1.5; no GT, native clipping, or changes to
    point identities. One fixed crop is used throughout this core call.
    """
    boxes=np.asarray(boxes,dtype=float)
    if boxes.ndim!=2 or boxes.shape[1]!=4 or not len(boxes) or not np.isfinite(boxes).all() or (boxes[:,2:]<=boxes[:,:2]).any():
        raise ValueError("Finite positive full-body boxes required")
    centers=(boxes[:,:2]+boxes[:,2:])/2
    sizes=np.ceil((boxes[:,2:]-boxes[:,:2])*1.5).astype(int)
    left=np.floor(centers-sizes/2).astype(int)
    lo=left.min(0);hi=(left+sizes).max(0)
    return CropTransform(int(lo[0]),int(lo[1]),int(hi[0]-lo[0]),int(hi[1]-lo[1]))

def run(config_path):
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    lock=read_binding(cfg["input_lock"]);manifest=read_binding(lock["video"]["frames_manifest"])
    source_seeds=read_binding(cfg["seed_result"])
    if cfg.get("seed_kind")=="ray_estimated":
        if source_seeds.get("schema")!="action4.frame4_provisional_ray_seeds.v1" or source_seeds.get("geometry_accepted") is not False:
            raise ValueError("Explicit provisional ray binding source required")
        source_input=read_binding(source_seeds["input"])
        seeds=[]
        from fractions import Fraction
        for row in source_seeds["rows"]:
            selected=[p for p in row["features"] if p["feature_id"] in row["selected_feature_ids"]]
            if len(selected)!=row["selected_count"]:raise ValueError("Ray seed selection domain differs")
            converted=[]
            for p in selected:
                if p["status"]!="provisional_fixed_binding" or p["geometry_accepted"] is not False:
                    raise ValueError("Ray source cannot imply accepted geometry")
                converted.append(dict(key=dict(track_id=row["track_id"],anchor_id="ray:"+p["binding_identity"],generation=0),
                    query_frame_id=row["pose"]["frame_id"],query_uv_native=p["native_uv"],
                    X_cad_m=p["X_cad_m"],mesh_triangle_id=p["triangle_id"],barycentric_coordinates=p["barycentric"],
                    spatial_grid_id=p["spatial_grid_id"],binding_quality="estimated_source_pose_raycast_not_truth",
                    XYZ_source="A0_estimated_keyframe_real_mesh_raycast",lifecycle_state="provisional",
                    source_pose_identity=p["source_pose_identity"],geometry_accepted=False))
            seeds.append(dict(frame_id=row["pose"]["frame_id"],track_id=row["track_id"],
                track_generation=row["registration"]["project_generation"],seeds=converted,
                geometry_accepted=False,pixel_space_id=row["registration"]["pixel_space_id"],
                time_base=str(Fraction(*row["registration"]["time_base"])),source_ray_result=cfg["seed_result"],
                source_surface=row["pose"]["surface"],source_models=row["pose"]["models"]))
    elif cfg.get("seed_kind","surface")=="surface":
        seeds=source_seeds["records"]
    else:raise ValueError("Unknown provisional seed origin")
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    core,assets=load_gpu_core(cfg["assets"],cfg["allocation"])
    reader=WindowImageReader();results=[];started=time.monotonic()
    for record in seeds:
        tid=record["track_id"];points=record["seeds"]
        if not points:
            results.append(dict(track_id=tid,status="NO_PROVISIONAL_SEEDS"));continue
        if record["frame_id"]!=cfg["start"] or record["geometry_accepted"] is not False:
            raise ValueError("Only exact-frame provisional input allowed")
        rows=tuple(manifest["frames"][cfg["start"]:cfg["end"]])
        initial=provider.rows[cfg["start"]]["fused"]["tracks"]
        box=np.asarray(next(x for x in initial if x["track_id"]==tid)["bbox_for_roi"],float)
        crop_policy=cfg.get("crop_policy","initial_prediction")
        crop_boxes=[box.tolist()];crop_frames=[cfg["start"]]
        if crop_policy=="offline_frozen_prediction_union":
            crop_boxes=[];crop_frames=[]
            for fid in range(cfg["start"],cfg["end"]):
                matched=[x for x in provider.rows[fid]["fused"]["tracks"] if x["track_id"]==tid]
                if len(matched)>1:raise ValueError("Duplicate instance prediction in crop domain")
                if matched:crop_boxes.append(matched[0]["bbox_for_roi"]);crop_frames.append(fid)
        elif crop_policy!="initial_prediction":raise ValueError("Unknown crop policy")
        crop=prediction_crop(crop_boxes)
        query=np.asarray([p["query_uv_native"] for p in points],float)
        request=TrackWindowRequest(window=WindowIdentity(tid,record["track_generation"],"native17_"+tid,cfg["start"],cfg["end"]),
            frames=rows,crop=crop,anchor_ids=tuple(AnchorKey(**p["key"]) for p in points),
            query_frame_ids=tuple(p["query_frame_id"] for p in points),query_uv_native=query,
            pixel_space_id=record["pixel_space_id"],query_role="material_candidates",time_base=record["time_base"])
        tracked=track_window_raw(core,request,reader=reader,iters=assets["iters"],model_identity=core._action4_native_identity)
        raw=tracked["merged"]["raw"];forward=tracked["raw_forward_model"];backward=tracked["raw_backward_model"]
        array=out/(tid+"_TRACKS.npz")
        with array.open("xb") as stream:np.savez_compressed(stream,uv_native=raw[0],visibility=raw[1],confidence=raw[2],
            is_query=tracked["merged"]["is_query"],raw_forward_xy=forward[0],raw_backward_xy=backward[0],
            raw_forward_visibility=forward[1],raw_backward_visibility=backward[1],
            raw_forward_confidence=forward[2],raw_backward_confidence=backward[2],
            queries_forward_model=tracked["queries_forward_model"],queries_backward_model=tracked["queries_backward_model"])
        frames=[]
        for i,f in enumerate(rows):
            masks,receipt=provider.current_native_masks(f["frame_id"])
            uv=raw[0][i];finite=np.isfinite(uv).all(-1)
            inside=finite&(uv[:,0]>=0)&(uv[:,0]<manifest["source"]["width"])&(uv[:,1]>=0)&(uv[:,1]<manifest["source"]["height"])
            owned=np.zeros(len(points),bool);cross=np.zeros(len(points),bool)
            xy=np.rint(np.where(np.isfinite(uv),uv,0)).astype(int)
            safe=inside&(xy[:,0]<manifest["source"]["width"])&(xy[:,1]<manifest["source"]["height"])
            if tid in masks:owned[safe]=masks[tid][xy[safe,1],xy[safe,0]]
            for key,mask in masks.items():
                if key!=tid:cross[safe]|=mask[xy[safe,1],xy[safe,0]]
            # Partial observation support, deliberately not lifecycle/geometric acceptance.
            supported=inside&owned&~cross&(raw[1][i]>=.9)&(raw[2][i]>=.6)&~tracked["merged"]["is_query"][i]
            frames.append(dict(frame_id=f["frame_id"],pts=f["pts"],query=tracked["merged"]["is_query"][i].tolist(),
                finite=finite.tolist(),same_instance_support=owned.tolist(),other_instance_support=cross.tolist(),
                partial_supported_candidate=supported.tolist(),surface_consistency_checked=False,cycle_checked=False,
                geometry_accepted=False))
        result=dict(track_id=tid,status=tracked["status"],identity=tracked["identity"],snapshot_identity=tracked["snapshot_identity"],
            crop_policy=crop_policy,crop_source_frames=crop_frames,crop_source_full_boxes=crop_boxes,crop_source="frozen_B0_SAM_predictions_no_GT",
            arrays=binding(array),seed_count=len(points),seed_record=record,frame_observations=frames,
            support_query_count=tracked["support_query_count"],support_has_CAD_XYZ=False,
            native_core_called=tracked["official_core_called_directly"],pose_modified=False,
            pending=["three_nonquery_surface_identity_confirmations","cycle_check","appearance_stability",">=12_points_and4_grids_for_geometry"])
        (out/(tid+"_RESULT.json")).write_text(json.dumps(result,indent=2,allow_nan=False));results.append(dict(track_id=tid,
            native_status=tracked["status"],seeds=len(points),partial_support_count=sum(sum(f["partial_supported_candidate"]) for f in frames),
            possible_nonquery_observations=(len(rows)-1)*len(points),geometry_accepted=False))
    done=dict(status="NATIVE_PROVISIONAL_TRACK_PROBE_COMPLETE_NOT_GEOMETRY_ACCEPTANCE",results=results,seconds=time.monotonic()-started,
              A0_calls=0,B0_calls=0,SAM_calls=0,CoTracker_native=True,default_modified=False,pose_modified=False,GT_used=False)
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2));return done
