"""CPU current-video material preparation; no tracker/model inference or promotion."""
import json
import hashlib
import time
from pathlib import Path
from fractions import Fraction
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT, contained_path
from data.action4_sam_roi_inputs import ImageFrame,array_digest
from data.action4_video_frame_binding import load_video_frame
from data.action4_video_material_support import VerifiedVideoMaterialSupport
from data.action4_surface_snapshot import load_snapshot
from model.action4_surface_anchors import seed_material_candidates,SeedPolicy
from model.action4_material_mesh import MaterialMeshBinder
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_roi_sweep import read_binding,binding


def inspect_frame(cfg,frame_id):
    started=time.monotonic();lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    expected={k:{x:v[x] for x in ("path","sha256")} for k,v in pair["models"].items()}
    source,timing=load_video_frame(lock["video"]["frames_manifest"]["path"],lock["video"]["frames_manifest"]["sha256"],frame_id)
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"]);_,context=provider.load(frame_id,K)
    ledger=json.loads(context.source_ledger_json);completion=read_binding(lock["video"]["track_replay"]["completion_ref"])
    status=contained_path(ROOT,provider.rows[frame_id]["native_status"],exists=True);status_ref=binding(status)
    mesh=MaterialMeshBinder.from_policy(cfg["mesh_policy"]);results=[]
    for track in cfg["tracks"]:
        try:
            surface_ref=cfg["surfaces"][str(frame_id)][track]
            path=contained_path(ROOT,surface_ref["path"],exists=True)
            if binding(path)!=surface_ref:raise ValueError("Current surface snapshot changed")
            surface=load_snapshot(path);m=surface.metadata
            if m["models"]!=expected:raise ValueError("Not the selected current raw A0/B0 snapshot")
            if ledger.get("native_status",{}).get(track)!="target_present":
                results.append(dict(frame_id=frame_id,track_id=track,status="CURRENT_VIDEO_SUPPORT_UNAVAILABLE",native_status=ledger.get("native_status",{}).get(track),seeds=[],geometry_accepted=False));continue
            owner=ledger["native_owners"][track];generation=owner["generation"]
            o=next(o for o in provider.owners if o["generation"]==generation)
            if generation==0:owner_path=o["native_root"]/"run/SESSION.json"
            else:
                event=next(e for e in completion["events"] if e["proposed_generation"]==generation and e["new_session"]==owner["session_id"])
                owner_path=provider.root/"run/events"/event["event_id"].replace(":","_")/"COMMIT.json"
            f=ImageFrame(m["image_id"],source.pixels,m["pixel_space_id"],source.source_sha256,m["variant_id"],array_digest(source.pixels))
            project_generation=ledger["project_body_registrations"][track]["project_generation"]
            support=VerifiedVideoMaterialSupport.from_current_response(frame=f,source_frame=source,frame_binding=timing,
                surface=surface,context=context,expected_models=expected,track_id=track,track_generation=project_generation,
                native_status_ref=status_ref,owner_ledger_ref={"path":str(owner_path),"sha256":owner["ledger_sha"]})
            r=seed_material_candidates(surface,f,support,expected_models=expected,expected_sources=support.sources,
                mesh=mesh,track_id=track,track_generation=project_generation,anchor_generation=0,
                frame_id=frame_id,pts=timing.pts,time_base=timing.time_base,track_confirmed=True,frame_binding=timing,
                policy=SeedPolicy(max_geometry_trials=cfg["max_geometry_trials"]))
            r.update(source_surface=surface_ref,diagnostic_budget_only=True,quality_gates_unchanged=True,
                CoTracker_started=False,model_forward_calls=0)
            results.append(r)
        except Exception as e:results.append(dict(frame_id=frame_id,track_id=track,status="INPUT_OR_PREPARATION_ERROR",reason=type(e).__name__+":"+str(e),geometry_accepted=False))
    out=contained_path(ROOT,cfg["output"],exists=True)/(str(frame_id)+"_RESULT.json")
    report=dict(frame_id=frame_id,records=results,requested_tracks=cfg["tracks"],seconds=time.monotonic()-started,
        purpose="material candidate data readiness only; no tracking",model_calls=0,GPU=False)
    with out.open("x") as f:json.dump(report,f,indent=2,allow_nan=False)
    return report
