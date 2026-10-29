"""Current native-support association of existing marker observations; no pose publication."""
import json
from collections import Counter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from model.action4_marker_observation_assignment import assign_markers

def flatten(row):
    return [dict(o,crop_track_id=inst["track_id"]) for inst in row["instances"] for o in inst["observations"]]

def run(config_path):
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    lock=read_binding(cfg["input_lock"]);provider=ReanchorROIProvider(**lock["video"]["track_replay"])
    source=contained_path(R,cfg["observations"]["path"],exists=True)
    if binding(source)!=cfg["observations"]:raise ValueError("Current feature evidence changed")
    rows=[json.loads(l) for l in source.open()]
    if len(rows)!=2850 or [r["frame_id"] for r in rows]!=list(range(2850)):raise ValueError("Full source domain required")
    policy=cfg["policy"];seed=rows[cfg["seed_frame"]];masks,receipt=provider.current_native_masks(seed["frame_id"])
    inspected=assign_markers(seed["frame_id"],flatten(seed),masks,{},**policy)
    registry={}
    for observation in inspected["observations"]:
        cov=sorted(observation["coverage"].items(),key=lambda p:(-p[1],p[0]))
        if not cov or cov[0][1]<policy["minimum_coverage"] or cov[0][1]-(cov[1][1] if len(cov)>1 else 0)<policy["minimum_margin"]:
            raise ValueError("Seed marker does not have unique current body support")
        if cov[0][0] in registry or observation["marker_id"] in registry.values():raise ValueError("Seed body/code identity ambiguous")
        registry[cov[0][0]]=observation["marker_id"]
    if len(registry)!=2:raise ValueError("This registered two-body development audit needs two actual seed bindings")
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    reg=dict(schema="action4.provisional_display_marker_registry.v1",seed_frame=seed["frame_id"],
        expected_marker_by_track=registry,seed_observations=seed,seed_native_receipt=receipt,
        physical_identity_verified=False,semantic_GT=False,uses_future_frames=False,
        valid_from_frame=seed["frame_id"],detection_variant_id=cfg["detection_variant_id"])
    (out/"REGISTRY.json").write_text(json.dumps(reg,indent=2))
    counts=Counter();reasons=Counter();body_counts=Counter()
    with (out/"ASSOCIATIONS.jsonl").open("x") as stream:
        for row in rows:
            fid=row["frame_id"];observations=flatten(row)
            if observations:
                masks,current=provider.current_native_masks(fid)
                if current["RGB_sha256"]!=row["source_RGB_sha256"] or current["pts"]!=row["pts"]:
                    raise ValueError("Marker/native frame identity mismatch")
                result=assign_markers(fid,observations,masks,registry if fid>=cfg["seed_frame"] else {},**policy)
                result["current_native_receipt"]=current
            else:
                result=dict(frame_id=fid,observations=[],input_observation_count=0,unique_observation_count=0,
                    status="no_marker_observation_not_target_absence")
            result.update(pts=row["pts"],source_RGB_sha256=row["source_RGB_sha256"],detection_variant_id=cfg["detection_variant_id"])
            counts["frames"]+=1;counts["raw_observations"]+=result["input_observation_count"]
            counts["unique_observations"]+=result["unique_observation_count"]
            for o in result["observations"]:
                counts[o["status"]]+=1;reasons[o["reason"]]+=1
                if o["track_id"] is not None:body_counts[o["track_id"]]+=1
                counts["duplicate_crop_observations"]+=o["duplicate_crop_observation_count"]
            stream.write(json.dumps(result,allow_nan=False)+chr(10))
    summary=dict(status="FULL_CURRENT_SUPPORT_ASSOCIATION_AUDIT_NOT_POSE_ACCEPTANCE",
        counts=dict(counts),reasons=dict(reasons),per_track_assigned=dict(body_counts),registry=reg,
        B0_predictions_modified=False,A0_predictions_modified=False,body_IDs_swapped=False,
        display_fixed_to_body_metric_verified=False)
    (out/"COMPLETE.json").write_text(json.dumps(summary,indent=2));return summary
