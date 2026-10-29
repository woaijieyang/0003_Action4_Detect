"""Paired fixed-model ROI-path diagnostics. Predicted tracks are never ground truth."""
import numpy as np

REQUIRED_IDENTITY=("A0_sha256","B0_sha256","code_sha256","camera_identity","canonical8_identity","decoder_identity","pixel_space_id","roi_expansion")

def compare(b0_rows,track_rows,*,b0_identity,track_identity,frame_manifest,review=None,comparison_mode="causal_forward_B0_assist"):
    if any(k not in b0_identity or k not in track_identity or b0_identity[k]!=track_identity[k] for k in REQUIRED_IDENTITY):
        raise ValueError("Paired model/ROI/camera/decoder identities differ")
    for k in ("A0_sha256","B0_sha256","code_sha256"):
        s=b0_identity[k]
        if not isinstance(s,str) or len(s)!=64 or any(c not in "0123456789abcdef" for c in s):raise ValueError("Invalid fixed SHA identity")
    if b0_identity["roi_expansion"]!=1.5:raise ValueError("This admitted comparison fixes1.5")
    if comparison_mode not in ("causal_forward_B0_assist","offline_bidirectional_SAM_diagnostic"):
        raise ValueError("Explicit registered temporal comparison mode required")
    expected_mode="causal_forward" if comparison_mode=="causal_forward_B0_assist" else "offline_bidirectional_clip"
    if (track_identity.get("temporal_mode")!=expected_mode
            or track_identity.get("causal") is not (comparison_mode=="causal_forward_B0_assist")):
        raise ValueError("Offline bidirectional evidence cannot masquerade as causal B0 tracking")
    expected={r["frame_id"]:r for r in frame_manifest}
    if len(expected)!=len(frame_manifest):raise ValueError("Duplicate manifest frame")
    maps=[]
    for rows in (b0_rows,track_rows):
        mapping={r["frame_id"]:r for r in rows}
        if len(mapping)!=len(rows) or set(mapping)!=set(expected):raise ValueError("Missing/duplicate paired frames cannot be silently removed")
        maps.append(mapping)
    review={} if review is None else review
    if any(v not in ("correct","incorrect","unknown") for v in review.values()):raise ValueError("Explicit visual review verdict required")
    events=[];consumed_reviews=set();totals={"b0_outputs":0,"track_outputs":0,"common_b0_candidate":0,
        "track_supported_without_current_b0":0,"track_selected_other_candidate":0,"b0_only_not_adopted":0,
        "reviewed_correct":0,"reviewed_incorrect":0,"review_unknown":0,"invalid_raw_pose_common":0}
    for fid,frame in expected.items():
        a,b=maps[0][fid],maps[1][fid]
        for row in (a,b):
            if row["pts"]!=frame["pts"] or row["RGB_sha256"]!=frame["RGB_sha256"]:raise ValueError("Paired native frame/PTS differs")
            if row.get("roi_expansion")!=1.5:raise ValueError("Per-frame expansion differs")
        ps=a["prediction"]["instances"];ts=b["prediction"]["instances"]
        totals["b0_outputs"]+=len(ps);totals["track_outputs"]+=len(ts)
        bygrid={}
        for x in ps:
            grid=x.get("grid_index")
            if type(grid) is not int or grid in bygrid:raise ValueError("Unique actual B0 grid identity required")
            bygrid[grid]=x
        for x in ps+ts:
            if x.get("pose_valid") and x.get("corner_ids")!=list(range(8)):
                raise ValueError("Fixed canonical corner IDs0-7 required without reordering")
        track_ids=[x["track_id"] for x in ts]
        if len(set(track_ids))!=len(track_ids):raise ValueError("Duplicate project track")
        used=set()
        for x in ts:
            grid=x.get("grid_index");tid=x["track_id"];key=str(fid)+":"+tid
            verdict=review.get(key,"unknown")
            if key in review:consumed_reviews.add(key)
            totals["review_unknown" if verdict=="unknown" else "reviewed_"+verdict]+=1
            if grid in bygrid:
                if grid in used:raise ValueError("Two tracks copied one B0 candidate")
                kind="common_b0_candidate";used.add(grid);raw=bygrid[grid]
            elif x.get("B0_current_observation") is False and x.get("temporal_status")=="sam_supported":
                kind="track_supported_without_current_b0";raw=None
            else:
                kind="track_selected_other_candidate";raw=None
            totals[kind]+=1
            event=dict(frame_id=fid,pts=frame["pts"],track_id=tid,kind=kind,visual_identity_review=verdict,
                grid_index=grid,pose_valid=bool(x.get("pose_valid")),roi_source=x.get("roi_source",x.get("origin")),
                bbox_for_a0=x["bbox_for_a0"],current_support=x.get("temporal_status"),
                corner_change_native_px=None,roi_center_change_native_px=None)
            if x.get("post_A0_geometry","none")!="none" or x.get("corners_final",x.get("corners_px"))!=x.get("corners_raw_from_resolved_roi",x.get("corners_px")):
                raise ValueError("Compare saved raw A0, not smoothed/geometry-selected poses")
            if raw is not None:
                box=np.asarray(raw["bbox_full_xyxy"],float);new=np.asarray(x["bbox_for_a0"],float)
                event["bbox_b0_raw"]=box.tolist()
                event["roi_center_change_native_px"]=float(np.linalg.norm((new[:2]+new[2:]-box[:2]-box[2:])/2))
                uv=np.asarray(raw.get("corners_px"),float);tv=np.asarray(x.get("corners_raw_from_resolved_roi",x.get("corners_px")),float)
                if raw.get("pose_valid") and x.get("pose_valid") and uv.shape==(8,2) and tv.shape==(8,2) and np.isfinite(uv).all() and np.isfinite(tv).all():
                    event["corner_change_native_px"]=np.linalg.norm(uv-tv,axis=1).tolist()
                else:totals["invalid_raw_pose_common"]+=1
            events.append(event)
        for grid,raw in bygrid.items():
            if grid not in used:
                totals["b0_only_not_adopted"]+=1
                events.append(dict(frame_id=fid,pts=frame["pts"],kind="b0_only_not_adopted",grid_index=grid,
                    bbox_b0_raw=raw["bbox_full_xyxy"],pose_valid=bool(raw.get("pose_valid"))))
    if consumed_reviews!=set(review):raise ValueError("Review contains unknown frame/track records")
    deltas=[v for e in events if e.get("corner_change_native_px") is not None for v in e["corner_change_native_px"]]
    return dict(schema="action4.roi_path_pair_diagnostic.v1",identity={k:b0_identity[k] for k in REQUIRED_IDENTITY},
        frame_count=len(expected),comparison_mode=comparison_mode,counts=totals,events=events,
        common_corner_delta=dict(point_count=len(deltas),median=float(np.median(deltas)) if deltas else None,p95=float(np.percentile(deltas,95)) if deltas else None),
        scope="Prediction-source correspondence and raw-output changes; shared B0 grid is not proof of physical identity",
        recovered_correct_instances=sum(e.get("visual_identity_review")=="correct" for e in events if e["kind"]=="track_supported_without_current_b0") if all(e.get("visual_identity_review")!="unknown" for e in events if e["kind"]=="track_supported_without_current_b0") else None,
        accuracy_GT_available=False,diagnostic_not_pose_accuracy=True)
