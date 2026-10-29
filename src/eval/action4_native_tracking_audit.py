"""Saved native tracking integrity and disagreement diagnostics. No model calls or GT."""
import json
from collections import Counter
import numpy as np
from data.action4_paths import PROJECT_ROOT,contained_path


def native_input_contract(manifest):
    kind=manifest.get("native_pose_input_kind","raw_A0")
    if kind not in ("raw_A0","selected_A0"):raise ValueError("Unknown native pose input kind")
    return kind,("raw_A0_pose" if kind=="raw_A0" else "selected_A0_pose"),("raw_corners" if kind=="raw_A0" else "selected_corners")


def recovery_contract(manifest):
    mode=manifest.get("recovery_mode")
    if mode not in (None,"current_evidence_dev_v1","current_evidence_optional_reseed_and_SAM_only_continuation","record_only_no_reseed"):
        raise ValueError("Unknown recovery contract")
    return mode


def _input_matrix(target,pose_key):
    pose=target[pose_key]
    if pose_key=="selected_A0_pose" and (pose.get("finite") is not True or pose.get("coordinate_frame")!="object_to_capture_cv_m"):
        raise ValueError("Selected input lacks resolved metric numeric contract")
    r=np.asarray(pose["rotation"],float);t=np.asarray(pose["translation_m"],float)
    if r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError("Malformed declared native input pose")
    T=np.eye(4);T[:3,:3]=r;T[:3,3]=t
    return T.astype(np.float32).astype(float)


def audit_records(manifest,records,events):
    kind,pose_key,corners_key=native_input_contract(manifest);mode=recovery_contract(manifest)
    frames=manifest["rows"];tracks=manifest["tracks"]
    if not frames or len(set(tracks))!=len(tracks):raise ValueError("Nonempty unique native domain required")
    expected=[(f["frame_id"],t) for f in frames for t in tracks]
    actual=[(r["frame_id"],r["track_id"]) for r in records]
    if actual!=expected or len(set(actual))!=len(actual):raise ValueError("Tracking output must retain exact ordered frame/body domain")
    if [e["frame_id"] for e in events]!=[f["frame_id"] for f in frames]:raise ValueError("Frame-event domain differs")
    by_key={(r["frame_id"],r["track_id"]):r for r in records};frame_lookup={f["frame_id"]:f for f in frames}
    targets={}
    for f,e in zip(frames,events):
        if e["pts"]!=f["pts"] or e["time_base"]!=f["time_base"] or not e["owner_audit"]["unchanged"]:raise ValueError("Actual native RGB/PTS consumption differs")
        if [i["track_id"] for i in f["instances"]]!=tracks:raise ValueError("Input frame instance order/uniqueness differs")
        for i in f["instances"]:targets[f["frame_id"],i["track_id"]]=i
    rows=[]
    for record in records:
        f=frame_lookup[record["frame_id"]];target=targets[record["frame_id"],record["track_id"]]
        if (record["pts"]!=f["pts"] or record["time_base"]!=f["time_base"] or record["source_RGB"]!=f["source_png"] or
            record["final_corners"]!=target[corners_key] or record["published"] is not False or
            record["raw_baseline_pose"]!=target["raw_A0_pose"] or record["raw_baseline_corners"]!=target["raw_corners"]):
            raise ValueError("Tracking raw source/declared fallback/publication changed")
        if record.get("native_pose_input_kind","raw_A0")!=kind:raise ValueError("Native output consumed another pose route")
        if kind=="selected_A0" or "selected_A0_input_pose" in record:
            if record.get("selected_A0_input_pose")!=target[pose_key]:raise ValueError("Recorded A0 input differs from declared route")
        if type(record.get("raw_A0_reset_this_frame")) is not bool:raise ValueError("Explicit legacy reset boolean required")
        rows.append(dict(frame_id=record["frame_id"],track_id=record["track_id"],status=record["status"],native_call_ok=record["native_call_ok"],raw_A0_reset_this_frame=record["raw_A0_reset_this_frame"]))
    support=[r for e in events for r in e.get("native_texture_consumption",[])]
    if support and len(support)!=len(records):raise ValueError("Incomplete actual support consumption records")
    for f,e in zip(frames,events):
        consumed=e.get("native_texture_consumption",[])
        if consumed and [r["track_id"] for r in consumed]!=tracks:raise ValueError("Native support track order differs")
        if any(r["support_frame_id"]!=f["frame_id"] or not r["all_retained_points_in_current_support"] or not 0<=r["after"]<=r["before"] for r in consumed):raise ValueError("Current support was not correctly consumed")
    first=frames[0]["frame_id"];init_name="initialize_raw_A0_once" if kind=="raw_A0" else "initialize_selected_A0_once"
    named=[(e["frame_id"],c.get("track_id"),c["name"]) for e in events for c in e["calls"] if c["name"] in ("initialize_raw_A0_once","initialize_selected_A0_once")]
    if named and named!=[(first,t,init_name) for t in tracks]:raise ValueError("Initialization event differs from declared pose route")
    if kind=="selected_A0" and not named:raise ValueError("Explicit selected initialization event required")
    if not any(c["name"] in ("StartModalities_reference_frame","StartModalities") and c.get("ok") is True for c in events[0]["calls"]):raise ValueError("Actual successful initialization call missing")
    for tid in tracks:
        if not np.array_equal(np.asarray(by_key[first,tid]["native_pose"]),_input_matrix(targets[first,tid],pose_key)):
            raise ValueError("Initial native pose did not consume the declared A0 input")
    expected_resets=[(first,t) for t in tracks];recoveries=[];suppressed=[];counts={t:0 for t in tracks}
    for f,e in zip(frames,events):
        restarts=[c for c in e["calls"] if c["name"] in ("RestartSupportedBody_current_evidence","RestartSupportedBody")]
        if mode=="record_only_no_reseed" and restarts:raise ValueError("Record-only mode attempted native reseed")
        decisions=e.get("recovery_decisions",[])
        if mode is None:
            if restarts or any(d.get("action")=="reseed" or d.get("native_restart_succeeded") for d in decisions):raise ValueError("Undeclared recovery")
            continue
        if [d["track_id"] for d in decisions]!=tracks:raise ValueError("Recovery decision domain differs")
        actual_reseed_tracks=[]
        for d in decisions:
            if d["frame_id"]!=f["frame_id"] or d["pts"]!=f["pts"]:raise ValueError("Stale recovery decision")
            if d.get("recovery_mode",mode)!=mode:raise ValueError("Recovery record mode differs")
            if by_key[f["frame_id"],d["track_id"]].get("recovery_decision")!=d:raise ValueError("Record/event recovery decision differs")
            if d["action"] not in ("continue","withhold","reseed","record_only_no_reseed"):raise ValueError("Unknown recovery action")
            if mode=="record_only_no_reseed":
                if d["action"]=="reseed" or d.get("native_restart_succeeded"):raise ValueError("Record-only mode actually reseeded")
                if not np.array_equal(np.asarray(d["native_pose_before_decision"]),np.asarray(d["native_pose_after_decision"])):raise ValueError("Record-only decision changed native pose")
            if d["action"]=="record_only_no_reseed":
                if mode!="record_only_no_reseed" or d.get("proposed_action")!="reseed" or d.get("action_suppressed") is not True or d.get("native_restart_succeeded") is not False:raise ValueError("Malformed suppressed reseed record")
                suppressed.append(d);continue
            if d["action"]!="reseed":
                if d.get("native_restart_succeeded"):raise ValueError("Undeclared successful restart")
                continue
            if (d.get("native_restart_succeeded") is not True or d.get("reason")!="two_frame_contradiction_with_current_consistent_A0_B0_SAM" or
                not all(d["current_evidence"]["current_support_status"].get(k) is True for k in ("B0","SAM"))):raise ValueError("Recovery lacks successful native action/current support")
            target=targets[f["frame_id"],d["track_id"]]
            if not np.array_equal(np.asarray(d["native_pose_after_decision"]),_input_matrix(target,pose_key)):raise ValueError("Recovery did not consume the declared current A0 seed")
            counts[d["track_id"]]+=1
            if d["recovery_id"]!=counts[d["track_id"]]:raise ValueError("Recovery generation count differs")
            expected_resets.append((f["frame_id"],d["track_id"]));recoveries.append(d);actual_reseed_tracks.append(d["track_id"])
        if [c.get("track_id") for c in restarts]!=actual_reseed_tracks or any(c.get("ok") is not True for c in restarts):raise ValueError("Actual restart call and recovery decision differ")
    if [(r["frame_id"],r["track_id"]) for r in rows if r["raw_A0_reset_this_frame"]]!=expected_resets:raise ValueError("Undeclared per-frame reset or missing actual recovery")
    return dict(status="NATIVE_SAVED_CHAIN_INTEGRITY_CHECKED_NOT_ACCURACY",frames=len(frames),records=len(records),initializations=len(tracks),
        native_pose_input_kind=kind,initialization_pose_source=pose_key,reseed_pose_source=pose_key,recovery_mode=mode,
        legacy_reset_flag="raw_A0_reset_this_frame",legacy_reset_flag_semantics="reset of manifest-declared A0 input; selected route is not raw-network consumption",
        initialization_evidence="explicit_route_event_and_exact_native_seed" if named else "legacy_StartModalities_and_exact_raw_seed",
        raw_pose_consumed_for_initialization=kind=="raw_A0",raw_baseline_preserved=True,input_final_preserved=True,
        recovery_count=len(recoveries),suppressed_recovery_count=len(suppressed),
        recovery_frames=[dict(frame_id=d["frame_id"],track_id=d["track_id"]) for d in recoveries],
        actual_tracking_steps=sum(any(c["name"]=="ExecuteTrackingStep" for c in e["calls"]) for e in events),
        per_track={t:dict(Counter(r["status"] for r in rows if r["track_id"]==t)) for t in tracks},
        native_call_failures=sum(not e["native_calls_ok"] for e in events),texture_SAM_supported_records=len(support),
        keypoints_before=sum(r["before"] for r in support),keypoints_after=sum(r["after"] for r in support),raw_final_preserved=True,
        raw_final_preserved_semantics="original raw baseline plus the manifest-declared final fallback are retained separately",
        ground_truth_used=False,accuracy_admitted=False)


def audit_native_run(run):
    path=contained_path(PROJECT_ROOT,run,exists=True)
    manifest=json.loads((path/"INPUT_MANIFEST.json").read_text())
    records=[json.loads(line) for line in (path/"POSES.jsonl").read_text().splitlines()]
    events=json.loads((path/"FRAME_EVENTS.json").read_text())
    return audit_records(manifest,records,events)


def compare_native_pose_records(reference,candidate):
    """Compare saved numeric records without interpreting disagreement as GT error."""
    keys=lambda rows:[(r["frame_id"],r["track_id"]) for r in rows]
    if keys(reference)!=keys(candidate) or len(set(keys(reference)))!=len(reference):
        raise ValueError("Exact paired ordered native domains required")
    rows=[]
    for a,b in zip(reference,candidate):
        if a["source_RGB"]!=b["source_RGB"] or a["pts"]!=b["pts"] or a["raw_baseline_pose"]!=b["raw_baseline_pose"]:
            raise ValueError("Paired run input/parent differs")
        x=np.asarray(a["native_pose"],float);y=np.asarray(b["native_pose"],float)
        if x.shape!=(4,4) or y.shape!=(4,4) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite numeric pose comparison requires explicit unavailable record")
        rows.append(dict(frame_id=b["frame_id"],track_id=b["track_id"],
            matrix_equal=bool(np.array_equal(x,y)),max_matrix_difference=float(abs(x-y).max()),
            candidate_orthogonality_error=float(abs(y[:3,:3].T@y[:3,:3]-np.eye(3)).max()),
            candidate_determinant_error=float(abs(np.linalg.det(y[:3,:3])-1)),
            reference_status=a["status"],candidate_status=b["status"],
            reference_recovery=a.get("recovery_decision",{}).get("action"),
            candidate_recovery=b.get("recovery_decision",{}).get("action")))
    return dict(scope="paired numeric/state disagreement, not GT accuracy",records=len(rows),
        exact_equal=sum(r["matrix_equal"] for r in rows),
        max_orthogonality_error=max((r["candidate_orthogonality_error"] for r in rows),default=None),
        max_determinant_error=max((r["candidate_determinant_error"] for r in rows),default=None),
        candidate_numeric_invalid=sum(r["candidate_status"]=="native_numeric_invalid" for r in rows),rows=rows)
