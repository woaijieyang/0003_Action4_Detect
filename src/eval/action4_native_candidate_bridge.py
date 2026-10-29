"""Attach saved native tracking candidates to the existing ROI-video stream.

This consumer never runs a model, selects a new pose, or changes raw/final output.
Only an explicitly admitted later stage may promote a candidate.
"""
import copy,json,hashlib
import numpy as np
from data.action4_paths import PROJECT_ROOT,contained_path


def _read(ref):
    p=contained_path(PROJECT_ROOT,ref["path"],exists=True)
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref["sha256"]:raise ValueError("Native candidate dependency changed")
    return raw


def _geometry_recipe(pair,lock,asset):
    """Two explicit producer contracts; never invent missing ROI/geometry defaults."""
    if all(k in pair for k in ("roi_expansion","CAD","camera")):
        if pair["CAD"]!=lock["CAD"] or pair["camera"]!=lock["video"]["K"]:
            raise ValueError("Pair camera/CAD source differs")
        return pair["roi_expansion"],dict(kind="legacy_pair_direct_geometry_refs",CAD=pair["CAD"],camera=pair["camera"])
    if pair.get("status") not in ("SEALED_USER_STOPPED_STAGE4_FIXED_FOR_STAGE5","SEALED_COMPLETED75_STAGE4_FIXED_FOR_STAGE5"):
        raise ValueError("Unknown model-pair geometry contract")
    trial=json.loads(_read(pair["trial_selection"]));sealed=json.loads(_read(pair["strict_reload"]))
    from eval.action4_video_roi_contract import trial_ratio
    selected_ratio=trial_ratio(trial,sealed)
    expected_status="SEALED_COMPLETED75_STAGE4_FIXED_FOR_STAGE5" if selected_ratio==1.15 else "SEALED_USER_STOPPED_STAGE4_FIXED_FOR_STAGE5"
    if (pair.get("status")!=expected_status or sealed.get("schema")!="action4.sealed_corner_all8_research.v1" or
        trial["sealed_research"]!=pair["strict_reload"] or pair["models"]["A0"]["sealed_research"]!=pair["strict_reload"] or
        trial["selected"]["step"]!=pair["models"]["A0"]["selected_step"] or trial["selected"]["checkpoint"]["sha256"]!=pair["models"]["A0"]["sha256"] or
        sealed["sha256"]!=pair["models"]["A0"]["sha256"] or sealed["selected_step"]!=pair["models"]["A0"]["selected_step"] or sealed["config_id"]!=pair["models"]["A0"]["config_id"] or
        any(sealed["model_B0"][k]!=pair["models"]["B0"][k] for k in ("path","sha256","selected_step","config_id")) or
        trial["B0"]!=pair["models"]["B0"]["path"]):
        raise ValueError("Sealed pair/selection model identities differ")
    roi=selected_ratio
    if isinstance(roi,bool) or not isinstance(roi,(int,float)) or not np.isfinite(roi) or roi<=0 or sealed["roi_expansion"]!=roi:
        raise ValueError("Sealed selection ROI recipe differs")
    canonical=np.asarray(asset["corners_3d_mm"],float)/1000.
    if canonical.shape!=(8,3) or not np.array_equal(canonical,np.asarray(sealed["canonical8_m"],float)):
        raise ValueError("Sealed canonical8 differs from locked CAD")
    return roi,dict(kind="sealed_research_and_bound_trial_selection",ROI_source=pair["trial_selection"],
        canonical_source=pair["strict_reload"],CAD=lock["CAD"],camera=lock["video"]["K"],
        camera_consistency="locked_K_equals_native_manifest_and_selected_final_corner_projection")


def join_candidate_rows(base_rows,native_rows,native_manifest):
    from eval.action4_native_tracking_audit import native_input_contract,recovery_contract
    kind,pose_key,corners_key=native_input_contract(native_manifest);mode=recovery_contract(native_manifest)
    expected=[(f["frame_id"],t) for f in native_manifest["rows"] for t in native_manifest["tracks"]]
    actual=[(r["frame_id"],r["track_id"]) for r in native_rows]
    if actual!=expected or len(set(actual))!=len(actual):raise ValueError("Incomplete native frame/instance domain")
    by_frame={r["frame_id"]:r for r in base_rows}
    if len(by_frame)!=len(base_rows):raise ValueError("Duplicate baseline frame")
    frame_manifest={r["frame_id"]:r for r in native_manifest["rows"]}
    grouped={}
    for row in native_rows:
        fid=row["frame_id"];track=row["track_id"]
        if fid not in by_frame:raise ValueError("Native candidate outside baseline domain")
        base=by_frame[fid];frame=frame_manifest[fid]
        if (base["pts"]!=row["pts"] or row["pts"]!=frame["pts"]
            or base["RGB_sha256"]!=row["source_RGB"]["sha256"]
            or row["source_RGB"]!=frame["source_png"] or row["time_base"]!=frame["time_base"]):
            raise ValueError("Native/baseline current frame timing or pixels differ")
        matches=[i for i in base["prediction"]["instances"] if i["track_id"]==track]
        if len(matches)!=1:raise ValueError("Native body has no unique baseline instance")
        inst=matches[0]
        if (row["raw_baseline_pose"]!=inst["raw_network_pose"]
            or row["raw_baseline_corners"]!=inst["corners_raw_from_resolved_roi"]
            or row["final_corners"]!=inst["corners_final"]
            or row["source_B0_box"]!=inst["bbox_b0_raw"] or row["published"] is not False):
            raise ValueError("Native candidate was derived from a different actual A0 ROI result")
        if row.get("native_pose_input_kind","raw_A0")!=kind:raise ValueError("Native candidate input route differs")
        if kind=="selected_A0":
            targets=[x for x in frame["instances"] if x["track_id"]==track]
            if len(targets)!=1:raise ValueError("Selected native input identity ambiguous")
            target=targets[0];selected=target[pose_key];final=inst.get("final_pose",{})
            if (row["final_corners"]!=target[corners_key] or row.get("selected_A0_input_pose")!=selected or
                target["raw_A0_pose"]!=inst["raw_network_pose"] or target["raw_corners"]!=inst["corners_raw_from_resolved_roi"] or
                any(selected.get(k)!=final.get(k) for k in ("coordinate_frame","rotation","translation_m"))):
                raise ValueError("Selected candidate does not bind authoritative final and preserved raw sources")
        decision=row.get("recovery_decision")
        eligible=(row["status"]=="native_numeric_candidate_not_accuracy" and row["native_call_ok"] is True
            and (decision is None or decision["action"] not in ("withhold","record_only_no_reseed")))
        candidate=dict(schema="action4.native_tracking_candidate.v1",track_id=track,frame_id=fid,pts=row["pts"],
            current_SAM_consumed_by_texture=row["SAM_consumed_by_native"],
            native_pose_raw=copy.deepcopy(row["native_pose"]),numeric_status=row["status"],
            numerical_error=row.get("numeric_error"),corners_candidate=copy.deepcopy(row.get("corners_candidate")),
            numerical_candidate_eligible=eligible,recovery_decision=copy.deepcopy(decision),
            deployment_admitted=False,final_output_replaced=False,native_pose_input_kind=kind,source_input_pose_field=pose_key,
            source_baseline_roi_decision_id=inst["roi_decision_id"])
        grouped[fid,track]=candidate
    result=copy.deepcopy(base_rows)
    for frame in result:
        for inst in frame["prediction"]["instances"]:
            key=(frame["frame_id"],inst["track_id"])
            if key in grouped:
                inst["native_tracking_candidate"]=grouped[key]
            else:
                inst["native_tracking_candidate"]=dict(status="outside_registered_candidate_domain",
                    deployment_admitted=False,final_output_replaced=False)
    return result


def attach_saved_native_candidates(base_path,binding,output):
    request=json.loads(_read(binding))
    if request.get("schema")!="action4.native_tracking_candidate_attachment.v1" or request.get("promotion_allowed") is not False:
        raise ValueError("Explicit nonpromoting candidate attachment required")
    path=contained_path(PROJECT_ROOT,base_path,exists=True)
    expected=contained_path(PROJECT_ROOT,request["baseline"]["path"],exists=True)
    if path!=expected:raise ValueError("Attachment targets another baseline stream")
    base=[json.loads(l) for l in _read(request["baseline"]).decode().splitlines()]
    native=[json.loads(l) for l in _read(request["native_results"]).decode().splitlines()]
    manifest=json.loads(_read(request["native_inputs"]))
    events=json.loads(_read(request["native_events"]))
    from eval.action4_native_tracking_audit import audit_records
    audit=audit_records(manifest,native,events)
    # Bind the declared selected models even though this consumer loads no weights.
    lock=json.loads(_read(request["input_lock"]))
    pair=json.loads(_read(lock["formal_pair_gate"]))
    if request["model_pair_sha256"]!=lock["formal_pair_gate"]["sha256"]:
        raise ValueError("Candidate model pair identity differs")
    camera=json.loads(_read(lock["video"]["K"]))
    asset=json.loads(_read(lock["CAD"]))
    roi_expansion,pair_geometry_contract=_geometry_recipe(pair,lock,asset)
    if manifest.get("baseline_predictions",request["baseline"])!=request["baseline"] or manifest.get("input_lock",request["input_lock"])!=request["input_lock"]:
        raise ValueError("Native manifest baseline/input-lock identity differs")
    for frame in base:
        identity=frame["prediction"]["identity"]
        for role in ("A0","B0"):
            if (identity[role+"_step"]!=pair["models"][role]["selected_step"]
                or identity[role+"_config_id"]!=pair["models"][role]["config_id"]):
                raise ValueError("Baseline actual model metadata differs from selected pair")
        if frame["roi_expansion"]!=roi_expansion:raise ValueError("Baseline ROI recipe differs")
    K=np.asarray(camera["K_native"],float);cad=np.asarray(asset["corners_3d_mm"],float)/1000.
    by_frame={f["frame_id"]:f for f in manifest["rows"]}
    for row in native:
        if not np.array_equal(K,np.asarray(by_frame[row["frame_id"]]["K"])):
            raise ValueError("Native and baseline camera domain differs")
        if row["status"]=="native_numeric_candidate_not_accuracy":
            T=np.asarray(row.get("converted_pose",row["native_pose"]),float)
            if (T.shape!=(4,4) or not np.isfinite(T).all()
                or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6,rtol=0)
                or abs(np.linalg.det(T[:3,:3])-1)>1e-6):
                raise ValueError("Candidate claims numeric validity for a nonrigid pose")
            xyz=cad@T[:3,:3].T+T[:3,3]
            if not (xyz[:,2]>1e-6).all():raise ValueError("Candidate depth invalid")
            h=xyz@K.T;uv=h[:,:2]/h[:,2:]
            observed=np.asarray(row["corners_candidate"],float)
            if observed.shape!=(8,2) or not np.isfinite(observed).all() or np.linalg.norm(uv-observed,axis=1).max()>1e-4:
                raise ValueError("Candidate fixed-ID corners do not match its actual pose")
    from eval.action4_native_tracking_audit import native_input_contract
    kind,pose_key,corners_key=native_input_contract(manifest)
    if kind=="selected_A0":
        from model.action4_final_pose_record import resolve_final_pose
        base_by_frame={r["frame_id"]:r for r in base}
        for frame in manifest["rows"]:
            by_track={x["track_id"]:x for x in base_by_frame[frame["frame_id"]]["prediction"]["instances"]}
            for target in frame["instances"]:
                resolved,resolution=resolve_final_pose(by_track[target["track_id"]],cad,K)
                if target[pose_key]!=resolved or target[corners_key]!=by_track[target["track_id"]]["corners_final"]:
                    raise ValueError("Prepared selected pose differs from verified authoritative final")
    joined=join_candidate_rows(base,native,manifest)
    out=contained_path(PROJECT_ROOT,output,exists=False);out.mkdir(exist_ok=False,parents=True)
    target=out/"predictions.jsonl"
    with target.open("x") as f:
        for row in joined:f.write(json.dumps(row,allow_nan=False)+chr(10))
    receipt=dict(status="NATIVE_CANDIDATE_ATTACHED_RAW_FINAL_UNCHANGED",frames=len(joined),
        native_records=len(native),audit=audit,attachment=copy.deepcopy(binding),
        predictions=dict(path=str(target.relative_to(PROJECT_ROOT)),sha256=hashlib.sha256(target.read_bytes()).hexdigest()),
        model_pair_sha256=request["model_pair_sha256"],pair_geometry_contract=pair_geometry_contract,model_calls=0,
        final_output_replaced=False,accuracy_accepted=False,native_pose_input_kind=kind,raw_baseline_preserved=True)
    (out/"COMPLETE.json").write_text(json.dumps(receipt,indent=2)+chr(10))
    return receipt
