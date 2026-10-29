"""Bounded native-part sequence diagnostic; failures retained, no mainline publication."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT, contained_path
from data.action4_sam_native_export import canonical, _pixel_hash
from eval.action4_roi_sweep import read_binding, binding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_sam_full_review import ReceiptVerifier
from model.action4_planar_landmark_pose import planar_pose_candidates
from model.action4_region_pose_initializers import refine_clear_seed_regions, _project


def json_lines(ref):
    path=ReceiptVerifier().verify(ref["path"],ref["sha256"])
    return [json.loads(x) for x in path.open() if x.strip()]


def current_part(ref, fid, track, current, K, *, expected_face_id=5):
    # Identity/source checks only; no learned adapter is loaded.
    if type(expected_face_id) is not int or not 0<=expected_face_id<6:raise ValueError("Explicit canonical face hypothesis ID required")
    snap=read_binding(ref["snapshot"]);seed=read_binding(ref["seed"])
    dep=read_binding(seed["prompt_ref"])["dependencies"]; body=dep["body"]
    s=snap["snapshot"]
    if (snap["identity"]!=canonical(s) or body["track_id"]!=track
        or body["scoped_body_id"]!=seed["body_track_id"] or seed["face_id"]!=expected_face_id
        or seed["face_generation"]!=s["generation"]
        or not np.array_equal(np.asarray(seed["K_native"]).reshape(3,3),K)):
        raise ValueError("Part seed/session/body/geometry identity differs")
    frames=[f for f in s["frames"] if f["global_frame_id"]==fid]
    if len(frames)!=1:raise ValueError("Requested current part frame absent or duplicate")
    f=frames[0]
    if f["pts"]!=current["pts"] or seed["time_base"]!=current["time_base"]:
        raise ValueError("Part/body PTS domain differs")
    if not f["available"]:
        return None,dict(status=f["status"],reason=f["reason"],frame_id=fid,track_id=track)
    reg=snap["registrations"][f["mask_identity"]];b=reg["binding"]
    if (b["global_frame_id"]!=fid or b["local_index"]!=f["local_index"] or b["pts"]!=f["pts"]
        or b["session_id"]!=s["session_id"] or b["obj_id"]!=s["obj_id"]
        or b["generation"]!=s["generation"] or b["track_id"]!=s["track_id"]
        or b["region_identity"]!=s["region_identity"]
        or b["pixel_identity"]!=current["pixel_space_id"]
        or reg["binding_identity"]!=canonical(b)):
        raise ValueError("Current part registration frame/owner/session differs")
    path=ReceiptVerifier().verify(reg["path"],reg["sha256"])
    with np.load(path,allow_pickle=False) as z:mask=z["mask"].copy()
    if mask.dtype!=np.bool_ or mask.ndim!=2 or _pixel_hash(mask)!=reg["pixel_sha256"]:
        raise ValueError("Registered current mask pixels differ")
    if f["mask_identity"]!="sam-mask-sha256:"+canonical({"binding":reg["binding_identity"],"pixels":reg["pixel_sha256"]}):
        raise ValueError("Native mask identity differs")
    rr_path=path.parent/"RAW_RECEIPT.json";raw=ReceiptVerifier().raw(json.loads(rr_path.read_text()),path.parent)
    ids=np.asarray(raw["outputs"]["out_obj_ids"]).tolist()
    if raw["frame_index"]!=f["local_index"] or ids.count(s["obj_id"])!=1:
        raise ValueError("Raw propagated response frame/object differs")
    if not np.array_equal(mask,np.asarray(raw["outputs"]["out_binary_masks"])[ids.index(s["obj_id"])]):
        raise ValueError("Registered mask differs from raw propagated response")
    return mask,dict(status="current_propagated_part_candidate",frame_id=fid,track_id=track,pts=f["pts"],
        binding=b,mask_identity=f["mask_identity"],mask_ref={"path":reg["path"],"sha256":reg["sha256"]},
        seed_ref=ref["seed"],snapshot_ref=ref["snapshot"],physical_full_face_verified=False)


def prepare(cfg):
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir();(out/"inputs").mkdir()
    lock=read_binding(cfg["input_lock"]);K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    cad=read_binding(lock["CAD"]);canonical8=np.asarray(cad["corners_3d_mm"])/1000.
    mesh=ReceiptVerifier().verify(cfg["mesh"]["path"],cfg["mesh"]["sha256"])
    if cad["mesh_data_sha256"]!=cfg["mesh"]["sha256"]:raise ValueError("CAD version mismatch")
    with np.load(mesh,allow_pickle=False) as z:
        vertices=z["vertices_mm"]/1000.;button=vertices[np.unique(z["faces"][z["face_components"]==10])]
    saved={r["frame_id"]:r for r in json_lines(cfg["predictions"])}
    markers={r["frame_id"]:r for r in json_lines(cfg["marker_associations"])}
    marker_model=read_binding(cfg["marker_binding"])
    if marker_model["detection_variant_id"]!=cfg["detection_variant_id"]:raise ValueError("Marker detector binding mismatch")
    review=read_binding(cfg["review"]);reviewed={(r["frame_id"],r["track_id"]):r for r in review["per_instance"]}
    provider=ReanchorROIProvider(**lock["video"]["track_replay"]);jobs=[]
    for fid in cfg["frame_ids"]:
        masks,current=provider.current_native_masks(fid);row=saved[fid];features=markers[fid]
        if row["RGB_sha256"]!=current["RGB_sha256"] or row["pts"]!=current["pts"] or features["source_RGB_sha256"]!=row["RGB_sha256"]:
            raise ValueError("Current RGB/PTS evidence mismatch")
        for track,ref in cfg["parts"].items():
            job=dict(frame_id=fid,pts=row["pts"],time_base=current["time_base"],track_id=track,
                     status="prepared",K_native=K.tolist(),parameters=cfg["parameters"],
                     canonical8=canonical8.tolist(),image_review=reviewed[(fid,track)],
                     use_recent_history=bool(cfg.get("recent_history_initialization",False)),
                     body_generation=read_binding(ref["seed"])["body_generation"])
            raw=next(p for p in row["prediction"]["instances"] if p["track_id"]==track)
            job["A0_raw"]=raw;job["body_receipt"]=current
            if cfg.get("region_observation_review") is not None:job["region_observation_review"]=cfg["region_observation_review"]
            try:
                part,part_receipt=current_part(ref,fid,track,current,K);job["part_receipt"]=part_receipt
                reviewed_parts=reviewed[(fid,track)]
                body_ok=reviewed_parts.get("body_boundary_assumption_allowed",reviewed_parts["full_region_assumption_allowed"])
                button_ok=reviewed_parts.get("button_boundary_assumption_allowed",reviewed_parts["full_region_assumption_allowed"])
                job["whole_body_boundary_eligible"]=body_ok
                job["excluded_observations"]=[] if body_ok else [dict(role="whole_body_boundary",reason=reviewed_parts["reason"])]
                if part is None:job.update(status="unavailable",reason=part_receipt["status"])
                elif not button_ok:job.update(status="unavailable",reason="button_boundary_assumption_unavailable")
                else:
                    obs=[o for o in features["observations"] if o["track_id"]==track and o["status"]=="assigned_current_image_evidence"]
                    if len(obs)!=1:job.update(status="unavailable",reason="current_marker_missing_or_ambiguous")
                    else:
                        m=next(m for m in marker_model["markers"] if m["id"]==obs[0]["marker_id"])
                        xyz=np.asarray([c["xyz_object_m"] for c in m["corners"]]);uv=np.asarray(obs[0]["corners_native"])
                        body_geometry=masks[track]
                        if cfg.get("body_component_association") is not None:
                            import cv2
                            from data.action4_region_support import associate_supported_component
                            marker_mask=np.zeros(part.shape,np.uint8)
                            cv2.fillConvexPoly(marker_mask,np.rint(uv).astype(np.int32),1)
                            associated,association=associate_supported_component(masks[track],
                                {"current_button":part,"current_marker":marker_mask.astype(bool)},
                                **cfg["body_component_association"])
                            job["body_component_association"]=association
                            if associated is None:
                                job["whole_body_boundary_eligible"]=False
                                job["excluded_observations"].append(dict(role="whole_body_boundary",reason=association["reason"]))
                            else:body_geometry=associated
                        pp=planar_pose_candidates(xyz,uv,K,canonical8)
                        initial=next((p for p in pp["candidates"] if p["solution_id"]==0 and p["valid"]),None)
                        if initial is None:job.update(status="unavailable",reason="IPPE0_unavailable")
                        else:
                            # Bounding crop only translates pixel domain; no resize or mask clipping.
                            yy,xx=np.nonzero(body_geometry|part);l=max(0,int(xx.min())-8);t=max(0,int(yy.min())-8)
                            r=min(part.shape[1],int(xx.max())+9);b=min(part.shape[0],int(yy.max())+9)
                            A=np.array([[1.,0,-l],[0,1.,-t],[0,0,1.]])
                            file=out/"inputs"/(str(fid)+"_"+track+".npz")
                            np.savez_compressed(file,body=body_geometry[t:b,l:r],button=part[t:b,l:r],
                                vertices=vertices,button_vertices=button,K=A@K,marker_xyz=xyz,marker_uv=uv-[l,t])
                            job.update(input=binding(file),initial_pose=initial,crop_native=[l,t,r,b],marker_observation=obs[0])
            except Exception as e:job.update(status="input_error",reason=type(e).__name__+":"+str(e))
            jobs.append(job)
    (out/"JOBS.json").write_text(json.dumps(jobs,indent=2,allow_nan=False));return jobs


def region_observation_identity(job,part):
    """Exact per-part working pixels, CAD and original source ownership."""
    from data.action4_sam_roi_inputs import array_digest
    receipt=job["part_receipt"]
    return dict(frame_id=job["frame_id"],track_id=job["track_id"],pts=job["pts"],
        time_base=job["time_base"],RGB_sha256=job["body_receipt"]["RGB_sha256"],
        body_generation=job.get("body_generation"),input=job["input"],crop_native=job["crop_native"],
        part_id=part["part_id"],mask_sha256=array_digest(part["mask"]),
        geometry_sha256=array_digest(np.asarray(part["vertices_m"])),
        eligibility_sha256=None if part.get("observed_boundary_eligibility") is None else array_digest(part["observed_boundary_eligibility"]),
        seed_ref=receipt["seed_ref"],snapshot_ref=receipt["snapshot_ref"],
        part_mask_ref=receipt["mask_ref"],part_mask_identity=receipt["mask_identity"],
        body_raw=job["body_receipt"]["raw"])


def resolve_region_observations(job,parts):
    """Only source-bound per-part review licenses full physical boundaries.

    Missing semantic evidence remains an explicitly unverified one-way research
    observation; it is never promoted to a complete physical boundary. Neither
    numerical fitting nor this review admits a formal pose decision.
    """
    allowed=[];states=[];review_ref=job.get("region_observation_review")
    review=None
    if review_ref is not None:
        review=read_binding(review_ref)
        if review.get("schema")!="action4.region_observation_review.v1":
            raise ValueError("Unregistered physical-boundary review schema")
        rows=review.get("parts",[])
        if not isinstance(rows,list) or len({r.get("part_id") for r in rows})!=len(rows):
            raise ValueError("Duplicate reviewed part identities")
        # Source evidence files must exist and retain their reviewed bytes.
        sources=review.get("source_evidence",[])
        if not sources:raise ValueError("Physical-boundary review requires independent observation evidence")
        inherited={job["input"]["path"]}|{job["part_receipt"][key]["path"] for key in ("seed_ref","snapshot_ref","mask_ref")}
        if any(ref["path"] in inherited for ref in sources):
            raise ValueError("Seed/mask provenance alone cannot certify physical boundary")
        for ref in sources:ReceiptVerifier().verify(ref["path"],ref["sha256"])
    identities=set()
    for part in parts:
        identity=region_observation_identity(job,part)
        # Independent labels cannot manufacture two observations from one mask.
        if identity["mask_sha256"] in identities:raise ValueError("Duplicated part mask is not independent observation")
        identities.add(identity["mask_sha256"])
        entry=None if review is None else next((r for r in review["parts"] if r["part_id"]==part["part_id"]),None)
        status=dict(part_id=part["part_id"],identity=identity,review_ref=review_ref,
                    formal_decision_allowed=False,full_boundary_verified=False)
        licensed=False
        if entry is not None:
            if entry.get("identity")!=identity:raise ValueError("Reviewed part frame/body/mask/CAD/source identity differs")
            # Verify referenced original source objects, not copied job booleans.
            for key in ("seed_ref","snapshot_ref","part_mask_ref"):
                ref=identity[key];ReceiptVerifier().verify(ref["path"],ref["sha256"])
            raw=identity["body_raw"]
            ReceiptVerifier().verify(raw["arrays"],raw["sha256"])
            ReceiptVerifier().verify(raw["header"],raw["header_sha256"])
            if (entry.get("semantic_correspondence_verified") is True
                    and entry.get("physical_boundary_verified") is True
                    and entry.get("full_visibility_verified") is True):
                licensed=True
        if licensed:
            configured=dict(part,full_visibility_seed_review=True,partial_boundary=False)
            status.update(status="source_reviewed_full_boundary",reason="exact_current_part_review",full_boundary_verified=True)
            allowed.append(configured)
        elif part.get("partial_boundary") is True and part.get("observed_boundary_eligibility") is not None:
            allowed.append(dict(part,full_visibility_seed_review=False))
            status.update(status="research_partial_available",reason="eligible_original_boundary_not_certified_physical_silhouette")
        else:
            # Existing raw region evidence remains usable as a one-way research
            # hypothesis. All-true eligibility selects ORIGINAL contour pixels;
            # it neither fills/rewrites masks nor asserts a physical silhouette.
            allowed.append(dict(part,full_visibility_seed_review=False,partial_boundary=True,
                                observed_boundary_eligibility=np.ones_like(part["mask"],dtype=bool)))
            status.update(status="research_partial_unverified_semantics",
                          reason="raw_region_hypothesis_not_complete_physical_boundary")
        states.append(status)
    return allowed,dict(schema="action4.region_observation_decision.v1",parts=states,
        all_requested_parts_available=len(allowed)==len(parts),formal_decision_allowed=False,
        geometric_accuracy_verified=False)


def fit_track(args):
    jobs,cpu=args;os.sched_setaffinity(0,{cpu});results=[];history=[];active_generation=None
    from scipy.spatial.transform import Rotation
    from model.action4_region_pose_initializers import recent_pose_initializer, select_clear_seed_region_fit
    for job in jobs:
        result=dict(job);result["published"]=False;result["final_corners"]=job["A0_raw"]["corners_raw_from_resolved_roi"]
        if job["status"]!="prepared":results.append(result);continue
        try:
            if job.get("use_recent_history",False):
                generation=job["body_generation"]
                if active_generation is not None and generation<active_generation:raise ValueError("Stale body generation")
                if generation!=active_generation:
                    history=[];result["history_generation_reset"]=True;active_generation=generation
            path=ReceiptVerifier().verify(job["input"]["path"],job["input"]["sha256"])
            with np.load(path,allow_pickle=False) as z:
                bounds=job.get("crop_native")
                if not isinstance(bounds,list) or len(bounds)!=4 or any(type(v) is not int for v in bounds):
                    raise ValueError("Explicit integer computational crop required")
                left,top,right,bottom=bounds
                if right<=left or bottom<=top or z["body"].shape!=(bottom-top,right-left):
                    raise ValueError("Mask grid differs from computational crop")
                expected_K=np.array([[1.,0,-left],[0,1.,-top],[0,0,1.]])@np.asarray(job["K_native"],float)
                if not np.allclose(z["K"],expected_K,atol=1e-8,rtol=0):raise ValueError("Computational crop K differs from native translation")
                role=job.get("observed_part_role","button")
                part_vertices=z["part_vertices"] if "part_vertices" in z.files else z["button_vertices"]
                part_mask=z["part_mask"] if "part_mask" in z.files else z["button"]
                if part_mask.shape!=z["body"].shape:raise ValueError("Current part/body mask pixel domains differ")
                normal=np.asarray(job.get("observed_normal_object",[0.,0.,1.]),float)
                if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9:raise ValueError("Observed part normal required")
                parts=[dict(part_id="body",vertices_m=z["vertices"],mask=z["body"],full_visibility_seed_review=False),
                       dict(part_id=role,vertices_m=part_vertices,mask=part_mask,full_visibility_seed_review=False)]
                if "body_edge_eligibility" in z.files:
                    parts[0].update(partial_boundary=True,observed_boundary_eligibility=z["body_edge_eligibility"],full_visibility_seed_review=False)
                if not job.get("whole_body_boundary_eligible",True):parts=[parts[1]]
                parts,decision=resolve_region_observations(job,parts)
                result["region_observation_decision"]=decision
                result["formal_decision_allowed"]=False
                result["consumed_region_roles"]=[]
                if not decision["all_requested_parts_available"]:
                    result.update(status="observation_unavailable",reason="source_bound_region_observation_incomplete",
                                  solver_called=False,raw_fallback_retained=True)
                    results.append(result)
                    continue
                result["consumed_region_roles"]=[p["part_id"] for p in parts]
                result["solver_called"]=True
                groups=[]
                if "marker_xyz" in z.files:
                    groups=[dict(points_object_m=z["marker_xyz"],pixels_native=z["marker_uv"],group_weight=1.,
                        XYZ_source="existing_appearance_texture_mapping_estimate",real_metric_calibration=False)]
                initial=job["initial_pose"];c=np.asarray(job["canonical8"])
                fit=refine_clear_seed_regions(z["K"],initial["R"],initial["t_m"],parts,
                    cad_diagonal_m=float(np.linalg.norm(np.ptp(c,axis=0))),landmark_groups=groups,**job["parameters"])
                result["history_initialization"]={"enabled":bool(job.get("use_recent_history",False)),"selected":False}
                if job.get("use_recent_history",False):
                    prior=recent_pose_initializer(history,track_id=job["track_id"],generation=job["body_generation"],
                        target_pts=job["pts"],time_base=job["time_base"])
                    result["history_initialization"]["prior"]=prior
                    if prior["status"]=="initializer_only":
                        x=np.r_[Rotation.from_matrix(np.asarray(prior["R"])@np.asarray(initial["R"]).T).as_rotvec(),
                            (np.asarray(prior["t_m"])-initial["t_m"])/np.linalg.norm(np.ptp(c,axis=0))]
                        bounds=np.r_[np.full(3,job["parameters"]["rotation_limit_rad"]),np.full(3,job["parameters"]["translation_limit_cad"])]
                        if (np.abs(x)<bounds).all():
                            other=refine_clear_seed_regions(z["K"],initial["R"],initial["t_m"],parts,
                                cad_diagonal_m=float(np.linalg.norm(np.ptp(c,axis=0))),landmark_groups=groups,
                                initial_increment=x,**job["parameters"])
                            result["history_initialization"].update(fresh_fit=fit,history_fit=other,
                                score_basis="same_current_observation_cost; no history residual, GT, or image error labels")
                            selection=select_clear_seed_region_fit(fit,other,c,parts[-1]["vertices_m"],observed_normal_object=normal)
                            result["history_initialization"]["selection_gates"]={k:v for k,v in selection.items() if k!="chosen"}
                            fit=selection["chosen"];result["history_initialization"]["selected"]=selection["history_selected"]
                        else:result["history_initialization"]["reason"]="outside_unchanged_current_search_domain"
            R=np.asarray(fit["R"]);t=np.asarray(fit["t_m"]);camera=-R.T@t
            # Directly observed top button must face capture camera; diagnostic gate only.
            part_facing=float(np.dot(camera-parts[-1]["vertices_m"].mean(0),normal))>0
            result.update(status="candidate",candidate_scope="research_only_not_formal_decision",fit=fit,observed_part_faces_camera=part_facing,
                observed_part_role=role,observed_normal_object=normal.tolist(),
                top_part_faces_camera=part_facing if role=="button" else None,
                corners_candidate=_project(c,R,t,np.asarray(job["K_native"])).tolist(),
                numeric_unbounded_candidate=bool(fit["optimizer_success"] and not fit["at_component_bound"] and part_facing))
            if job.get("use_recent_history",False) and result["numeric_unbounded_candidate"]:
                history.append(dict(track_id=job["track_id"],generation=job["body_generation"],pts=job["pts"],
                    R=fit["R"],t_m=fit["t_m"],observation_id=job["part_receipt"]["mask_identity"],
                    current_image_supported=True,prediction_only=False))
                history=history[-5:]
        except Exception as e:result.update(status="fit_error",reason=type(e).__name__+":"+str(e))
        results.append(result)
    return results


def run(cfg):
    from scipy.spatial.transform import Rotation
    from collections import Counter
    jobs=prepare(cfg);out=contained_path(ROOT,cfg["output"],exists=True)
    cpus=cfg["cpus"]
    if len(cpus)!=len(cfg["parts"]) or len(set(cpus))!=len(cpus) or not set(cpus)<=set(range(24,96)):raise ValueError("CPU budget")
    work=[([j for j in jobs if j["track_id"]==t],cpu) for t,cpu in zip(cfg["parts"],cpus)]
    with ProcessPoolExecutor(max_workers=len(work)) as pool:results=[r for group in pool.map(fit_track,work) for r in group]
    results.sort(key=lambda r:(r["frame_id"],r["track_id"]))
    with (out/"RESULTS.jsonl").open("x") as f:
        for r in results:f.write(json.dumps(r,allow_nan=False)+"\n")
    per_track={}
    for track in cfg["parts"]:
        rows=[r for r in results if r["track_id"]==track];changes=[]
        for a,b in zip(rows,rows[1:]):
            if a["status"]==b["status"]=="candidate":
                rawR0=np.asarray(a["A0_raw"]["raw_network_pose"]["rotation"]);rawR1=np.asarray(b["A0_raw"]["raw_network_pose"]["rotation"])
                Ra=np.asarray(a["fit"]["R"]);Rb=np.asarray(b["fit"]["R"])
                changes.append(dict(from_frame=a["frame_id"],to_frame=b["frame_id"],
                    raw_rotation_change_deg=float(Rotation.from_matrix(rawR1@rawR0.T).magnitude()*180/np.pi),
                    candidate_rotation_change_deg=float(Rotation.from_matrix(Rb@Ra.T).magnitude()*180/np.pi)))
        per_track[track]=dict(requested=len(rows),statuses=dict(Counter(r["status"] for r in rows)),
            bounds=sum(r.get("fit",{}).get("at_component_bound",False) for r in rows),
            optimizer_failure=sum(not r["fit"]["optimizer_success"] for r in rows if "fit" in r),
            numeric_unbounded_candidates=sum(r.get("numeric_unbounded_candidate",False) for r in rows),
            history_selected=sum(r.get("history_initialization",{}).get("selected",False) for r in rows),
            max_history_count=max([r.get("history_initialization",{}).get("prior",{}).get("history_count",0) for r in rows]),
            adjacent_pose_changes_diagnostic_only=changes)
    report=dict(status="CURRENT_PROPAGATED_PART_SEQUENCE_DIAGNOSTIC_NOT_ACCEPTED",requested=len(jobs),
        recorded=len(results),per_track=per_track,published=0,no_smoothing=True,history_initialization=bool(cfg.get("recent_history_initialization",False)),history_loss_prior=False,
        current_part_masks_consumed=True,full_video_accuracy_verified=False,training_executed=False,new_SAM_calls=0)
    (out/"COMPLETE.json").write_text(json.dumps(report,indent=2));return report


def prepare_front_region_probe(cfg):
    """Saved native lens-like region + current A0; no rear-marker fallback.

    CAD component13 is a declared front-cover hypothesis, not certified image
    boundary GT. Other-body missing native support stays in the request domain.
    """
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir();(out/"inputs").mkdir()
    lock=read_binding(cfg["input_lock"]);K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    cad=read_binding(lock["CAD"]);canonical8=np.asarray(cad["corners_3d_mm"])/1000.
    mesh=ReceiptVerifier().verify(cfg["mesh"]["path"],cfg["mesh"]["sha256"])
    if cfg["mesh"]["sha256"]!=cad["mesh_data_sha256"]:raise ValueError("Physical component CAD changed")
    if cfg["part_component_ids"]!=[13] or cfg["observed_normal_object"]!=[0,-1,0]:
        raise ValueError("Explicit frozen front-cover hypothesis required")
    with np.load(mesh,allow_pickle=False) as z:
        vertices=z["vertices_mm"]/1000.;part_vertices=vertices[np.unique(z["faces"][np.isin(z["face_components"],cfg["part_component_ids"])])]
    saved={r["frame_id"]:r for r in json_lines(cfg["predictions"])}
    provider=ReanchorROIProvider(**lock["video"]["track_replay"]);jobs=[]
    for fid in cfg["frame_ids"]:
        masks,current=provider.current_native_masks(fid);row=saved[fid]
        if row["RGB_sha256"]!=current["RGB_sha256"] or row["pts"]!=current["pts"]:raise ValueError("Current image mismatch")
        for track in cfg["expected_tracks"]:
            raw=next(p for p in row["prediction"]["instances"] if p["track_id"]==track)
            job=dict(frame_id=fid,pts=row["pts"],time_base=current["time_base"],track_id=track,status="unavailable",
                reason="no_registered_current_part_hypothesis",A0_raw=raw,K_native=K.tolist(),canonical8=canonical8.tolist(),
                parameters=cfg["parameters"],body_receipt=current,use_recent_history=True,
                observed_part_role="front_lens_cover_component13_hypothesis",observed_normal_object=cfg["observed_normal_object"],
                physical_boundary_calibrated=False,initializer_source="current_best_A0_raw_not_rear_marker")
            if cfg.get("region_observation_review") is not None:job["region_observation_review"]=cfg["region_observation_review"]
            if track not in cfg["parts"]:jobs.append(job);continue
            try:
                ref=cfg["parts"][track];part,part_receipt=current_part(ref,fid,track,current,K,expected_face_id=2)
                job.update(part_receipt=part_receipt,body_generation=read_binding(ref["seed"])["body_generation"])
                if part is None:job["reason"]=part_receipt["status"]
                elif not raw["raw_network_pose"]["finite"]:job["reason"]="A0_initializer_nonfinite"
                else:
                    yy,xx=np.nonzero(masks[track]|part);l=max(0,int(xx.min())-8);t=max(0,int(yy.min())-8)
                    r=min(part.shape[1],int(xx.max())+9);b=min(part.shape[0],int(yy.max())+9)
                    A=np.array([[1.,0,-l],[0,1.,-t],[0,0,1.]])
                    file=out/"inputs"/(str(fid)+"_"+track+".npz")
                    np.savez_compressed(file,body=masks[track][t:b,l:r],part_mask=part[t:b,l:r],
                        vertices=vertices,part_vertices=part_vertices,K=A@K)
                    job.update(status="prepared",reason=None,input=binding(file),whole_body_boundary_eligible=True,
                        clear_region_assumption="limited inspected front-view development candidate; not an automatic occlusion proof",
                        initial_pose={"R":raw["raw_network_pose"]["rotation"],"t_m":raw["raw_network_pose"]["translation_m"]},
                        crop_native=[l,t,r,b],excluded_observations=[dict(role="rear_marker",reason="front-view profile does not consume rear correspondence")])
            except Exception as e:job.update(status="input_error",reason=type(e).__name__+":"+str(e))
            jobs.append(job)
    (out/"JOBS.json").write_text(json.dumps(jobs,indent=2));return jobs
