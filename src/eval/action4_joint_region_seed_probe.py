"""Clear-seed joint region diagnostic. No GT, fitting markers, or final pose publication."""
import json
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT, contained_path
from eval.action4_roi_sweep import read_binding, binding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from model.action4_region_pose_initializers import refine_clear_seed_regions, _project


def load_button_seed(binding_record, *, frame_id, track_id, current_receipt, K_native):
    """Authenticate add_prompt seed pixels, separately from propagated frame0."""
    from eval.action4_sam_full_review import ReceiptVerifier
    from data.action4_sam_roi_inputs import array_digest
    from data.action4_sam_native_export import canonical
    from model.action4_prompt_component_evidence import seed_component_evidence
    seed=read_binding(binding_record["face_seed"])
    request=read_binding(binding_record["prompt_request"])
    snapshot=read_binding(binding_record["native_snapshot"])
    proposal=read_binding(seed["prompt_ref"]); dep=proposal["dependencies"]; body=dep["body"]
    eligibility=read_binding(binding_record["component_review"])
    if (eligibility["frame_id"]!=frame_id or eligibility["track_id"]!=track_id
        or body["track_id"]!=track_id or body["scoped_body_id"]!=seed["body_track_id"]
        or body["generation"]!=seed["body_generation"] or seed["face_id"]!=5
        or seed["global_frame_id"]!=frame_id or dep["frame_id"]!=frame_id
        or seed["pts"]!=current_receipt["pts"] or seed["rgb_sha256"]!=current_receipt["RGB_sha256"]
        or seed["time_base"]!=current_receipt["time_base"]
        or seed["pixel_space_id"]!=current_receipt["pixel_space_id"]
        or not np.array_equal(np.asarray(seed["K_native"]).reshape(3,3),K_native)):
        raise ValueError("Seed part frame/PTS/body/pixel/K ownership mismatch")
    snap=snapshot["snapshot"]
    if (snapshot["identity"]!=canonical(snap) or request["type"]!="add_prompt"
        or request["session_id"]!=snap["session_id"] or request["obj_id"]!=snap["obj_id"]
        or seed["face_generation"]!=snap["generation"]
        or request["frame_index"]!=seed["local_index"] or request["rel_coordinates"] is not True):
        raise ValueError("Seed operation/session/generation mismatch")
    receipt=read_binding(binding_record["raw_receipt"])
    raw=ReceiptVerifier().raw(receipt,contained_path(ROOT,receipt["arrays"],exists=True).parent)
    ids=np.asarray(raw["outputs"]["out_obj_ids"]).tolist()
    if raw["frame_index"]!=seed["local_index"] or ids.count(request["obj_id"])!=1:
        raise ValueError("Native seed response obj/frame mismatch")
    mask=np.asarray(raw["outputs"]["out_binary_masks"])[ids.index(request["obj_id"])].astype(bool)
    points=np.asarray(request["points"])*[mask.shape[1],mask.shape[0]]
    eligible,report=seed_component_evidence(mask,points,request["point_labels"])
    expected=eligibility["component_eligibility"]["button"]
    if report!=expected:raise ValueError("Seed component differs from actual prompt-response pixels")
    path=contained_path(ROOT,binding_record["mask"]["path"],exists=True)
    if binding(path)!=binding_record["mask"]:raise ValueError("Eligible seed mask changed")
    with np.load(path,allow_pickle=False) as z:saved=z["mask"]
    if not np.array_equal(saved,eligible):raise ValueError("Eligible mask differs from replayed prompt component")
    return eligible,dict(frame_id=frame_id,track_id=track_id,pts=seed["pts"],
        session_id=snap["session_id"],generation=snap["generation"],obj_id=snap["obj_id"],
        operation="add_prompt",propagation_frame0_not_substituted=True,
        seed_mask_sha256=array_digest(mask),eligible_mask_sha256=array_digest(eligible),
        physical_boundary_calibrated=False)


def run(config):
    import cv2
    from scipy.spatial.transform import Rotation
    cv2.setNumThreads(1)
    if config["frame_id"] not in config.get("clear_seed_reviewed_frames",[]):
        raise ValueError("Explicit per-frame clear-seed review required; not a generic video tracker")
    lock=read_binding(config["input_lock"])
    pair=read_binding(lock["formal_pair_gate"])
    cad=read_binding(lock["CAD"])
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"],float)
    mesh=contained_path(ROOT,config["mesh"]["path"],exists=True)
    assert binding(mesh)==config["mesh"] and config["mesh"]["sha256"]==cad["mesh_data_sha256"]
    with np.load(mesh,allow_pickle=False) as z:
        vertices=z["vertices_mm"]/1000.; faces=z["faces"]; components=z["face_components"]
        button=vertices[np.unique(faces[components==10])]
    canonical=np.asarray(cad["corners_3d_mm"])/1000.
    provider=ReanchorROIProvider(**lock["video"]["track_replay"])
    masks,receipt=provider.current_native_masks(config["frame_id"])
    source=read_binding(config["pose_source"])
    marker_model=read_binding(config["marker_binding"])
    output=contained_path(ROOT,config["output"],exists=False);output.mkdir()
    rows=[]
    for track,part_binding in config["button_regions"].items():
        button_mask,part_receipt=load_button_seed(part_binding,frame_id=config["frame_id"],
            track_id=track,current_receipt=receipt,K_native=K)
        record=next(r for r in source["records"] if r["frame_id"]==config["frame_id"] and r["track_id"]==track)
        initial=record["A0_raw"]["raw_network_pose"]
        seed=initial
        if config.get("initialization","A0_raw") == "marker_IPPE_0":
            candidate=next(c for c in record["candidates"] if c["solution_id"]==0)
            if not candidate["valid"]:raise ValueError("Invalid estimated marker initializer")
            seed=dict(rotation=candidate["R"],translation_m=candidate["t_m"])
        elif config.get("initialization","A0_raw") != "A0_raw":
            raise ValueError("Unregistered initializer")
        parts=[dict(part_id="whole_body_convex_envelope",vertices_m=vertices,mask=masks[track],full_visibility_seed_review=True),
               dict(part_id="top_button_component10_estimated_boundary",vertices_m=button,mask=button_mask,full_visibility_seed_review=True)]
        marker=next(m for m in marker_model["markers"] if m["id"]==record["marker_id"])
        xyz=np.array([c["xyz_object_mm"] for c in marker["corners"]])/1000.
        group=[]
        if config.get("estimated_marker_group_weight",0)>0:
            group=[dict(points_object_m=xyz,pixels_native=record["observed_marker_corners_native"],
                group_weight=config["estimated_marker_group_weight"],
                XYZ_source="existing_appearance_texture_mapping_estimate",real_metric_calibration=False)]
        fit=refine_clear_seed_regions(K,seed["rotation"],seed["translation_m"],parts,landmark_groups=group,
            cad_diagonal_m=float(np.linalg.norm(np.ptp(canonical,axis=0))),**config["parameters"])
        diagnostic=[]
        for name,R,t in [("A0_raw",np.array(initial["rotation"]),np.array(initial["translation_m"])),
                         ("joint_region_candidate",np.array(fit["R"]),np.array(fit["t_m"]))]:
            per_part={}
            for part in parts:
                uv=_project(part["vertices_m"],R,t,K);render=np.zeros(part["mask"].shape,np.uint8)
                if uv is None:raise ValueError("Nonpositive candidate geometry")
                cv2.fillConvexPoly(render,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
                pred=render.astype(bool);obs=part["mask"]
                per_part[part["part_id"]]={"predicted_region_IoU":float((pred&obs).sum()/max(1,(pred|obs).sum()))}
            corners=_project(canonical,R,t,K)
            marker=next(m for m in marker_model["markers"] if m["id"]==record["marker_id"])
            # Marker is unused when group_weight=0; otherwise it is an estimated fitting observation.
            xyz=np.array([c["xyz_object_mm"] for c in marker["corners"]])/1000.
            mproj=_project(xyz,R,t,K);observed=np.asarray(record["observed_marker_corners_native"])
            diagnostic.append(dict(name=name,R=R.tolist(),t_m=t.tolist(),corners_native=corners.tolist(),
                regions=per_part,estimated_marker_RMS_px=float(np.sqrt(np.mean(np.sum((mproj-observed)**2,axis=1))))))
        rows.append(dict(track_id=track,frame_id=config["frame_id"],fit=fit,diagnostic=diagnostic,
            rotation_change_deg=float(Rotation.from_matrix(np.asarray(fit["R"])@np.asarray(initial["rotation"]).T).magnitude()*180/np.pi),
            part_receipt=part_receipt,initialization=config.get("initialization","A0_raw"),marker_binding_is_estimate=True,marker_used_in_fit=bool(group),real_pose_GT=False,final_corners_replaced=False))
    result=dict(status="TWO_REGION_SEED_DIAGNOSTIC_NOT_SIX_FACE_OR_TRACK_ACCEPTANCE",config=config,
        model_pair=pair,native_body_receipt=receipt,rows=rows,trained=False,new_native_calls=0,
        current_image_parts_only=True,independent_3D_accuracy=False)
    (output/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
