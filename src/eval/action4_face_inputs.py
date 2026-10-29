"""Consume authenticated A0 surface output as face priors, never face ground truth."""
import copy
from dataclasses import dataclass
import numpy as np
from data.action4_surface_snapshot import validate_snapshot,roi_identity
from data.action4_roi_contract import RoiSpec
from model.action4_face_geometry import project_six_faces
from data.action4_video_roi_context import validate_project_body_registration,validate_project_body_scope

@dataclass(frozen=True)
class FaceInputs:
    metadata: dict
    geometry_prior: dict
    xyz_object_m: np.ndarray
    valid_image_support112: np.ndarray
    mask_logits: np.ndarray
    region_logits: np.ndarray

def prepare_face_inputs(snapshot,*,frame,expected_models,expected_prediction_id,roi_spec,contract,
                        min_depth_m,grazing_cos_threshold):
    if not isinstance(roi_spec,RoiSpec):raise TypeError("Actual consumed RoiSpec required")
    expected_roi=roi_identity(frame.image_id,expected_prediction_id,list(roi_spec.full_body_bbox),roi_spec.affine())
    snapshot=validate_snapshot(snapshot,frame=frame,expected_models=expected_models,
        expected_prediction_id=expected_prediction_id,expected_roi_id=expected_roi)
    a,m=snapshot.arrays,snapshot.metadata
    if not np.allclose(a["A448"],roi_spec.affine(),atol=1e-8,rtol=0):
        raise ValueError("Face inputs differ from actual RGB ROI transform")
    geometry=project_six_faces(a["canonical_m"],contract,a["rotation"],a["translation_m"],a["K_native"],
        units="m",pixel_space_id=frame.pixel_space_id,image_size_wh=(frame.pixels.shape[1],frame.pixels.shape[0]),
        min_depth_m=min_depth_m,grazing_cos_threshold=grazing_cos_threshold)
    xyz=a["low_m"].astype(float)[:,None,None]+a["xyz_normalized"].astype(float)*a["extent_m"].astype(float)[:,None,None]
    if not np.isfinite(xyz).all():raise ValueError("Face input XYZ overflow")
    arrays=[xyz,a["valid_image_support112"].copy(),a["mask_logits"].copy(),a["region_logits"].copy()]
    for x in arrays:x.setflags(write=False)
    video=m.get("video_context");body=None
    if video is not None:
        ledger=video["source_ledger"];tid=video["candidate"]["track_id"]
        registrations=ledger.get("project_body_registrations",{})
        if not isinstance(registrations,dict):raise ValueError("Snapshot project body registration map malformed")
        scope=ledger.get("project_body_scope")
        if scope is not None:
            validate_project_body_scope(scope)
            for registered_tid,registration in registrations.items():
                validate_project_body_registration(registration,scope,track_id=registered_tid,
                    frame_id=video["frame_id"],pts=ledger.get("pts"),time_base=ledger.get("time_base"),
                    pixel_space_id=frame.pixel_space_id,replay_identity=video["replay_identity"],
                    native_owner=(ledger.get("native_owners") or {}).get(registered_tid))
        elif registrations:
            raise ValueError("Snapshot body registrations lack sequence scope")
        event=registrations.get(tid)
        body={"track_id":tid,"generation":None,"frame_id":video["frame_id"],
              "sequence_identity":None,"scoped_body_id":None,"persistent_identity_ready":False,
              "association_semantics":"unregistered_unknown_not_semantic_gt","physical_left_right_verified":False}
        if event is not None:
            validate_project_body_registration(event,ledger.get("project_body_scope"),track_id=tid,
                frame_id=video["frame_id"],pts=ledger.get("pts"),time_base=ledger.get("time_base"),
                pixel_space_id=frame.pixel_space_id,replay_identity=video["replay_identity"],
                native_owner=(ledger.get("native_owners") or {}).get(tid))
            body.update(generation=event["project_generation"],sequence_identity=event["sequence_identity"],
                scoped_body_id=event["scoped_body_id"],persistent_identity_ready=True,
                registration_event_identity=event["event_identity"],
                association_semantics=event["semantics"])
    persistent_faces=None
    if body and body["persistent_identity_ready"]:
        from model.action4_sam_face_contract import persistent_face_identity
        persistent_faces=[dict(face_id=i,persistent_face_id=persistent_face_identity(
            body["scoped_body_id"],body["generation"],i),
            semantic_face_verified=False,source="fixed_canonical_topology_with_registered_body_identity")
            for i in range(6)]
    metadata={"persistent_face_entities":persistent_faces,
        "face_session_identity_separate_from_entity":True,
        "schema":"action4.face_inputs.v1","surface_receipt":m["receipt"],
        "models":copy.deepcopy(expected_models),"image_id":m["image_id"],"prediction_id":m["prediction_id"],
        "pixel_space_id":m["pixel_space_id"],"roi_spec":roi_spec.record(),"roi_id":expected_roi,
        "body_identity":body,"persistent_identity_ready":bool(body and body["persistent_identity_ready"]),"source_kind":"A0_current_image_geometry_prior",
        "evidence_group_id":m["receipt"],"independent_observation":False,"semantic_face_labels_available":False,
        "XYZ_semantics":"unclamped model estimate in object metres; physical patch assignment still required",
        "face_masks_available":False,"face_tracking_accepted":False}
    return FaceInputs(metadata,geometry,*arrays)
