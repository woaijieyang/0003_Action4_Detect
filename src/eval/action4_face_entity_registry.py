"""Source-bound persistent entity names; not a claim that a physical face was observed."""
import json,hashlib
from data.action4_paths import PROJECT_ROOT,contained_path
from model.action4_sam_face_contract import FaceSeedBinding,persistent_face_identity,persistent_corner_identity
from model.action4_face_geometry import FaceGeometryContract


def read_binding(ref):
    path=contained_path(PROJECT_ROOT,ref["path"],exists=True);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref["sha256"]:raise ValueError("Entity source changed")
    return raw


def build_registry(seed_refs,contract_ref):
    doc=json.loads(read_binding(contract_ref));asset=read_binding(doc["parent_asset"])
    FaceGeometryContract.from_bytes(read_binding(contract_ref),asset)
    bodies={};sequence=None
    for ref in seed_refs:
        seed=FaceSeedBinding(**json.loads(read_binding(ref)))
        proposal=json.loads(read_binding(seed.prompt_ref));body=proposal["dependencies"]["body"]
        if (body["scoped_body_id"]!=seed.body_track_id or body["generation"]!=seed.body_generation
            or not body["persistent_identity_ready"] or body["frame_id"]!=seed.global_frame_id):
            raise ValueError("Seed and registered body identity differ")
        if sequence is None:sequence=body["sequence_identity"]
        if body["sequence_identity"]!=sequence or body["track_id"] in bodies:
            raise ValueError("Mixed sequences or duplicate body identity")
        bodies[body["track_id"]]=dict(scoped_body_id=seed.body_track_id,body_generation=seed.body_generation,
            source_seed=ref,identity_available_from_frame=seed.global_frame_id,
            faces=[dict(face_id=f["face_id"],persistent_face_id=persistent_face_identity(seed.body_track_id,seed.body_generation,f["face_id"]),
                corner_ids=f["corner_ids"],axis=f["object_axis_name"]) for f in doc["faces"]],
            corners=[dict(corner_id=i,persistent_corner_id=persistent_corner_identity(seed.body_track_id,seed.body_generation,i)) for i in range(8)],
            semantic_face_verified=False,physical_left_right_verified=False)
    if len({b["scoped_body_id"] for b in bodies.values()})!=len(bodies):
        raise ValueError("One body cannot be renamed as two independent instances")
    return dict(schema="action4.persistent_body_face_corner_registry.v1",sequence_identity=sequence,
        contract=contract_ref,bodies=bodies,semantic_face_tracking_accepted=False)


def label_pose_records(records,registry):
    result=[]
    for row in records:
        track=row["track_id"]
        if track not in registry["bodies"]:raise ValueError("Unregistered body")
        body=registry["bodies"][track]
        if row["frame_id"]<body["identity_available_from_frame"]:raise ValueError("Future seed identity used")
        valid=row["status"]=="native_numeric_candidate_not_accuracy"
        faces=[dict(f,observation_status="inferred_from_body_pose" if valid else "unavailable_invalid_pose",
            directly_observed=False,semantic_assignment_verified=False) for f in body["faces"]]
        corners=[dict(c,measurement_kind="body_pose_projection_not_independent_point_measurement" if valid else "unavailable_invalid_pose") for c in body["corners"]]
        result.append(dict(frame_id=row["frame_id"],pts=row["pts"],track_id=track,
            body_generation=body["body_generation"],scoped_body_id=body["scoped_body_id"],
            faces=faces,corners=corners,
            geometry_valid=valid,
            pose_source=dict(path=row["pose_native_file"],sha256=row["pose_native_sha256"]),
            native_recovery_id=(row.get("recovery_decision") or {}).get("recovery_id"),
            face_entities_unchanged_by_session_restart=True,source_status=row["status"],
            physical_face_observation_available=False))
    return result
