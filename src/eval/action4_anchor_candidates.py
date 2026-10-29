"""Stable body/face/corner namespaces for actual A0 pose projections."""
import copy
from dataclasses import dataclass
from eval.action4_face_inputs import FaceInputs

def _body(inputs):
    if not isinstance(inputs,FaceInputs):raise TypeError("Validated FaceInputs required")
    m=inputs.metadata;b=m.get("body_identity")
    if not m.get("persistent_identity_ready") or not b or not b.get("persistent_identity_ready"):
        raise ValueError("Authoritative registered project body identity is not available")
    if type(b.get("generation")) is not int or b["generation"]<0:
        raise ValueError("Explicit project body generation required")
    for k in ("sequence_identity","scoped_body_id","track_id","registration_event_identity"):
        if not isinstance(b.get(k),str) or not b[k]:raise ValueError("Missing body identity field: "+k)
    return b

@dataclass(frozen=True)
class BodyAnchorSlot:
    slot: int
    sequence_identity: str
    scoped_body_id: str
    track_id: str
    generation: int
    registration_event_identity: str

class AnchorIdentityLayout:
    """Caller explicitly binds registered bodies once; never sort by screen position."""
    def __init__(self,slot_inputs):
        if not 1<=len(slot_inputs)<=2:raise ValueError("One/two registered bodies only; do not invent missing body")
        bindings=[]
        for slot,inputs in slot_inputs:
            if type(slot) is not int or slot not in (0,1):raise ValueError("Explicit body slot0/1 required")
            b=_body(inputs)
            bindings.append(BodyAnchorSlot(slot,**{k:b[k] for k in ("sequence_identity","scoped_body_id","track_id","generation","registration_event_identity")}))
        if len({b.slot for b in bindings})!=len(bindings) or len({b.scoped_body_id for b in bindings})!=len(bindings):
            raise ValueError("Duplicate body slot/identity")
        if len({b.sequence_identity for b in bindings})!=1:raise ValueError("Cannot mix independent sequence scopes")
        self.bindings=tuple(bindings)

    def packet(self,inputs):
        b=_body(inputs);found=[x for x in self.bindings if x.scoped_body_id==b["scoped_body_id"]]
        if len(found)!=1:raise ValueError("Body has no registered stable anchor slot")
        slot=found[0]
        if any(getattr(slot,k)!=b[k] for k in ("sequence_identity","track_id","generation","registration_event_identity")):
            raise ValueError("Body identity/generation changed; explicit reidentification required")
        g=inputs.geometry_prior
        if [x["corner_id"] for x in g["corners"]]!=list(range(8)) or [x["face_id"] for x in g["faces"]]!=list(range(6)):
            raise ValueError("Canonical local IDs changed")
        faces=[];corners=[]
        for f in g["faces"]:
            faces.append(dict(face_id=slot.slot*6+f["face_id"],local_face_id=f["face_id"],
                corner_ids=[slot.slot*8+k for k in f["corner_ids"]],
                geometric_front_facing=f["geometric_front_facing"],
                visibility_state="geometry_candidate_only",occlusion_state="unknown",
                observation_source="A0_pose_projection",direct_observation=False,probability=None))
        for c in g["corners"]:
            incident=[f["face_id"] for f in faces if slot.slot*8+c["corner_id"] in f["corner_ids"]]
            if len(incident)!=3:raise ValueError("Cuboid corner must belong to exactly3 fixed faces")
            corners.append(dict(corner_id=slot.slot*8+c["corner_id"],local_corner_id=c["corner_id"],
                incident_face_ids=incident,native_uv=copy.deepcopy(c["native_uv"]),
                positive_depth=c["positive_depth"],visibility_state="unknown",
                observation_source="rigid_pose_projection",direct_observation=False,probability=None,
                evidence_group_id=inputs.metadata["surface_receipt"]))
        return dict(schema="action4.fixed_anchor_candidates.v1",body_slot=slot.slot,body_identity=copy.deepcopy(b),
            face_id_namespace="two_bodies_times6",corner_id_namespace="two_bodies_times8",
            physical_left_right_verified=False,faces=faces,corners=corners,
            pose_candidates=[dict(candidate_id="A0_raw",R=copy.deepcopy(g["R"]),t_m=copy.deepcopy(g["t_m"]),
                source="current_image_A0_pose",corner_ids=[c["corner_id"] for c in corners],
                image_probability=None,corner_probabilities_available=False)],
            candidate_count=1,history_reranking_applied=False,
            history_reranking_reason="single_pose_path_has_no_per_corner_multipeak_probabilities",
            evidence_group_id=inputs.metadata["surface_receipt"],independent_corner_evidence_count=0,
            source_models=copy.deepcopy(inputs.metadata["models"]),roi_id=inputs.metadata["roi_id"],
            new_accuracy_or_track_acceptance=False)
