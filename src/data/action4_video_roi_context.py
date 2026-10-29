"""Immutable video-only ROI transport. No SAM mask is treated as a full box."""
from dataclasses import dataclass,field
import hashlib
import json
import numpy as np

def rgb_digest(rgb):
    a=rgb.detach().cpu().numpy() if hasattr(rgb,"detach") else np.asarray(rgb)
    if a.ndim!=3 or a.shape[0]!=3 or a.dtype!=np.uint8:
        raise ValueError("Native CHW uint8 RGB required")
    return hashlib.sha256(str(a.shape).encode()+b"\0"+a.tobytes(order="C")).hexdigest()



_SAM_ADMITTED=object()

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or value=="0"*64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Explicit SAM evidence SHA256 required")

@dataclass(frozen=True)
class VideoSAMSupportReceipt:
    """Caller-authoritative current ledger binding, not a class/quality certificate."""
    frame_id:int
    pts:int
    time_base:tuple
    rgb_identity:str
    pixel_space_id:str
    track_id:str
    native_obj_id:int
    session_id:str
    track_generation:int
    mask_generation:int
    replay_identity:str
    source_sha256:str
    ledger_sha256:str
    mask_sha256:str
    bbox_full_xyxy:tuple

@dataclass(frozen=True)
class VideoSAMSupportAdmission:
    """Explicit opt-in: expected ledger receipts plus independently supplied mask responses.

    Caller must authenticate the current frame/track/session ledger and source.
    Equality of self-declared objects alone is not physical identity proof.
    """
    pixel_space_id:str
    time_base:tuple
    source_sha256:str
    ledger_sha256:str
    expected:tuple
    predictions:tuple  # pairs (actual VideoSAMSupportReceipt, native bool mask)

def _admit_support(admission,row,rgb,K,replay_identity):
    if admission is None:return {}
    if not isinstance(admission,VideoSAMSupportAdmission):raise ValueError("Typed authoritative SAM support admission required")
    for value in (admission.source_sha256,admission.ledger_sha256):_sha(value)
    if not isinstance(admission.pixel_space_id,str) or not admission.pixel_space_id:
        raise ValueError("Native pixel space required")
    if (type(admission.time_base) is not tuple or len(admission.time_base)!=2
            or any(type(v) is not int or v<=0 for v in admission.time_base)):
        raise ValueError("Explicit source time base required")
    if type(admission.expected) is not tuple or type(admission.predictions) is not tuple:
        raise ValueError("Immutable receipt lists required")
    from dataclasses import asdict
    from data.action4_sam_roi_inputs import array_digest
    native=rgb.detach().cpu().numpy() if hasattr(rgb,"detach") else np.asarray(rgb)
    rid=rgb_digest(rgb);frame=row["frame"];tracks={x["track_id"]:x for x in row["fused"]["tracks"]}
    if type(frame.get("global_frame_id")) is not int or type(frame.get("pts")) is not int:
        raise ValueError("Current SAM frame and PTS must be integers")
    if ("pixel_space_id" in row and row["pixel_space_id"]!=admission.pixel_space_id
            or "time_base" in row and tuple(row["time_base"])!=admission.time_base):
        raise ValueError("Replay pixel/time domain contradicts SAM admission")
    expected={}
    for receipt in admission.expected:
        if not isinstance(receipt,VideoSAMSupportReceipt) or receipt.track_id in expected:
            raise ValueError("Duplicate or invalid expected SAM receipt")
        expected[receipt.track_id]=receipt
    supplied={};native_ids=set();mask_ids=set()
    for pair in admission.predictions:
        if not isinstance(pair,tuple) or len(pair)!=2:raise ValueError("Receipt/native-mask pair required")
        receipt,mask=pair
        if not isinstance(receipt,VideoSAMSupportReceipt) or receipt.track_id in supplied:
            raise ValueError("Duplicate or invalid SAM response receipt")
        if receipt.track_id not in expected or receipt!=expected[receipt.track_id]:
            raise ValueError("Stale/swapped generation, identity or source receipt")
        for v in (receipt.rgb_identity,receipt.replay_identity,receipt.source_sha256,receipt.ledger_sha256,receipt.mask_sha256):_sha(v)
        for v in (receipt.frame_id,receipt.native_obj_id,receipt.track_generation,receipt.mask_generation):
            if type(v) is not int or v<0:raise ValueError("Invalid native/frame/generation identity")
        if not all(isinstance(v,str) and v.strip() for v in (receipt.track_id,receipt.session_id,receipt.pixel_space_id)):
            raise ValueError("Explicit project/native session identity required")
        if (receipt.frame_id!=frame["global_frame_id"] or type(receipt.pts) is not int or receipt.pts!=frame.get("pts")
                or receipt.time_base!=admission.time_base or receipt.rgb_identity!=rid
                or receipt.pixel_space_id!=admission.pixel_space_id or receipt.replay_identity!=replay_identity
                or receipt.source_sha256!=admission.source_sha256 or receipt.ledger_sha256!=admission.ledger_sha256):
            raise ValueError("SAM source/frame/RGB/pixel-space/replay binding differs")
        if ("session_id" in row and row["session_id"]!=receipt.session_id
                or "generation" in row and row["generation"]!=receipt.mask_generation):
            raise ValueError("Current replay session/mask generation differs")
        track=tracks.get(receipt.track_id)
        if (track is None or track.get("status")!="sam_supported" or track.get("sam_current_support") is not True
                or track.get("candidate_id") is not None or track.get("raw_detection") is not None
                or track.get("support_rejection") is not None or receipt.track_id in row.get("native_unavailable",[])):
            raise ValueError("Only current SAM-supported state without B0 observation may use this opt-in")
        box=np.asarray(track["bbox_for_roi"],float)
        if (type(receipt.bbox_full_xyxy) is not tuple or box.shape!=(4,) or not np.isfinite(box).all()
                or not (box[2:]>box[:2]).all() or not np.array_equal(box,np.asarray(receipt.bbox_full_xyxy,float))):
            raise ValueError("Estimated body ROI differs from authoritative receipt")
        key=(receipt.session_id,receipt.native_obj_id)
        if key in native_ids or receipt.mask_sha256 in mask_ids:raise ValueError("Duplicate native ID/shared identical SAM support")
        native_ids.add(key);mask_ids.add(receipt.mask_sha256)
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=native.shape[1:]:
            raise ValueError("SAM mask must match native RGB dimensions and bool dtype")
        owned=mask.copy()
        if not owned.any() or array_digest(owned)!=receipt.mask_sha256:
            raise ValueError("Empty/changed SAM support mask")
        payload={"schema":"action4.video_sam_geometry_admission.v1","receipt":asdict(receipt),
            "mask_pixels":int(owned.sum()),"K_native":K.reshape(-1).tolist(),
            "mask_quality_certified":False,"geometry_accuracy_verified":False}
        supplied[receipt.track_id]={"payload":payload,"identity":_digest(payload)}
    if set(supplied)!=set(expected):raise ValueError("Expected SAM support response missing")
    return supplied

def validate_project_body_scope(scope):
    if not isinstance(scope,dict) or set(scope)!={"schema","clip_id","region_identity","producer_identity","sequence_identity"}:
        raise ValueError("Explicit project body sequence scope required")
    if scope["schema"]!="action4.project_body_scope.v1" or not isinstance(scope["clip_id"],str) or not scope["clip_id"]:
        raise ValueError("Invalid project body sequence scope")
    for key in ("region_identity","producer_identity","sequence_identity"):_sha(scope[key])
    expected_scope=_digest({k:scope[k] for k in ("clip_id","region_identity","producer_identity")})
    if scope["sequence_identity"]!=expected_scope:
        raise ValueError("Project body sequence identity changed")
    return scope


def validate_project_body_registration(event, scope, *, track_id, frame_id, pts,
                                       time_base, pixel_space_id, replay_identity, native_owner=None):
    """Validate explicit initial project identity, independently of native generation.

    Caller authenticates replay/native receipts. Hashes bind this transported
    registration; they are not semantic or physical object truth certificates.
    """
    validate_project_body_scope(scope)
    keys={"schema","event_type","sequence_identity","producer_identity","replay_identity",
          "track_id","scoped_body_id","project_generation","creation_global_frame","creation_pts",
          "time_base","pixel_space_id","seed_fingerprint","seed_binding_ref","seeds_ref",
          "semantics","semantic_gt","physical_left_right_verified","event_identity"}
    if not isinstance(event,dict) or set(event)!=keys:
        raise ValueError("Complete explicit project body creation event required")
    if (event["schema"]!="action4.project_body_registration.v1" or event["event_type"]!="project_body_created"
            or event["semantics"]!="provisional_seed_association_not_semantic_ground_truth"
            or event["semantic_gt"] is not False or event["physical_left_right_verified"] is not False
            or type(event["project_generation"]) is not int
            or event["project_generation"]!=0):
        raise ValueError("Only explicit initial project creation is currently supported")
    if (event["event_identity"]!=_digest({k:v for k,v in event.items() if k!="event_identity"})
            or event["sequence_identity"]!=scope["sequence_identity"]
            or event["producer_identity"]!=scope["producer_identity"] or event["replay_identity"]!=replay_identity
            or event["track_id"]!=track_id):
        raise ValueError("Project body registration source/track identity differs")
    if event["scoped_body_id"]!=_digest({"sequence_identity":scope["sequence_identity"],"track_id":track_id}):
        raise ValueError("Scoped project body ID differs")
    for key in ("creation_global_frame","creation_pts"):
        if type(event[key]) is not int:raise ValueError("Integer body creation frame/PTS required")
    if (event["creation_global_frame"]<0 or type(frame_id) is not int or type(pts) is not int
            or event["creation_global_frame"]>frame_id or event["creation_pts"]>pts
            or event["time_base"]!=list(time_base) or event["pixel_space_id"]!=pixel_space_id):
        raise ValueError("Future creation or body frame/time/pixel-domain mismatch")
    _sha(event["seed_fingerprint"])
    for key in ("seed_binding_ref","seeds_ref"):
        ref=event[key]
        if not isinstance(ref,dict) or set(ref)!={"path","sha256"} or not isinstance(ref["path"],str) or not ref["path"]:
            raise ValueError("Explicit bound seed source reference required")
        _sha(ref["sha256"])
    if not isinstance(native_owner,dict) or native_owner.get("project_seed")!={
            "seed_fingerprint":event["seed_fingerprint"],"creation_global_frame":event["creation_global_frame"],
            "creation_pts":event["creation_pts"],"seed_binding_ref":event["seed_binding_ref"],"seeds_ref":event["seeds_ref"]}:
        raise ValueError("Project creation differs from authenticated seed owner")
    return event


@dataclass(frozen=True)
class VideoROIContext:
    frame_id:int
    rgb_identity:str
    b0_sha256:str
    replay_identity:str
    K_native:tuple
    detections_json:str
    withheld_json:str
    sam_support_json:str|None=None
    source_ledger_json:str|None=None
    _sam_admitted:object=field(default=None,init=False,repr=False,compare=False)
    _project_ledger_identity:str|None=field(default=None,init=False,repr=False,compare=False)

    @classmethod
    def from_replay(cls,row,rgb,K,*,b0_sha256,replay_identity,sam_support=None):
        frame_id=row["frame"]["global_frame_id"]
        if row["fused"]["mode"]!="video_inference" or row["fused"]["frame_id"]!=frame_id:
            raise ValueError("Video replay/frame mismatch")
        K=np.asarray(K,float)
        if K.shape!=(3,3) or not np.isfinite(K).all():raise ValueError("Finite native K required")
        for identity in [b0_sha256,replay_identity]:
            if len(identity)!=64 or any(x not in "0123456789abcdef" for x in identity):
                raise ValueError("Explicit model and replay identities required")
        admitted=_admit_support(sam_support,row,rgb,K,replay_identity)
        dets=[];withheld=[];ids=set()
        for track in row["fused"]["tracks"]:
            tid=track["track_id"]
            if tid in ids:raise ValueError("Duplicate track identity")
            ids.add(tid)
            if track["status"]=="sam_supported" and tid in admitted:
                evidence=admitted[tid];box=list(evidence["payload"]["receipt"]["bbox_full_xyxy"])
                dets.append({"prediction_id":f"video:{tid}:frame:{frame_id}","track_id":tid,
                    "source_detection_ids":[],"score":None,"grid_index":None,
                    "bbox_full_xyxy":box,"raw_bbox_full_xyxy":None,
                    "origin":"SAM_tracking_geometry","box_origin":"SAM_tracking_geometry",
                    "B0_current_observation":False,"sam_role":"geometry_support_only",
                    "temporal_status":"sam_supported","full_box_estimated":True,
                    "full_box_quality_calibrated":False,"full_box_accuracy_verified":False,
                    "sam_support_admission_id":evidence["identity"],"sam_support_evidence":evidence["payload"]})
                continue
            if track["status"]!="observed":
                withheld.append({"track_id":tid,"status":track["status"],"reason":"no_admitted_current_observation_for_A0"})
                continue
            raw=track["raw_detection"];box=np.asarray(track["bbox_for_roi"],float)
            if box.shape!=(4,) or not np.isfinite(box).all() or not (box[2:]>box[:2]).all():
                raise ValueError("Invalid smoothed full-body ROI")
            expected=f"b0:frame{frame_id}:grid:{raw['grid_index']}"
            if track["candidate_id"]!=expected or raw["candidate_id"]!=expected:
                raise ValueError("Candidate frame/grid mismatch")
            if not np.isfinite(raw["score"]) or not .3<=raw["score"]<=1:
                raise ValueError("Current detector score required")
            dets.append({"prediction_id":f"video:{tid}:frame:{frame_id}","track_id":tid,
                "source_detection_ids":[expected],"score":raw["score"],"grid_index":raw["grid_index"],
                "bbox_full_xyxy":box.tolist(),"raw_bbox_full_xyxy":raw["bbox_full_xyxy"],
                "box_origin":"video_history_smoothed_B0_full_box","sam_role":"association_only",
                "temporal_status":"observed","full_box_quality_calibrated":False})
        visible=dict(row.get("_roi_visible_boxes",{}))
        if sam_support is not None:
            for receipt,mask in sam_support.predictions:
                ys,xs=np.nonzero(mask)
                box=None if not len(xs) else [float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]
                if receipt.track_id in visible and visible[receipt.track_id]!=box:raise ValueError("Current visible-box evidence differs")
                visible[receipt.track_id]=box
        trackmap={t["track_id"]:t for t in row["fused"]["tracks"]}
        for d in dets:
            t=trackmap[d["track_id"]];last=row.get("_roi_last_trusted_full",{}).get(d["track_id"],{})
            d.update(bbox_b0_raw=d.get("raw_bbox_full_xyxy"),bbox_sam_visible=visible.get(d["track_id"]),
                bbox_for_a0=list(d["bbox_full_xyxy"]),roi_source=d.get("box_origin",d.get("origin")),
                frame_id=frame_id,pts=row["frame"].get("pts"),time_base=row.get("time_base"),
                current_support_status={"B0":t["status"]=="observed","SAM":t.get("sam_current_support")},
                last_trusted_full_frame_id=last.get("frame_id"),last_trusted_full_pts=last.get("pts"),
                last_prediction_calibration=row.get("_roi_last_prediction_calibration",{}).get(d["track_id"]),

                roi_accept_reason=t.get("source","current_B0_observation" if t["status"]=="observed" else "admitted_current_SAM_support"),
                sam_reject_reason=t.get("support_rejection"),roi_reject_reason=None,
                bbox_sam_visible_convention="native_pixel_index_half_open_not_amodal")
            d["roi_decision_id"]=_digest({k:d[k] for k in ("track_id","frame_id","bbox_b0_raw","bbox_sam_visible","bbox_for_a0","roi_source","current_support_status","last_trusted_full_frame_id","last_trusted_full_pts","last_prediction_calibration")})
        ledger={"frame_id":frame_id,"pts":row["frame"].get("pts"),
                "native_session_id":row.get("session_id"),"mask_generation":row.get("generation"),
                "native_status":row.get("native_status"),"native_owners":row.get("native_owners"),
                "project_track_generation":None}
        if "time_base" in row:ledger["time_base"]=row["time_base"]
        scope=row.get("project_body_scope")
        registrations=row.get("project_body_registrations",{})
        if not isinstance(registrations,dict):raise ValueError("Project body registration map required")
        if scope is not None or registrations:
            validate_project_body_scope(scope)
            for tid,event in registrations.items():
                if tid not in trackmap:raise ValueError("Body registration belongs to another frame track")
                validate_project_body_registration(event,scope,track_id=tid,frame_id=frame_id,pts=row["frame"].get("pts"),
                    time_base=row.get("time_base"),pixel_space_id=row.get("pixel_space_id"),replay_identity=replay_identity,
                    native_owner=(row.get("native_owners") or {}).get(tid))
            ledger.update(project_body_scope=scope,project_body_registrations=registrations,
                          time_base=row.get("time_base"),pixel_space_id=row.get("pixel_space_id"))
        result=cls(frame_id,rgb_digest(rgb),b0_sha256,replay_identity,tuple(K.reshape(-1)),
                   json.dumps(dets,allow_nan=False),json.dumps(withheld,allow_nan=False),
                   json.dumps(admitted,allow_nan=False) if admitted else None,
                   json.dumps(ledger,allow_nan=False))
        if scope is not None:object.__setattr__(result,"_project_ledger_identity",_digest(ledger))
        if admitted:object.__setattr__(result,"_sam_admitted",_SAM_ADMITTED)
        return result

    def consume(self,rgb,K,*,b0_sha256):
        if rgb_digest(rgb)!=self.rgb_identity:raise ValueError("Video ROI RGB differs")
        if b0_sha256!=self.b0_sha256:raise ValueError("Video ROI B0 model differs")
        if not np.array_equal(np.asarray(K,float).reshape(-1),np.asarray(self.K_native)):
            raise ValueError("Video ROI native K differs")
        rows=json.loads(self.detections_json)
        self._check_support_rows(rows)
        return rows

    def _check_support_rows(self,rows):
        ledger=json.loads(self.source_ledger_json) if self.source_ledger_json is not None else {}
        if ledger.get("project_body_scope") is not None:
            if self._project_ledger_identity!=_digest(ledger):
                raise ValueError("Project body source ledger changed or lacks local admission")
            for tid,event in ledger.get("project_body_registrations",{}).items():
                validate_project_body_registration(event,ledger["project_body_scope"],track_id=tid,
                    frame_id=self.frame_id,pts=ledger.get("pts"),time_base=ledger.get("time_base"),
                    pixel_space_id=ledger.get("pixel_space_id"),replay_identity=self.replay_identity,
                    native_owner=(ledger.get("native_owners") or {}).get(tid))
        for p in rows:
            if p.get("frame_id")!=self.frame_id or p.get("bbox_for_a0")!=p.get("bbox_full_xyxy") or p.get("bbox_b0_raw")!=p.get("raw_bbox_full_xyxy"):
                raise ValueError("Resolved A0 ROI fields changed")
            expected=_digest({k:p[k] for k in ("track_id","frame_id","bbox_b0_raw","bbox_sam_visible","bbox_for_a0","roi_source","current_support_status","last_trusted_full_frame_id","last_trusted_full_pts","last_prediction_calibration")})
            if p.get("roi_decision_id")!=expected:raise ValueError("ROI decision identity changed")
        admitted=json.loads(self.sam_support_json) if self.sam_support_json is not None else {}
        supported=[p for p in rows if p.get("temporal_status")=="sam_supported"]
        if supported and self._sam_admitted is not _SAM_ADMITTED:
            raise ValueError("SAM support context requires local authoritative admission")
        if {p["track_id"] for p in supported}!=set(admitted):raise ValueError("Support admission/row coverage differs")
        for p in supported:
            entry=admitted[p["track_id"]];payload=entry["payload"];receipt=payload["receipt"]
            if (entry["identity"]!=_digest(payload) or p.get("sam_support_admission_id")!=entry["identity"] or p.get("sam_support_evidence")!=payload
                    or p.get("score") is not None or p.get("grid_index") is not None
                    or p.get("source_detection_ids")!=[] or p.get("raw_bbox_full_xyxy") is not None
                    or p.get("origin")!="SAM_tracking_geometry" or p.get("B0_current_observation") is not False
                    or receipt["bbox_full_xyxy"]!=p["bbox_full_xyxy"] or receipt["frame_id"]!=self.frame_id
                    or receipt["rgb_identity"]!=self.rgb_identity or receipt["replay_identity"]!=self.replay_identity
                    or payload["K_native"]!=list(self.K_native)):
                raise ValueError("SAM geometry/B0-absence evidence changed")

    def select_for_a0(self,rows):
        """Video-only selection; a missing B0 score is never invented."""
        if rows!=json.loads(self.detections_json):raise ValueError("Changed video ROI candidates")
        self._check_support_rows(rows)
        selected=[]
        for p in rows:
            status=p.get("temporal_status")
            if status=="sam_supported":selected.append(p)
            elif status in (None,"observed"):
                if p["score"]>=.3:selected.append(p)
            else:raise ValueError("Predicted/lost temporal candidates cannot reach A0")
        return selected
