"""Prediction-side one-to-one association; no models, GT, track births or ROI publication."""
from dataclasses import dataclass
import math
import numpy as np
from model.action4_guidance_contract import FrameIdentity,SourceIdentity,box,name,probability

def vector(value):
    a=np.asarray(value,dtype=float)
    if a.shape!=(128,) or not np.isfinite(a).all() or np.linalg.norm(a)<=1e-12:
        raise ValueError("Available finite nonzero B0 appearance128 required")
    a=a/np.linalg.norm(a)
    return tuple(map(float,a))

def mask(value,frame):
    if value is None:return None
    a=np.asarray(value)
    if a.dtype!=np.bool_ or a.shape!=(frame.height,frame.width) or not a.any():
        raise ValueError("Native nonempty boolean predicted mask required")
    return np.frombuffer(a.tobytes(),dtype=bool).reshape(a.shape)

@dataclass(frozen=True)
class TrackEvidence:
    frame: FrameIdentity
    track_id: str
    generation: int
    support_box: tuple
    appearance128: tuple
    diagonal_px: float
    last_observed_frame: int
    last_observed_seconds: float
    support_origin: str
    support_id: str
    identity_confirmed: bool
    appearance_source: SourceIdentity
    predicted_mask: object = None
    def __post_init__(self):
        if not isinstance(self.frame,FrameIdentity):raise ValueError("Typed current frame required")
        if not isinstance(self.appearance_source,SourceIdentity) or self.appearance_source.producer!="b0" or self.appearance_source.weight_identity_kind!="checkpoint_sha256":
            raise ValueError("Frozen B0 appearance source identity required")
        name(self.track_id,"track ID");name(self.support_id,"support source")
        if type(self.generation) is not int or self.generation<0 or type(self.identity_confirmed) is not bool:
            raise ValueError("Explicit generation/confirmation required")
        if self.support_origin not in ("sam_prediction","history_prediction"):raise ValueError("Prediction-only support origin required")
        if type(self.last_observed_frame) is not int or not 0<=self.last_observed_frame<=self.frame.frame_id:
            raise ValueError("Invalid observation frame")
        if not math.isfinite(self.last_observed_seconds) or not math.isfinite(self.diagonal_px) or self.diagonal_px<=0:
            raise ValueError("Finite time and frozen full-object scale required")
        if self.support_origin=="history_prediction" and self.predicted_mask is not None:
            raise ValueError("History box cannot masquerade as current SAM mask")
        if self.support_origin=="sam_prediction" and self.predicted_mask is None:
            raise ValueError("SAM support requires its actual current mask")
        object.__setattr__(self,"support_box",box(self.support_box))
        object.__setattr__(self,"appearance128",None if self.appearance128 is None else vector(self.appearance128))
        object.__setattr__(self,"predicted_mask",mask(self.predicted_mask,self.frame))

@dataclass(frozen=True)
class CandidateEvidence:
    frame: FrameIdentity
    candidate_id: str
    support_box: tuple
    appearance128: tuple
    detector_score: float
    association_eligible: bool
    support_id: str
    appearance_source: SourceIdentity
    predicted_mask: object = None
    def __post_init__(self):
        if not isinstance(self.frame,FrameIdentity):raise ValueError("Typed current frame required")
        if not isinstance(self.appearance_source,SourceIdentity) or self.appearance_source.producer!="b0" or self.appearance_source.weight_identity_kind!="checkpoint_sha256":
            raise ValueError("Frozen B0 appearance source identity required")
        name(self.candidate_id,"candidate ID");name(self.support_id,"support source")
        if type(self.association_eligible) is not bool:raise ValueError("Explicit eligibility required")
        probability(self.detector_score,"detector score")
        object.__setattr__(self,"support_box",box(self.support_box))
        object.__setattr__(self,"appearance128",None if self.appearance128 is None else vector(self.appearance128))
        object.__setattr__(self,"predicted_mask",mask(self.predicted_mask,self.frame))

def iou_box(a,b):
    a=np.asarray(a);b=np.asarray(b)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection))

def associate(frame,seconds,tracks,candidates):
    """Confirmed-track input is caller-validated; does not claim automatic identity initialization."""
    if not isinstance(frame,FrameIdentity) or isinstance(seconds,(bool,np.bool_)) or not math.isfinite(seconds) or seconds<0:raise ValueError("Current frame/time required")
    tracks=tuple(tracks);candidates=tuple(candidates)
    if any(not isinstance(t,TrackEvidence) for t in tracks) or any(not isinstance(c,CandidateEvidence) for c in candidates):
        raise ValueError("Typed association evidence required")
    tracks=sorted(tracks,key=lambda t:t.track_id);candidates=sorted(candidates,key=lambda c:c.candidate_id)
    if len(tracks)>8 or len(candidates)>200:raise ValueError("Explicit association budget exceeded; no silent truncation")
    if any(not isinstance(t,TrackEvidence) or t.frame!=frame for t in tracks) or any(not isinstance(c,CandidateEvidence) or c.frame!=frame for c in candidates):
        raise ValueError("Association RGB/frame/pixel identity mismatch")
    if len({t.track_id for t in tracks})!=len(tracks) or len({c.candidate_id for c in candidates})!=len(candidates):
        raise ValueError("Duplicate track/candidate identity")
    blocked={};edges={};trace=[]
    for i,t in enumerate(tracks):
        elapsed=seconds-t.last_observed_seconds
        if elapsed<0:raise ValueError("Observation time is in the future")
        if not t.identity_confirmed:blocked[i]="unconfirmed_identity"
        elif t.appearance128 is None:blocked[i]="appearance_unavailable"
        elif frame.frame_id-t.last_observed_frame>=15 or elapsed>=.5:blocked[i]="lost"
    for i,t in enumerate(tracks):
        if t.predicted_mask is None:continue
        for j,u in enumerate(tracks[:i]):
            if u.predicted_mask is not None and (t.support_id==u.support_id or np.array_equal(t.predicted_mask,u.predicted_mask)):
                blocked[i]=blocked[j]="ambiguous_shared_mask"
    for i,t in enumerate(tracks):
        if i in blocked:continue
        for j,c in enumerate(candidates):
            if not c.association_eligible:continue
            if t.appearance_source!=c.appearance_source:
                trace.append({"track_id":t.track_id,"candidate_id":c.candidate_id,"rejected":"appearance_source_mismatch"})
                continue
            if c.appearance128 is None:
                trace.append({"track_id":t.track_id,"candidate_id":c.candidate_id,"rejected":"appearance_unavailable"})
                continue
            center_a=(np.array(t.support_box[:2])+t.support_box[2:])/2
            center_b=(np.array(c.support_box[:2])+c.support_box[2:])/2
            motion=float(np.linalg.norm(center_b-center_a)/t.diagonal_px)
            if motion>.75:
                trace.append({"track_id":t.track_id,"candidate_id":c.candidate_id,"rejected":"motion_gate","motion_ratio":motion})
                continue
            if t.predicted_mask is not None and c.predicted_mask is not None:
                overlap=float(np.logical_and(t.predicted_mask,c.predicted_mask).sum()/np.logical_or(t.predicted_mask,c.predicted_mask).sum())
                basis="mask"
            else:overlap=iou_box(t.support_box,c.support_box);basis="local_box"
            cosine=float(np.clip(np.dot(t.appearance128,c.appearance128),-1,1))
            cost=.5*(1-overlap)+.3*min(motion,1)+.2*(1-cosine)
            edge={"track_id":t.track_id,"candidate_id":c.candidate_id,"cost":cost,"support_iou":overlap,
                  "motion_ratio":motion,"appearance_cosine":cosine,"match_basis":basis,
                  "support_origin":t.support_origin,"track_support_id":t.support_id,
                  "candidate_support_id":c.support_id,"detector_score":c.detector_score}
            edges[i,j]=edge;trace.append(edge)
    for i in range(len(tracks)):
        costs=sorted(e["cost"] for (ti,_),e in edges.items() if ti==i)
        if len(costs)>1 and costs[1]-costs[0]<.15:blocked[i]="ambiguous_candidate_margin"
    # A shared candidate with indistinguishable track evidence is not assigned by array order.
    ambiguous_candidates=set()
    for j,c in enumerate(candidates):
        if c.predicted_mask is None:continue
        for k,other in enumerate(candidates[:j]):
            if other.predicted_mask is not None and (c.support_id==other.support_id or np.array_equal(c.predicted_mask,other.predicted_mask)):
                ambiguous_candidates.update((j,k))
    for j in range(len(candidates)):
        costs=sorted(e["cost"] for (i,cj),e in edges.items() if cj==j and i not in blocked)
        if len(costs)>1 and costs[1]-costs[0]<.15:ambiguous_candidates.add(j)
    states={0:(0.,())}
    for j in range(len(candidates)):
        if j in ambiguous_candidates:continue
        nxt=dict(states)
        for used,(cost,pairs) in states.items():
            for i in range(len(tracks)):
                if i in blocked or used&(1<<i) or (i,j) not in edges:continue
                key=used|(1<<i);value=(cost+edges[i,j]["cost"],tuple(sorted(pairs+((i,j),))))
                if key not in nxt or value<nxt[key]:nxt[key]=value
        states=nxt
    used,(cost,pairs)=min(states.items(),key=lambda item:(-item[0].bit_count(),item[1][0],item[1][1]))
    matches=[dict(edges[i,j],generation=tracks[i].generation) for i,j in pairs]
    assigned_tracks={i for i,j in pairs};assigned_candidates={j for i,j in pairs}
    return {"schema":"action4.temporal_association.v1","frame_id":frame.frame_id,
        "frame_identity":{"image_id":frame.image_id,"frame_id":frame.frame_id,"pixel_space_id":frame.pixel_space_id,
                          "width":frame.width,"height":frame.height,"rgb_sha256":frame.rgb_sha256},
        "matches":matches,"unmatched_tracks":[{"track_id":t.track_id,"reason":blocked.get(i,"no_unique_observation")} for i,t in enumerate(tracks) if i not in assigned_tracks],
        "unassigned_candidates":[c.candidate_id for j,c in enumerate(candidates) if j not in assigned_candidates],
        "ambiguous_candidates":[candidates[j].candidate_id for j in sorted(ambiguous_candidates)],
        "edge_diagnostics":trace,"candidate_count":len(candidates),"association_cost":cost,
        "automatic_track_birth_implemented":False,"A0_ROI_publication_ready":False,
        "score_semantics":"association cost, not a replacement calibrated class probability"}
