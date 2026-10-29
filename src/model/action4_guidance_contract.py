"""Prediction-only Stage2 contracts. No inference, GT, IO or implicit promotion."""
from dataclasses import dataclass, field
from math import isfinite
import numpy as np

CHANNELS = ("text_positive", "visual_positive", "negative")
VECTOR_ORDER = tuple(x + "_score" for x in CHANNELS) + tuple(x + "_available" for x in CHANNELS)


def name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be explicit")


def sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or value == "0" * 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " must be lowercase SHA256")


def probability(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(label + " must be a finite probability")


def box(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or a[2] <= a[0] or a[3] <= a[1]:
        raise ValueError("Ordered finite native XYXY required")
    return tuple(map(float, a))


@dataclass(frozen=True)
class FrameIdentity:
    image_id: str
    frame_id: int
    pixel_space_id: str
    width: int
    height: int
    rgb_sha256: str

    def __post_init__(self):
        name(self.image_id, "image_id"); name(self.pixel_space_id, "pixel_space_id")
        for k in ("frame_id", "width", "height"):
            v = getattr(self, k)
            if type(v) is not int or v < (0 if k == "frame_id" else 1):
                raise ValueError("Invalid " + k)
        sha(self.rgb_sha256, "RGB")


@dataclass(frozen=True)
class SourceIdentity:
    producer: str
    model_sha256: str
    source_sha256: str
    profile_id: str | None = None
    profile_sha256: str | None = None
    formal_admission: bool = False
    weight_identity_kind: str = "checkpoint_sha256"
    run_config_sha256: str | None = None
    optimizer_step: int | None = None

    def __post_init__(self):
        if self.producer not in ("b0", "yolo_text", "yolo_visual", "yolo_negative"):
            raise ValueError("Undeclared prediction producer")
        sha(self.model_sha256, "model"); sha(self.source_sha256, "source")
        if self.producer != "b0":
            name(self.profile_id, "YOLO profile"); sha(self.profile_sha256, "profile")
        elif (self.profile_id is None) != (self.profile_sha256 is None):
            raise ValueError("Partial profile identity")
        elif self.profile_id is not None:
            name(self.profile_id, "profile"); sha(self.profile_sha256, "profile")
        if type(self.formal_admission) is not bool:
            raise ValueError("Explicit formal admission required")
        if self.weight_identity_kind=="initialization_sha256_plus_update":
            if self.producer!="b0":raise ValueError("Only live B0 has update-addressed weights")
            sha(self.run_config_sha256,"live run config")
            if type(self.optimizer_step) is not int or self.optimizer_step<0:raise ValueError("Live optimizer step required")
        elif self.weight_identity_kind!="checkpoint_sha256" or self.run_config_sha256 is not None or self.optimizer_step is not None:
            raise ValueError("Unknown or mixed weight identity kind")



@dataclass(frozen=True)
class EvidenceMatch:
    score: float = 0.0
    available: bool = False
    match_status: str = "missing"
    reason: str = "not_constructed"
    source_candidate_ids: tuple = ()

    def __post_init__(self):
        probability(self.score, "evidence score")
        if type(self.available) is not bool: raise ValueError("Boolean availability required")
        if self.match_status not in ("matched", "missing", "ambiguous", "unsupported"):
            raise ValueError("Unknown evidence match status")
        name(self.reason, "match reason")
        object.__setattr__(self, "source_candidate_ids", tuple(self.source_candidate_ids))
        if len(set(self.source_candidate_ids)) != len(self.source_candidate_ids):
            raise ValueError("Duplicate source candidate IDs")
        for v in self.source_candidate_ids: name(v, "source candidate ID")
        if self.available != (self.match_status == "matched"):
            raise ValueError("Availability and match status disagree")
        if not self.available and self.score != 0:
            raise ValueError("Missing/ambiguous evidence must have score0")
        if self.available and not self.source_candidate_ids:
            raise ValueError("Matched evidence needs source IDs")


@dataclass(frozen=True)
class LocalEvidence:
    text_positive: EvidenceMatch = field(default_factory=EvidenceMatch)
    visual_positive: EvidenceMatch = field(default_factory=EvidenceMatch)
    negative: EvidenceMatch = field(default_factory=EvidenceMatch)

    def __post_init__(self):
        if any(not isinstance(getattr(self, c), EvidenceMatch) for c in CHANNELS):
            raise ValueError("Typed evidence matches required")

    @property
    def vector(self):
        return tuple(float(getattr(self, c).score) for c in CHANNELS) + tuple(float(getattr(self, c).available) for c in CHANNELS)


@dataclass(frozen=True)
class CandidateRecord:
    frame: FrameIdentity
    candidate_id: str
    source_detection_id: str
    source: SourceIdentity
    bbox_candidate: tuple
    raw_scores: tuple
    parent_ids: tuple = ()
    support_kind: str = "local_box"
    source_mask_ref: str | None = None
    evidence_channel: str | None = None
    evidence_score: float | None = None
    local_evidence: LocalEvidence = field(default_factory=LocalEvidence)
    p_action4: float | None = None
    bbox_full: tuple | None = None
    q_full: float | None = None
    full_box_source: str | None = None

    def __post_init__(self):
        if not isinstance(self.frame, FrameIdentity) or not isinstance(self.source, SourceIdentity):
            raise ValueError("Typed frame/source identity required")
        name(self.candidate_id, "candidate_id"); name(self.source_detection_id, "source_detection_id")
        object.__setattr__(self, "bbox_candidate", box(self.bbox_candidate))
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        if len(set(self.parent_ids)) != len(self.parent_ids): raise ValueError("Duplicate parent IDs")
        for v in self.parent_ids: name(v, "parent ID")
        scores = tuple(self.raw_scores)
        if len({k for k, v in scores}) != len(scores): raise ValueError("Duplicate raw score names")
        for k, v in scores:
            name(k, "raw score"); probability(v, "raw score")
        object.__setattr__(self, "raw_scores", tuple((k, float(v)) for k, v in scores))
        if self.support_kind not in ("mask", "local_box"): raise ValueError("Unknown support kind")
        if self.support_kind == "mask":
            name(self.source_mask_ref, "mask ref")
            if self.source.producer == "b0": raise ValueError("B0 box cannot masquerade as predicted instance mask")
        elif self.source_mask_ref is not None: raise ValueError("local_box has no pixel mask")
        if self.evidence_channel is not None:
            if self.evidence_channel not in CHANNELS or self.source.producer == "b0":
                raise ValueError("External evidence channel must come from YOLO prediction")
            probability(self.evidence_score, "external score")
        elif self.evidence_score is not None: raise ValueError("Score without evidence channel")
        if not isinstance(self.local_evidence, LocalEvidence): raise ValueError("Typed local evidence required")
        if self.p_action4 is not None: probability(self.p_action4, "p_action4")
        if self.bbox_full is not None:
            object.__setattr__(self, "bbox_full", box(self.bbox_full))
            name(self.full_box_source, "full box source")
        if self.q_full is not None:
            probability(self.q_full, "q_full")
            if self.bbox_full is None: raise ValueError("q_full requires its own full box/source")
        if self.bbox_full is None and self.full_box_source is not None: raise ValueError("Full source without box")


@dataclass(frozen=True)
class InstanceProposal:
    project_instance_id: str
    member_candidate_ids: tuple
    identity_status: str
    separation_status: str
    support_kind: str
    support_ref: str | None = None
    bbox_full: tuple | None = None
    q_full: float | None = None
    rejection_reasons: tuple = ()
    confirmation_source: str | None = None

    def __post_init__(self):
        name(self.project_instance_id, "project instance")
        object.__setattr__(self, "member_candidate_ids", tuple(self.member_candidate_ids))
        if not self.member_candidate_ids or len(set(self.member_candidate_ids)) != len(self.member_candidate_ids):
            raise ValueError("Unique nonempty members required")
        for v in self.member_candidate_ids: name(v, "member ID")
        if self.identity_status not in ("unassessed", "provisional", "confirmed", "rejected"):
            raise ValueError("Unknown identity status")
        if self.separation_status not in ("unassessed", "independent", "ambiguous", "merged"):
            raise ValueError("Unknown separation status")
        if self.support_kind not in ("mask", "local_box"): raise ValueError("Unknown proposal support")
        if self.support_kind == "mask": name(self.support_ref, "mask support ref")
        if self.identity_status == "confirmed":
            name(self.confirmation_source, "independent confirmation source")
            if self.separation_status != "independent": raise ValueError("Ambiguous support cannot be confirmed")
        if self.bbox_full is not None: object.__setattr__(self, "bbox_full", box(self.bbox_full))
        if self.q_full is not None:
            probability(self.q_full, "full quality")
            if self.bbox_full is None: raise ValueError("Full quality without full box")
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))


@dataclass(frozen=True)
class BudgetRecord:
    source_id: str
    observed: int
    limit: int
    truncated_candidate_ids: tuple = ()
    reason: str = "candidate_resource_budget_not_object_count"

    def __post_init__(self):
        name(self.source_id, "budget source")
        if type(self.observed) is not int or self.observed < 0 or type(self.limit) is not int or self.limit < 1:
            raise ValueError("Invalid candidate budget")
        object.__setattr__(self, "truncated_candidate_ids", tuple(self.truncated_candidate_ids))
        name(self.reason, "budget reason")

    @property
    def touched_limit(self): return self.observed >= self.limit


@dataclass(frozen=True)
class DetectionPacket:
    frame: FrameIdentity
    sources: tuple
    candidates: tuple
    instances: tuple = ()
    provisional: tuple = ()
    budgets: tuple = ()
    status: str = "provisional"

    def __post_init__(self):
        if not isinstance(self.frame, FrameIdentity): raise ValueError("Typed packet frame required")
        for k in ("sources", "candidates", "instances", "provisional", "budgets"):
            object.__setattr__(self, k, tuple(getattr(self, k)))
        if not self.sources or any(not isinstance(x, SourceIdentity) for x in self.sources):
            raise ValueError("Packet source identities required")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)): raise ValueError("Duplicate candidate identity")
        for c in self.candidates:
            if not isinstance(c, CandidateRecord) or c.frame != self.frame or c.source not in self.sources:
                raise ValueError("Packet candidate frame/source mismatch")
        proposals = self.instances + self.provisional
        pids = [p.project_instance_id for p in proposals]
        if len(pids) != len(set(pids)): raise ValueError("Duplicate instance identity")
        used = set()
        for p in proposals:
            if not isinstance(p, InstanceProposal) or not set(p.member_candidate_ids) <= set(ids):
                raise ValueError("Unknown proposal members")
            if used & set(p.member_candidate_ids): raise ValueError("Candidate assigned to multiple proposals")
            used.update(p.member_candidate_ids)
        for p in self.instances:
            if p.identity_status != "confirmed" or p.separation_status != "independent":
                raise ValueError("Formal instance requires independent confirmation")
        if any(not isinstance(x, BudgetRecord) for x in self.budgets): raise ValueError("Typed budget required")
        if self.status not in ("no_prediction", "provisional", "confirmed"): raise ValueError("Unknown packet status")
        if (self.status == "no_prediction") != (not self.candidates):
            raise ValueError("no_prediction must reflect actual candidate absence")
        if self.status == "confirmed" and not self.instances: raise ValueError("No confirmed instances")
