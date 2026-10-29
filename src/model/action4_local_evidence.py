"""Local predicted support association; never decides class, merges boxes or deletes candidates."""
from dataclasses import dataclass, replace
import hashlib
import numpy as np
from .action4_guidance_contract import (
    CHANNELS, FrameIdentity, CandidateRecord, EvidenceMatch, LocalEvidence, name,
)

@dataclass(frozen=True)
class PredictedSupport:
    frame: FrameIdentity
    reference: str
    pixels: np.ndarray

    def __post_init__(self):
        if not isinstance(self.frame, FrameIdentity): raise ValueError("Typed support frame required")
        name(self.reference, "support reference")
        a = self.pixels
        if not isinstance(a, np.ndarray) or a.dtype != np.bool_ or a.shape != (self.frame.height, self.frame.width):
            raise ValueError("Support must be native bool HxW, not resized or bbox raster")
        a = np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.bool_).reshape(a.shape)
        object.__setattr__(self, "pixels", a)

    @property
    def digest(self):
        return hashlib.sha256(self.pixels.tobytes()).hexdigest()


@dataclass(frozen=True)
class AssociationPolicy:
    support_overlap: float = .7
    center_ratio: float = .25
    single_coverage: float = .8
    area_ratio_min: float = .5
    area_ratio_max: float = 2.
    auxiliary_bbox_iou: float = .3
    negative_iou: float = .7

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Positive finite association policy required")
        if self.area_ratio_min > self.area_ratio_max or any(v > 1 for v in (self.support_overlap, self.single_coverage, self.auxiliary_bbox_iou, self.negative_iou)):
            raise ValueError("Invalid association thresholds")


@dataclass(frozen=True)
class LocalEvidenceBatch:
    candidate_ids: tuple
    evidence: tuple
    pair_diagnostics: tuple

    @property
    def values(self):
        return np.asarray([x.vector for x in self.evidence], dtype=np.float32).reshape(-1, 6)

    def attach(self, candidates):
        candidates = tuple(candidates)
        if tuple(c.candidate_id for c in candidates) != self.candidate_ids:
            raise ValueError("Evidence rows must be explicitly aligned by candidate ID")
        return tuple(replace(c, local_evidence=e) for c, e in zip(candidates, self.evidence))


def _area(b): return (b[2] - b[0]) * (b[3] - b[1])


def _iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return inter / (_area(a) + _area(b) - inter)


def _inside(mask, box):
    # Integer centres inside continuous [left,right), exactly matching the old
    # nonzero-coordinate comparisons, including negative/fractional boundaries.
    h,w=mask.shape
    # Clip in floating point before integer conversion to preserve finite
    # very large off-image boxes without int64 overflow.
    xa,ya,xb,yb=np.clip(np.ceil(np.asarray(box,dtype=float)),[0,0,0,0],[w,h,w,h]).astype(np.int64)
    return int(np.count_nonzero(mask[ya:yb,xa:xb]))


def _mask_stats(masks, supports=None):
    result={}
    for ref,mask in masks.items():
        yy,xx=np.nonzero(mask)
        result[ref]={'area':len(xx),'center':np.array([xx.mean(),yy.mean()]) if len(xx) else None,
                     'digest':supports[ref].digest if supports is not None else hashlib.sha256(mask.tobytes()).hexdigest()}
    return result


def _center(mask, bbox):
    if mask is None: return (np.asarray(bbox[:2]) + bbox[2:]) / 2
    yy, xx = np.nonzero(mask)
    return np.array([xx.mean(), yy.mean()])


def _pair(target, source, masks, policy, stats=None, intersections=None):
    stats=_mask_stats(masks) if stats is None else stats
    intersections={} if intersections is None else intersections
    tm = masks.get(target.source_mask_ref)
    sm = masks.get(source.source_mask_ref)
    if sm is None or stats[source.source_mask_ref]["area"]==0:
        return dict(eligible=False, spans=False, reason="source_has_no_nonempty_predicted_mask")
    if tm is not None and stats[target.source_mask_ref]["area"]==0:
        return dict(eligible=False, spans=False, reason="target_empty_predicted_mask")
    sa = stats[source.source_mask_ref]["area"]; ta = stats[target.source_mask_ref]["area"] if tm is not None else _area(target.bbox_candidate)
    distance = float(np.linalg.norm((stats[target.source_mask_ref]["center"] if tm is not None else _center(None,target.bbox_candidate)) - stats[source.source_mask_ref]["center"]))
    scale = min(np.linalg.norm(np.subtract(target.bbox_candidate[2:], target.bbox_candidate[:2])),
                np.linalg.norm(np.subtract(source.bbox_candidate[2:], source.bbox_candidate[:2])))
    center = distance / scale
    biou = _iou(target.bbox_candidate, source.bbox_candidate)
    if tm is not None:
        pair=tuple(sorted((target.source_mask_ref,source.source_mask_ref)))
        if pair not in intersections:
            intersections[pair]=sa if pair[0]==pair[1] else int((tm & sm).sum())
        inter=intersections[pair]
    else:inter=_inside(sm,target.bbox_candidate)
    overlap = inter / min(ta, sa)
    bbox_ratio = _area(source.bbox_candidate) / _area(target.bbox_candidate)
    mask_ratio = sa / ta if tm is not None else None
    # A local_box is not a target mask: use candidate geometry for scale/IoU.
    # Visible foreground fill is diagnostic, never the single-sided scale gate.
    ratio = mask_ratio if tm is not None else bbox_ratio
    mask_iou = inter / (ta + sa - inter) if tm is not None else None
    local_iou = mask_iou if tm is not None else biou
    visible_fill = sa / _area(target.bbox_candidate)
    spans = overlap >= policy.support_overlap
    if tm is not None:
        ok = overlap >= policy.support_overlap and center <= policy.center_ratio
        reason = "dual_predicted_support"
    else:
        ok = inter / sa >= policy.single_coverage and center <= policy.center_ratio and policy.area_ratio_min <= ratio <= policy.area_ratio_max and biou >= policy.auxiliary_bbox_iou
        reason = "single_predicted_support_in_local_box"
    if source.evidence_channel == "negative":
        ok = ok and local_iou >= policy.negative_iou and policy.area_ratio_min <= ratio <= policy.area_ratio_max
    return dict(eligible=bool(ok), spans=bool(spans), reason=reason if ok else "local_support_gate_failed",
                support_overlap=float(overlap), support_in_target=float(inter / sa),
                area_ratio=float(ratio),
                area_ratio_kind="mask_area_ratio" if tm is not None else "candidate_bbox_area_ratio",
                candidate_bbox_area_ratio=float(bbox_ratio), mask_area_ratio=mask_ratio,
                visible_fill=float(visible_fill), visible_fill_is_diagnostic=True,
                local_iou=float(local_iou),
                local_iou_kind="predicted_mask_iou" if tm is not None else "candidate_bbox_iou",
                mask_iou=mask_iou, center_ratio=center,
                auxiliary_bbox_iou=float(biou), target_kind=target.support_kind)


def build_local_evidence(targets, evidence_sources, supports, *, policy=AssociationPolicy()):
    """Return P x 6 with frozen row IDs and reasons; input scores are never class acceptance.

    Targets must be the caller's candidate set, not GT. Competing targets remain
    ambiguous: this function does not deduplicate high-IoU boxes or promote them.
    Exact duplicate source pixels are grouped across IDs/profiles/channels, so a
    copied mask cannot acquire two independent target assignments.
    """
    targets, sources, supports = tuple(targets), tuple(evidence_sources), tuple(supports)
    if not isinstance(policy, AssociationPolicy): raise ValueError("Typed policy required")
    records = targets + sources
    if any(not isinstance(c, CandidateRecord) for c in records): raise ValueError("Typed prediction candidates required")
    for group in (targets, sources):
        if len({c.candidate_id for c in group}) != len(group): raise ValueError("Duplicate stable candidate ID")
    if not records:
        if supports: raise ValueError("Support without registered predictions")
        return LocalEvidenceBatch((), (), ())
    frame = records[0].frame
    if any(c.frame != frame for c in records): raise ValueError("Cross-image/frame/pixel-space/RGB association forbidden")
    registered = {}
    for candidate in records:
        previous = registered.get(candidate.candidate_id)
        if previous is not None and previous != candidate:
            raise ValueError("Conflicting records share the same stable candidate ID")
        registered[candidate.candidate_id] = candidate
    lookup = {}
    for s in supports:
        if not isinstance(s, PredictedSupport) or s.frame != frame or s.reference in lookup:
            raise ValueError("Support coordinate identity mismatch or duplicate reference")
        lookup[s.reference] = s
    for c in records:
        if c.support_kind == "mask" and c.source_mask_ref not in lookup:
            raise ValueError("Registered mask missing")
    if any(c.evidence_channel not in CHANNELS for c in sources): raise ValueError("Source requires explicit external channel")
    masks = {k: v.pixels for k, v in lookup.items()}
    stats=_mask_stats(masks,lookup);intersections={}
    groups = {}
    for s in sources:
        key = stats[s.source_mask_ref]["digest"] if s.source_mask_ref in lookup else "unavailable:" + s.candidate_id
        groups.setdefault(key, []).append(s)
    diagnostics, eligible, spans, conflicted = [], {}, {}, set()
    for key, members in groups.items():
        supported_targets = set()
        for s in members:
            for t in targets:
                d = _pair(t, s, masks, policy,stats,intersections)
                diagnostics.append(dict(target_id=t.candidate_id, source_id=s.candidate_id, channel=s.evidence_channel, **d))
                eligible[(t.candidate_id, s.candidate_id)] = d["eligible"]
                spans[(t.candidate_id, s.candidate_id)] = d["spans"]
                if d["spans"] or d["eligible"]: supported_targets.add(t.candidate_id)
        if len(supported_targets) > 1:
            conflicted.add(key)
    results = []
    for t in targets:
        matches = {}
        for channel in CHANNELS:
            candidates, ambiguous = [], []
            for key, members in groups.items():
                local = [s for s in members if s.evidence_channel == channel]
                if key in conflicted:
                    ambiguous.extend(s.candidate_id for s in local if spans.get((t.candidate_id, s.candidate_id)) or eligible.get((t.candidate_id, s.candidate_id)))
                    continue
                same = [s for s in local if eligible.get((t.candidate_id, s.candidate_id))]
                if same: candidates.append(same)
            if ambiguous:
                matches[channel] = EvidenceMatch(0., False, "ambiguous", "one_support_spans_multiple_targets", tuple(sorted(ambiguous)))
            elif len(candidates) > 1:
                matches[channel] = EvidenceMatch(0., False, "ambiguous", "multiple_distinct_supports_compete", tuple(sorted(s.candidate_id for g in candidates for s in g)))
            elif candidates:
                group = candidates[0]
                matches[channel] = EvidenceMatch(float(max(s.evidence_score for s in group)), True, "matched", "unique_local_predicted_support", tuple(sorted(s.candidate_id for s in group)))
            else:
                reason = "channel_missing" if not any(s.evidence_channel == channel for s in sources) else "no_unique_eligible_local_support"
                matches[channel] = EvidenceMatch(0., False, "missing", reason)
        results.append(LocalEvidence(**matches))
    return LocalEvidenceBatch(tuple(t.candidate_id for t in targets), tuple(results), tuple(diagnostics))
