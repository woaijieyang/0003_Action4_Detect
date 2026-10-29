"""Pure CPU SAM session, replay and complete-region replacement contracts.

No SAM imports, worker spawning, mask reads or model calls occur here. The caller
executes official API requests and supplies explicit release acknowledgements.
"""
from dataclasses import dataclass, replace, asdict, field
import hashlib
import json
import math


def _name(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} required")


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"integer {label} >= {minimum} required")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    frame_ids: tuple
    pts: tuple
    time_base: tuple
    manifest_identity: str
    pixel_identity: str
    width: int
    height: int

    def __post_init__(self):
        for field in ("region_id", "manifest_identity", "pixel_identity"):
            _name(getattr(self, field), field)
        for field in ("frame_ids", "pts", "time_base"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.frame_ids or len(self.frame_ids) != len(self.pts) or len(self.time_base) != 2:
            raise ValueError("nonempty aligned region frame/PTS and rational time base required")
        for x in self.frame_ids: _int(x, "frame_id")
        for x in self.pts:
            if type(x) is not int: raise ValueError("integer PTS required")
        for x in self.time_base: _int(x, "time_base", 1)
        for field in ("width", "height"): _int(getattr(self, field), field, 1)
        if any(b <= a for a, b in zip(self.frame_ids, self.frame_ids[1:])) or any(b <= a for a, b in zip(self.pts, self.pts[1:])):
            raise ValueError("region frame IDs and PTS must strictly increase")

    @property
    def identity(self):
        return _fingerprint(asdict(self))

    def frame(self, local_index):
        _int(local_index, "local_index")
        if local_index >= len(self.frame_ids): raise ValueError("frame outside isolated region")
        return self.frame_ids[local_index], self.pts[local_index]


@dataclass(frozen=True)
class SeedBinding:
    track_id: str
    region_identity: str
    candidate_id: str
    local_index: int
    global_frame_id: int
    box_xywh01: tuple
    evidence_ids: tuple
    source: str

    def __post_init__(self):
        for field in ("track_id", "region_identity", "candidate_id"):
            _name(getattr(self, field), field)
        _int(self.local_index, "seed local_index"); _int(self.global_frame_id, "seed global_frame")
        box = tuple(self.box_xywh01); evidence = tuple(self.evidence_ids)
        if len(box) != 4 or not all(math.isfinite(v) for v in box): raise ValueError("finite normalized XYWH required")
        x, y, w, h = box
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1 and y + h <= 1):
            raise ValueError("seed normalized box outside image")
        if self.source not in {"frozen_b0", "frozen_b0_yolo"} or not evidence or any(not isinstance(e, str) or not e for e in evidence):
            raise ValueError("independent frozen image detection evidence required")
        object.__setattr__(self, "box_xywh01", box); object.__setattr__(self, "evidence_ids", evidence)

    @property
    def fingerprint(self): return _fingerprint(asdict(self))

    def validate_region(self, region):
        if self.region_identity != region.identity or region.frame(self.local_index)[0] != self.global_frame_id:
            raise ValueError("seed region/frame mismatch")


@dataclass(frozen=True)
class PromptEvent:
    event_id: str
    track_id: str
    region_identity: str
    seed_fingerprint: str
    local_index: int
    global_frame_id: int
    points_xy_native: tuple
    points_xy01: tuple
    labels: tuple
    evidence_ids: tuple
    trigger_reason: str
    error_segment_id: str
    evidence_source: str

    def __post_init__(self):
        for field in ("event_id", "track_id", "region_identity", "seed_fingerprint", "trigger_reason", "error_segment_id"):
            _name(getattr(self, field), field)
        if self.evidence_source not in {"frozen_b0", "frozen_b0_yolo"}:
            raise ValueError("correction evidence must be frozen independent image detections, never GT/A0 pose")
        _int(self.local_index, "prompt local_index"); _int(self.global_frame_id, "prompt global_frame")
        for field in ("points_xy_native", "points_xy01"):
            points = tuple(tuple(p) for p in getattr(self, field))
            if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
                raise ValueError("finite XY point pairs required")
            object.__setattr__(self, field, points)
        labels = tuple(self.labels); evidence = tuple(self.evidence_ids)
        if len(labels) != len(self.points_xy_native) or len(labels) != len(self.points_xy01):
            raise ValueError("point/label lengths differ")
        if any(type(v) is not int or v not in (0, 1) for v in labels) or not (2 <= labels.count(0) <= 4 and 2 <= labels.count(1) <= 4):
            raise ValueError("each event requires 2-4 positive and 2-4 negative points")
        if any(not 0 <= v <= 1 for p in self.points_xy01 for v in p): raise ValueError("normalized point outside image")
        if not evidence or any(not isinstance(e, str) or not e for e in evidence): raise ValueError("prompt evidence IDs required")
        object.__setattr__(self, "labels", labels); object.__setattr__(self, "evidence_ids", evidence)


@dataclass(frozen=True)
class PVSBoxPromptEvent(PromptEvent):
    """Versioned frozen-B0-only PVS box corners; legacy point events stay unchanged.

    This contract does not implement shared-session lifecycle or execute SAM.
    """
    protocol: str = "action4.sam.pvs_box_corners.v1"

    def __post_init__(self):
        for name in ("event_id", "track_id", "region_identity", "seed_fingerprint", "trigger_reason", "error_segment_id"):
            _name(getattr(self, name), name)
        if self.protocol != "action4.sam.pvs_box_corners.v1":
            raise ValueError("Explicit PVS box-corner protocol required")
        if self.evidence_source != "frozen_b0":
            raise ValueError("PVS box evidence requires frozen B0 only; no GT/A0/YOLO")
        _int(self.local_index, "prompt local_index"); _int(self.global_frame_id, "prompt global_frame")
        for name in ("points_xy_native", "points_xy01"):
            points = tuple(tuple(p) for p in getattr(self, name))
            if len(points) != 2 or any(len(p) != 2 or any(isinstance(v, bool) or not math.isfinite(v) for v in p) for p in points):
                raise ValueError("Exactly two finite ordered box-corner pairs required")
            if not (0 <= points[0][0] < points[1][0] and 0 <= points[0][1] < points[1][1]):
                raise ValueError("PVS box corners require positive XY extent")
            if name == "points_xy01" and any(v > 1 for pair in points for v in pair):
                raise ValueError("Normalized box corner outside image")
            object.__setattr__(self, name, points)
        labels = tuple(self.labels); evidence = tuple(self.evidence_ids)
        if labels != (2, 3) or any(type(v) is not int for v in labels):
            raise ValueError("PVS box-only labels must be exactly (2,3)")
        if not evidence or any(not isinstance(e, str) or not e for e in evidence):
            raise ValueError("Frozen B0 evidence references required")
        object.__setattr__(self, "labels", labels); object.__setattr__(self, "evidence_ids", evidence)

    def validate_binding(self, region, seed):
        seed.validate_region(region)
        if (seed.source != "frozen_b0" or self.track_id != seed.track_id
                or self.seed_fingerprint != seed.fingerprint or self.region_identity != region.identity
                or region.frame(self.local_index)[0] != self.global_frame_id):
            raise ValueError("PVS box event must match the frozen B0 seed/region/frame")

    def to_request(self, region, seed, *, session_id, obj_id):
        from model.action4_sam_seed_extent import box_corner_prompt, PROTOCOL
        self.validate_binding(region, seed)
        if self.protocol != PROTOCOL:raise ValueError("Request converter protocol mismatch")
        box = self.points_xy_native[0] + self.points_xy_native[1]
        request = box_corner_prompt(session_id, self.local_index, obj_id, box, region.width, region.height)
        if any(abs(a-b)>1e-9 for supplied, converted in zip(self.points_xy01,request["points"]) for a,b in zip(supplied,converted)):
            raise ValueError("Native/normalized PVS box coordinate mismatch")
        return request


@dataclass(frozen=True)
class ReplayPlan:
    region_identity: str
    seed_fingerprints: tuple
    events: tuple
    source_generation: int = 0

    def __post_init__(self):
        _name(self.region_identity, "region identity")
        object.__setattr__(self, "seed_fingerprints", tuple(tuple(x) for x in self.seed_fingerprints))
        object.__setattr__(self, "events", tuple(self.events))
        if self.source_generation != 0 or len(self.events) > 4 or any(not isinstance(e, PromptEvent) for e in self.events):
            raise ValueError("one bounded immutable gen0 correction plan required")

    @property
    def identity(self): return _fingerprint(asdict(self))


def lock_replay_plan(region, seeds, events):
    """Freeze prediction-only gen0 corrections; enforce region/track event budgets.

    Caller supplies events already selected by gen0 diagnostics and frozen B0/YOLO.
    This function validates provenance references, not the underlying image evidence.
    """
    seeds = tuple(seeds); events = tuple(events)
    for seed in seeds: seed.validate_region(region)
    bytrack = {s.track_id: s for s in seeds}
    if len(bytrack) != len(seeds) or len(seeds) > 8: raise ValueError("unique regional track seeds, maximum eight")
    if len(events) > 4 or len({e.event_id for e in events}) != len(events): raise ValueError("regional correction budget or duplicate event")
    ordered = sorted(events, key=lambda e: (region.frame(e.local_index)[1], e.event_id))
    if events != tuple(ordered): raise ValueError("corrections must be frozen in PTS/event-ID order")
    histories = {}
    for event in events:
        seed = bytrack.get(event.track_id)
        if seed is None or event.seed_fingerprint != seed.fingerprint or event.region_identity != region.identity or region.frame(event.local_index)[0] != event.global_frame_id:
            raise ValueError("correction does not belong to frozen regional seed")
        for native, normalized in zip(event.points_xy_native, event.points_xy01):
            if not (0 <= native[0] <= region.width and 0 <= native[1] <= region.height) or not all(abs(a - b) <= 1e-9 for a, b in zip((native[0] / region.width, native[1] / region.height), normalized)):
                raise ValueError("native/normalized correction coordinate mismatch")
        if isinstance(event, PVSBoxPromptEvent): event.validate_binding(region, seed)
        prior = histories.setdefault(event.track_id, [])
        if len(prior) >= 2 or any(e.error_segment_id == event.error_segment_id for e in prior): raise ValueError("track correction budget or repeated error segment")
        if prior and event.global_frame_id - prior[-1].global_frame_id < 15: raise ValueError("correction interval below 15 frames")
        prior.append(event)
    return ReplayPlan(region.identity, tuple(sorted((s.track_id, s.fingerprint) for s in seeds)), events)


@dataclass(frozen=True)
class SamSessionRecord:
    record_id: str
    track_id: str
    seed_fingerprint: str
    generation: int
    attempt: int
    region_identity: str
    worker_id: str
    device_identity: str
    dependency_identity: str
    prompt_log_identity: str
    state: str = "PLANNED"
    session_id: str | None = None
    obj_id: int | None = None
    snapshot_identity: str | None = None
    termination: str | None = None
    failure_kind: str | None = None
    active_operation: str | None = None
    resource_reserved: bool = False
    completed_rounds: int = 0
    applied_event_ids: tuple = ()
    published: bool = False
    creation_concurrency: int = 0
    failure_concurrency: int = 0
    retry_concurrency_limit: int | None = None
    restored_from_receipt_identity: str | None = None


@dataclass(frozen=True)
class AbsenceDecision:
    """Explicit versioned upstream decision, never inferred from public empty IDs.

    This is provenance for a project absence decision, not a ground-truth label.
    The producer is responsible for its evidence; the contract validates binding.
    """
    decision_id: str
    source_identity: str
    rule_version: str
    track_id: str
    region_identity: str
    session_id: str
    generation: int
    attempt: int
    local_index: int
    global_frame_id: int
    obj_id: int
    schema_version: str = "action4.sam_absence_decision.v1"

    def __post_init__(self):
        if self.schema_version != "action4.sam_absence_decision.v1": raise ValueError("versioned absence decision schema required")
        for name in ("decision_id", "source_identity", "rule_version", "track_id", "region_identity", "session_id"):
            _name(getattr(self, name), name)
        for name in ("generation", "attempt", "local_index", "global_frame_id", "obj_id"):
            _int(getattr(self, name), name)

    @property
    def identity(self): return _fingerprint(asdict(self))

    def validate(self, record, region, local_index):
        expected = (record.track_id, record.region_identity, record.session_id, record.generation,
                    record.attempt, local_index, region.frame(local_index)[0], record.obj_id)
        actual = (self.track_id, self.region_identity, self.session_id, self.generation,
                  self.attempt, self.local_index, self.global_frame_id, self.obj_id)
        if actual != expected: raise ValueError("absence decision does not bind this exact frame/object/session")


@dataclass(frozen=True)
class FrameResponse:
    region_identity: str
    session_id: str
    generation: int
    local_index: int
    status: str
    obj_id: int | None = None
    mask_identity: str | None = None
    reason: str | None = None
    failure_kind: str | None = None
    absence_decision: AbsenceDecision | None = None
    available: bool = field(init=False)
    use_as_training_negative: bool = field(default=False, init=False)

    def __post_init__(self):
        for field in ("region_identity", "session_id"): _name(getattr(self, field), field)
        for field in ("generation", "local_index"): _int(getattr(self, field), field)
        if self.status not in {"target_present", "target_absent", "target_visibility_unknown", "ambiguous_identity", "inference_error", "session_lost"}:
            raise ValueError("explicit response status required; missing_response is created only for absent responses")
        if self.status == "target_present":
            _int(self.obj_id, "object ID"); _name(self.mask_identity, "mask identity")
        elif self.mask_identity is not None: raise ValueError("non-present status cannot carry a mask")
        if self.status in {"target_absent", "target_visibility_unknown"}: _int(self.obj_id, "bound object ID")
        if self.status == "target_visibility_unknown" and not self.reason: raise ValueError("visibility unknown requires a reason")
        if self.status == "target_absent":
            evidence = self.absence_decision
            if not isinstance(evidence, AbsenceDecision): raise ValueError("target_absent requires explicit versioned decision evidence")
            if (evidence.region_identity, evidence.session_id, evidence.generation, evidence.local_index, evidence.obj_id) != (self.region_identity, self.session_id, self.generation, self.local_index, self.obj_id):
                raise ValueError("absence decision response binding mismatch")
        elif self.absence_decision is not None: raise ValueError("absence decision supplied for a non-absence status")
        object.__setattr__(self, "available", self.status == "target_present")
        if self.status in {"ambiguous_identity", "inference_error", "session_lost"} and not self.reason:
            raise ValueError("failure/ambiguity reason required")
        kinds = {"inference_error": {"transient", "oom", "schema"}, "session_lost": {"worker"}, "ambiguous_identity": {"identity"}}
        if self.status in kinds and self.failure_kind not in kinds[self.status]: raise ValueError("explicit compatible failure kind required")
        if self.status not in kinds and self.failure_kind is not None: raise ValueError("normal response cannot carry a failure kind")


@dataclass(frozen=True)
class SnapshotFrame:
    global_frame_id: int
    pts: int
    local_index: int
    status: str
    mask_identity: str | None
    reason: str | None
    failure_kind: str | None = None
    available: bool = False
    use_as_training_negative: bool = False
    absence_decision_identity: str | None = None


@dataclass(frozen=True)
class SamVideoSnapshot:
    region_identity: str
    region_id: str
    track_id: str
    seed_fingerprint: str
    session_id: str
    obj_id: int
    generation: int
    attempt: int
    dependency_identity: str
    prompt_log_identity: str
    propagation_round: int
    frames: tuple
    prompt_diagnostics: tuple
    responses_complete: bool
    mask_available_count: int = 0
    mask_available_coverage: float = 0.0

    @property
    def complete(self):
        """Legacy alias: response-domain completeness, NEVER tracking success."""
        return self.responses_complete

    @property
    def identity(self): return _fingerprint(asdict(self))


class RoundCollector:
    """Propagation wins over prompt diagnostics; conflicting stream duplicates reject."""
    def __init__(self, record, region, round_index):
        if record.region_identity != region.identity or record.session_id is None or record.obj_id is None:
            raise ValueError("bound session and exact region required")
        _int(round_index, "round index")
        self.record, self.region, self.round_index = record, region, round_index
        self._stream, self._prompts = {}, []
        self._sealed = False
        self._conflicted = False

    def invalidate(self, reason):
        """A raw-response adapter may reject before it can build FrameResponse."""
        _name(reason, "round invalidation reason")
        self._conflicted = True

    def add(self, response, *, source="propagation"):
        if self._sealed: raise ValueError("round already sealed")
        try:
            if not isinstance(response, FrameResponse) or source not in {"propagation", "prompt"}: raise ValueError("typed response and source required")
            r = self.record
            if (response.region_identity, response.session_id, response.generation) != (r.region_identity, r.session_id, r.generation):
                raise ValueError("response belongs to another region/session/generation")
            self.region.frame(response.local_index)
            if response.obj_id is not None and response.obj_id != r.obj_id: raise ValueError("response target object binding differs")
            if response.absence_decision is not None:
                response.absence_decision.validate(r, self.region, response.local_index)
        except (ValueError, TypeError, AttributeError):
            # Catching one bad response in the caller must never turn schema or
            # identity failure into a successful round after later valid frames.
            self._conflicted = True
            raise
        if source == "prompt":
            self._prompts.append(response)
            return
        old = self._stream.get(response.local_index)
        if old is not None and old != response:
            self._conflicted = True
            raise ValueError("conflicting duplicate propagation response")
        self._stream[response.local_index] = response

    def seal(self):
        if self._sealed or self._conflicted: raise ValueError("sealed or conflicting round cannot be published")
        self._sealed = True
        rows = []
        for i, (frame, pts) in enumerate(zip(self.region.frame_ids, self.region.pts)):
            response = self._stream.get(i)
            rows.append(SnapshotFrame(frame, pts, i, response.status if response else "missing_response",
                                      response.mask_identity if response else None, response.reason if response else "no_propagation_response",
                                      response.failure_kind if response else "transient",
                                      response.available if response else False, False,
                                      response.absence_decision.identity if response and response.absence_decision else None))
        r = self.record
        return SamVideoSnapshot(r.region_identity, self.region.region_id, r.track_id, r.seed_fingerprint,
                                r.session_id, r.obj_id, r.generation, r.attempt, r.dependency_identity,
                                r.prompt_log_identity, self.round_index, tuple(rows), tuple(self._prompts),
                                all(row.status in {"target_present", "target_absent", "target_visibility_unknown"} for row in rows),
                                sum(row.available for row in rows), sum(row.available for row in rows) / len(rows))


def replace_generation_snapshot(previous, revised):
    """Internal candidate-only structural check; NOT a publication API.

    A complete snapshot alone does not establish identity confirmation or READY.
    Formal consumers must use SessionLedger.publish_replacement, which validates
    live record eligibility and switches the whole-region publication atomically.
    """
    fields = ("region_identity", "region_id", "track_id", "seed_fingerprint", "dependency_identity")
    if any(getattr(previous, f) != getattr(revised, f) for f in fields): raise ValueError("replacement ownership/input identity mismatch")
    if previous.generation != 0 or revised.generation != 1: raise ValueError("only one gen0 -> gen1 correction replacement allowed")
    if revised.propagation_round != 1 or previous.session_id == revised.session_id:
        raise ValueError("replacement requires corrected full propagation in a newly bound session")
    if [(r.global_frame_id, r.pts, r.local_index) for r in previous.frames] != [(r.global_frame_id, r.pts, r.local_index) for r in revised.frames]:
        raise ValueError("replacement responsibility domain changed")
    if not revised.responses_complete or any(r.status not in {"target_present", "target_absent", "target_visibility_unknown"} for r in revised.frames):
        raise ValueError("incomplete or erroneous replacement; missing is not absent")
    return revised


class BudgetExceeded(RuntimeError): pass


class SessionLedger:
    """Single controller ledger. Serializes operations per session; no backend work.

    begin_create reserves capacity before calling a backend. close requires an
    explicit acknowledgement that backend session memory is released; failed close
    keeps capacity reserved. A dead worker release must likewise be acknowledged.
    """
    def __init__(self, *, per_worker_limit=2, global_limit=4, region_track_limit=8):
        for name, value, maximum in (("per_worker", per_worker_limit, 2), ("global", global_limit, 4), ("regional", region_track_limit, 8)):
            _int(value, name, 1)
            if value > maximum: raise ValueError("budget exceeds registered protocol")
        self.per_worker_limit, self.global_limit, self.region_track_limit = per_worker_limit, global_limit, region_track_limit
        self._records, self._regions, self._seeds, self._plans, self._snapshots = {}, {}, {}, {}, {}
        self._publications = {}
        self._events = []

    @property
    def audit_log(self): return tuple(self._events)

    @property
    def active_count(self): return sum(r.resource_reserved for r in self._records.values())

    def get(self, record_id): return self._records[record_id]

    def snapshot(self, record_id, propagation_round=None):
        r = self.get(record_id)
        index = r.completed_rounds - 1 if propagation_round is None else propagation_round
        return self._snapshots[(record_id, index)]

    def restore_closed_published_baseline(self, snapshot, region, seed, *, expected_snapshot_identity,
                                        source_receipt_identity, worker_id, device_identity):
        """Restore a sealed gen0 publication record, NEVER backend session memory.

        The caller verifies the saved raw/receipt source, normalizes its slots and
        supplies their identity. Restoration reserves no resources and dispatches
        no API requests; its audit event is distinct from a newly executed session.
        """
        for name, value in (("source receipt", source_receipt_identity), ("worker", worker_id), ("device", device_identity)):
            _name(value, name)
        seed.validate_region(region)
        if not isinstance(snapshot, SamVideoSnapshot) or snapshot.identity != expected_snapshot_identity:
            raise ValueError("sealed baseline snapshot identity mismatch")
        if snapshot.generation != 0 or snapshot.propagation_round != 0 or not snapshot.responses_complete:
            raise ValueError("only a complete gen0 baseline publication can be restored")
        if (snapshot.region_identity, snapshot.region_id, snapshot.track_id, snapshot.seed_fingerprint) != (region.identity, region.region_id, seed.track_id, seed.fingerprint):
            raise ValueError("baseline region/track/seed mismatch")
        expected = [(f, p, i) for i, (f, p) in enumerate(zip(region.frame_ids, region.pts))]
        if [(f.global_frame_id, f.pts, f.local_index) for f in snapshot.frames] != expected:
            raise ValueError("baseline responsibility frame/PTS domain mismatch")
        for frame in snapshot.frames:
            if frame.status not in {"target_present", "target_absent", "target_visibility_unknown"}:
                raise ValueError("baseline contains failed response slots")
            if frame.available != (frame.status == "target_present") or frame.use_as_training_negative:
                raise ValueError("baseline availability/label flags inconsistent")
            if (frame.mask_identity is not None) != frame.available:
                raise ValueError("baseline mask availability mismatch")
            if frame.status == "target_absent" and not frame.absence_decision_identity:
                raise ValueError("baseline absence lacks decision provenance")
        count = sum(f.available for f in snapshot.frames)
        if snapshot.mask_available_count != count or snapshot.mask_available_coverage != count / len(snapshot.frames):
            raise ValueError("baseline coverage inconsistent")
        if (region.identity, seed.track_id) in self._publications:
            raise ValueError("baseline publication already exists")
        _name(snapshot.session_id, "baseline session ID"); _int(snapshot.obj_id, "baseline object ID")
        _int(snapshot.attempt, "baseline attempt"); _name(snapshot.dependency_identity, "baseline dependencies")
        if snapshot.prompt_log_identity != _fingerprint({"seed": seed.fingerprint, "corrections": None}):
            raise ValueError("baseline prompt-log identity mismatch")
        tracks = {x.track_id for x in self._records.values() if x.region_identity == region.identity}
        if seed.track_id in tracks or len(tracks) >= self.region_track_limit:
            raise ValueError("baseline track already registered or regional track budget exhausted")
        new = self._new(region, seed, worker_id, device_identity, snapshot.dependency_identity, 0, snapshot.attempt, None)
        restored = self._update(new, "restored_closed_baseline", state="CLOSED", published=True,
                                session_id=snapshot.session_id, obj_id=snapshot.obj_id, completed_rounds=1,
                                snapshot_identity=snapshot.identity, restored_from_receipt_identity=source_receipt_identity)
        self._snapshots[(restored.record_id, 0)] = snapshot
        self._publications[(region.identity, seed.track_id)] = restored.record_id
        return restored

    def current_publication(self, region_identity, track_id):
        """Return only the actively published snapshot, never a temporary round."""
        record_id = self._publications[(region_identity, track_id)]
        record = self.get(record_id)
        snapshot = self.snapshot(record_id)
        if not record.published or record.state not in {"READY", "CLOSED"} or record.snapshot_identity != snapshot.identity:
            raise ValueError("published snapshot eligibility/identity invalidated")
        return snapshot

    def publish_replacement(self, previous_record_id, revised_record_id):
        """Only READY + identity-confirmed gen1 can replace the active gen0 region."""
        old, new = self.get(previous_record_id), self.get(revised_record_id)
        if old.generation != 0 or old.state != "CLOSED" or not old.published:
            raise ValueError("replacement requires valid published and closed gen0")
        if new.generation != 1 or new.state != "READY" or not new.published or new.active_operation:
            raise ValueError("replacement requires READY identity-confirmed gen1 record")
        previous, revised = self.snapshot(previous_record_id), self.snapshot(revised_record_id)
        if old.snapshot_identity != previous.identity or new.snapshot_identity != revised.identity:
            raise ValueError("record and stored snapshot identity mismatch")
        key = (old.region_identity, old.track_id)
        if self._publications.get(key) != previous_record_id:
            raise ValueError("gen0 is not the current publication")
        result = replace_generation_snapshot(previous, revised)
        self._publications[key] = revised_record_id
        self._events.append(("replacement_published", revised_record_id, new.state, None))
        return result

    def _update(self, record, action, **changes):
        updated = replace(record, **changes); self._records[record.record_id] = updated
        self._events.append((action, updated.record_id, updated.state, updated.termination))
        return updated

    def plan(self, region, seed, *, worker_id, device_identity, dependency_identity):
        seed.validate_region(region)
        for label, value in (("worker", worker_id), ("device", device_identity), ("dependencies", dependency_identity)): _name(value, label)
        tracks = {r.track_id for r in self._records.values() if r.region_identity == region.identity}
        if seed.track_id in tracks: raise ValueError("track already planned; use retry/replacement contract")
        if len(tracks) >= self.region_track_limit: raise BudgetExceeded("regional track budget exhausted")
        return self._new(region, seed, worker_id, device_identity, dependency_identity, 0, 0, None)

    def _new(self, region, seed, worker, device, dependency, generation, attempt, replay_plan):
        key = _fingerprint([region.identity, seed.track_id, generation, attempt])
        if key in self._records: raise ValueError("duplicate session attempt identity")
        prompt_identity = _fingerprint({"seed": seed.fingerprint, "corrections": replay_plan.identity if replay_plan else None})
        record = SamSessionRecord(key, seed.track_id, seed.fingerprint, generation, attempt, region.identity,
                                  worker, device, dependency, prompt_identity)
        self._records[key] = record; self._regions[key] = region; self._seeds[key] = seed; self._plans[key] = replay_plan
        self._events.append(("planned", key, "PLANNED", None))
        return record

    def begin_create(self, record_id):
        r = self.get(record_id)
        if r.state != "PLANNED": raise ValueError("creation requires PLANNED")
        active = [x for x in self._records.values() if x.resource_reserved]
        if len(active) >= self.global_limit or sum(x.worker_id == r.worker_id for x in active) >= self.per_worker_limit:
            self._events.append(("queued_budget", r.record_id, r.state, None))
            raise BudgetExceeded("session capacity unavailable; keep queued PLANNED")
        if r.retry_concurrency_limit is not None and len(active) + 1 > r.retry_concurrency_limit:
            raise BudgetExceeded("OOM retry reduced-concurrency limit exceeded")
        # Existing OOM retry reservations also constrain later session admissions.
        if any(x.retry_concurrency_limit is not None and len(active) + 1 > x.retry_concurrency_limit for x in active):
            raise BudgetExceeded("active OOM retry holds reduced-concurrency reservation")
        return self._update(r, "creating", state="CREATING", resource_reserved=True, active_operation="start_session", creation_concurrency=len(active) + 1)

    def created(self, record_id, session_id):
        r = self.get(record_id); _name(session_id, "session_id")
        if r.state != "CREATING" or r.active_operation != "start_session": raise ValueError("unexpected session creation response")
        if any(x.session_id == session_id for x in self._records.values()): raise ValueError("session ID reused across attempts/tracks")
        return self._update(r, "created", session_id=session_id, active_operation=None)

    def seed_request(self, record_id, *, points_xy01=None, point_labels=None, obj_id=None, point_evidence_id=None):
        r = self.get(record_id); seed = self._seeds[record_id]
        if r.state != "CREATING" or not r.session_id or r.active_operation: raise ValueError("seed requires created idle session")
        if points_xy01 is None:
            if point_labels is not None or obj_id is not None or point_evidence_id is not None:
                raise ValueError("Partial point seed contract")
            self._update(r, "seed_requested", active_operation="seed")
            return {"type": "add_prompt", "session_id": r.session_id, "frame_index": seed.local_index,
                    "bounding_boxes": [list(seed.box_xywh01)], "bounding_box_labels": [1]}
        import numpy as np
        points=np.asarray(points_xy01,float);labels=tuple(point_labels or ())
        _int(obj_id,"point seed object");_name(point_evidence_id,"predicted point source")
        if (points.ndim!=2 or points.shape[1:]!=(2,) or len(points)!=len(labels)
                or not np.isfinite(points).all() or np.any((points<0)|(points>1))
                or any(type(v) is not int or v not in (0,1) for v in labels)
                or not 2<=labels.count(1)<=4 or not 2<=labels.count(0)<=4):
            raise ValueError("Predicted point seed requires2-4 positive and2-4 negative normalized points")
        request={"type":"add_prompt","session_id":r.session_id,"frame_index":seed.local_index,
                 "obj_id":obj_id,"points":points.tolist(),"point_labels":list(labels),
                 "rel_coordinates":True,"clear_old_points":True}
        log=_fingerprint({"previous":r.prompt_log_identity,"request":request,"evidence":point_evidence_id})
        self._update(r,"instance_point_seed_requested",active_operation="seed",prompt_log_identity=log)
        return request

    def bind_seed(self, record_id, unique_candidate_obj_ids):
        r = self.get(record_id); ids = tuple(unique_candidate_obj_ids)
        if r.active_operation != "seed": raise ValueError("no seed request pending")
        if len(ids) != 1:
            return self._update(r, "seed_identity_failed", state="FAILED", active_operation=None,
                                termination="ambiguous_identity", failure_kind="identity")
        _int(ids[0], "object ID")
        return self._update(r, "seed_bound", state="SEEDED", obj_id=ids[0], active_operation=None)

    def correction_request(self, record_id, event_id):
        r = self.get(record_id); plan = self._plans[record_id]
        if r.generation != 1 or plan is None or r.state != "ASSOCIATED" or r.active_operation or r.completed_rounds != 1:
            raise ValueError("correction requires new gen1 session with completed seed propagation and association")
        events = [e for e in plan.events if e.track_id == r.track_id]
        index = len(r.applied_event_ids)
        if index >= len(events) or events[index].event_id != event_id: raise ValueError("correction log order/membership differs")
        event = events[index]
        if isinstance(event, PVSBoxPromptEvent):
            request = event.to_request(self._regions[record_id], self._seeds[record_id], session_id=r.session_id, obj_id=r.obj_id)
            self._update(r, "correction_requested", active_operation="correction:" + event_id)
            return request
        self._update(r, "correction_requested", active_operation="correction:" + event_id)
        return {"type": "add_prompt", "session_id": r.session_id, "frame_index": event.local_index,
                "obj_id": r.obj_id, "points": [list(p) for p in event.points_xy01], "point_labels": list(event.labels),
                "rel_coordinates": True, "clear_old_points": True}

    def correction_applied(self, record_id, event_id):
        r = self.get(record_id)
        if r.active_operation != "correction:" + event_id: raise ValueError("correction response mismatch")
        return self._update(r, "correction_applied", active_operation=None, applied_event_ids=r.applied_event_ids + (event_id,))

    def begin_propagation(self, record_id):
        r = self.get(record_id)
        if r.state not in {"SEEDED", "ASSOCIATED"} or r.active_operation: raise ValueError("session busy or not seeded")
        if r.completed_rounds:
            plan = self._plans[record_id]
            expected = tuple(e.event_id for e in plan.events if e.track_id == r.track_id) if plan else ()
            if r.generation != 1 or r.completed_rounds != 1 or not expected or r.applied_event_ids != expected:
                raise ValueError("only one complete fixed-correction propagation allowed")
        record = self._update(r, "propagating", state="PROPAGATING", active_operation="propagate_both")
        return RoundCollector(record, self._regions[record_id], r.completed_rounds)

    def finish_propagation(self, record_id, collector):
        r = self.get(record_id)
        if r.state != "PROPAGATING" or collector.record.record_id != record_id or collector.round_index != r.completed_rounds:
            raise ValueError("propagation collector identity/round mismatch")
        snapshot = collector.seal(); self._snapshots[(record_id, r.completed_rounds)] = snapshot
        # A missing early frame must not hide a later OOM/schema/identity failure
        # and accidentally grant a less restrictive transient retry.
        failures = [f for f in snapshot.frames if f.status not in {"target_present", "target_absent", "target_visibility_unknown"}]
        priority = {"schema": 0, "identity": 1, "oom": 2, "worker": 3, "transient": 4}
        failed_frame = min(failures, key=lambda f: (priority[f.failure_kind], f.local_index)) if failures else None
        failed = failed_frame.status if failed_frame else None
        self._update(r, "propagation_completed", state="FAILED" if failed else "ASSOCIATED", active_operation=None,
                     termination=failed, failure_kind=failed_frame.failure_kind if failed_frame else None,
                     failure_concurrency=self.active_count if failed else 0,
                     completed_rounds=r.completed_rounds + 1, snapshot_identity=snapshot.identity)
        return snapshot

    def publish(self, record_id, *, identity_confirmed):
        r = self.get(record_id)
        if type(identity_confirmed) is not bool or not identity_confirmed:
            return self.fail(record_id, status="ambiguous_identity", failure_kind="identity")
        if r.state != "ASSOCIATED" or r.active_operation: raise ValueError("publish requires complete idle association")
        plan = self._plans[record_id]
        expected = tuple(e.event_id for e in plan.events if e.track_id == r.track_id) if plan else ()
        if r.generation == 1 and (r.completed_rounds != 2 or r.applied_event_ids != expected):
            raise ValueError("gen1 requires seed propagation, locked corrections and full revised propagation")
        key = (r.region_identity, r.track_id)
        if r.generation == 0 and key in self._publications and self._publications[key] != record_id:
            raise ValueError("another record is already published for this region/track")
        result = self._update(r, "ready", state="READY", published=True)
        if r.generation == 0: self._publications[key] = record_id
        # Gen1 readiness does not silently replace gen0; controlled publication is
        # a separate whole-domain step after both record identities are checked.
        return result

    def fail(self, record_id, *, status, failure_kind):
        r = self.get(record_id)
        allowed = {"inference_error": {"transient", "oom", "schema"}, "session_lost": {"worker"},
                   "ambiguous_identity": {"identity"}, "missing_response": {"transient"}}
        if status not in allowed or failure_kind not in allowed[status] or (r.state == "CLOSED" and not r.published): raise ValueError("invalid failure status/kind")
        key = (r.region_identity, r.track_id)
        if self._publications.get(key) == record_id:
            del self._publications[key]
        return self._update(r, "failed", state="FAILED", termination=status, failure_kind=failure_kind,
                            active_operation=None, failure_concurrency=self.active_count, published=False)

    def close(self, record_id, *, released):
        r = self.get(record_id)
        if type(released) is not bool: raise ValueError("explicit backend memory release acknowledgement required")
        if r.state == "CLOSED": return r
        if r.active_operation and r.state != "FAILED": raise ValueError("cannot close a busy session without failure/cancellation")
        if not released:
            self._events.append(("close_not_acknowledged", record_id, r.state, r.termination))
            return r
        return self._update(r, "closed", state="CLOSED", resource_reserved=False, active_operation=None)

    def retry(self, record_id, *, worker_id=None, oom_concurrency_limit=None):
        r = self.get(record_id)
        if r.state != "CLOSED" or r.resource_reserved or r.attempt >= 1 or r.failure_kind not in {"transient", "worker", "oom"}:
            raise ValueError("retry requires released retryable failure and at most one retry")
        if r.failure_kind == "oom":
            _int(oom_concurrency_limit, "OOM reduced concurrency", 1)
            if oom_concurrency_limit >= r.failure_concurrency:
                raise ValueError("OOM retry must reduce measured session concurrency; single-session OOM cannot reduce further")
        elif oom_concurrency_limit is not None:
            raise ValueError("reduced concurrency override only applies to OOM retry")
        worker = r.worker_id if worker_id is None else worker_id; _name(worker, "worker")
        new = self._new(self._regions[record_id], self._seeds[record_id], worker, r.device_identity,
                        r.dependency_identity, r.generation, r.attempt + 1, self._plans[record_id])
        if oom_concurrency_limit is not None:
            new = self._update(new, "oom_retry_cap", retry_concurrency_limit=oom_concurrency_limit)
        return new

    def replacement(self, record_id, replay_plan, *, worker_id=None):
        r = self.get(record_id)
        if r.generation != 0 or r.state != "CLOSED" or r.resource_reserved or r.termination is not None or not r.snapshot_identity or not r.published:
            raise ValueError("gen1 replacement requires closed successfully propagated gen0")
        if not isinstance(replay_plan, ReplayPlan) or replay_plan.source_generation != 0 or replay_plan.region_identity != r.region_identity:
            raise ValueError("locked gen0 replay plan from same region required")
        candidates = {(s.track_id, s.fingerprint): s for s in self._seeds.values() if s.region_identity == r.region_identity}
        if any(key not in candidates for key in replay_plan.seed_fingerprints): raise ValueError("replay plan includes unregistered regional seed")
        validated = lock_replay_plan(self._regions[record_id], [candidates[k] for k in replay_plan.seed_fingerprints], replay_plan.events)
        if validated != replay_plan or (r.track_id, r.seed_fingerprint) not in replay_plan.seed_fingerprints or not any(e.track_id == r.track_id for e in validated.events):
            raise ValueError("replacement must replay exact seed and nonempty frozen corrections")
        worker = r.worker_id if worker_id is None else worker_id; _name(worker, "worker")
        return self._new(self._regions[record_id], self._seeds[record_id], worker,
                         r.device_identity, r.dependency_identity, 1, 0, replay_plan)
