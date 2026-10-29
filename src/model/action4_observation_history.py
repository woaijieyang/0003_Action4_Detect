"""Bounded accepted-image observation history for diagnostics/association only.

This module never accepts/rejects measurements by residual magnitude, smooths
a current observation, or publishes independently predicted corners. Acceptance
and evidence authenticity are responsibilities of the caller.
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

_IMAGE_SOURCES = frozenset(("image_keypoint", "image_face_feature", "image_detection_center",
                            "image_rigid_pose_projection"))
_NON_EVIDENCE = frozenset(("predicted_only", "rejected", "missing", "raw_mask_centroid"))


def _nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative integer")


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Explicit " + label + " required")


@dataclass(frozen=True)
class ObservationKey:
    body_track_id: str
    body_generation: int
    node_kind: str
    node_id: int
    node_generation: int

    def __post_init__(self):
        _name(self.body_track_id, "body track ID")
        _nonnegative(self.body_generation, "body generation")
        _nonnegative(self.node_generation, "node generation")
        limits = {"body_center": 1, "face": 6, "corner": 8}
        if self.node_kind not in limits or type(self.node_id) is not int or not 0 <= self.node_id < limits[self.node_kind]:
            raise ValueError("Fixed body-center(0), face(0..5), or corner(0..7) node ID required")


@dataclass(frozen=True)
class _Observation:
    key: ObservationKey
    points: tuple
    shape: tuple
    frame_id: int
    pts: int
    time_base: tuple
    pixel_space_id: str
    source_kind: str
    evidence_id: str
    evidence_group_id: str

    @property
    def timestamp(self):
        return Fraction(self.pts * self.time_base[0], self.time_base[1])

    def array(self):
        return np.asarray(self.points, dtype=float).reshape(self.shape)


class AcceptedObservationHistory:
    def __init__(self, *, max_horizon_seconds, max_gap_seconds):
        for value in (max_horizon_seconds, max_gap_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Explicit positive finite horizon and gap required")
        self.max_horizon_seconds = float(max_horizon_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self._history = {}
        self._active = {}
        self._body_generation = {}
        # Processing state is independent of accepted observation storage.
        # One watermark per logical node includes its current full identity.
        self._processed = {}

    @staticmethod
    def _base(key):
        if not isinstance(key, ObservationKey):
            raise TypeError("Typed ObservationKey required")
        return key.body_track_id, key.node_kind, key.node_id

    @staticmethod
    def _domain(frame_id, pts, time_base, pixel_space_id):
        _nonnegative(frame_id, "frame ID")
        if type(pts) is not int:
            raise ValueError("Integer source PTS required")
        if (not isinstance(time_base, tuple) or len(time_base) != 2
                or any(type(x) is not int or x <= 0 for x in time_base)):
            raise ValueError("Exact positive rational time_base tuple required")
        if pixel_space_id not in ("raw_video_native_integer_centers", "native_integer_centers"):
            raise ValueError("Explicit native integer-centre pixel space required")
        return Fraction(pts * time_base[0], time_base[1])

    @staticmethod
    def _points(position):
        array = np.asarray(position, dtype=float)
        if (array.shape != (2,) and (array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1)
                or not np.isfinite(array).all()):
            raise ValueError("Finite native (2,) point or nonempty (N,2) point array required")
        return array

    @staticmethod
    def _unknown(reason, count=0):
        return {"status": "unknown", "reason": reason, "position": None,
                "velocity_px_per_second": None, "prediction_only": True, "publishable": False,
                "history_count": count, "accepted_observation_count": count,
                "used_observation_count": 0, "used_segment_count": 0, "independent_ground_truth": False}

    @staticmethod
    def _check_domain(last, time_base, pixel_space_id, shape=None):
        if last.time_base != time_base or last.pixel_space_id != pixel_space_id:
            raise ValueError("Observation time base/pixel space differs")
        if shape is not None and last.shape != shape:
            raise ValueError("Observation point-array shape differs")

    def prior(self, key, *, frame_id, pts, time_base, pixel_space_id):
        """Median adjacent-dt velocity from all retained accepted observations."""
        base = self._base(key)
        target = self._domain(frame_id, pts, time_base, pixel_space_id)
        generation = (key.body_generation, key.node_generation)
        if self._body_generation.get(key.body_track_id) != key.body_generation or self._active.get(base) != generation:
            return self._unknown("no_history_or_generation_mismatch")
        history = self._history.get(key, ())
        if not history:
            return self._unknown("no_accepted_history")
        last = history[-1]
        self._check_domain(last, time_base, pixel_space_id)
        if frame_id <= last.frame_id or target <= last.timestamp:
            raise ValueError("Prior target must strictly follow accepted history")
        if len(history) < 2:
            return self._unknown("insufficient_history", len(history))
        horizon = float(target - last.timestamp)
        if horizon > self.max_horizon_seconds:
            return self._unknown("maximum_horizon_exceeded", len(history))
        if horizon > self.max_gap_seconds:
            return self._unknown("observation_gap_exceeded", len(history))
        retained = list(history)
        velocities, segments = [], []
        for previous, current in zip(retained, retained[1:]):
            dt = float(current.timestamp - previous.timestamp)
            if not math.isfinite(dt) or dt <= 0 or dt > self.max_gap_seconds:
                return self._unknown("history_gap_exceeded", len(history))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                segment_velocity = (current.array() - previous.array()) / dt
            if not np.isfinite(segment_velocity).all():
                # Do not silently discard a bad segment to manufacture a prior.
                return self._unknown("nonfinite_extrapolation", len(history))
            velocities.append(segment_velocity)
            segments.append({"from_frame_id": previous.frame_id, "to_frame_id": current.frame_id,
                "from_pts": previous.pts, "to_pts": current.pts,
                "time_base": list(current.time_base), "dt_seconds": dt,
                "velocity_px_per_second": segment_velocity.tolist(),
                "from_evidence_id": previous.evidence_id, "to_evidence_id": current.evidence_id,
                "evidence_group_ids": list(dict.fromkeys((previous.evidence_group_id,current.evidence_group_id)))})
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = np.median(np.stack(velocities), axis=0)
            position = last.array() + horizon * velocity
        if not np.isfinite(velocity).all() or not np.isfinite(position).all():
            return self._unknown("nonfinite_extrapolation", len(history))
        sources = [self._record(row) for row in retained]
        return {"status": "available", "position": position.tolist(),
                "velocity_px_per_second": velocity.tolist(), "horizon_seconds": horizon,
                "velocity_interval_seconds": segments[0]["dt_seconds"] if len(segments)==1 else None,
                "velocity_segments": segments, "prediction_only": True, "publishable": False,
                "history_count": len(history), "accepted_observation_count": len(history),
                "used_observation_count": len(history), "used_segment_count": len(segments),
                "extrapolation_anchor_frame_id": last.frame_id, "extrapolation_anchor_pts": last.pts,
                "method": "all_retained_accepted_adjacent_actual_PTS_componentwise_median_velocity",
                "source_observations": sources,
                "evidence_group_ids": list(dict.fromkeys(s["evidence_group_id"] for s in sources)),
                "independent_ground_truth": False,
                "correlated": True if any(s["correlated"] is True for s in sources) else None}

    @staticmethod
    def _record(row):
        rigid = row.source_kind == "image_rigid_pose_projection"
        return {"key": row.key.__dict__.copy(), "position": row.array().tolist(),
                "frame_id": row.frame_id, "pts": row.pts, "time_base": list(row.time_base),
                "pixel_space_id": row.pixel_space_id, "source_kind": row.source_kind,
                "evidence_id": row.evidence_id, "evidence_group_id": row.evidence_group_id,
                "accepted_current_image": True, "correlated": True if rigid else None,
                "independent_ground_truth": False,
                "physical_corner_observed": False if rigid else None,
                "semantics": "image-derived rigid-pose projection; hidden corner not a direct observation" if rigid
                             else "caller-accepted current-image feature; not independently certified ground truth"}

    def observations(self, key):
        self._base(key)
        return [self._record(row) for row in self._history.get(key, ())]

    def update(self, key, position, *, frame_id, pts, time_base, pixel_space_id,
               source_kind, evidence_id, accepted_current_image, evidence_group_id=None):
        """Return the unchanged measurement, diagnostic innovation, and admission.

        accepted_current_image is a caller decision, never inferred by this
        module. A large innovation never vetoes an accepted current measurement.
        Every valid event advances processing time/identity, even if not stored.
        One identical-frame/PTS/domain non-observation-to-accepted upgrade is
        allowed. Other changed same-frame events reject; exact repeats are
        no-ops. A retained exact accepted repeat may arrive late as a no-op but
        cannot rewind processing state. prior() queries never advance it.
        """
        base = self._base(key)
        timestamp = self._domain(frame_id, pts, time_base, pixel_space_id)
        if type(accepted_current_image) is not bool:
            raise ValueError("Explicit current-image acceptance boolean required")
        if source_kind not in _IMAGE_SOURCES | _NON_EVIDENCE:
            raise ValueError("Unknown observation source semantics")
        _name(evidence_id, "evidence ID")
        if accepted_current_image and source_kind not in _IMAGE_SOURCES:
            raise ValueError("Predicted/rejected/missing/raw-mask-centroid is not accepted image evidence")
        if source_kind == "image_rigid_pose_projection":
            _name(evidence_group_id, "shared rigid-pose/frame evidence_group_id")
        if evidence_group_id is None:
            evidence_group_id = evidence_id
        _name(evidence_group_id, "evidence group ID")
        points = None if position is None else self._points(position)
        if accepted_current_image and points is None:
            raise ValueError("Accepted current observation cannot be missing")
        generation = (key.body_generation, key.node_generation)
        active = self._active.get(base)
        body_generation = self._body_generation.get(key.body_track_id)
        if body_generation is not None and key.body_generation < body_generation:
            raise ValueError("Stale body generation cannot re-enter any node history")
        if active is not None and generation < active:
            raise ValueError("Stale body/node generation cannot re-enter history")
        history = self._history.get(key, ())
        row = None if points is None else _Observation(key, tuple(points.reshape(-1)), points.shape,
            frame_id, pts, time_base, pixel_space_id, source_kind, evidence_id, evidence_group_id)
        if history and active == generation:
            last = history[-1]
            self._check_domain(last, time_base, pixel_space_id, None if points is None else points.shape)
            # Exact retained duplicate is idempotent, even if delivered late.
            same = [x for x in history if x.frame_id == frame_id]
            if same:
                if accepted_current_image and row == same[0]:
                    return {"stored": False, "idempotent": True, "measurement": points.tolist(),
                            "prior": self._unknown("already_ingested", len(history)), "innovation": None,
                            "prediction_only": True, "publishable": False}
                raise ValueError("Conflicting observation for the same frame")
            if frame_id <= last.frame_id or timestamp <= last.timestamp:
                raise ValueError("Accepted observation frame and PTS must strictly increase")
            if any(x.evidence_id == evidence_id for x in history):
                raise ValueError("Evidence ID reused for a different current frame")
        signature = (key, frame_id, pts, time_base, pixel_space_id,
                     None if points is None else (points.shape, tuple(points.reshape(-1))),
                     source_kind, evidence_id, evidence_group_id, accepted_current_image)
        watermark = self._processed.get(base)
        upgrade = False
        if watermark is not None:
            if watermark["time_base"] != time_base or watermark["pixel_space_id"] != pixel_space_id:
                raise ValueError("Processed event time base/pixel space differs")
            if (points is not None and watermark["key"] == key and watermark["shape"] is not None
                    and watermark["shape"] != points.shape):
                raise ValueError("Processed event point-array shape differs")
            previous_time = Fraction(watermark["pts"] * time_base[0], time_base[1])
            if frame_id < watermark["frame_id"] or timestamp < previous_time:
                raise ValueError("Event precedes processed frame/PTS watermark")
            if frame_id == watermark["frame_id"] or timestamp == previous_time:
                if frame_id != watermark["frame_id"] or timestamp != previous_time:
                    raise ValueError("Same-frame processed event requires identical PTS")
                if watermark["signature"] == signature:
                    return {"stored": False, "idempotent": True,
                            "measurement": None if points is None else points.tolist(),
                            "prior": self._unknown("already_processed",len(history)),
                            "innovation": None, "prediction_only": True, "publishable": False}
                if watermark["key"] == key:
                    upgrade = not watermark["accepted"] and accepted_current_image
                    if not upgrade:
                        raise ValueError("Conflicting same-frame processed event; only non-observation to accepted upgrade allowed")
                # A new generation may begin on the same source frame. It has
                # its own identity and never inherits the old accepted deque.
        prior = self.prior(key, frame_id=frame_id, pts=pts, time_base=time_base,
                           pixel_space_id=pixel_space_id)
        innovation = None
        if points is not None and prior["status"] == "available":
            with np.errstate(over="ignore", invalid="ignore"):
                delta = points - np.asarray(prior["position"], float)
            if np.isfinite(delta).all():
                innovation = delta.tolist()
        reset = None
        # Every validated event, including missing/predicted/rejected events,
        # advances generation ownership and the processing watermark.
        if body_generation != key.body_generation:
            for old in [k for k in self._history if k.body_track_id == key.body_track_id]:
                del self._history[old]
            for old in [b for b in self._active if b[0] == key.body_track_id]:
                del self._active[old]
            self._body_generation[key.body_track_id] = key.body_generation
            active = None
        if active != generation:
            for old in [k for k in self._history if self._base(k) == base]:
                del self._history[old]
            self._active[base] = generation
            self._history[key] = deque(maxlen=5)
            reset = "new_generation"
        if accepted_current_image:
            if history and reset is None and float(timestamp - history[-1].timestamp) > self.max_gap_seconds:
                self._history[key].clear()
                reset = "observation_gap"
            self._history[key].append(row)
        self._processed[base] = {"key": key, "frame_id": frame_id, "pts": pts,
            "time_base": time_base, "pixel_space_id": pixel_space_id,
            "shape": points.shape if points is not None else
                     watermark["shape"] if watermark is not None and watermark["key"] == key else None,
            "accepted": accepted_current_image, "signature": signature}
        return {"stored": accepted_current_image, "idempotent": False, "same_frame_upgrade": upgrade,
                "not_stored_reason": None if accepted_current_image else "not_accepted_current_image_evidence",
                "measurement": None if points is None else points.tolist(),
                "measurement_evidence": {"source_kind": source_kind, "evidence_id": evidence_id,
                    "evidence_group_id": evidence_group_id,
                    "correlated": True if source_kind == "image_rigid_pose_projection" else None,
                    "physical_corner_observed": False if source_kind == "image_rigid_pose_projection" else None,
                    "independent_ground_truth": False}, "prior": prior,
                "innovation": innovation, "innovation_semantics": "measurement_minus_prior_no_acceptance_gate",
                "history_reset": reset, "history_count": len(self._history.get(key, ())),
                "prediction_only": True, "publishable": False, "independent_ground_truth": False}
