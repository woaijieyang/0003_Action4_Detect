"""Prediction-only material qualification; no tracker, GT, or pose solver calls.

Consumes action4.provisional_material_seeds.v1 and returns existing Binding keys.
Activation establishes repeated prediction consistency, not independent CAD truth.
"""
from dataclasses import dataclass, replace, asdict
import hashlib
import json
import math
from numbers import Real
from model.action4_phase2_tracking_contracts import AnchorKey, Binding

EXCLUSIONS = ('reflection', 'screen', 'strap', 'mount')


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Nonempty prediction provenance required')
    return value


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer required')
    return value


def _vector(value, length, *, finite=True):
    result = tuple(value)
    if len(result) != length or any(isinstance(v, bool) or not isinstance(v, Real) for v in result):
        raise ValueError('Numeric coordinate vector required')
    result = tuple(float(v) for v in result)
    if finite and not all(math.isfinite(v) for v in result):
        raise ValueError('Finite binding coordinates required')
    return result


@dataclass(frozen=True)
class LifecyclePolicy:
    min_nonquery_frames: int = 3
    visibility_min: float = .9
    confidence_min: float = .6
    distance_fraction: float = .02

    def __post_init__(self):
        if type(self.min_nonquery_frames) is not int or self.min_nonquery_frames < 3:
            raise ValueError('At least three distinct nonquery confirmations required')
        for value in (self.visibility_min, self.confidence_min, self.distance_fraction):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('Finite registered positive thresholds required')
        if self.distance_fraction > .02:
            raise ValueError('Binding tolerance cannot exceed registered two percent CAD diagonal')


@dataclass(frozen=True)
class PredictedMaterialObservation:
    key: AnchorKey
    track_generation: int
    frame_id: int
    source_id: str
    raw_uv: tuple
    visibility: float
    confidence: float
    is_query: bool
    nonpadding: bool
    visibility_state: str
    sam_attribution: str
    sam_track_id: str | None
    surface_xyz_m: tuple | None
    surface_triangle_id: int | None
    surface_supported: bool
    exclusions: tuple
    exclusion_source_id: str
    surface_source_id: str
    sam_source_id: str
    origin: str = 'prediction'
    track_state: str = 'observed'

    def __post_init__(self):
        if not isinstance(self.key, AnchorKey): raise ValueError('Typed material key required')
        for v in (self.track_generation, self.frame_id): _integer(v)
        for v in (self.source_id, self.exclusion_source_id, self.surface_source_id, self.sam_source_id): _name(v)
        if self.origin != 'prediction': raise ValueError('GT or solver qualification is forbidden')
        if self.track_state not in ('observed', 'lost'): raise ValueError('Explicit track state required')
        if self.visibility_state not in ('visible', 'occluded', 'unknown'): raise ValueError('Explicit visibility state required')
        if self.sam_attribution not in ('same_instance', 'other_instance', 'unknown'): raise ValueError('Explicit SAM attribution required')
        if self.sam_track_id is not None: _name(self.sam_track_id)
        for v in (self.is_query, self.nonpadding, self.surface_supported):
            if type(v) is not bool: raise ValueError('Explicit boolean gates required')
        for v in (self.visibility, self.confidence):
            if isinstance(v, bool) or not isinstance(v, Real): raise ValueError('Continuous v/c numbers required')
        object.__setattr__(self, 'visibility', float(self.visibility))
        object.__setattr__(self, 'confidence', float(self.confidence))
        object.__setattr__(self, 'raw_uv', _vector(self.raw_uv, 2, finite=False))
        if self.surface_xyz_m is not None:
            object.__setattr__(self, 'surface_xyz_m', _vector(self.surface_xyz_m, 3, finite=False))
        if self.surface_triangle_id is not None: _integer(self.surface_triangle_id)
        exclusions = tuple(tuple(row) for row in self.exclusions)
        if (len(exclusions) != 4 or {r[0] for r in exclusions if len(r) == 2} != set(EXCLUSIONS)
                or any(len(r) != 2 or r[1] not in ('clear', 'excluded', 'unknown') for r in exclusions)):
            raise ValueError('All four exclusion categories need explicit clear/excluded/unknown evidence')
        object.__setattr__(self, 'exclusions', tuple(sorted(exclusions)))

    @property
    def fingerprint(self):
        # NaN remains a diagnostic value and hashes deterministically for duplicates.
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=True).encode()).hexdigest()


@dataclass(frozen=True)
class MaterialLifecycle:
    binding: Binding
    track_generation: int
    triangle_id: int
    cad_diagonal_m: float
    policy: LifecyclePolicy
    seed_dependency_id: str
    compatible_triangle_ids: tuple
    face_compatibility_source: str
    state: str = 'provisional'
    reason: str = 'awaiting_nonquery_confirmation'
    confirmed_frames: tuple = ()
    seen_frames: tuple = ()
    last_observation_eligible: bool = False
    last_observation_frame_id: int | None = None

    @property
    def key(self): return self.binding.key

    @property
    def current_geometry_eligible(self):
        return self.state == 'active_for_geometry' and self.last_observation_eligible

    def qualified_binding(self, *, frame_id):
        _integer(frame_id)
        if frame_id != self.last_observation_frame_id:
            raise ValueError('Geometry frame differs from the explicit last observation')
        if not self.current_geometry_eligible:
            raise ValueError('Current observation is not admitted for geometry')
        return self.binding

    def assert_seed_unchanged(self, seed):
        if (AnchorKey(**seed['key']) != self.key or tuple(seed['X_cad_m']) != self.binding.xyz_m
                or seed['mesh_triangle_id'] != self.triangle_id or seed['track_generation'] != self.track_generation
                or seed['query_frame_id'] != self.binding.binding_frame):
            raise ValueError('Binding identity/XYZ/face changed; retire and initialize a new generation')

    def observe(self, observation):
        if not isinstance(observation, PredictedMaterialObservation):
            raise ValueError('Prediction-only typed observation required')
        if observation.key != self.key:
            raise ValueError('Observation belongs to another anchor generation')
        fingerprint = observation.fingerprint
        seen = dict(self.seen_frames)
        if observation.frame_id in seen:
            if seen[observation.frame_id] != fingerprint:
                raise ValueError('Contradictory duplicate frame observation')
            return self
        if self.state in ('rejected', 'retired'):
            raise ValueError('Terminal material lifecycle cannot be reused')
        if observation.track_generation != self.track_generation or observation.track_state == 'lost':
            return replace(self, state='retired', reason='track_generation_changed' if observation.track_generation != self.track_generation else 'identity_lost',
                           seen_frames=self.seen_frames + ((observation.frame_id, fingerprint),), last_observation_eligible=False, last_observation_frame_id=observation.frame_id)
        reason = self._qualification_reason(observation)
        confirmations = self.confirmed_frames
        if reason is None:
            confirmations = tuple(sorted(confirmations + (observation.frame_id,)))
        state = 'active_for_geometry' if len(confirmations) >= self.policy.min_nonquery_frames else 'provisional'
        if reason == 'excluded_material': state = 'rejected'
        return replace(self, state=state, reason=reason or ('prediction_consistency_confirmed' if state == 'active_for_geometry' else 'awaiting_nonquery_confirmation'),
                       confirmed_frames=confirmations, seen_frames=self.seen_frames + ((observation.frame_id, fingerprint),), last_observation_eligible=reason is None, last_observation_frame_id=observation.frame_id)

    def _qualification_reason(self, obs):
        if any(status == 'excluded' for _, status in obs.exclusions): return 'excluded_material'
        if obs.is_query or obs.frame_id == self.binding.binding_frame: return 'query_frame'
        if not all(math.isfinite(v) for v in obs.raw_uv): return 'nonfinite_track'
        if not obs.nonpadding: return 'padding'
        if obs.visibility_state != 'visible': return obs.visibility_state
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (obs.visibility, obs.confidence)): return 'invalid_continuous_scores'
        if obs.visibility < self.policy.visibility_min or obs.confidence < self.policy.confidence_min: return 'low_continuous_scores'
        if obs.sam_attribution != 'same_instance': return 'sam_' + obs.sam_attribution
        if obs.sam_track_id != self.key.track_id: return 'sam_identity_mismatch'
        exclusions = dict(obs.exclusions)
        if 'excluded' in exclusions.values(): return 'excluded_material'
        if 'unknown' in exclusions.values(): return 'exclusion_unknown'
        if not obs.surface_supported or obs.surface_xyz_m is None or obs.surface_triangle_id is None: return 'surface_unknown'
        if not all(math.isfinite(v) for v in obs.surface_xyz_m): return 'surface_nonfinite'
        if obs.surface_triangle_id not in self.compatible_triangle_ids: return 'surface_face_incompatible'
        distance = math.dist(obs.surface_xyz_m, self.binding.xyz_m)
        if distance > self.policy.distance_fraction * self.cad_diagonal_m: return 'surface_binding_distance'
        return None

    def retire(self, reason):
        _name(reason)
        return replace(self, state='retired', reason=reason, last_observation_eligible=False)


def lifecycles_from_seed_packet(packet, *, cad_diagonal_m, policy=LifecyclePolicy(), face_compatibility=None):
    """Direct consumer of seed_material_candidates output; no new seed convention.

    face_compatibility maps seed triangle ID -> {'triangle_ids': [...], 'source_id': ...}.
    Omitted mapping accepts only the same mesh triangle (conservative).
    Caller evidence must come from frozen CAD/prediction logic, never solver/GT.
    """
    if packet.get('schema') != 'action4.provisional_material_seeds.v1' or packet.get('query_role') != 'material_candidates':
        raise ValueError('Actual material-candidate selector packet required')
    if packet.get('synthetic_fixture_allowed') is not False or packet.get('time_identity_verified') is not True:
        raise ValueError('Verified source-video seed timing required')
    if not isinstance(policy, LifecyclePolicy) or isinstance(cad_diagonal_m, bool) or not math.isfinite(cad_diagonal_m) or cad_diagonal_m <= 0:
        raise ValueError('Fixed positive CAD diagonal and policy required')
    dependency = _name(packet.get('mesh_binding_identity'))
    if packet.get('geometry_accepted') is not False: raise ValueError('Seed packet must be provisional')
    result = {}
    for seed in packet['seeds']:
        key = AnchorKey(**seed['key'])
        if key in result or key.track_id != packet['track_id'] or seed['track_generation'] != packet['track_generation'] or key.generation != packet['anchor_generation']:
            raise ValueError('Duplicate or mismatched selector identity')
        if seed.get('lifecycle_state') != 'provisional' or seed.get('geometry_accepted') is not False:
            raise ValueError('Only provisional selector seeds accepted')
        if seed.get('XYZ_source') != 'A0_estimated_surface_nearest_fixed_mesh' or seed.get('binding_identity') != dependency:
            raise ValueError('Prediction-derived fixed mesh provenance required')
        triangle = _integer(seed['mesh_triangle_id']); generation = _integer(seed['track_generation'])
        xyz = _vector(seed['X_cad_m'], 3)
        binding = Binding(key, xyz, seed['XYZ_source'], _integer(seed['query_frame_id']), _name(seed['binding_quality']))
        compatible = (triangle,); compatibility_source = dependency + ':same_triangle_only'
        if face_compatibility is not None and triangle in face_compatibility:
            specification = face_compatibility[triangle]
            compatible = tuple(sorted({_integer(v) for v in specification['triangle_ids']}))
            compatibility_source = _name(specification['source_id'])
            if triangle not in compatible: raise ValueError('Compatibility must include the fixed source face')
        result[key] = MaterialLifecycle(binding, generation, triangle, float(cad_diagonal_m), policy,
            dependency, compatible, compatibility_source)
    return result


def replace_binding(previous, new_lifecycle):
    """Explicit generation switch; old immutable state is retired, never overwritten."""
    if not isinstance(previous, MaterialLifecycle) or not isinstance(new_lifecycle, MaterialLifecycle):
        raise ValueError('Typed lifecycle states required')
    if (previous.key.track_id, previous.key.anchor_id) != (new_lifecycle.key.track_id, new_lifecycle.key.anchor_id) or new_lifecycle.key.generation <= previous.key.generation:
        raise ValueError('Rebinding requires strictly newer generation of the same material ID')
    if new_lifecycle.state != 'provisional' or new_lifecycle.seen_frames:
        raise ValueError('New binding must be fresh provisional state')
    return previous.retire('binding_generation_replaced'), new_lifecycle
