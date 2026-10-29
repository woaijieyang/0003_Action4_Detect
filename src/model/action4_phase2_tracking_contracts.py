"""Pure CPU Stage 6/7 contracts. No model, annotation reader, or solver is loaded."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np


@dataclass(frozen=True, order=True)
class AnchorKey:
    track_id: str
    anchor_id: str
    generation: int

    def __post_init__(self):
        if not isinstance(self.track_id, str) or not self.track_id or not isinstance(self.anchor_id, str) or not self.anchor_id or type(self.generation) is not int or self.generation < 0:
            raise ValueError("nonempty IDs and nonnegative integer generation required")


@dataclass(frozen=True)
class Binding:
    key: AnchorKey
    xyz_m: tuple
    xyz_source: str
    binding_frame: int
    binding_quality: str
    heldout: bool = False

    def __post_init__(self):
        xyz = np.asarray(self.xyz_m, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("finite XYZ[3] required")
        if not isinstance(self.key, AnchorKey) or not isinstance(self.xyz_source, str) or not self.xyz_source or not isinstance(self.binding_quality, str) or not self.binding_quality or type(self.binding_frame) is not int or type(self.heldout) is not bool:
            raise ValueError("binding provenance required")
        object.__setattr__(self, "xyz_m", tuple(float(x) for x in xyz))


class AnchorRegistry:
    """Immutable full-identity registry; rebinding requires a new generation."""
    __slots__ = ("_bindings",)

    def __init__(self, bindings):
        entries = {}
        for binding in bindings:
            if not isinstance(binding, Binding) or binding.key in entries:
                raise ValueError("invalid or duplicate binding identity")
            entries[binding.key] = binding
        object.__setattr__(self, "_bindings", MappingProxyType(entries))

    def __setattr__(self, name, value):
        raise AttributeError("registry is immutable")

    def __getitem__(self, key):
        return self._bindings[key]

    @property
    def heldout_ids(self):
        return frozenset(k for k, v in self._bindings.items() if v.heldout)


def _array(value, shape, name, finite=False):
    a = np.asarray(value)
    if a.shape != shape or (finite and not np.isfinite(a).all()):
        raise ValueError(f"invalid {name}: expected {shape}")
    return a


def _bool(value, shape, name):
    a = _array(value, shape, name)
    if a.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean")
    return a


def _readonly(value):
    a = np.asarray(value).copy()
    # A bytes-backed array cannot be made writable by a downstream consumer.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class FitPacket:
    ids: tuple
    xyz_m: np.ndarray
    predicted_uv: np.ndarray
    weights: np.ndarray
    grid_ids: tuple
    count: int
    grid_count: int
    eligible: bool


def partition_geometry(registry, ids, predicted_uv, observed, is_query, weights, grid_ids,
                       human_uv=None):
    """Filter BEFORE all solvers/acceptance/fallback; human UV never enters FitPacket.

    The independent score packet may contain heldout UV. Do not pass it downstream
    to a solver. A0-derived XYZ supports fixed-estimated-binding 2D consistency,
    not independent true 3D ground truth.
    """
    ids = tuple(ids)
    n = len(ids)
    if any(not isinstance(k, AnchorKey) for k in ids):
        raise ValueError("full AnchorKey identities required")
    if len({k.track_id for k in ids}) > 1:
        raise ValueError("one FitPacket cannot combine multiple track instances")
    if len({(k.track_id, k.anchor_id) for k in ids}) != n:
        raise ValueError("one frame cannot contain multiple generations of the same material")
    if len(set(ids)) != n or len(grid_ids) != n:
        raise ValueError("duplicate identities or wrong grid length")
    uv = _array(predicted_uv, (n, 2), "predicted_uv")
    obs = _bool(observed, (n,), "observed")
    query = _bool(is_query, (n,), "is_query")
    weight = _array(weights, (n,), "weights")
    bindings = [registry[k] for k in ids]  # Unknown generations fail closed.
    fit = obs & ~query & np.array([not b.heldout for b in bindings], dtype=bool)
    # Heldout fields are never validated to select fallback or affect fit counts.
    if not np.isfinite(uv[fit]).all() or not np.isfinite(weight[fit]).all() or (weight[fit] <= 0).any():
        raise ValueError("observed fit rows require finite UV and positive weights")
    indices = np.flatnonzero(fit)
    grids = tuple(grid_ids[i] for i in indices)
    for grid in grids:
        if type(grid) is not int or grid < 0:
            raise ValueError("fit grid IDs must be nonnegative integers")
    fit_ids = tuple(ids[i] for i in indices)
    packet = FitPacket(fit_ids, _readonly(np.array([bindings[i].xyz_m for i in indices]).reshape(-1, 3)),
                       _readonly(uv[fit]), _readonly(weight[fit]), grids, len(indices),
                       len(set(grids)), len(indices) >= 12 and len(set(grids)) >= 4)
    held = np.array([b.heldout for b in bindings], dtype=bool)
    human = None if human_uv is None else _array(human_uv, (n, 2), "human_uv")[held].copy()
    scoring = {"ids": tuple(k for k, h in zip(ids, held) if h), "predicted_uv": uv[held].copy(),
               "human_uv": human, "bindings": tuple(b for b in bindings if b.heldout),
               "interpretation": "fixed_estimated_binding_2d_consistency_not_true_3d_gt"}
    assert_consumers_exclude_heldout(registry, solver_ids=packet.ids,
                                    acceptance_ids=packet.ids, fallback_ids=packet.ids)
    return packet, scoring


def assert_consumers_exclude_heldout(registry, *, solver_ids, acceptance_ids, fallback_ids):
    for name, ids in (("solver", solver_ids), ("acceptance", acceptance_ids), ("fallback", fallback_ids)):
        keys = tuple(ids)
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate {name} identity")
        for key in keys:
            registry[key]
        if set(keys) & registry.heldout_ids:
            raise AssertionError(f"heldout leakage into {name}")


@dataclass(frozen=True)
class CropTransform:
    x0: int
    y0: int
    width: int
    height: int

    def __post_init__(self):
        if any(type(x) is not int for x in (self.x0, self.y0, self.width, self.height)) or min(self.width, self.height) < 2:
            raise ValueError("integer crop with width,height >= 2 required")

    def to_model(self, xy_native):
        xy = np.asarray(xy_native, dtype=float)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("finite XY required")
        return (xy - [self.x0, self.y0]) * [511 / (self.width - 1), 383 / (self.height - 1)]

    def to_native(self, xy_model):
        xy = np.asarray(xy_model, dtype=float)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("finite XY required")
        return xy * [(self.width - 1) / 511, (self.height - 1) / 383] + [self.x0, self.y0]

    @property
    def affine(self):
        sx, sy = 511 / (self.width - 1), 383 / (self.height - 1)
        return np.array([[sx, 0, -self.x0 * sx], [0, sy, -self.y0 * sy], [0, 0, 1.]])


def support_queries(crop, native_width, native_height):
    if type(native_width) is not int or type(native_height) is not int or min(native_width, native_height) < 1:
        raise ValueError("positive image dimensions required")
    x, y = np.meshgrid(np.linspace(8, 504, 6), np.linspace(8, 376, 6))
    xy = np.stack([x.ravel(), y.ravel()], axis=-1)
    native = crop.to_native(xy)
    valid = (native[:, 0] >= 0) & (native[:, 0] <= native_width - 1) & (native[:, 1] >= 0) & (native[:, 1] <= native_height - 1)
    return np.column_stack([np.zeros(valid.sum()), xy[valid]])


def query_local_times(frame_ids, query_frame_ids):
    frames = tuple(frame_ids)
    queries = tuple(query_frame_ids)
    integer = lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
    if any(not integer(x) for x in frames) or any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("strictly increasing integer frame IDs required")
    if any(not integer(x) for x in queries):
        raise ValueError("integer query frame IDs required")
    lookup = {f: i for i, f in enumerate(frames)}
    try:
        return np.array([lookup[f] for f in queries], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("query frame outside window") from exc


def reverse_query_times(query_times, length):
    q = np.asarray(query_times)
    if type(length) is not int or length < 1 or q.ndim != 1 or q.dtype.kind not in "iu" or ((q < 0) | (q >= length)).any():
        raise ValueError("local integer query times required")
    return length - 1 - q


def merge_bidirectional(forward, backward_reversed, query_times, query_xy):
    """Inputs each (xy[T,N,2], continuous visibility[T,N], confidence[T,N])."""
    xy = np.asarray(forward[0])
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("XY must be [T,N,2]")
    t, n = xy.shape[:2]
    q = np.asarray(query_times)
    reverse_query_times(q, t)
    _array(q, (n,), "query_times")
    query_xy = _array(query_xy, (n, 2), "query_xy", True)
    if len(forward) != 3 or len(backward_reversed) != 3:
        raise ValueError("xy,visibility,confidence required")
    replace = np.arange(t)[:, None] < q[None, :]
    raw = []
    for j, shape in enumerate(((t, n, 2), (t, n), (t, n))):
        f = _array(forward[j], shape, "forward")
        b = _array(backward_reversed[j], shape, "backward")[::-1]
        if j and (f.dtype.kind != "f" or b.dtype.kind != "f"):
            raise ValueError("visibility/confidence must remain continuous floats")
        raw.append(np.where(replace[..., None] if j == 0 else replace, b, f).copy())
    is_query = np.arange(t)[:, None] == q[None, :]
    corrected = raw[0].copy()
    corrected[q, np.arange(n)] = query_xy
    probability_valid = tuple(np.isfinite(x) & (x >= 0) & (x <= 1) for x in raw[1:])
    return {"raw": tuple(raw), "query_corrected_xy": corrected, "is_query": is_query,
            "nonquery_candidate_mask": ~is_query if t > 1 else np.zeros((t, n), bool),
            "coordinate_finite": np.isfinite(raw[0]).all(-1),
            "visibility_probability_valid": probability_valid[0],
            "confidence_probability_valid": probability_valid[1],
            "direction": replace.astype(np.uint8)}


@dataclass(frozen=True)
class GateScale:
    diagonal_px: float
    kind: str
    source_frame: int
    valid: bool
    upstream_identity: str

    def __post_init__(self):
        if self.kind not in {"full", "history", "candidate_support"} or not isinstance(self.upstream_identity, str) or not self.upstream_identity or type(self.source_frame) is not int or type(self.valid) is not bool:
            raise ValueError("frozen prediction-side scale provenance required")
        if self.valid and (not np.isfinite(self.diagonal_px) or self.diagonal_px <= 0):
            raise ValueError("positive finite valid scale required")

    @property
    def sigma(self):
        if not self.valid:
            raise ValueError("gate scale unavailable")
        return max(2., .005 * self.diagonal_px)


def assert_shared_gate_scales(scales_by_run):
    """Reject C0/C1/C2 scale drift before comparative inference/scoring.

    Mappings are keyed by exact (track_id, frame_id); every run must use the same
    frozen mapping, including invalid-scale provenance. NaN invalid values compare
    as equivalent only when all provenance and validity fields also agree.
    """
    if not scales_by_run:
        raise ValueError("at least one run required")
    baseline = None
    for records in scales_by_run.values():
        canonical = {}
        for key, scale in records.items():
            if not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], str) or not key[0] or type(key[1]) is not int or not isinstance(scale, GateScale):
                raise ValueError("full track/frame scale identity required")
            canonical[key] = (scale.kind, scale.source_frame, scale.valid, scale.upstream_identity,
                              float(scale.diagonal_px) if np.isfinite(scale.diagonal_px) else "nonfinite")
        if baseline is None:
            baseline = canonical
        elif baseline != canonical:
            raise ValueError("gate scale or provenance changed between configurations")
    return True


def owner_boundaries(windows):
    windows = tuple(windows)
    for s, e in windows:
        if type(s) is not int or type(e) is not int or e <= s:
            raise ValueError("nonempty integer half-open window required")
    for (sa, ea), (sb, eb) in zip(windows, windows[1:]):
        if not sa < sb < ea < eb:
            raise ValueError("windows require increasing starts/ends and overlap")
    if not windows:
        return []
    boundaries = [windows[0][0]] + [(b[0] + a[1]) // 2 for a, b in zip(windows, windows[1:])] + [windows[-1][1]]
    return list(zip(boundaries, boundaries[1:]))


@dataclass(frozen=True)
class WindowIdentity:
    track_id: str
    track_generation: int
    window_id: str
    start: int
    end: int

    def __post_init__(self):
        if (not isinstance(self.track_id, str) or not self.track_id or
            type(self.track_generation) is not int or self.track_generation < 0 or
            not isinstance(self.window_id, str) or not self.window_id or
            type(self.start) is not int or type(self.end) is not int or self.end <= self.start):
            raise ValueError("full valid track lifecycle/window identity required")


def validate_neighbor_windows(a, b, frame_ids):
    if not isinstance(a, WindowIdentity) or not isinstance(b, WindowIdentity):
        raise ValueError("WindowIdentity records required")
    if (a.track_id, a.track_generation) != (b.track_id, b.track_generation) or a.window_id == b.window_id:
        raise ValueError("different windows of same track lifecycle required")
    if not a.start < b.start < a.end < b.end:
        raise ValueError("ordered overlapping neighbor windows required")
    frames = tuple(frame_ids)
    if any(type(x) is not int for x in frames) or any(y <= x for x, y in zip(frames, frames[1:])):
        raise ValueError("strictly ordered common frame IDs required")
    if any(not b.start <= f < a.end for f in frames):
        raise ValueError("frame outside window intersection")


def overlap_diagnostic(xy_a, xy_b, valid_a, valid_b, query_a, query_b, scales, *,
                       frame_ids_a, frame_ids_b, anchor_key_a, anchor_key_b,
                       window_a, window_b):
    if tuple(frame_ids_a) != tuple(frame_ids_b):
        raise ValueError("common frame alignment differs")
    validate_neighbor_windows(window_a, window_b, frame_ids_a)
    if (not isinstance(anchor_key_a, AnchorKey) or anchor_key_a != anchor_key_b or
        anchor_key_a.track_id != window_a.track_id):
        raise ValueError("same full material identity within window track required")
    a, b = np.asarray(xy_a, float), np.asarray(xy_b, float)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape or len(scales) != len(a) or len(frame_ids_a) != len(a):
        raise ValueError("aligned per-anchor common-frame XY/scales required")
    shape = (len(a),)
    mask = (_bool(valid_a, shape, "valid_a") & _bool(valid_b, shape, "valid_b") &
            ~_bool(query_a, shape, "query_a") & ~_bool(query_b, shape, "query_b") &
            np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.array([s.valid for s in scales]))
    if mask.sum() < 3:
        return {"status": "unknown", "count": int(mask.sum()), "median_z": None, "p95_z": None, "allow_carry": False}
    z = np.linalg.norm(a[mask] - b[mask], axis=1) / np.array([s.sigma for s, m in zip(scales, mask) if m])
    med, p95 = float(np.median(z)), float(np.percentile(z, 95))
    good = med <= 1 and p95 <= 2
    return {"status": "consistent" if good else "conflict", "count": len(z), "median_z": med, "p95_z": p95, "allow_carry": good}


def cycle_diagnostic(start_xy, returned_xy, scale):
    if returned_xy is None or not scale.valid:
        return {"status": "cycle_unavailable", "error_px": None, "passed": None}
    a = _array(start_xy, (2,), "start_xy", True)
    b = _array(returned_xy, (2,), "returned_xy")
    if not np.isfinite(b).all():
        return {"status": "cycle_unavailable", "error_px": None, "passed": None}
    error = float(np.linalg.norm(a - b))
    return {"status": "measured", "error_px": error, "passed": error <= scale.sigma}
