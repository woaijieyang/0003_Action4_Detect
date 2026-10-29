"""Pose scheduling and diagnostics, never smoothing/solver/precision acceptance."""
import numpy as np
from model.action4_phase2_tracking_contracts import validate_neighbor_windows


def pose_windows_with_commits(start, end):
    if type(start) is not int or type(end) is not int:
        raise ValueError("integer half-open range required")
    if end <= start:
        return []
    windows, s = [], start
    while True:
        e = min(s + 15, end)
        windows.append((s, e))
        if e == end:
            break
        s += 5
    return [{"window": (s, e), "commit": (start if i == 0 else s + 5,
                                          end if i == len(windows) - 1 else s + 10)}
            for i, (s, e) in enumerate(windows)]


def split_pose_segments(frame_ids, timestamps_s, identities, states, normal_interval_s):
    """Return index ranges, split before gaps >2 normal intervals and identity changes.

    Caller retains original frame IDs and actual PTS. Only observed_pose/pose_only
    frames belong to optimization segments. Uneven positive PTS remain unchanged.
    """
    ids, ts = np.asarray(frame_ids), np.asarray(timestamps_s, float)
    n = len(ids)
    if ids.ndim != 1 or (n and ids.dtype.kind not in "iu") or ts.shape != (n,) or len(identities) != n or len(states) != n:
        raise ValueError("aligned sequence arrays required")
    if not np.isfinite(normal_interval_s) or normal_interval_s <= 0 or not np.isfinite(ts).all():
        raise ValueError("finite times and positive normal interval required")
    if n > 1 and ((np.diff(ids) <= 0).any() or (np.diff(ts) <= 0).any()):
        raise ValueError("strictly increasing frame IDs and PTS required")
    valid_states = {"observed_pose", "pose_only", "motion_predicted", "lost", "identity_unknown"}
    if any(s not in valid_states for s in states):
        raise ValueError("unknown pose state")
    result, start = [], None
    for i in range(n):
        valid = states[i] in {"observed_pose", "pose_only"} and identities[i] is not None
        boundary = i > 0 and (identities[i] != identities[i - 1] or ts[i] - ts[i - 1] > 2 * normal_interval_s)
        if start is not None and (not valid or boundary):
            result.append((start, i))
            start = None
        if valid and start is None:
            start = i
    if start is not None:
        result.append((start, n))
    return result


def _rotation(rotation):
    r = np.asarray(rotation, float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-7, rtol=0):
        raise ValueError("proper rotation matrix required")
    return r


def so3_log(rotation):
    """Principal SO(3) logarithm in radians, stable near zero and pi."""
    r = _rotation(rotation)
    theta = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-7:
        return .5 * skew
    if np.pi - theta < 1e-5:
        values, vectors = np.linalg.eigh((r + r.T) / 2)
        axis = vectors[:, np.argmax(values)]
        if np.linalg.norm(skew) > 1e-10 and np.dot(axis, skew) < 0:
            axis = -axis
        return theta * axis
    return theta / (2 * np.sin(theta)) * skew


def _translation(value):
    t = np.asarray(value, float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("finite translation [3] in meters required")
    return t


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {name} required")
    return float(value)


def same_frame_window_disagreement(rotation_a, translation_a, rotation_b, translation_b,
                                   *, frame_id_a, frame_id_b, identity_a, identity_b,
                                   cad_diagonal_m, gate_scale, material_ids_a=(),
                                   xyz_a=(), material_ids_b=(), xyz_b=(), heldout_ids=(),
                                   camera_matrix_a=None, camera_matrix_b=None,
                                   window_a, window_b, fixed_material_ids=None,
                                   fixed_material_manifest_id=None):
    """Compare neighboring-window candidates for the SAME frame and track lifecycle.

    identity_a/b are (track_id, track_generation). Material identities separately
    use (track_id, anchor_id, anchor_generation); different materials can have
    different anchor generations within one track lifecycle.

    Projection uses the explicitly preregistered fixed nonheldout material set F
    and its manifest identity, with identical XYZ and K in both windows. Missing
    any F member makes projection not_measured, never a smaller intersection.
    Without a supplied F, projection is not_measured. No GT/P1/motion is inferred.
    """
    if (type(frame_id_a) is not int or type(frame_id_b) is not int or frame_id_a != frame_id_b or
        identity_a != identity_b or not isinstance(identity_a, tuple) or len(identity_a) != 2 or
        not isinstance(identity_a[0], str) or not identity_a[0] or type(identity_a[1]) is not int or identity_a[1] < 0):
        raise ValueError("same integer frame and (track_id,track_generation) required")
    validate_neighbor_windows(window_a, window_b, [frame_id_a])
    if identity_a != (window_a.track_id, window_a.track_generation):
        raise ValueError("pose and window track lifecycle differ")
    ra, rb = _rotation(rotation_a), _rotation(rotation_b)
    ta, tb = _translation(translation_a), _translation(translation_b)
    diag = _positive(cad_diagonal_m, "CAD diagonal")
    angle = float(np.linalg.norm(so3_log(ra @ rb.T)))
    shift = float(np.linalg.norm(ta - tb) / diag)
    ids_a, ids_b, heldout_ids = tuple(material_ids_a), tuple(material_ids_b), tuple(heldout_ids)
    fixed = None if fixed_material_ids is None else tuple(fixed_material_ids)
    for key in ids_a + ids_b + heldout_ids + (() if fixed is None else fixed):
        if not isinstance(key, tuple) or len(key) != 3 or not isinstance(key[0], str) or not key[0] or not isinstance(key[1], str) or not key[1] or type(key[2]) is not int or key[2] < 0:
            raise ValueError("full material identity (track_id,anchor_id,generation) required")
        if key[0] != identity_a[0]:
            raise ValueError("material belongs to a different track")
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise ValueError("duplicate material identities")
    for keys in (ids_a, ids_b, () if fixed is None else fixed):
        if len({(k[0], k[1]) for k in keys}) != len(keys):
            raise ValueError("one frame cannot contain multiple versions of one material")
    xa, xb = np.asarray(xyz_a, float), np.asarray(xyz_b, float)
    if xa.shape == (0,): xa = xa.reshape(0, 3)
    if xb.shape == (0,): xb = xb.reshape(0, 3)
    if xa.shape != (len(ids_a), 3) or xb.shape != (len(ids_b), 3):
        raise ValueError("material XYZ alignment mismatch")
    projection = {"status": "not_measured", "reason": "fixed_material_set_absent", "count": 0,
                  "median_px": None, "p95_px": None, "fixed_material_manifest_id": fixed_material_manifest_id,
                  "required_count": 0 if fixed is None else len(fixed), "missing_ids_a": [], "missing_ids_b": []}
    if fixed is not None:
        if not isinstance(fixed_material_manifest_id, str) or not fixed_material_manifest_id:
            raise ValueError("preregistered fixed material manifest identity required")
        if set(fixed) & set(heldout_ids):
            raise ValueError("fixed projection set must exclude heldout materials")
        projection["missing_ids_a"] = [key for key in fixed if key not in ids_a]
        projection["missing_ids_b"] = [key for key in fixed if key not in ids_b]
        projection["reason"] = "fixed_material_set_empty" if not fixed else "fixed_material_members_missing"
    complete_fixed = (fixed is not None and len(fixed) > 0 and
                      not projection["missing_ids_a"] and not projection["missing_ids_b"])
    if complete_fixed:
        k = np.asarray(camera_matrix_a, float)
        kb = np.asarray(camera_matrix_b, float)
        if k.shape != (3, 3) or kb.shape != (3, 3) or not np.isfinite(k).all() or not np.array_equal(k, kb) or abs(np.linalg.det(k)) < 1e-12:
            raise ValueError("same finite invertible frozen K required")
        points = []
        for material in fixed:
            a, b = xa[ids_a.index(material)], xb[ids_b.index(material)]
            if not np.isfinite(a).all() or not np.array_equal(a, b):
                raise ValueError("same frozen XYZ required")
            points.append(a)
        points = np.asarray(points)
        ca, cb = points @ ra.T + ta, points @ rb.T + tb
        if (ca[:, 2] <= 0).any() or (cb[:, 2] <= 0).any():
            projection["reason"] = "common_material_not_projectable"
        else:
            pa, pb = ca @ k.T, cb @ k.T
            errors = np.linalg.norm(pa[:, :2] / pa[:, 2:] - pb[:, :2] / pb[:, 2:], axis=1)
            projection.update(status="measured", reason=None, count=len(points),
                              median_px=float(np.median(errors)), p95_px=float(np.percentile(errors, 95)))
    triggers = []
    if angle > np.deg2rad(5):
        triggers.append("rotation_gt_5deg")
    if shift > .05:
        triggers.append("translation_gt_0.05_cad_diagonal")
    if projection["status"] == "measured" and gate_scale.valid and projection["median_px"] > gate_scale.sigma:
        triggers.append("projection_median_gt_gate_sigma")
    return {"kind": "same_frame_cross_window_disagreement", "rotation_rad": angle,
            "rotation_deg": float(np.rad2deg(angle)), "translation_over_cad_diagonal": shift,
            "projection": projection, "diagnostic_trigger": bool(triggers), "trigger_reasons": triggers,
            "projection_gate_status": "available" if gate_scale.valid else "not_measured",
            "precision_acceptance": "not_measured", "adjacent_motion": "not_measured",
            "p1_difference": "not_measured", "known_gt_error": "not_measured"}


def temporal_kinematics(rotations, translations_m, timestamps_s, *, gap_after=None):
    """Relative capture-camera pose velocities/accelerations using actual PTS.

    Invalid time or declared gaps skip affected terms explicitly, never fill zero.
    """
    r, t, ts = np.asarray(rotations, float), np.asarray(translations_m, float), np.asarray(timestamps_s, float)
    n = len(ts)
    if ts.shape != (n,) or r.shape != (n, 3, 3) or t.shape != (n, 3):
        raise ValueError("aligned pose sequence required")
    for item in r:
        _rotation(item)
    if not np.isfinite(t).all():
        raise ValueError("finite translations required")
    gaps = np.zeros(max(0, n - 1), bool) if gap_after is None else np.asarray(gap_after)
    if gaps.shape != (max(0, n - 1),) or gaps.dtype != np.bool_:
        raise ValueError("gap_after boolean[N-1] required")
    velocities, accelerations = [], []
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        reason = "gap" if gaps[i - 1] else "invalid_time" if not np.isfinite(dt) or dt <= 0 else None
        record = {"end_index": i, "midpoint_s": None, "linear_m_s": None, "angular_rad_s": None,
                  "status": "skipped" if reason else "measured", "reason": reason}
        if reason is None:
            record.update(midpoint_s=float((ts[i] + ts[i - 1]) / 2), linear_m_s=(t[i] - t[i - 1]) / dt,
                          angular_rad_s=so3_log(r[i] @ r[i - 1].T) / dt)
        velocities.append(record)
    for a, b in zip(velocities, velocities[1:]):
        item = {"end_index": b["end_index"], "status": "skipped", "reason": "adjacent_velocity_unavailable",
                "linear_m_s2": None, "angular_rad_s2": None}
        if a["status"] == b["status"] == "measured":
            dt = b["midpoint_s"] - a["midpoint_s"]
            item.update(status="measured", reason=None, linear_m_s2=(b["linear_m_s"] - a["linear_m_s"]) / dt,
                        angular_rad_s2=(b["angular_rad_s"] - a["angular_rad_s"]) / dt)
        accelerations.append(item)
    return {"interpretation": "relative_capture_pose_not_world_motion", "velocities": velocities,
            "accelerations": accelerations, "acceleration_status": "insufficient_frames" if n < 3 else
            "measured" if any(a["status"] == "measured" for a in accelerations) else "not_measured"}


def material_observability(material_jacobian_px, *, cad_diagonal_m, gate_scale,
                           normalized_row_weights=None):
    """Input J is material-only dr_px/d[rotation_rad,translation_m].

    No prior/damping parameters are accepted. Caller must assemble only filtered
    nonheldout observation rows, and provide already grid/frame normalized weights.
    """
    j = np.asarray(material_jacobian_px, float)
    if j.ndim != 2 or j.shape[1] != 6 or not np.isfinite(j).all():
        raise ValueError("finite material-only Jacobian [residual_rows,6] required")
    diagonal = _positive(cad_diagonal_m, "CAD diagonal")
    weights = np.ones(len(j)) if normalized_row_weights is None else np.asarray(normalized_row_weights, float)
    if weights.shape != (len(j),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("nonnegative finite row weights required")
    scaled = j * np.array([np.deg2rad(5)] * 3 + [.05 * diagonal] * 3) / gate_scale.sigma
    jw = scaled * np.sqrt(weights[:, None])
    singular = np.linalg.svd(jw, compute_uv=False)
    rank = int((singular > singular[0] * 1e-8).sum()) if len(singular) and singular[0] > 0 else 0
    condition = float((singular[0] / singular[-1]) ** 2) if rank == 6 else None
    return {"rank": rank, "h_obs_condition": condition, "well_conditioned": rank == 6 and condition <= 1e8,
            "relative_svd_tolerance": 1e-8, "material_only": True, "h_obs": jw.T @ jw}
