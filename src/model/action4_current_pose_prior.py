"""Current predicted A0 pose -> optional native soft prior. No ground truth or smoothing."""
import numpy as np


def current_pose_prior(instance, *, track_id,frame_id,pts,canonical8,pose_key="raw_A0_pose"):
    evidence=instance["recovery_evidence"]
    if (instance["track_id"]!=track_id or evidence["frame_id"]!=frame_id or evidence["pts"]!=pts):
        raise ValueError("Same current instance/frame/PTS required for A0 prior")
    support=evidence["current_support_status"]
    if pose_key not in ("raw_A0_pose","selected_A0_pose"):raise ValueError("Explicit registered A0 pose input required")
    source=instance[pose_key]
    packet=dict(track_id=track_id,frame_id=frame_id,pts=pts,available=False,
        source="current_raw_A0_from_resolved_ROI" if pose_key=="raw_A0_pose" else "current_selected_A0_from_resolved_ROI",roi_decision_id=evidence["roi_decision_id"],
        independent_geometric_evidence=False,ground_truth=False,calibrated_uncertainty=False)
    if not (support.get("B0") is True or support.get("SAM") is True):
        return dict(packet,reason="no_current_observation",pose=None)
    if evidence.get("pose_valid") is not True or source.get("finite") is not True:
        return dict(packet,reason="invalid_current_A0",pose=None)
    if source.get("coordinate_frame")!="object_to_capture_cv_m":
        raise ValueError("Explicit metre object-to-capture pose required")
    R=np.asarray(source["rotation"],float);t=np.asarray(source["translation_m"],float)
    if (R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0) or abs(np.linalg.det(R)-1)>1e-6 or t[2]<=0):
        return dict(packet,reason="nonfinite_nonrigid_or_nonpositive_current_A0",pose=None)
    cad=np.asarray(canonical8,float)
    if cad.shape!=(8,3) or not np.isfinite(cad).all():raise ValueError("Fixed metre canonical8 required")
    if not ((cad@R.T+t)[:,2]>1e-6).all():
        return dict(packet,reason="nonpositive_canonical_depth",pose=None)
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    return dict(packet,available=True,reason="prediction_prior_not_GT",pose=T.tolist())


def native_prior_directions(mode, available):
    """Native SoftConstraint order: rotation XYZ, then translation XYZ."""
    if mode not in ("continuous", "disabled_continuous", "translation_only"):
        raise ValueError("Unregistered native prior mode")
    if type(available) is not bool:
        raise ValueError("Explicit current-prior availability required")
    if not available or mode == "disabled_continuous":
        return [False] * 6
    if mode == "translation_only":
        return [False] * 3 + [True] * 3
    return [True] * 6
