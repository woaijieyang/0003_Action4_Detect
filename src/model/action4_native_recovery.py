"""Evidence-driven native pose recovery proposal; never an accuracy certificate."""
from dataclasses import dataclass
import numpy as np

DEFAULT_POLICY=dict(schema="action4.native_pose_recovery.dev_v1",min_texture_points=6,
 disagreement_iou=.1,consecutive_bad_frames=2,min_reseed_pose_box_iou=.3,
 min_reseed_mask_inside_box=.5,b0_score=.3,cooldown_frames=10)

# Finite development candidate: earlier contradiction detection, not looser reseed admission.
EARLY_DISAGREEMENT_POLICY=dict(DEFAULT_POLICY,
    schema="action4.native_pose_recovery.dev_early_v1",disagreement_iou=.3)


def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite full boxes required")
    if (a[2:]<=a[:2]).any() or (b[2:]<=b[:2]).any():raise ValueError("Positive box extents required")
    i=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(i/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-i))


@dataclass
class NativeRecoveryController:
    track_id: str
    generation: int
    policy: dict
    last_frame: int = -1
    last_pts: int = -1
    bad_streak: int = 0
    last_reseed_frame: int = -1000000
    recovery_count: int = 0
    pending: object = None
    allow_sam_only_continuation: bool = False
    support_count_kind: str = "retained_keypoints_legacy"

    def observe(self, *, frame_id,pts,track_id,generation,texture_points,
                current_B0,current_SAM,b0_score,native_box,b0_box,
                seed_pose,canonical8,K,mask_inside_box):
        if self.support_count_kind not in ("retained_keypoints_legacy","matched_within_native_scale"):
            raise ValueError("Declared texture support-count semantics required")
        if self.pending is not None:raise ValueError("Prior recovery must be acknowledged before next frame")
        if (track_id!=self.track_id or generation!=self.generation
            or type(frame_id) is not int or frame_id<=self.last_frame or type(pts) is not int or pts<=self.last_pts
            or type(texture_points) is not int or texture_points<0):
            raise ValueError("Ordered current frame and exact independent track/generation required")
        if self.policy not in (DEFAULT_POLICY,EARLY_DISAGREEMENT_POLICY):raise ValueError("Explicit preregistered development policy required")
        if self.last_frame>=0 and frame_id!=self.last_frame+1:self.bad_streak=0
        self.last_frame=frame_id;self.last_pts=pts
        current=current_B0 is True and current_SAM is True
        overlap=box_iou(native_box,b0_box) if current and native_box is not None and b0_box is not None else None
        contradictory=current and (native_box is None or
            (overlap is not None and overlap<self.policy["disagreement_iou"] and texture_points<self.policy["min_texture_points"]))
        self.bad_streak=self.bad_streak+1 if contradictory else 0
        answer=dict(track_id=track_id,frame_id=frame_id,pts=pts,generation=generation,
            native_B0_box_iou=overlap,bad_streak=self.bad_streak,texture_points=texture_points,support_count_kind=self.support_count_kind,
            action="continue",reason="current_observation_not_contradictory",
            pose_accuracy_verified=False,publication_admitted=False)
        if not current:
            if type(self.allow_sam_only_continuation) is not bool:
                raise ValueError("Explicit SAM-only continuation policy required")
            if (self.allow_sam_only_continuation and current_SAM is True and current_B0 is False
                and texture_points>=self.policy["min_texture_points"] and native_box is not None):
                # A current SAM observation is not a prediction-only missing-frame hold.
                # This continues the research tracker; it does not certify pose accuracy.
                box_iou(native_box,native_box)
                answer.update(action="continue",reason="current_SAM_texture_support_without_B0",
                    recovery_allowed=False)
                return answer
            answer.update(action="withhold",reason="missing_current_independent_B0_SAM_recovery_support")
            return answer
        if not contradictory:return answer
        answer.update(action="withhold",reason="native_disagrees_and_texture_support_low")
        if self.bad_streak<self.policy["consecutive_bad_frames"]:return answer
        if frame_id-self.last_reseed_frame<self.policy["cooldown_frames"]:
            answer["reason"]="recovery_cooldown";return answer
        T=np.asarray(seed_pose,float);cad=np.asarray(canonical8,float);camera=np.asarray(K,float)
        if (T.shape!=(4,4) or cad.shape!=(8,3) or camera.shape!=(3,3)
            or not np.isfinite(T).all() or not np.isfinite(cad).all() or not np.isfinite(camera).all()
            or not np.array_equal(T[3],[0,0,0,1])
            or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6,rtol=0)
            or abs(np.linalg.det(T[:3,:3])-1)>1e-6):
            answer["reason"]="invalid_current_A0_seed";return answer
        xyz=cad@T[:3,:3].T+T[:3,3]
        if not (xyz[:,2]>1e-6).all():answer["reason"]="nonpositive_current_A0_depth";return answer
        uv=xyz@camera.T;uv=uv[:,:2]/uv[:,2:]
        seed_iou=box_iou(np.r_[uv.min(0),uv.max(0)],b0_box)
        if (not np.isfinite([b0_score,mask_inside_box]).all() or not 0<=mask_inside_box<=1
            or b0_score<self.policy["b0_score"] or seed_iou<self.policy["min_reseed_pose_box_iou"]
            or mask_inside_box<self.policy["min_reseed_mask_inside_box"]):
            answer.update(reason="current_seed_not_consistent_with_B0_SAM",seed_box_iou=seed_iou);return answer
        self.pending=frame_id
        answer.update(action="reseed",reason="two_frame_contradiction_with_current_consistent_A0_B0_SAM",
            seed_box_iou=seed_iou,recovery_id=self.recovery_count+1)
        return answer

    def acknowledge(self,frame_id,success):
        if self.pending!=frame_id or type(success) is not bool:raise ValueError("Exact pending native recovery acknowledgement required")
        self.pending=None
        if success:self.recovery_count+=1;self.last_reseed_frame=frame_id;self.bad_streak=0
