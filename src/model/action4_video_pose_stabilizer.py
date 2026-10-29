"""Video-only rigid pose stabilization; raw A0 and ROI records remain intact.

Uses a causal five-history-plus-current SO(3) medoid and speed-adaptive low-pass.
Low-pass equations follow the 1 Euro filter, https://gery.casiez.net/1euro/.
Rotation extension and ROI-relative coordinates are project-specific.
"""
from dataclasses import dataclass,asdict
from collections import deque
from fractions import Fraction
import copy,hashlib,json,math
import numpy as np
from model.action4_native_rotation_history import rotation_exp,rotation_log

@dataclass(frozen=True)
class StabilizerConfig:
    window:int=5
    rotation_coherence_gain:bool=False
    incoherent_cutoff_hz:float=.5
    incoherent_speed_deg_s:float=60.
    rotation_min_cutoff_hz:float=3.
    rotation_beta:float=.5
    coordinate_min_cutoff_hz:float=3.
    coordinate_beta:float=.5
    derivative_cutoff_hz:float=1.
    rotation_cost_cap_rad:float=math.pi/4
    recency_weight:float=.8
    max_gap_seconds:float=.25
    rotation_speed_clock:str="output_tick"
    motion_veto_enabled:bool=False
    motion_veto_hold_frames:int=0
    require_reference_shape:bool=False
    require_rigid_texture:bool=False
    mask_motion_gain_enabled:bool=False
    mask_motion_policy_id:str|None=None
    weak_rotation_max_cutoff_hz:float|None=None
    weak_rotation_max_speed_deg_s:float|None=None
    image_orientation_enabled:bool=False
    image_orientation_policy_id:str|None=None
    measurement_source:str="raw_network_pose"
    rotation_max_cutoff_hz:float|None=None
    rotation_max_speed_deg_s:float|None=None
    coordinate_max_speed:tuple|None=None
    def __post_init__(self):
        if type(self.rotation_coherence_gain) is not bool:raise ValueError("Explicit coherence gain switch required")
        for v in (self.incoherent_cutoff_hz,self.incoherent_speed_deg_s):
            if isinstance(v,bool) or not math.isfinite(v) or v<=0:raise ValueError("Positive incoherent gain bounds required")
        if self.rotation_coherence_gain and (self.rotation_max_cutoff_hz is None or self.rotation_max_speed_deg_s is None or self.incoherent_cutoff_hz>self.rotation_max_cutoff_hz or self.incoherent_speed_deg_s>self.rotation_max_speed_deg_s):raise ValueError("Coherence gain requires explicit ordinary bounds")
        if type(self.mask_motion_gain_enabled) is not bool:raise ValueError("Explicit weak-gain switch required")
        weak=(self.weak_rotation_max_cutoff_hz,self.weak_rotation_max_speed_deg_s)
        if self.mask_motion_gain_enabled:
            if not self.image_orientation_enabled:raise ValueError("Weak gain needs explicit stronger-evidence status")
            if not isinstance(self.mask_motion_policy_id,str) or len(self.mask_motion_policy_id)!=64 or any(c not in "0123456789abcdef" for c in self.mask_motion_policy_id):raise ValueError("Bound weak-motion policy identity required")
            if any(v is None or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in weak):raise ValueError("Positive weak gain bounds required")
            if self.rotation_max_cutoff_hz is None or self.rotation_max_speed_deg_s is None or weak[0]>self.rotation_max_cutoff_hz or weak[1]>self.rotation_max_speed_deg_s:raise ValueError("Weak gain may only reduce ordinary bounds")
        elif any(v is not None for v in (*weak,self.mask_motion_policy_id)):raise ValueError("Unused weak-gain configuration")
        if type(self.image_orientation_enabled) is not bool:raise ValueError("Explicit image orientation evidence switch required")
        if self.image_orientation_enabled:
            from model.action4_image_orientation_evidence import DEFAULT_POLICY,policy_identity
            if self.image_orientation_policy_id!=policy_identity(DEFAULT_POLICY):raise ValueError("Explicit bound image orientation policy required")
        elif self.image_orientation_policy_id is not None:raise ValueError("Unused image orientation policy")
        if self.measurement_source not in ("raw_network_pose","selected_A0_pose"):raise ValueError("Explicit input pose source required")
        for value in (self.rotation_max_cutoff_hz,self.rotation_max_speed_deg_s):
            if value is not None and (isinstance(value,bool) or not math.isfinite(value) or value<=0):raise ValueError("Positive optional rotation bound required")
        if self.coordinate_max_speed is not None:
            if len(self.coordinate_max_speed)!=3 or any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in self.coordinate_max_speed):raise ValueError("Three positive ROI coordinate speeds required")
            object.__setattr__(self,"coordinate_max_speed",tuple(self.coordinate_max_speed))
        if any(type(v) is not bool for v in (self.require_reference_shape,self.require_rigid_texture)) or ((self.require_reference_shape or self.require_rigid_texture) and not self.motion_veto_enabled):raise ValueError("Evidence requirements need enabled motion consumer")
        if type(self.motion_veto_hold_frames) is not int or not 0<=self.motion_veto_hold_frames<=5 or (self.motion_veto_hold_frames and not self.motion_veto_enabled):raise ValueError("Bounded motion hold requires enabled evidence gate")
        if type(self.motion_veto_enabled) is not bool:raise ValueError("Explicit motion veto flag required")
        if self.window!=5:raise ValueError("Fixed five-frame version1 history required")
        if self.rotation_speed_clock not in ("output_tick","selected_measurement"):raise ValueError("Unknown angular-speed clock")
        for k in ("rotation_min_cutoff_hz","coordinate_min_cutoff_hz","derivative_cutoff_hz","rotation_cost_cap_rad","max_gap_seconds"):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0:raise ValueError("Invalid positive filter parameter")
        if not 0<self.recency_weight<=1 or not math.isfinite(self.rotation_beta) or not math.isfinite(self.coordinate_beta) or self.rotation_beta<0 or self.coordinate_beta<0:raise ValueError("Invalid adaptation policy")
    @property
    def identity(self):
        fields=asdict(self)
        if not fields["rotation_coherence_gain"]:
            for key in ("rotation_coherence_gain","incoherent_cutoff_hz","incoherent_speed_deg_s"):fields.pop(key)
        if not fields["mask_motion_gain_enabled"]:fields.pop("mask_motion_gain_enabled")
        for key in ("mask_motion_policy_id","weak_rotation_max_cutoff_hz","weak_rotation_max_speed_deg_s"):
            if fields[key] is None:fields.pop(key)
        if not fields["image_orientation_enabled"]:fields.pop("image_orientation_enabled")
        if fields["image_orientation_policy_id"] is None:fields.pop("image_orientation_policy_id")
        if fields["measurement_source"]=="raw_network_pose":fields.pop("measurement_source")
        for key in ("rotation_max_cutoff_hz","rotation_max_speed_deg_s","coordinate_max_speed"):
            if fields[key] is None:fields.pop(key)
        for key in ("require_reference_shape","require_rigid_texture"):
            if not fields[key]:fields.pop(key)
        if fields["motion_veto_hold_frames"]==0:fields.pop("motion_veto_hold_frames")
        if not fields["motion_veto_enabled"]:fields.pop("motion_veto_enabled")
        if fields["rotation_speed_clock"]=="output_tick":fields.pop("rotation_speed_clock")
        return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()

def _timebase(value):
    try:
        if isinstance(value,str):result=Fraction(value)
        elif isinstance(value,(list,tuple)) and len(value)==2 and all(type(x) is int for x in value):result=Fraction(*value)
        else:raise ValueError("Exact rational timebase required")
        if result<=0:raise ValueError("Positive rational timebase required")
        return result
    except (ZeroDivisionError,TypeError):raise ValueError("Invalid rational timebase")


def alpha(cutoff,dt):
    return 1./(1.+1./(2.*math.pi*cutoff*dt))

class VideoPoseStabilizer:
    def __init__(self,config):
        if not isinstance(config,StabilizerConfig):raise TypeError("Explicit version1 config")
        self.config=config;self.states={};self.last_frame=-1;self.last_time=None
    def apply_prediction(self,prediction,*,frame_id,pts,time_base,K_native,canonical_m,mode,motion_observations=None,current_supports=None,image_orientation_observations=None):
        if self.config.image_orientation_enabled:
            if not isinstance(image_orientation_observations,dict):raise ValueError("Current image orientation evidence map required")
        elif image_orientation_observations is not None:raise ValueError("Unused image orientation evidence")
        if self.config.require_reference_shape:
            if (not isinstance(current_supports,dict) or current_supports.get("frame_id")!=frame_id
                or current_supports.get("pts")!=pts or current_supports.get("time_base")!=list(time_base)
                or not isinstance(current_supports.get("masks"),dict)):
                raise ValueError("Same-frame native owned-mask support required")
        elif current_supports is not None:raise ValueError("Unused reference-shape input")
        if self.config.motion_veto_enabled and not isinstance(motion_observations,dict):raise ValueError("Motion evidence map required")
        if not self.config.motion_veto_enabled and motion_observations is not None:raise ValueError("Motion observations have no enabled consumer")
        if mode!="video":raise ValueError("Stabilization is video-only")
        if type(frame_id) is not int or type(pts) is not int or len(time_base)!=2 or any(type(x) is not int or x<=0 for x in time_base):
            raise ValueError("Exact frame/PTS domain required")
        now=pts*time_base[0]/time_base[1]
        if frame_id<=self.last_frame or (self.last_time is not None and now<=self.last_time):raise ValueError("Nonincreasing video timeline")
        K=np.asarray(K_native,float);cad=np.asarray(canonical_m,float)
        if K.shape!=(3,3) or cad.shape!=(8,3) or not np.isfinite(K).all() or not np.isfinite(cad).all() or K[0,0]<=0 or K[1,1]<=0:
            raise ValueError("Invalid native camera/canonical8")
        camera_id=hashlib.sha256(K.tobytes()+cad.tobytes()).hexdigest()
        if hasattr(self,"camera_id") and self.camera_id!=camera_id:raise ValueError("Camera/CAD changed within sequence")
        self.camera_id=camera_id
        result=copy.deepcopy(prediction);ids=[x["track_id"] for x in result["instances"]]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate body track ID")
        for inst in result["instances"]:
            if inst.get("frame_id")!=frame_id or inst.get("pts")!=pts or tuple(inst.get("time_base",()))!=tuple(time_base):
                raise ValueError("Instance frame/PTS mismatch")
            if inst.get("corner_ids")!=list(range(8)):raise ValueError("Canonical IDs changed")
            allowed=("none",) if self.config.measurement_source=="raw_network_pose" else ("corner_candidate_raw_fallback","corner_candidate_research_only")
            if inst.get("post_A0_geometry") not in allowed:raise ValueError("Do not stack over an unknown pose correction")
            observation=(motion_observations or {}).get(inst["track_id"])
            if observation is not None:
                if observation.get("track_id")!=inst["track_id"] or observation.get("frame_id")!=frame_id or observation.get("pts")!=pts or observation.get("time_base")!=list(time_base):raise ValueError("Cross-instance or stale motion observation")
            orientation=(image_orientation_observations or {}).get(inst["track_id"])
            if self.config.image_orientation_enabled:
                if (not isinstance(orientation,dict) or orientation.get("track_id")!=inst["track_id"] or orientation.get("frame_id")!=frame_id
                    or orientation.get("pts")!=pts or _timebase(orientation.get("time_base"))!=_timebase(time_base)
                    or orientation.get("policy_id")!=self.config.image_orientation_policy_id
                    or orientation.get("current_RGB_sha256")!=prediction.get("RGB_sha256")
                    or not prediction.get("RGB_sha256")):
                    raise ValueError("Cross-instance/stale/different-RGB image orientation evidence")
                if any(type(orientation.get(k)) is not bool for k in ("available","hold_rotation","current_mask_authenticated")):
                    raise ValueError("Explicit image evidence booleans required")
                if orientation["hold_rotation"] and orientation.get("previous_frame_id")!=frame_id-1:raise ValueError("Nonconsecutive image evidence")
                if orientation["hold_rotation"] and not (orientation["available"] and orientation["current_mask_authenticated"]):
                    raise ValueError("Cannot hold from unknown current image support")
            if self.config.mask_motion_gain_enabled:
                hint=orientation.get("mask_motion_hint")
                if (not isinstance(hint,dict) or hint.get("track_id")!=inst["track_id"] or hint.get("frame_id")!=frame_id
                    or hint.get("pts")!=pts or _timebase(hint.get("time_base"))!=_timebase(time_base)
                    or hint.get("current_RGB_sha256")!=prediction.get("RGB_sha256")
                    or hint.get("policy_id")!=self.config.mask_motion_policy_id):raise ValueError("Wrong weak-motion source identity")
                if any(type(hint.get(k)) is not bool for k in ("eligible_for_gain","weak_low_motion","current_mask_authenticated")):
                    raise ValueError("Explicit weak-motion booleans required")
                if hint["eligible_for_gain"] and hint.get("previous_frame_id")!=frame_id-1:raise ValueError("Nonconsecutive mask hint")
                if hint["eligible_for_gain"] and (orientation["available"] or orientation["hold_rotation"] or not orientation["current_mask_authenticated"] or not hint["current_mask_authenticated"] or not hint["weak_low_motion"]):
                    raise ValueError("Weak hint cannot override strong evidence or missing masks")
            self._apply_instance(inst,frame_id,now,K,cad,observation,current_supports,orientation)
        self.last_frame=frame_id;self.last_time=now
        for tid in list(self.states):
            if now-self.states[tid]["time"]>self.config.max_gap_seconds:del self.states[tid]
        result["video_pose_stabilization"]=dict(schema="action4.video_pose_stabilization.v1",config_id=self.config.identity,
            frame_id=frame_id,pts=pts,time_base=list(time_base),causal=True,GT_used=False,
            raw_pose_preserved=True,actual_image_ROI_unchanged=True,renderer_changes_pose=False)
        return result
    def _apply_instance(self,inst,fid,now,K,cad,observation=None,current_supports=None,orientation_observation=None):
        tid=inst["track_id"];raw=inst.get("raw_network_pose",{}) if self.config.measurement_source=="raw_network_pose" else inst.get("final_pose",{});support=inst.get("current_support_status",{})
        if self.config.measurement_source=="selected_A0_pose":
            inst["input_pose_before_stabilization"]=copy.deepcopy(raw)
            inst["input_corners_before_stabilization"]=copy.deepcopy(inst["corners_final"])
            inst["input_post_A0_geometry"]=inst["post_A0_geometry"]
        if not inst.get("published",False) or not inst.get("pose_valid") or raw.get("finite") is not True or not (support.get("B0") or support.get("SAM")):
            self.states.pop(tid,None);inst["pose_stabilization"]=dict(status="raw_passthrough_no_valid_current_support");return
        r=np.asarray(raw["rotation"],float);t=np.asarray(raw["translation_m"],float);box=np.asarray(inst["bbox_for_a0"],float)
        if r.shape!=(3,3) or t.shape!=(3,) or box.shape!=(4,) or not np.isfinite(r).all() or not np.isfinite(t).all() or not np.isfinite(box).all():
            raise ValueError("Nonfinite supported pose/ROI")
        if not np.allclose(r.T@r,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(r)-1)>1e-5 or t[2]<=0 or np.any(box[2:]<=box[:2]):
            raise ValueError("Invalid supported rigid pose/ROI")
        center=(box[:2]+box[2:])/2;diagonal=float(np.linalg.norm(box[2:]-box[:2]));focal=math.sqrt(K[0,0]*K[1,1])
        cad_diag=float(np.linalg.norm(cad.max(0)-cad.min(0)))
        if cad_diag<=0:raise ValueError("Degenerate CAD")
        h=K@t;uv=h[:2]/h[2]
        q=np.r_[(uv-center)/diagonal,math.log(t[2]*diagonal/(focal*cad_diag))]
        state=self.states.get(tid);reset=state is None or now-state["time"]>self.config.max_gap_seconds
        if reset:
            state=dict(time=now,history=deque(maxlen=5),rotation=r.copy(),coordinates=q.copy(),
                       last_target_rotation=r.copy(),last_target_coordinates=q.copy(),omega=np.zeros(3),velocity=np.zeros(3),
                       last_forward_rotation=r.copy(),last_forward_time=now)
            self.states[tid]=state
        previous_rotation=state["rotation"].copy()
        previous_coordinates=state["coordinates"].copy()
        previous_translation=state.get("last_output_translation")
        previous_output_frame=state.get("last_output_frame")
        weak_gain=bool(self.config.mask_motion_gain_enabled and orientation_observation is not None and orientation_observation["mask_motion_hint"]["eligible_for_gain"])
        weak_gain_state_ready=not reset and previous_output_frame==fid-1
        weak_gain=weak_gain and weak_gain_state_ready
        current=dict(frame_id=fid,time=now,pts=inst["pts"],time_base=copy.deepcopy(inst["time_base"]),rotation=r.copy(),coordinates=q.copy())
        history=list(state["history"])+[current];weights=self.config.recency_weight**np.arange(len(history)-1,-1,-1)
        costs=[sum(w*min(float(np.linalg.norm(rotation_log(a["rotation"]@b["rotation"].T))),self.config.rotation_cost_cap_rad)
                   for w,b in zip(weights,history)) for a in history]
        chosen=min(range(len(history)),key=lambda i:(costs[i],-history[i]["frame_id"]))
        coherence=None
        coherence_cutoff=None;coherence_speed=None
        if self.config.rotation_coherence_gain and len(history)>=4:
            increments=np.stack([rotation_log(b["rotation"]@a["rotation"].T) for a,b in zip(history,history[1:])])
            total=float(np.linalg.norm(increments,axis=1).sum())
            coherence=1. if total<=1e-12 else min(1.,float(np.linalg.norm(increments.sum(axis=0)))/total)
            gain=coherence**2
            coherence_cutoff=self.config.incoherent_cutoff_hz+gain*(self.config.rotation_max_cutoff_hz-self.config.incoherent_cutoff_hz)
            coherence_speed=self.config.incoherent_speed_deg_s+gain*(self.config.rotation_max_speed_deg_s-self.config.incoherent_speed_deg_s)
        target_r=history[chosen]["rotation"]
        target_q=np.median(np.stack([x["coordinates"] for x in history[-3:]]),axis=0)
        source_time=history[chosen]["time"]
        source_dt=source_time-state["last_forward_time"]
        source_status="initial"
        dt=now-state["time"]
        fc_r=self.config.rotation_min_cutoff_hz;fc_q=self.config.coordinate_min_cutoff_hz
        if not reset:
            ad=alpha(self.config.derivative_cutoff_hz,dt)
            if self.config.rotation_speed_clock=="selected_measurement":
                if source_dt>0:
                    observed_w=rotation_log(target_r@state["last_forward_rotation"].T)/source_dt
                    state["last_forward_time"]=source_time;state["last_forward_rotation"]=target_r.copy()
                    source_status="new_forward_measurement"
                else:
                    observed_w=np.zeros(3);source_status="held_or_older_measurement_no_speed_boost"
            else:
                observed_w=rotation_log(target_r@state["last_target_rotation"].T)/dt
                source_status="legacy_output_tick"
                state["last_forward_time"]=source_time;state["last_forward_rotation"]=target_r.copy()
            observed_v=(target_q-state["last_target_coordinates"])/dt
            state["omega"]=ad*observed_w+(1-ad)*state["omega"]
            state["velocity"]=ad*observed_v+(1-ad)*state["velocity"]
            fc_r+=self.config.rotation_beta*float(np.linalg.norm(state["omega"]))
            if self.config.rotation_max_cutoff_hz is not None:fc_r=min(fc_r,self.config.rotation_max_cutoff_hz)
            if weak_gain:fc_r=min(fc_r,self.config.weak_rotation_max_cutoff_hz)
            if coherence_cutoff is not None:fc_r=min(fc_r,coherence_cutoff)
            fc_q+=self.config.coordinate_beta*float(np.linalg.norm(state["velocity"]))
            state["rotation"]=rotation_exp(alpha(fc_r,dt)*rotation_log(target_r@state["rotation"].T))@state["rotation"]
            aq=alpha(fc_q,dt);state["coordinates"]=aq*target_q+(1-aq)*state["coordinates"]
        previous_frames=[x["frame_id"] for x in state["history"]]
        state["history"].append(current)
        state.update(time=now,last_target_rotation=target_r.copy(),last_target_coordinates=target_q.copy())
        coordinate_rate_limited=False
        if not reset and self.config.coordinate_max_speed is not None:
            delta=state["coordinates"]-previous_coordinates;limit=np.asarray(self.config.coordinate_max_speed)*dt
            bounded=np.clip(delta,-limit,limit);coordinate_rate_limited=not np.array_equal(delta,bounded)
            state["coordinates"]=previous_coordinates+bounded
        uv_final=center+state["coordinates"][:2]*diagonal
        depth=math.exp(float(state["coordinates"][2]))*focal*cad_diag/diagonal
        ray=np.linalg.solve(K,np.r_[uv_final,1.]);translation=ray*(depth/ray[2])
        reference_consistency=None
        reference_compatible=True
        if self.config.require_reference_shape:
            mask=current_supports["masks"].get(tid)
            if mask is None:
                reference_consistency=dict(available=False,reference_compatible=False,reasons=["current_body_mask_missing"])
            else:
                others=np.zeros_like(mask,dtype=bool)
                for key,other in current_supports["masks"].items():
                    if key!=tid:others|=other
                from model.action4_pose_support_consistency import pose_support_consistency,SupportPolicy
                reference_consistency=pose_support_consistency(mask,K,cad,previous_rotation,translation,
                    policy=SupportPolicy(),other_instance_mask=others)
            reference_compatible=reference_consistency["reference_compatible"]
        motion_decision=dict(available=False,reject_candidate=False,reason="motion_veto_disabled")
        if self.config.motion_veto_enabled:
            motion_decision=dict(available=False,reject_candidate=False,reason="no_admitted_image_motion")
            if (not reset and observation is not None and observation.get("available") is True
                and (not self.config.require_rigid_texture or observation.get("rigid_support_qualified") is True)
                and reference_compatible):
                if observation.get("previous_frame_id")!=previous_output_frame or observation.get("same_generation") is not True:
                    raise ValueError("Motion evidence crosses pose frame or session generation")
                if (observation.get("inlier_ratio",0)<.6 or observation.get("coverage",0)<.1
                    or observation.get("points",0)<12):
                    raise ValueError("Unqualified motion marked available")
                from model.action4_video_motion_evidence import score_pose_motion
                motion_decision=score_pose_motion(observation["previous_uv"],observation["current_uv"],K,cad,
                    previous_rotation,previous_translation,state["rotation"],translation,diagonal)
                if motion_decision["reject_candidate"]:
                    state["last_image_veto_frame"]=fid
                    state["rotation"]=previous_rotation
                    state["omega"]=np.zeros(3)
                    state["last_target_rotation"]=previous_rotation.copy()
                    state["last_forward_rotation"]=previous_rotation.copy()
                    state["last_forward_time"]=now
            if motion_decision.get("available") and not motion_decision["reject_candidate"]:
                state.pop("last_image_veto_frame",None)
            if not reference_compatible:
                state.pop("last_image_veto_frame",None)
                motion_decision=dict(motion_decision,reason="historical_reference_not_shape_compatible")
            if self.config.require_rigid_texture and observation is not None and observation.get("rigid_support_qualified") is not True:
                motion_decision=dict(motion_decision,texture_reason="rigid_texture_identity_unqualified")
            if reference_consistency is not None:
                motion_decision["reference_consistency"]=reference_consistency
            last_veto=state.get("last_image_veto_frame")
            age=None if last_veto is None else fid-last_veto
            if (not reset and not motion_decision.get("available") and age is not None
                and 0<age<=self.config.motion_veto_hold_frames):
                state["rotation"]=previous_rotation.copy();state["omega"]=np.zeros(3)
                state["last_target_rotation"]=previous_rotation.copy()
                state["last_forward_rotation"]=previous_rotation.copy();state["last_forward_time"]=now
                motion_decision=dict(motion_decision,bounded_rotation_hold=True,hold_age_frames=age,
                    hold_limit_frames=self.config.motion_veto_hold_frames,current_rotation_observed=False,
                    fallback_reason="recent_image_veto_followed_by_missing_texture")
            elif age is not None and age>self.config.motion_veto_hold_frames:
                state.pop("last_image_veto_frame",None)
                motion_decision=dict(motion_decision,bounded_rotation_hold=False,hold_expired=True,
                    current_rotation_observed=False)
            state["last_motion_decision"]=motion_decision
        rotation_rate_limited=False
        if not reset and self.config.rotation_max_speed_deg_s is not None:
            increment=rotation_log(state["rotation"]@previous_rotation.T);angle=float(np.linalg.norm(increment))
            speed_limit=self.config.weak_rotation_max_speed_deg_s if weak_gain else self.config.rotation_max_speed_deg_s
            if coherence_speed is not None:speed_limit=min(speed_limit,coherence_speed)
            limit=math.radians(speed_limit)*dt
            if angle>limit:
                state["rotation"]=rotation_exp(increment*(limit/angle))@previous_rotation
                rotation_rate_limited=True
        image_rotation_held=False
        image_hold_state_ready=not reset and previous_output_frame==fid-1
        if orientation_observation is not None and orientation_observation["hold_rotation"] and image_hold_state_ready:
            state["rotation"]=previous_rotation.copy();state["omega"]=np.zeros(3)
            state["last_target_rotation"]=previous_rotation.copy()
            state["last_forward_rotation"]=previous_rotation.copy();state["last_forward_time"]=now
            image_rotation_held=True
            state["image_hold_run"]=state.get("image_hold_run",0)+1
            state.setdefault("image_hold_origin_frame",previous_output_frame)
        else:
            state["image_hold_run"]=0;state.pop("image_hold_origin_frame",None)
        rotation=state["rotation"];camera=cad@rotation.T+translation
        if not np.isfinite(camera).all() or np.any(camera[:,2]<=1e-6):
            self.states.pop(tid,None);inst["pose_stabilization"]=dict(status="raw_passthrough_invalid_filtered_depth");return
        projected=camera@K.T;corners=(projected[:,:2]/projected[:,2,None]).tolist()
        state["last_output_translation"]=translation.copy();state["last_output_frame"]=fid
        inst["corners_final"]=corners;inst["corners_px"]=corners
        inst["final_pose"]=dict(coordinate_frame="object_to_capture_cv_m",finite=True,rotation=rotation.tolist(),translation_m=translation.tolist())
        inst["rotation"]=copy.deepcopy(inst["final_pose"]["rotation"])
        inst["translation_m"]=copy.deepcopy(inst["final_pose"]["translation_m"])
        from model.action4_final_pose_record import CONTRACT
        inst["pose_output_contract"]=CONTRACT
        inst["post_A0_geometry"]="video_pose_stabilization_v1"
        inst["pose_stabilization"]=dict(status="initialized" if reset else "filtered",config_id=self.config.identity,
            history_frames=previous_frames,candidate_frames=[x["frame_id"] for x in history],selected_rotation_frame=history[chosen]["frame_id"],
            selection_is_probability=False,rotation_selection_cost=costs[chosen],
            measurement_source=self.config.measurement_source,rotation_rate_limited=rotation_rate_limited,
            coordinate_rate_limited=coordinate_rate_limited,
            rotation_speed_clock=self.config.rotation_speed_clock,rotation_speed_source_status=source_status,
            rotation_speed_source_dt_seconds=source_dt if source_dt>0 else None,
            selected_rotation_pts=history[chosen]["pts"],selected_rotation_time_base=history[chosen]["time_base"],
            raw_rotation_change_rad=float(np.linalg.norm(rotation_log(r@rotation.T))),
            dt_seconds=None if reset else dt,rotation_cutoff_hz=fc_r,coordinate_cutoff_hz=fc_q,
            coordinate_frame="current_full_body_ROI_relative_center_and_log_metric_depth_ratio",
            current_ROI_source=inst["roi_source"],current_support=copy.deepcopy(support))
        if self.config.rotation_coherence_gain:
            inst["pose_stabilization"]["rotation_coherence_gain"]=dict(available=coherence is not None,coherence=coherence,cutoff_cap_hz=coherence_cutoff,speed_cap_deg_s=coherence_speed,source="same_instance_last_five_raw_rotation_increments",is_image_motion_measurement=False,GT_used=False)
        if self.config.motion_veto_enabled:inst["pose_stabilization"]["image_motion_decision"]=motion_decision
        if orientation_observation is not None:
            inst["pose_stabilization"]["image_orientation_evidence"]=copy.deepcopy(orientation_observation)
            inst["pose_stabilization"].update(image_rotation_held=image_rotation_held,image_hold_run=state["image_hold_run"],
                image_hold_state_ready=image_hold_state_ready,weak_gain_state_ready=weak_gain_state_ready,
                image_hold_origin_frame=state.get("image_hold_origin_frame"),
                final_rotation_source="current_image_stationarity_hold" if image_rotation_held else "weak_silhouette_low_gain_filter" if weak_gain else "bounded_selected_pose_filter",
                weak_silhouette_gain_applied=weak_gain,
                effective_rotation_speed_limit_deg_s=min([v for v in (self.config.weak_rotation_max_speed_deg_s if weak_gain else self.config.rotation_max_speed_deg_s,coherence_speed) if v is not None],default=None))
