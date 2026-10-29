"""Fixed-ID video box updates from B0 observations and independently tracked SAM support."""
from collections import deque
import math
import numpy as np
from model.action4_temporal_association import iou_box
from model.action4_five_frame_tracker import match
from model.action4_video_box_history import _center,_wh,_box

def mask_geometry(mask):
    if mask.dtype!=np.bool_ or mask.ndim!=2:raise ValueError("Native boolean instance mask required")
    ys,xs=np.nonzero(mask)
    return (None,0,None) if not len(xs) else (np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],float),len(xs),np.array([xs.mean(),ys.mean()]))

def geometry_support(mask, *, minimum_component_fraction=.005):
    """Remove only tiny disconnected islands for geometry; never modify raw SAM.

    The ratio is relative to the largest component, so disconnected substantial
    visible body surfaces remain. This is support cleanup, not amodal inference.
    """
    if mask.dtype!=np.bool_ or mask.ndim!=2:
        raise ValueError("Native boolean instance mask required")
    if not math.isfinite(minimum_component_fraction) or not 0<=minimum_component_fraction<=.05:
        raise ValueError("Explicit small geometry-component fraction required")
    from scipy.ndimage import label
    ys=np.flatnonzero(mask.any(axis=1));xs=np.flatnonzero(mask.any(axis=0))
    if not len(xs):
        return mask,{"policy":"tiny_disconnected_components_geometry_only_v1",
            "minimum_component_fraction":minimum_component_fraction,"raw_pixels":0,
            "retained_pixels":0,"removed_pixels":0,"components":0,"retained_components":0,
            "raw_mask_preserved":True}
    # Outside the native foreground bounds every pixel is zero. Eight-connected
    # labeling on this exact subarray has identical components, avoiding the
    # previous full8MP labels/bincount allocation for small camera supports.
    y0,y1=int(ys[0]),int(ys[-1])+1;x0,x1=int(xs[0]),int(xs[-1])+1
    labels,count=label(mask[y0:y1,x0:x1],np.ones((3,3),dtype=bool))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    keep=sizes>=max(1,int(sizes.max())*minimum_component_fraction)
    clean=mask
    if int(keep.sum())!=count:
        clean=mask.copy();clean[y0:y1,x0:x1]=keep[labels]
    return clean,{"policy":"tiny_disconnected_components_geometry_only_v1",
        "minimum_component_fraction":minimum_component_fraction,"raw_pixels":int(sizes.sum()),
        "retained_pixels":int(sizes[keep].sum()),"removed_pixels":int(sizes.sum()-sizes[keep].sum()),
        "components":int(count),"retained_components":int(keep.sum()),"raw_mask_preserved":True}

def mask_box(mask):return mask_geometry(mask)[0]

def coverage(mask,b,total_area=None):
    h,w=mask.shape;x0,y0=np.maximum(np.floor(b[:2]),0).astype(int);x1,y1=np.minimum(np.ceil(b[2:]),[w,h]).astype(int)
    return 0. if x1<=x0 or y1<=y0 else float(mask[y0:y1,x0:x1].sum()/max(1,mask.sum() if total_area is None else total_area))

class SamSupportedBoxes:
    def __init__(self,seeds,*,window=5,support_enclosure_aspect_limit=4.0,shared_mask_iou_threshold=.9,geometry_mode="legacy"):
        if geometry_mode not in ("legacy","sam_primary_v1","sam_b0_fallback_v2"):raise ValueError("Explicit registered geometry mode required")
        self.geometry_mode=geometry_mode
        self.fallback_policy={"minimum_score":.7,"minimum_quality":.8,"minimum_appearance_cosine":.8,
            "minimum_box_occupancy":.5,"maximum_motion":.35,"maximum_other_mask_coverage":.2,
            "fallback_cost_penalty":.15,"extent_recovery_frames":3,"extent_recovery_max_side_ratio":1.25,"extent_recovery_max_center_shift":.15}
        self.geometry_policy={"offset_ema_alpha":.1,"calibration_window":5,"raw_SAM_velocity_window":5,"center_alpha":.65,"log_size_alpha":.2}
        if window not in (5,10):raise ValueError("Registered history window required")
        if len(seeds)>2:raise ValueError("At most two registered DJI identities")
        if not math.isfinite(support_enclosure_aspect_limit) or support_enclosure_aspect_limit<1:
            raise ValueError("Finite support enclosure aspect limit required")
        if not math.isfinite(shared_mask_iou_threshold) or not .5<=shared_mask_iou_threshold<=1:
            raise ValueError("Explicit shared mask IoU threshold required")
        self.shared_mask_iou_threshold=float(shared_mask_iou_threshold)
        self.enclosure_aspect_limit=float(support_enclosure_aspect_limit)
        self.window=window;self.states={};self.frame=None;self.seconds=None
        for seed in seeds:
            tid=seed["track_id"]
            if not isinstance(tid,str) or not tid or tid in self.states:raise ValueError("Invalid/duplicate project ID")
            b=np.asarray(seed["bbox_native"],float)
            if b.shape!=(4,) or not np.isfinite(b).all() or not (_wh(b)>0).all():raise ValueError("Invalid seed body box")
            self.states[tid]={"seed_box":np.array(seed["bbox_native"],float),"calibration":deque(maxlen=window),"calibration_frames":deque(maxlen=window),"sizes":deque(maxlen=window),
                "history":deque(maxlen=window),"areas":deque(maxlen=window),"last_box":None,"missing":0}
            if self.geometry_mode in ("sam_primary_v1","sam_b0_fallback_v2"):
                self.states[tid].update(calibration=deque(maxlen=5),calibration_frames=deque(maxlen=5),sizes=deque(maxlen=5),
                    offset_ema=None,raw_sam_centroids=deque(maxlen=5))
            if self.geometry_mode=="sam_b0_fallback_v2":
                self.states[tid].update(extent_conflicted=False,extent_recovery_streak=0)
                appearance=np.asarray(seed.get("initial_appearance128",[]),float)
                self.states[tid]["seed_appearance"]=(appearance/np.linalg.norm(appearance)
                    if appearance.shape==(128,) and np.isfinite(appearance).all() and np.linalg.norm(appearance)>0 else None)
    def _primary_velocity(self,state,frame_id):
        raw=list(state["raw_sam_centroids"])
        if len(raw)>=2 and raw[-1][0]==frame_id:
            return np.median([(b[2]-a[2])/(b[1]-a[1]) for a,b in zip(raw,raw[1:])],axis=0),"contiguous_raw_SAM_centroids"
        history=list(state["history"])
        if len(history)>=3:
            return np.median([(_center(b[1])-_center(a[1]))/(b[0]-a[0]) for a,b in zip(history,history[1:])],axis=0),"fallback_filtered_box_history"
        return np.zeros(2),"insufficient_history_zero_velocity"

    def update(self,frame_id,seconds,candidates,masks):
        if type(frame_id) is not int or frame_id<0 or not math.isfinite(seconds):
            raise ValueError("Finite frame/time required")
        ids=set()
        for candidate in candidates:
            cid=candidate["candidate_id"];b=np.asarray(candidate["bbox_full_xyxy"],float)
            if cid in ids or not isinstance(cid,str) or not cid:raise ValueError("Unique candidate IDs required")
            ids.add(cid)
            if b.shape!=(4,) or not np.isfinite(b).all() or not (_wh(b)>0).all():raise ValueError("Invalid candidate box")
            if not 0<=candidate["score"]<=1 or not 0<=candidate.get("refined_quality_score",0)<=1:
                raise ValueError("Finite probability range required")
        if len(ids)>200:raise ValueError("Candidate budget exceeded")
        if self.frame is not None and (frame_id!=self.frame+1 or seconds<=self.seconds):
            raise ValueError("Continuous frame/time domain required")
        tids=sorted(self.states);proposals={};valid={};reasons={};edges={}
        shapes={m.shape for m in masks.values() if m is not None}
        if len(shapes)>1:raise ValueError("SAM supports have different pixel spaces")
        cleaned={tid:geometry_support(m) for tid,m in masks.items() if m is not None}
        supports={tid:pair[0] for tid,pair in cleaned.items()}
        geometries={tid:mask_geometry(m) for tid,m in supports.items()}
        copied=set()
        for i,tid in enumerate(tids):
            if masks.get(tid) is None:continue
            for other in tids[:i]:
                if masks.get(other) is not None:
                    intersection=int(np.count_nonzero(supports[tid]&supports[other]))
                    union=geometries[tid][1]+geometries[other][1]-intersection
                    if union and intersection/union>=self.shared_mask_iou_threshold:
                        copied.update([tid,other])
                        reasons[tid]=reasons[other]="ambiguous_shared_instance_support"
        for tid in tids:
            state=self.states[tid];mask=masks.get(tid)
            mb,area,centroid=geometries.get(tid,(None,0,None))
            usable=mb is not None and tid not in copied
            if usable and state["areas"] and area<.25*np.median(state["areas"]):
                usable=False;reasons[tid]="support_area_collapse"
            if usable and not state["calibration"]:
                b=state["seed_box"];state["calibration"].append(_center(b)-centroid)
                state["sizes"].append(_wh(b));state["calibration_frames"].append(frame_id)
            if usable:
                offset=np.median(np.stack(state["calibration"]),axis=0)
                if self.geometry_mode in ("sam_primary_v1","sam_b0_fallback_v2"):
                    if state["offset_ema"] is None:state["offset_ema"]=offset.copy()
                    offset=state["offset_ema"]
                size=np.median(np.stack(state["sizes"]),axis=0)
                estimate=_box(centroid+offset,size)
                if state["last_box"] is not None and np.linalg.norm(_center(estimate)-_center(state["last_box"]))>.75*np.linalg.norm(_wh(state["last_box"])):
                    usable=False;reasons[tid]="support_motion_conflict"
                else:proposals[tid]=estimate
            valid[tid]=usable
            if self.geometry_mode in ("sam_primary_v1","sam_b0_fallback_v2"):
                raw=state["raw_sam_centroids"]
                if not usable:raw.clear()
                else:
                    if raw and raw[-1][0]!=frame_id-1:raw.clear()
                    raw.append((frame_id,seconds,centroid.copy()))
            if not usable:
                if state["last_box"] is None:proposals[tid]=state["seed_box"].copy()
                else:
                    velocity=np.zeros(2);history=list(state["history"])
                    if len(history)>=3:velocity=np.median([(_center(b[1])-_center(a[1]))/(b[0]-a[0]) for a,b in zip(history,history[1:])],axis=0)
                    proposals[tid]=_box(_center(state["last_box"])+velocity*(seconds-self.seconds),_wh(state["last_box"]))
        fallback_edges={}
        eligible=[c for c in candidates if c["score"]>=.3 and c.get("refined_quality_score",0)>=.5]
        eligible=sorted(eligible,key=lambda c:c["candidate_id"])
        for i,tid in enumerate(tids):
            estimate=proposals[tid]
            for j,c in enumerate(eligible):
                b=np.asarray(c["bbox_full_xyxy"],float)
                motion=np.linalg.norm(_center(b)-_center(estimate))/np.linalg.norm(_wh(estimate))
                ratios=_wh(b)/_wh(estimate)
                if motion>.5 or np.any(ratios>1.6) or np.any(ratios<.5):continue
                if valid[tid]:
                    own=coverage(supports[tid],b,geometries[tid][1])
                    other=max([coverage(supports[t],b,geometries[t][1]) for t in tids if t!=tid and valid[t]] or [0.])
                    if own<.8 or other>=.5:
                        if self.geometry_mode!="sam_b0_fallback_v2":continue
                        template=self.states[tid].get("seed_appearance")
                        appearance=np.asarray(c.get("appearance_128",[]),float)
                        cosine=(-1. if template is None or appearance.shape!=(128,) or not np.isfinite(appearance).all() or np.linalg.norm(appearance)==0
                                else float(np.dot(template,appearance)/np.linalg.norm(appearance)))
                        occupancy=float(own*geometries[tid][1]/_wh(b).prod())
                        fp=self.fallback_policy
                        admitted=(c["score"]>=fp["minimum_score"] and c["refined_quality_score"]>=fp["minimum_quality"]
                            and cosine>=fp["minimum_appearance_cosine"] and occupancy>=fp["minimum_box_occupancy"]
                            and motion<=fp["maximum_motion"] and other<=fp["maximum_other_mask_coverage"])
                        if not admitted:continue
                        fallback_edges[i,j]={"accepted":True,"mask_in_box":own,"box_occupancy":occupancy,
                            "appearance_cosine":cosine,"motion":float(motion),"other_mask_coverage":other,
                            "reason":"current_B0_supported_despite_SAM_overextent","semantic_accuracy_verified":False}
                edges[i,j]=float(.5*motion+.5*(1-iou_box(estimate,b)) + (self.fallback_policy["fallback_cost_penalty"] if (i,j) in fallback_edges else 0.))
        pairs=match(len(tids),len(eligible),edges)
        assigned={tids[i]:eligible[j] for i,j in pairs}
        fallbacks={tids[i]:fallback_edges[i,j] for i,j in pairs if (i,j) in fallback_edges}
        result=[]
        for tid in tids:
            state=self.states[tid];candidate=assigned.get(tid);estimate=proposals[tid]
            fallback=fallbacks.get(tid)
            enclosure_applied=False;enclosure_reason="no_current_support"
            primary=self.geometry_mode in ("sam_primary_v1","sam_b0_fallback_v2")
            calibration_evidence={"accepted":False,"reason":"no_current_B0" if candidate is None else "not_high_score_or_no_valid_SAM"}
            geometry_trace=None
            if candidate is not None:
                b=np.asarray(candidate["bbox_full_xyxy"],float)
                measurement=estimate.copy() if primary and valid[tid] else (.7*b+.3*estimate if valid[tid] else b)
                status="observed";source="current_B0_and_SAM" if valid[tid] else "current_B0"
                if valid[tid] and candidate["score"]>=.9 and candidate["refined_quality_score"]>=.8:
                    mb=geometries[tid][0]
                    offset=_center(b)-geometries[tid][2]
                    visibility=float(coverage(supports[tid],b,geometries[tid][1])*geometries[tid][1]/_wh(b).prod())
                    if primary:
                        calibration_evidence={"candidate_id":candidate["candidate_id"],"raw_B0_box":b.tolist(),
                            "raw_B0_minus_SAM_centroid":offset.tolist(),"in_box_occupancy":visibility,
                            "score":candidate["score"],"refined_quality_score":candidate["refined_quality_score"],
                            "accepted":False,"reason":"occupancy_or_offset_gate"}
                    if visibility>=.4 and np.linalg.norm(offset)<=.35*np.linalg.norm(_wh(b)):
                        if primary:calibration_evidence.update(accepted=True,reason="trusted_B0_calibration")
                        if state["calibration_frames"][-1]==frame_id:
                            state["calibration"][-1]=offset;state["sizes"][-1]=_wh(b)
                        else:
                            state["calibration"].append(offset);state["sizes"].append(_wh(b))
                            state["calibration_frames"].append(frame_id)
            elif valid[tid]:
                measurement=estimate;status="sam_supported";source="SAM_motion_and_B0_body_calibration"
            else:
                measurement=estimate;state["missing"]+=1
                status="predicted" if state["missing"]<=5 else "lost";source="history_only"
            if self.geometry_mode=="sam_b0_fallback_v2":
                if fallback:
                    state["extent_conflicted"]=True;state["extent_recovery_streak"]=0
                elif state["extent_conflicted"]:
                    clean=False
                    if candidate is not None and valid[tid] and candidate["score"]>=self.fallback_policy["minimum_score"] and candidate["refined_quality_score"]>=self.fallback_policy["minimum_quality"]:
                        sb=geometries[tid][0]
                        clean=(np.max(_wh(sb)/_wh(b))<=self.fallback_policy["extent_recovery_max_side_ratio"]
                               and np.linalg.norm(_center(sb)-_center(b))<=self.fallback_policy["extent_recovery_max_center_shift"]*np.linalg.norm(_wh(b)))
                    state["extent_recovery_streak"]=state["extent_recovery_streak"]+1 if clean else 0
                    if state["extent_recovery_streak"]>=self.fallback_policy["extent_recovery_frames"]:state["extent_conflicted"]=False
            if fallback:
                offset=_center(b)-geometries[tid][2]
                state["calibration"].clear();state["calibration"].append(offset)
                state["sizes"].clear();state["sizes"].append(_wh(b))
                state["calibration_frames"].clear();state["calibration_frames"].append(frame_id)
                state["offset_ema"]=offset.copy()
                calibration_evidence={"accepted":True,"reason":"multisignal_B0_mask_conflict_reset",
                    "candidate_id":candidate["candidate_id"],"raw_B0_box":b.tolist(),
                    "score":candidate["score"],"refined_quality_score":candidate["refined_quality_score"],
                    "fallback":dict(fallback)}
            if primary:
                trusted_size=np.median(np.stack(state["sizes"]),axis=0) if state["sizes"] else _wh(state["seed_box"])
                if valid[tid]:
                    median_offset=np.median(np.stack(state["calibration"]),axis=0)
                    state["offset_ema"]=(1-self.geometry_policy["offset_ema_alpha"])*state["offset_ema"]+self.geometry_policy["offset_ema_alpha"]*median_offset
                    measurement=_box(geometries[tid][2]+state["offset_ema"],trusted_size)
                elif status=="observed":measurement=_box(_center(measurement),trusted_size)
                velocity,velocity_source=self._primary_velocity(state,frame_id)
                if fallback:
                    measurement=b.copy();velocity=np.zeros(2)
                    velocity_source="zero_velocity_during_conflicted_mask_B0_reset"
                geometry_trace={"mode":self.geometry_mode,"measurement_center_source":"current_raw_SAM_centroid_plus_slow_offset" if valid[tid] else source,
                    "raw_SAM_centroid":None if not valid[tid] else geometries[tid][2].tolist(),
                    "slow_offset":None if state["offset_ema"] is None else state["offset_ema"].tolist(),
                    "calibration_median_offset":None if not state["calibration"] else np.median(np.stack(state["calibration"]),axis=0).tolist(),
                    "trusted_size_median":trusted_size.tolist(),"raw_calibration_evidence":calibration_evidence,
                    "raw_SAM_history_frames":[x[0] for x in state["raw_sam_centroids"]],
                    "velocity_native_px_per_second":velocity.tolist(),"velocity_source":velocity_source,
                    "unsmoothed_measurement_center":_center(measurement).tolist(),"unsmoothed_measurement_size":_wh(measurement).tolist()}
            if status in ("observed","sam_supported"):
                state["missing"]=0
                if state["last_box"] is not None:
                    if primary:
                        pred_center=_center(state["last_box"])+velocity*(seconds-self.seconds)
                        center=self.geometry_policy["center_alpha"]*_center(measurement)+(1-self.geometry_policy["center_alpha"])*pred_center
                        alpha=self.geometry_policy["log_size_alpha"]
                        size=(trusted_size.copy() if fallback else
                              np.exp(alpha*np.log(trusted_size)+(1-alpha)*np.log(_wh(state["last_box"]))))
                        measurement=_box(center,size)
                    else:
                        history=list(state["history"]);velocity=np.zeros(2)
                        if len(history)>=3:velocity=np.median([(_center(b[1])-_center(a[1]))/(b[0]-a[0]) for a,b in zip(history,history[1:])],axis=0)
                        pred=_box(_center(state["last_box"])+velocity*(seconds-self.seconds),_wh(state["last_box"]))
                        measurement=.65*measurement+.35*pred
                if primary:
                    geometry_trace["pre_enclosure_box"]=measurement.tolist()
                if fallback:
                    measurement=np.r_[np.minimum(measurement[:2],b[:2]),np.maximum(measurement[2:],b[2:])]
                    enclosure_applied=True;enclosure_reason="current_B0_extent_after_mask_conflict"
                elif self.geometry_mode=="sam_b0_fallback_v2" and state["extent_conflicted"]:
                    enclosure_reason="mask_extent_quarantined_until_three_consistent_frames"
                elif valid[tid]:
                    support_box=geometries[tid][0];aspect=_wh(support_box)[0]/_wh(support_box)[1]
                    if 1/self.enclosure_aspect_limit<=aspect<=self.enclosure_aspect_limit:
                        enclosed=np.r_[np.minimum(measurement[:2],support_box[:2]),
                                       np.maximum(measurement[2:],support_box[2:])]
                        enclosure_applied=bool(np.any(enclosed!=measurement))
                        measurement=enclosed;enclosure_reason="current_visible_support_enclosure_only"
                    else:enclosure_reason="elongated_partial_support_no_full_extent_update"
                state["history"].append((seconds,measurement.copy()))
            state["last_box"]=measurement
            if valid[tid]:state["areas"].append(geometries[tid][1])
            result.append({"track_id":tid,"status":status,"bbox_for_roi":None if status=="lost" else measurement.tolist(),
                "candidate_id":None if candidate is None else candidate["candidate_id"],"raw_detection":candidate,
                "source":source,"sam_current_support":valid[tid],"support_rejection":reasons.get(tid),
                "support_geometry":cleaned[tid][1] if tid in cleaned else None,
                "support_enclosure_applied":enclosure_applied,"support_enclosure_reason":enclosure_reason,
                "full_box_estimated":True,"full_box_accuracy_verified":False,
                "calibration_frames":list(state["calibration_frames"]),
                "calibration_kind":"native_pixel_center_offset_and_trusted_B0_size"})
            if self.geometry_mode=="sam_b0_fallback_v2":
                result[-1]["B0_mask_conflict_fallback"]=dict(fallback) if fallback else {"accepted":False}
                result[-1]["B0_mask_conflict_fallback"].update(extent_conflicted=state["extent_conflicted"],
                    extent_recovery_streak=state["extent_recovery_streak"])
            if primary:
                if fallback:geometry_trace["measurement_center_source"]="current_B0_multisignal_fallback"
                geometry_trace["final_box"]=measurement.tolist()
                geometry_trace["visible_enclosure_override"]=enclosure_applied
                result[-1]["geometry_trace"]=geometry_trace
        self.frame=frame_id;self.seconds=seconds
        output={"frame_id":frame_id,"tracks":result,"mode":"video_inference","window":self.window,
                "IDs_reassigned":False,"raw_B0_boxes_modified":False,
                "support_enclosure_aspect_limit":self.enclosure_aspect_limit,
                "shared_mask_iou_threshold":self.shared_mask_iou_threshold}
        if self.geometry_mode in ("sam_primary_v1","sam_b0_fallback_v2"):output.update(geometry_mode=self.geometry_mode,geometry_policy=dict(self.geometry_policy))
        if self.geometry_mode=="sam_b0_fallback_v2":output["fallback_policy"]=dict(self.fallback_policy)
        return output
