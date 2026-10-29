"""Video-only five-observation box prior, association and smooth ROI update."""
from collections import deque
import math
import numpy as np
from model.action4_guidance_contract import box
from model.action4_temporal_association import iou_box

def _center(b):return (np.asarray(b[:2])+b[2:])/2
def _wh(b):return np.asarray(b[2:])-b[:2]
def _box(center,wh):return np.r_[center-wh/2,center+wh/2]

class VideoBoxHistory:
    def __init__(self,*,mode,window=5,alpha=.65,max_missing=5,merged_box_area_ratio=1.4,merged_box_coverage=.7):
        if mode!="video_inference":raise ValueError("Box history is forbidden in image training/eval")
        if type(window) is not int or not 3<=window<=15 or not 0<alpha<=1 or type(max_missing) is not int or not 1<=max_missing<=15:
            raise ValueError("Invalid video history policy")
        if not math.isfinite(merged_box_area_ratio) or merged_box_area_ratio<=1 or not 0<merged_box_coverage<=1:
            raise ValueError("Explicit merged-candidate geometry thresholds required")
        self.merged_box_area_ratio=float(merged_box_area_ratio)
        self.merged_box_coverage=float(merged_box_coverage)
        self.window,self.alpha,self.max_missing=window,float(alpha),max_missing
        self.tracks={};self.frame=None;self.seconds=None
    def initialize(self,frame_id,seconds,seeds):
        if self.tracks or self.frame is not None:raise ValueError("Explicit new sequence required")
        if type(frame_id) is not int or frame_id<0 or not math.isfinite(seconds):raise ValueError("Frame/time required")
        if len(seeds)>8:raise ValueError("Track budget exceeded")
        for seed in seeds:
            tid=seed["track_id"]
            if not isinstance(tid,str) or not tid or tid in self.tracks:raise ValueError("Unique track IDs required")
            b=np.asarray(box(seed["bbox_full_xyxy"]))
            self.tracks[tid]={"history":deque([(frame_id,seconds,b.copy())],maxlen=self.window),"filtered":b.copy(),
                              "missing":0,"status":"observed"}
        self.frame,self.seconds=frame_id,seconds
    def reinitialize_confirmed(self,track_id,observations):
        """Reopen a lost slot only from five consecutive current strong observations."""
        if track_id not in self.tracks or self.tracks[track_id]["status"]!="lost":
            raise ValueError("Only a lost track slot can be reinitialized")
        if len(observations)!=5 or observations[-1]["frame"]!=self.frame:
            raise ValueError("Five observations ending at the current frame required")
        for a,b in zip(observations,observations[1:]):
            if b["frame"]!=a["frame"]+1 or b["seconds"]<=a["seconds"]:
                raise ValueError("Reacquisition evidence must be consecutive")
        history=[]
        for item in observations:
            c=item["candidate"];q=c.get("refined_quality_score",float("nan"))
            if not .9<=c["score"]<=1 or not .8<=q<=1 or not math.isfinite(item["seconds"]):
                raise ValueError("Strong finite detector and quality evidence required")
            history.append((item["frame"],item["seconds"],np.asarray(box(c["bbox_full_xyxy"]))))
        state=self.tracks[track_id]
        state["association_epoch"]=state.get("association_epoch",0)+1
        state["identity_continuity"]="unverified_after_reacquisition"
        state["history"]=deque(history,maxlen=self.window)
        state["filtered"]=history[-1][2].copy();state["missing"]=0;state["status"]="observed"
        return state["association_epoch"]

    def update(self,frame_id,seconds,candidates,*,sam_support_boxes=None):
        if self.frame is None or type(frame_id) is not int or frame_id<=self.frame or not math.isfinite(seconds) or seconds<=self.seconds:
            raise ValueError("Strictly ordered initialized video frames required")
        dt=seconds-self.seconds;supports=sam_support_boxes or {}
        tracks=sorted(self.tracks);raw={}
        for c in candidates:
            cid=c["candidate_id"]
            if not isinstance(cid,str) or not cid or cid in raw:raise ValueError("Unique frame candidate ID required")
            b=np.asarray(box(c["bbox_full_xyxy"]));score=c["score"]
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError("Finite detector score required")
            if score>=.3:raw[cid]=(c,b)
        ids=sorted(raw);predicted={};edges={};support_status={}
        if len(ids)>200:raise ValueError("Candidate budget exceeded, no silent top2 truncation")
        for i,tid in enumerate(tracks):
            state=self.tracks[tid];history=list(state["history"]);last=state["filtered"]
            if state["status"]=="lost" or frame_id-history[-1][0]>self.max_missing or seconds-history[-1][1]>=.5:
                state["status"]="lost";predicted[tid]=last.copy();continue
            velocity=np.zeros(2)
            if len(history)>=3:
                velocity=np.median([(_center(b[2])-_center(a[2]))/(b[1]-a[1]) for a,b in zip(history,history[1:])],axis=0)
            diag=float(np.linalg.norm(_wh(last)));move=velocity*dt
            norm=float(np.linalg.norm(move))
            if norm>.75*diag:move*=.75*diag/norm
            pred=_box(_center(last)+move,_wh(last));predicted[tid]=pred
            if state["status"]=="lost":continue
            support=None
            if tid in supports:
                sb=np.asarray(box(supports[tid]))
                if np.linalg.norm(_center(sb)-_center(pred))<=.75*diag:
                    support=sb;support_status[tid]="used_for_association_only"
                else:support_status[tid]="history_conflict_ignored"
            for j,cid in enumerate(ids):
                c,b=raw[cid];motion=float(np.linalg.norm(_center(b)-_center(pred))/diag)
                if motion>.75:continue
                cost=.7*motion+.3*(1-iou_box(pred,b))
                if support is not None:
                    intersection=np.maximum(0,np.minimum(b[2:],support[2:])-np.maximum(b[:2],support[:2])).prod()
                    cost+=.2*(1-float(intersection/_wh(support).prod()))
                edges[i,j]=float(cost)
        candidate_rejections=[]
        # A single enlarged hypothesis spanning two established tracks is not
        # independent instance evidence. Keep separate predictions; do not merge IDs.
        for j,cid in enumerate(ids):
            b=raw[cid][1];area=float(_wh(b).prod());covered=[]
            for tid in tracks:
                if self.tracks[tid]["status"]=="lost":continue
                pred=predicted[tid];pred_area=float(_wh(pred).prod())
                intersection=float(np.maximum(0,np.minimum(b[2:],pred[2:])-np.maximum(b[:2],pred[:2])).prod())
                if intersection/pred_area>=self.merged_box_coverage:
                    covered.append((tid,pred_area))
            if len(covered)>=2 and area>=self.merged_box_area_ratio*max(a for _,a in covered):
                candidate_rejections.append({"candidate_id":cid,"reason":"shared_large_box",
                    "covered_track_ids":[tid for tid,_ in covered]})
                for i in range(len(tracks)):edges.pop((i,j),None)
        states={0:(0.,())}
        for j in range(len(ids)):
            nxt=dict(states)
            for used,(cost,pairs) in states.items():
                for i in range(len(tracks)):
                    if used&(1<<i) or (i,j) not in edges:continue
                    key=used|(1<<i);value=(cost+edges[i,j],pairs+((i,j),))
                    if key not in nxt or value<nxt[key]:nxt[key]=value
            states=nxt
        _,(_,pairs)=min(states.items(),key=lambda item:(-item[0].bit_count(),item[1][0],item[1][1]))
        chosen={tracks[i]:ids[j] for i,j in pairs};ambiguous=set()
        for i,(ta,ca) in enumerate(sorted(chosen.items())):
            for tb,cb in sorted(chosen.items())[:i]:
                a,b=raw[ca][1],raw[cb][1]
                if iou_box(a,b)>.95 and np.linalg.norm(_center(a)-_center(b))<.05*min(np.linalg.norm(_wh(a)),np.linalg.norm(_wh(b))):
                    ambiguous.update((ta,tb))
        result=[]
        for tid in tracks:
            state=self.tracks[tid];pred=predicted[tid];cid=chosen.get(tid)
            observed=cid is not None and tid not in ambiguous
            candidate=None
            if observed:
                candidate,b=raw[cid]
                center=self.alpha*_center(b)+(1-self.alpha)*_center(pred)
                wh=np.exp(self.alpha*np.log(_wh(b))+(1-self.alpha)*np.log(_wh(pred)))
                state["filtered"]=_box(center,wh);state["history"].append((frame_id,seconds,b.copy()))
                state["missing"]=0;state["status"]="observed"
            else:
                state["missing"]+=1;state["filtered"]=pred
                age=frame_id-state["history"][-1][0];elapsed=seconds-state["history"][-1][1]
                state["status"]="lost" if state["status"]=="lost" or age>=self.max_missing or elapsed>=.5 else "predicted"
            result.append({"track_id":tid,"candidate_id":cid if observed else None,"status":state["status"],
                "bbox_for_roi":None if state["status"]=="lost" else state["filtered"].tolist(),
                "raw_detection":candidate,"history_length":len(state["history"]),
                "sam_support":support_status.get(tid,"unavailable"),
                "reason":"duplicate_instance_ambiguity" if tid in ambiguous else None,
                "box_quality_scope":"raw_detection_only; temporal_box_not_independently_calibrated",
                "redetection_required":state["status"]=="lost",
                "association_epoch":state.get("association_epoch",0),
                "last_observed_bbox":state["history"][-1][2].tolist(),
                "last_observed_frame":state["history"][-1][0],
                "identity_continuity":state.get("identity_continuity","initial_seed_only")})
        self.frame,self.seconds=frame_id,seconds
        return {"frame_id":frame_id,"tracks":result,"window":self.window,"alpha":self.alpha,
                "mode":"video_inference","automatic_reinitialization":False,"candidate_rejections":candidate_rejections,
                "merged_box_policy":{"area_ratio":self.merged_box_area_ratio,
                                     "track_coverage":self.merged_box_coverage}}
