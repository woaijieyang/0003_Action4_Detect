"""Five-frame position tracker with bounded, explicit lost-slot reacquisition."""
from collections import deque
import numpy as np
from model.action4_video_box_history import VideoBoxHistory,_center,_wh
from model.action4_temporal_association import iou_box

def same_hypothesis(a,b):
    aa=np.asarray(a["bbox_full_xyxy"]);bb=np.asarray(b["bbox_full_xyxy"])
    if a["candidate_id"]==b["candidate_id"]:return True
    if abs(a.get("grid_index",-1000)-b.get("grid_index",1000))!=1:return False
    if iou_box(aa,bb)<.95 or np.linalg.norm(_center(aa)-_center(bb))>.05*min(np.linalg.norm(_wh(aa)),np.linalg.norm(_wh(bb))):return False
    av=np.asarray(a.get("appearance_128",[]));bv=np.asarray(b.get("appearance_128",[]))
    return (av.shape==bv.shape==(128,) and np.linalg.norm(av)>0 and np.linalg.norm(bv)>0
            and float(np.dot(av,bv)/np.linalg.norm(av)/np.linalg.norm(bv))>=.995)

def match(left_count,right_count,edges):
    states={0:(0.,())}
    for j in range(right_count):
        nxt=dict(states)
        for used,(cost,pairs) in states.items():
            for i in range(left_count):
                if used&(1<<i) or (i,j) not in edges:continue
                key=used|(1<<i);value=(cost+edges[i,j],pairs+((i,j),))
                if key not in nxt or value<nxt[key]:nxt[key]=value
        states=nxt
    return min(states.items(),key=lambda item:(-item[0].bit_count(),item[1]))[1][1]

class FiveFrameTracker:
    def __init__(self,*,mode="video_inference",window=5):
        if window not in (5,10):raise ValueError("Registered comparison supports 5 or10 history frames")
        self.core=VideoBoxHistory(mode=mode,window=window)
        self.provisional=[];self.last_frame=None
    def initialize(self,frame_id,seconds,seeds):
        self.core.initialize(frame_id,seconds,seeds);self.last_frame=frame_id
    def update(self,frame_id,seconds,candidates,*,sam_support_boxes=None):
        output=self.core.update(frame_id,seconds,candidates,sam_support_boxes=sam_support_boxes)
        active=[t["raw_detection"] for t in output["tracks"] if t["status"]=="observed"]
        needs=[t for t in output["tracks"] if t["status"]!="observed"]
        rejected={r["candidate_id"] for r in output["candidate_rejections"]}
        eligible=[]
        for c in sorted(candidates,key=lambda x:(-x["score"],x["candidate_id"])):
            if not (.9<=c["score"]<=1 and .8<=c.get("refined_quality_score",-1)<=1):continue
            if c["candidate_id"] in rejected or any(same_hypothesis(c,a) for a in active+eligible):continue
            box=np.asarray(c["bbox_full_xyxy"]);covered=[]
            for t in output["tracks"]:
                if frame_id-t["last_observed_frame"]>15:continue
                prior=np.asarray(t["last_observed_bbox"]);area=float(_wh(prior).prod())
                intersection=float(np.maximum(0,np.minimum(box[2:],prior[2:])-np.maximum(box[:2],prior[:2])).prod())
                if intersection/area>=.7:covered.append(area)
            if len(covered)>=2 and float(_wh(box).prod())>=1.4*max(covered):
                continue
            eligible.append(c)
        overflow=max(0,len(eligible)-8);eligible=eligible[:8]
        if not needs:
            self.provisional=[];output["reacquisition_events"]=[];self.last_frame=frame_id
            return output
        if frame_id!=self.last_frame+1:self.provisional=[]
        edges={}
        for i,chain in enumerate(self.provisional):
            a=np.asarray(chain[-1]["candidate"]["bbox_full_xyxy"]);diag=np.linalg.norm(_wh(a))
            for j,c in enumerate(eligible):
                b=np.asarray(c["bbox_full_xyxy"]);motion=np.linalg.norm(_center(a)-_center(b))/diag
                if motion<=.75:edges[i,j]=float(.7*motion+.3*(1-iou_box(a,b)))
        pairs=match(len(self.provisional),len(eligible),edges)
        chains=[];assigned=set()
        for i,j in pairs:
            chain=deque(self.provisional[i],maxlen=5)
            chain.append({"frame":frame_id,"seconds":seconds,"candidate":eligible[j]})
            chains.append(chain);assigned.add(j)
        for j,c in enumerate(eligible):
            if j not in assigned:chains.append(deque([{"frame":frame_id,"seconds":seconds,"candidate":c}],maxlen=5))
        self.provisional=chains
        lost=[t for t in needs if t["status"]=="lost"]
        mature=[chain for chain in chains if len(chain)==5]
        edges={}
        for i,t in enumerate(lost):
            a=np.asarray(t["last_observed_bbox"]);diag=np.linalg.norm(_wh(a))
            for j,chain in enumerate(mature):
                b=np.asarray(chain[-1]["candidate"]["bbox_full_xyxy"])
                edges[i,j]=float(np.linalg.norm(_center(a)-_center(b))/diag)
        pairs=match(len(lost),len(mature),edges)
        events=[];used_chains=set()
        for i,j in pairs:
            t=lost[i];chain=list(mature[j]);candidate=chain[-1]["candidate"]
            epoch=self.core.reinitialize_confirmed(t["track_id"],chain)
            t.update(candidate_id=candidate["candidate_id"],status="observed",bbox_for_roi=candidate["bbox_full_xyxy"],
                     raw_detection=candidate,history_length=5,reason="five_frame_reacquisition",
                     association_epoch=epoch,redetection_required=False,
                     identity_continuity="unverified_after_reacquisition",sam_support="unavailable",
                     last_observed_bbox=candidate["bbox_full_xyxy"],last_observed_frame=frame_id)
            events.append({"frame_id":frame_id,"track_id":t["track_id"],"association_epoch":epoch,
                "evidence_ids":[x["candidate"]["candidate_id"] for x in chain],
                "identity_continuity":"unverified_after_gap","SAM_confirmation_used":False})
            used_chains.add(id(mature[j]))
        self.provisional=[chain for chain in self.provisional if id(chain) not in used_chains]
        output.update(reacquisition_events=events,provisional_candidates_truncated=overflow,
                      automatic_reinitialization=True,reacquisition_policy="five_strong_consecutive_observations")
        self.last_frame=frame_id
        return output
