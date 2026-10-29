"""Causal correction proposals; only verified native rebinding may commit them."""
from collections import deque
import math
import numpy as np

class RecoveryGate:
    def __init__(self,track_id,*,confirmations=5,bad_frames=3,cooldown=15):
        self.track_id=track_id;self.confirmations=confirmations;self.bad_frames=bad_frames;self.cooldown=cooldown
        self.history=deque(maxlen=confirmations);self.last_frame=None;self.bad=0
        self.pending=None;self.last_attempt=-10**9;self.generation=0
    def observe(self,frame_id,mask_status,observation):
        if type(frame_id) is not int or frame_id<0 or (self.last_frame is not None and frame_id<=self.last_frame):
            raise ValueError("Strict ordered video frame required")
        if mask_status not in ("present_unscored","degraded","missing","inference_error","ambiguous"):
            raise ValueError("Explicit SAM state required")
        if self.last_frame is not None and frame_id!=self.last_frame+1:self.history.clear();self.bad=0
        self.last_frame=frame_id
        if mask_status in ("inference_error","ambiguous"):
            self.history.clear();self.bad=0
            return {"action":"no_automatic_correction","reason":mask_status}
        self.bad=self.bad+1 if mask_status in ("missing","degraded") else 0
        raw=observation.get("raw_detection") if observation is not None else None
        eligible=(observation is not None and observation.get("track_id")==self.track_id
                  and observation.get("status")=="observed" and raw is not None)
        if eligible:
            eligible=(math.isfinite(raw["score"]) and math.isfinite(raw["refined_quality_score"])
                      and raw["score"]>=.9 and raw["refined_quality_score"]>=.8
                      and observation["candidate_id"]==f"b0:frame{frame_id}:grid:{raw['grid_index']}")
        if not eligible:self.history.clear()
        else:
            box=np.asarray(raw["bbox_full_xyxy"],float)
            if box.shape!=(4,) or not np.isfinite(box).all() or not (box[2:]>box[:2]).all():
                raise ValueError("Invalid current full-body evidence")
            if self.history:
                prior=np.asarray(self.history[-1]["bbox"])
                motion=np.linalg.norm((box[:2]+box[2:]-prior[:2]-prior[2:])/2)/np.linalg.norm(prior[2:]-prior[:2])
                if motion>.75:self.history.clear()
            self.history.append({"frame":frame_id,"candidate_id":observation["candidate_id"],"bbox":box.tolist()})
        if self.pending is not None:return {"action":"await_native_binding","event_id":self.pending["event_id"]}
        if self.bad<self.bad_frames or len(self.history)<self.confirmations or frame_id-self.last_attempt<self.cooldown:
            return {"action":"collect_evidence","bad_frames":self.bad,"confirmed_observations":len(self.history)}
        self.last_attempt=frame_id
        self.pending={"event_id":f"{self.track_id}:g{self.generation+1}:frame{frame_id}",
            "action":"propose_native_rebinding","track_id":self.track_id,"frame_id":frame_id,
            "source_generation":self.generation,"proposed_generation":self.generation+1,
            "evidence":list(self.history),"bbox_prompt":self.history[-1]["bbox"],
            "requires_unique_native_binding":True,"identity_accepted":False}
        return dict(self.pending)
    def resolve(self,event_id,*,binding_accepted):
        if self.pending is None or self.pending["event_id"]!=event_id or type(binding_accepted) is not bool:
            raise ValueError("Explicit pending event and native binding decision required")
        if binding_accepted:self.generation=self.pending["proposed_generation"];self.bad=0
        self.pending=None;self.history.clear()
