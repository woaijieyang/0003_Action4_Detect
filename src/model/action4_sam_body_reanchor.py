"""Causal B0-derived body-scope replay proposals; native validation stays external."""
from collections import deque
from copy import deepcopy
import math
import numpy as np

class BodyReanchorGate:
    def __init__(self,native_obj_map,*,source_session_id,confirmation_window=5,
                 minimum_confirmations=3,missing_frames=3,max_lookback=15,cooldown=30):
        if (not isinstance(native_obj_map,dict) or not 1<=len(native_obj_map)<=2
                or any(not isinstance(k,str) or not k for k in native_obj_map)
                or any(type(v) is not int or v<0 for v in native_obj_map.values())
                or len(set(native_obj_map.values()))!=len(native_obj_map)):
            raise ValueError("Unique project/native instance mapping required")
        if not isinstance(source_session_id,str) or not source_session_id:
            raise ValueError("Current native session required")
        if (confirmation_window,minimum_confirmations,missing_frames,max_lookback,cooldown)!=(5,3,3,15,30):
            raise ValueError("Only registered bounded development policy supported")
        self.native_obj_map=dict(native_obj_map);self.session_id=source_session_id
        self.window=confirmation_window;self.confirmations=minimum_confirmations
        self.missing_frames=missing_frames;self.lookback=max_lookback;self.cooldown=cooldown
        self.history={k:deque(maxlen=self.window) for k in native_obj_map}
        self.missing={k:0 for k in native_obj_map}
        self.last_frame=None;self.anchor=None;self.pending=None;self.generation=0
        self.last_attempt=-10**9;self.last_anchor_used=None
        self.events=[]

    def _clear_evidence(self):
        for h in self.history.values():h.clear()
        self.anchor=None
        self.missing={k:0 for k in self.native_obj_map}

    @staticmethod
    def _eligible(track):
        if track.get("status")!="observed":return False
        d=track.get("raw_detection")
        c=track.get("geometry_trace",{}).get("raw_calibration_evidence",{})
        if d is None:return False
        if c.get("accepted") is not True:return False
        if (not isinstance(d.get("candidate_id"),str) or not d["candidate_id"]
                or c.get("candidate_id")!=d["candidate_id"]
                or track.get("candidate_id")!=d["candidate_id"]):
            raise ValueError("Calibration/candidate identity mismatch")
        b=np.asarray(d.get("bbox_full_xyxy"),float);cb=np.asarray(c.get("raw_B0_box"),float)
        if (b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all()
                or cb.shape!=(4,) or not np.array_equal(b,cb)):
            raise ValueError("Current body calibration box mismatch")
        for key in ("score","refined_quality_score"):
            v=d.get(key)
            if (isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v)
                    or not 0<=v<=1 or c.get(key)!=v):
                raise ValueError("Current finite probability/calibration binding required")
            if v<.9:return False
        return True

    def observe(self,row):
        frame=row["frame"]["global_frame_id"]
        if (type(frame) is not int or frame<0 or
                self.last_frame is not None and frame<=self.last_frame):
            raise ValueError("Strict ordered frames required")
        if self.last_frame is not None and frame!=self.last_frame+1:self._clear_evidence()
        self.last_frame=frame
        if row.get("native_state") in ("inference_error","ambiguous_identity","session_lost","missing_response"):
            self._clear_evidence()
            return {"action":"no_automatic_reanchor","reason":row["native_state"]}
        fused=row["fused"]
        if fused.get("frame_id")!=frame or fused.get("mode")!="video_inference":
            raise ValueError("Current video frame binding required")
        tracks={t["track_id"]:t for t in fused["tracks"]}
        if len(tracks)!=len(fused["tracks"]) or set(tracks)!=set(self.native_obj_map):
            raise ValueError("Do not drop, duplicate, or invent established instances")
        eligible={}
        for tid,t in tracks.items():
            if t.get("status") not in ("observed","sam_supported","predicted","lost"):
                raise ValueError("Explicit track observation state required")
            eligible[tid]=self._eligible(t)
            self.history[tid].append((frame,eligible[tid]))
            self.missing[tid]=0 if t["status"]=="observed" else self.missing[tid]+1
        if all(eligible.values()) and all(sum(ok for _,ok in h)>=self.confirmations for h in self.history.values()):
            self.anchor={"frame":frame,"seed_tracks":deepcopy(tracks),
                         "confirmed_frames":{k:[f for f,ok in h if ok] for k,h in self.history.items()}}
        if self.pending is not None:
            return {"action":"await_native_validation","event_id":self.pending["event_id"]}
        reasons=[k for k,v in self.missing.items() if v>=self.missing_frames]
        if (not reasons or self.anchor is None or frame-self.anchor["frame"]>self.lookback
                or frame-self.last_attempt<self.cooldown or self.anchor["frame"]==self.last_anchor_used):
            return {"action":"observe","missing_counts":dict(self.missing)}
        self.last_attempt=frame;self.last_anchor_used=self.anchor["frame"]
        self.pending={"action":"propose_reanchor","event_id":f"body:g{self.generation+1}:frame{frame}",
            "trigger_frame":frame,"seed_frame":self.anchor["frame"],"responsibility_start":frame,
            "source_generation":self.generation,"proposed_generation":self.generation+1,
            "source_session_id":self.session_id,"native_obj_map":dict(self.native_obj_map),
            "seed_tracks":deepcopy(self.anchor["seed_tracks"]),
            "confirmed_frames":deepcopy(self.anchor["confirmed_frames"]),
            "reason":"current_B0_missing_with_recent_confirmed_body_scope","trigger_tracks":reasons,
            "GT_used":False,"native_execution_complete":False,"accuracy_accepted":False}
        return deepcopy(self.pending)

    def resolve(self,event_id,*,binding=None,failure=None):
        if self.pending is None or event_id!=self.pending["event_id"]:
            raise ValueError("Exact pending proposal required")
        event=deepcopy(self.pending)
        if failure is not None:
            if binding is not None or not isinstance(failure,str) or not failure:
                raise ValueError("Explicit exclusive failure required")
            event.update(outcome="failed",failure=failure)
        else:
            if (not isinstance(binding,dict) or binding.get("status")!="native_binding_validated"
                    or binding.get("generation")!=event["proposed_generation"]
                    or binding.get("frame_id")!=event["trigger_frame"]
                    or binding.get("native_obj_map")!=self.native_obj_map
                    or not isinstance(binding.get("session_id"),str) or not binding["session_id"]
                    or binding["session_id"]==self.session_id):
                raise ValueError("New joint-session generation/current-frame binding required")
            self.session_id=binding["session_id"];self.generation=event["proposed_generation"]
            event.update(outcome="native_bound",binding=deepcopy(binding))
        self.events.append(event);self.pending=None
        return deepcopy(event)
