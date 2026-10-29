"""Video ROI geometry with body-size state independent of mask enclosure.

Input identity/association remains upstream. This layer never changes native
masks, B0 candidate selection, or a track's evidence status. It is not a
certificate of full-body accuracy.
"""
from copy import deepcopy
import math
import numpy as np

POLICY={"center_alpha":.65,"log_size_alpha":.2,"max_enclosure_side_ratio":1.35,
        "max_enclosure_area_ratio":1.6,"max_enclosure_center_shift_per_diagonal":.15}

class BodyBoxProjection:
    def __init__(self,policy=None):
        self.policy=dict(POLICY if policy is None else policy)
        if set(self.policy)!=set(POLICY) or any(not math.isfinite(v) for v in self.policy.values()):
            raise ValueError("Explicit finite body-box policy required")
        if not (0<self.policy["center_alpha"]<=1 and 0<self.policy["log_size_alpha"]<=1
                and self.policy["max_enclosure_side_ratio"]>=1 and self.policy["max_enclosure_area_ratio"]>=1
                and self.policy["max_enclosure_center_shift_per_diagonal"]>=0):
            raise ValueError("Invalid body-box update bounds")
        self.states={}

    def update(self,frame_id,seconds,track,support_box):
        if type(frame_id) is not int or frame_id<0 or not math.isfinite(seconds):
            raise ValueError("Finite frame/time required")
        tid=track["track_id"]
        if not isinstance(tid,str) or not tid:raise ValueError("Established project identity required")
        result=deepcopy(track);previous=self.states.get(tid)
        if previous is not None and (frame_id<=previous["frame"] or seconds<=previous["seconds"]):
            raise ValueError("Strict frame/time ordering required")
        if track["status"] not in ("observed","sam_supported"):
            self.states.pop(tid,None)
            result["body_box_projection"]={"applied":False,"reason":"no_current_evidence","accuracy_verified":False}
            return result
        trace=track["geometry_trace"]
        center=np.asarray(trace["unsmoothed_measurement_center"],float)
        size=np.asarray(trace["trusted_size_median"],float)
        velocity=np.asarray(trace["velocity_native_px_per_second"],float)
        if (any(v.shape!=(2,) or not np.isfinite(v).all() for v in (center,size,velocity))
                or not (size>0).all()):
            raise ValueError("Finite native body center, trusted size and velocity required")
        if previous is not None and frame_id==previous["frame"]+1:
            a=self.policy["center_alpha"];b=self.policy["log_size_alpha"]
            predicted=previous["center"]+velocity*(seconds-previous["seconds"])
            center=a*center+(1-a)*predicted
            size=np.exp(b*np.log(size)+(1-b)*np.log(previous["size"]))
        body=np.r_[center-size/2,center+size/2];published=body.copy()
        decision="no_current_mask_extent";diagnostics={}
        if support_box is not None:
            support=np.asarray(support_box,float)
            if support.shape!=(4,) or not np.isfinite(support).all() or not (support[2:]>support[:2]).all():
                raise ValueError("Finite current support extent required")
            enclosed=np.r_[np.minimum(body[:2],support[:2]),np.maximum(body[2:],support[2:])]
            ratio=(enclosed[2:]-enclosed[:2])/size
            shift=float(np.linalg.norm((enclosed[:2]+enclosed[2:])/2-center)/np.linalg.norm(size))
            diagnostics={"side_ratio":ratio.tolist(),"area_ratio":float(ratio.prod()),
                "center_shift_per_diagonal":shift,"proposed_enclosure":enclosed.tolist()}
            accepted=(float(ratio.max())<=self.policy["max_enclosure_side_ratio"]
                and float(ratio.prod())<=self.policy["max_enclosure_area_ratio"]
                and shift<=self.policy["max_enclosure_center_shift_per_diagonal"])
            if accepted:
                published=enclosed;decision="accepted_bounded_extent"
            else:decision="rejected_extent_conflict"
        # Crucially, the enclosing rectangle is never a future body-size sample.
        self.states[tid]={"frame":frame_id,"seconds":seconds,"center":center.copy(),"size":size.copy()}
        result["bbox_for_roi"]=published.tolist()
        result["body_box_projection"]={"applied":True,"policy":"action4.body_box_projection.v1",
            "original_bbox_for_roi":deepcopy(track["bbox_for_roi"]),
            "filtered_body_box":body.tolist(),"current_support_box":None if support_box is None else list(support_box),
            "enclosure_decision":decision,"enclosure_diagnostics":diagnostics,
            "enclosure_fed_back_to_state":False,"association_and_native_ID_unchanged":True,
            "accuracy_verified":False}
        return result
