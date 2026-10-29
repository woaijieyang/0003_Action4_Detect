"""Validate a new joint session's bounded replay before changing ownership."""
import numpy as np

def native_masks(raw,*,frame_id,height,width,native_obj_map):
    if raw.get("frame_index")!=frame_id:raise ValueError("Native frame mismatch")
    out=raw["outputs"];ids=np.asarray(out["out_obj_ids"])
    masks=np.asarray(out["out_binary_masks"]);prob=np.asarray(out["out_probs"]);boxes=np.asarray(out["out_boxes_xywh"])
    if (ids.ndim!=1 or ids.dtype.kind not in "iu" or len(set(ids.tolist()))!=len(ids)
            or not set(ids.tolist())<=set(native_obj_map.values())):
        raise ValueError("Unknown/duplicated native instance ID")
    if (masks.dtype!=np.bool_ or masks.shape!=(len(ids),height,width)
            or prob.shape!=(len(ids),) or boxes.shape!=(len(ids),4)
            or not np.isfinite(prob).all() or not np.isfinite(boxes).all()
            or ((prob<0)|(prob>1)).any()):
        raise ValueError("Invalid native shape, pixel domain, or numeric output")
    return {tid:(masks[ids.tolist().index(oid)] if oid in ids else None)
            for tid,oid in native_obj_map.items()}

class CandidateReplay:
    def __init__(self,*,session_id,generation,seed_frame,trigger_frame,native_obj_map,height,width):
        if not isinstance(session_id,str) or not session_id:raise ValueError("New session required")
        if (any(type(v) is not int or v<0 for v in (generation,seed_frame,trigger_frame))
                or seed_frame>trigger_frame or trigger_frame-seed_frame>15):
            raise ValueError("Bounded replay identity required")
        if (not isinstance(native_obj_map,dict) or not native_obj_map
                or len(set(native_obj_map.values()))!=len(native_obj_map)
                or any(type(v) is not int or v<0 for v in native_obj_map.values())
                or any(not isinstance(k,str) or not k for k in native_obj_map)):
            raise ValueError("Unique established native mapping required")
        if any(type(v) is not int or v<=0 for v in (height,width)):
            raise ValueError("Native dimensions required")
        self.session_id=session_id;self.generation=generation
        self.next_frame=seed_frame;self.trigger_frame=trigger_frame
        self.native_obj_map=dict(native_obj_map);self.height=height;self.width=width;self.last_masks=None

    def consume(self,session_id,generation,raw):
        if (session_id!=self.session_id or generation!=self.generation
                or self.next_frame>self.trigger_frame):
            raise ValueError("Replay belongs to another session/generation or is complete")
        masks=native_masks(raw,frame_id=self.next_frame,height=self.height,width=self.width,native_obj_map=self.native_obj_map)
        self.last_masks={k:(None if v is None else v.copy()) for k,v in masks.items()}
        self.next_frame+=1
        return masks

    def binding(self):
        if self.next_frame!=self.trigger_frame+1:raise ValueError("Incomplete current-frame replay")
        masks=list(self.last_masks.values())
        if any(m is None or not m.any() for m in masks):
            raise ValueError("Current confirmed identities unavailable; do not adopt candidate")
        for i,a in enumerate(masks):
            for b in masks[i+1:]:
                if float((a&b).sum()/max(1,(a|b).sum()))>=.9:
                    raise ValueError("Shared mask is not independent instance evidence")
        return {"session_id":self.session_id,"generation":self.generation,
                "frame_id":self.trigger_frame,"native_obj_map":dict(self.native_obj_map),
                "status":"native_binding_validated"}
