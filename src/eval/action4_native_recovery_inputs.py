"""Current video evidence -> native recovery arguments. Missing B0 stays missing."""
import numpy as np


def current_recovery_inputs(instance,mask,*,frame_id,pts,track_id):
    if (instance["frame_id"]!=frame_id or instance["pts"]!=pts or instance["track_id"]!=track_id):
        raise ValueError("Same current frame/instance required")
    state=instance["current_support_status"]
    b0=state.get("B0");sam=state.get("SAM")
    if type(b0) is not bool or type(sam) is not bool:raise ValueError("Explicit current observation flags required")
    box=instance.get("bbox_b0_raw");score=instance.get("score")
    if b0:
        b=np.asarray(box,float)
        if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any() or type(score) not in (float,int) or not np.isfinite(score):
            raise ValueError("Current B0 requires its actual finite raw box and score")
    elif box is not None or score is not None:
        raise ValueError("Absent B0 cannot inherit a historical score or substitute a SAM box")
    if sam and (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=2 or not mask.any()):
        raise ValueError("Declared current SAM requires actual nonempty native support")
    overlap=None
    if b0 and sam:
        yy,xx=np.nonzero(mask)
        overlap=float(((xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3])).mean())
    return dict(current_B0=b0,current_SAM=sam,b0_score=score,b0_box=box,mask_inside_box=overlap)
