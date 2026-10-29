"""Read native per-frame object logits without confusing them with seed scores."""
import math

def tracking_scores(state,frame_idx):
    meta=state.get("tracker_metadata",{})
    values=meta.get("obj_id_to_sam2_score_frame_wise",{}).get(frame_idx)
    if values is None:return {"frame":frame_idx,"available":False,"reason":"native_frame_score_missing"}
    result={}
    for obj_id,value in values.items():
        value=float(value.detach().cpu().item() if hasattr(value,"detach") else value)
        if not math.isfinite(value):raise ValueError("Nonfinite native object score")
        prob=1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))
        result[str(int(obj_id))]={"object_logit_raw":value,"sigmoid_once":prob,
                                 "is_DJI_identity_probability":False,"is_calibrated_mask_quality":False}
    return {"frame":frame_idx,"available":True,"objects":result,
            "source":"native PVS propagation object_score_logits","public_out_probs_not_used":True}
