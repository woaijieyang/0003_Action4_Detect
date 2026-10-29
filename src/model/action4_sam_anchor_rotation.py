"""Retire an obsolete correction through the native prompt-removal API."""
from model.action4_sam_history_window import prune_history

def retire_previous_anchor(state,tracker,*,obj_id,initial_frame,old_frame,new_frame,binding):
    if len({initial_frame,old_frame,new_frame})!=3 or old_frame==initial_frame:
        raise ValueError("Never retire initial or current correction anchor")
    if (binding.get("obj_id")!=obj_id or binding.get("frame_index")!=new_frame
        or binding.get("reason")!="bound_unique_prediction_support"):
        raise ValueError("Current native unique binding required before retiring old anchor")
    cores=[c for c in state["sam2_inference_states"] if obj_id in c["obj_id_to_idx"]]
    if len(cores)!=1:raise ValueError("Unique native object state required")
    core=cores[0]
    if set(core["output_dict"]["cond_frame_outputs"])!={initial_frame,old_frame,new_frame}:
        raise ValueError("Unexpected conditioning domain; do not delete unknown anchors")
    tracker.clear_all_points_in_frame(core,old_frame,obj_id,need_output=False)
    if set(core["output_dict"]["cond_frame_outputs"])!={initial_frame,new_frame}:
        raise RuntimeError("Native prompt retirement did not preserve required anchors")
    receipt=prune_history(state,new_frame,after_frame=True,window=15,max_anchors=2)
    return {"retired_frame":old_frame,"kept_anchors":[initial_frame,new_frame],
            "native_operation":"clear_all_points_in_frame(need_output=False)",
            "history_trim":receipt,"archived_predictions_modified":False}
