"""Optional frozen B0 scoring policies; no parameter or box geometry changes."""
import torch
from data.action4_dataset import prepare_full_images
from model.action4_online_b0 import proposals
from eval.action4_soft_nms import select

REFINED_QUALITY_PROTOCOL={
    "schema":"action4.B0_refined_full_box_quality.v1",
    "source":"BoxRefinement.quality_logit.sigmoid",
    "training_target":"current_refined_full_box_IoU_detached;negative_zero",
    "calibration":"not_calibrated_by_score_export",
}

@torch.inference_mode()
def _candidate_rows(model,image,cfg,device,policy,*,include_refined_quality=False,appearance=False):
    if type(include_refined_quality) is not bool:raise ValueError("Explicit refined quality diagnostic flag required")
    if policy not in ("center","product","learned"):
        raise ValueError("Explicit B0 scoring policy required")
    x,aff=prepare_full_images([image],size=tuple(cfg["preprocess"]["full_size"]),device=device)
    dense,feature=model(x)
    captured=[]
    handle=None
    if appearance:
        handle=model.refiner.appearance.appearance.register_forward_hook(lambda _module,_args,value:captured.append(value.detach().clone()))
    try:
        rows,_,output=proposals(model,dense,feature,aff[0],cfg)
    finally:
        if handle is not None:handle.remove()
    vectors=None;norms=None
    if appearance:
        if len(captured)!=1 or captured[0].shape!=(len(rows),128) or not torch.isfinite(captured[0]).all():
            raise ValueError("Same-forward B0 appearance must be finite N x 128")
        norms=captured[0].norm(dim=1)
        vectors=torch.nn.functional.normalize(captured[0],dim=1,eps=1e-12).cpu().tolist()
        norms=norms.cpu().tolist()
    refined=output[0].cpu().tolist();learned=output[1].sigmoid().cpu().tolist()
    candidates=[]
    for index,row in enumerate(rows):
        score=row["center_score"] if policy=="center" else row["old_product_score"] if policy=="product" else learned[index]
        item=dict(row,bbox_full_xyxy=refined[index],score=float(score))
        if include_refined_quality:
            import math
            if not math.isfinite(learned[index]) or not 0<=learned[index]<=1:raise ValueError("Invalid refined box quality")
            item.update(refined_quality_score=float(learned[index]),refined_quality_protocol=dict(REFINED_QUALITY_PROTOCOL))
        if appearance:
            item.update(appearance_128=vectors[index],appearance_available=norms[index]>1e-12,
                        appearance_kind="refiner.appearance.appearance_l2_normalized")
        candidates.append(item)
    return candidates


def association_pool(candidates):
    """Keep every decoded hypothesis; single-frame top2 is a separate view."""
    import copy
    rows=copy.deepcopy(candidates)
    grids=[p["grid_index"] for p in rows]
    if any(type(g) is not int or g<0 for g in grids) or len(set(grids))!=len(grids):
        raise ValueError("Unique frame-local native grid identities required")
    return {"schema":"action4.b0_association_pool.v1","association_candidates":rows,
            "single_frame_selected":select(rows,"soft_gaussian50"),
            "decoded_candidate_count":len(rows),"post_decode_candidates_dropped":0,
            "native_decoder_budget_unchanged":True,"frame_and_model_binding_required":True,
            "identity_confirmed":False,"tracking_applied":False,"publication_ready":False}


@torch.inference_mode()
def predict_candidate_pool(model,image,cfg,device,policy):
    """Opt-in association input. No top2 deletion or appearance rerun."""
    if model.training:raise ValueError("Association pool requires evaluation-mode B0")
    return association_pool(_candidate_rows(model,image,cfg,device,policy,
                                            include_refined_quality=True,appearance=True))


@torch.inference_mode()
def predict(model,image,cfg,device,policy,*,include_refined_quality=False):
    # Preserve baseline/public API and exact default dictionaries.
    return select(_candidate_rows(model,image,cfg,device,policy,
                  include_refined_quality=include_refined_quality),"soft_gaussian50")
