"""Read-only development quality diagnostics; never trains or publishes a gate."""
import numpy as np
from eval.action4_metrics import bbox_iou


def summarize_quality(q, iou):
    q=np.asarray(q,dtype=float);iou=np.asarray(iou,dtype=float)
    if q.ndim!=1 or q.shape!=iou.shape or not np.isfinite(q).all() or not np.isfinite(iou).all():
        raise ValueError("Equal finite quality/IoU vectors required")
    if ((q<0)|(q>1)|(iou<0)|(iou>1)).any():raise ValueError("Quality and IoU must be in0..1")
    if not len(q):return {"count":0,"MAE":None,"signed_bias":None,"pearson":None,"bins":[]}
    bins=[]
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        use=(q>=lo)&((q<hi) if hi<1 else (q<=hi))
        bins.append(dict(lower=float(lo),upper=float(hi),count=int(use.sum()),
            mean_q=float(q[use].mean()) if use.any() else None,
            mean_IoU=float(iou[use].mean()) if use.any() else None))
    return dict(count=len(q),MAE=float(np.abs(q-iou).mean()),signed_bias=float((q-iou).mean()),
        pearson=float(np.corrcoef(q,iou)[0,1]) if np.std(q)>0 and np.std(iou)>0 else None,
        bins=bins)


def prediction_quality_targets(predictions,references,sidecar_rows):
    """Max full-box IoU is a quality diagnostic, NOT one-to-one detector matching."""
    ids=[r["sample_id"] for r in references]
    if len(set(ids))!=len(ids) or ids!=[r["sample_id"] for r in predictions] or ids!=[r["image"]["sample_id"] for r in sidecar_rows]:
        raise ValueError("Exact ordered unique image domain required")
    output=[]
    from data.action4_sam_native_export import canonical
    for p,r,s in zip(predictions,references,sidecar_rows):
        qs={x["grid_index"]:x for x in s["qualities"]}
        if len(qs)!=len(s["qualities"]) or len(qs)!=len(p["instances"]):raise ValueError("Quality candidate domain differs")
        gs=[g for g in r["instances"] if not g.get("ignore",False)]
        for x in p["instances"]:
            q=qs[x["grid_index"]]
            if q["base_row_sha256"]!=canonical(x):raise ValueError("Quality/base identity differs")
            overlaps=[bbox_iou(x["bbox_full_xyxy"],g["bbox_full_xyxy"]) for g in gs]
            # Empty/ignore-only is left unknown: no invented background label.
            output.append(dict(sample_id=r["sample_id"],grid_index=x["grid_index"],
                q=q["refined_quality_score"],max_nonignore_full_box_IoU=max(overlaps) if overlaps else None,
                quality_target_kind="max_nonignore_full_box_IoU_not_detection_assignment",
                GT_count=len(gs)))
    return output
