"""Source-aligned train/eval prediction-crop distribution diagnostics."""
import numpy as np
from data.action4_instance_assignment import iou,match_geometry
from data.action4_dataset import virtual_roi,is_ignored

def distribution(predictions,records,expansion=1.5):
    mapping={r["sample_id"]:r["instances"] for r in predictions}
    if len(mapping)!=len(predictions) or set(mapping)!={r["sample_id"] for r in records}:
        raise ValueError("Exact split image domain required")
    rows=[];missing=unmatched=0
    for record in records:
        gs=[g for g in record["instances"] if not is_ignored(g)]
        ps=[p for p in mapping[record["sample_id"]] if p["score"]>=.3]
        pairs=match_geometry([np.array(p["bbox_full_xyxy"]) for p in ps],[np.array(g["bbox_full_xyxy"]) for g in gs],
            [p["grid_index"] for p in ps],[g["instance_id"] for g in gs])
        missing+=len(gs)-len(pairs);unmatched+=len(ps)-len(pairs)
        for pi,gi in pairs:
            p,g=ps[pi],gs[gi];a=np.array(p["bbox_full_xyxy"]);b=np.array(g["bbox_full_xyxy"])
            gt_wh=b[2:]-b[:2];wh=a[2:]-a[:2];diag=np.linalg.norm(gt_wh)
            roi=np.array(virtual_roi(a,expansion));intersection=np.maximum(0,np.minimum(roi[2:],b[2:])-np.maximum(roi[:2],b[:2]))
            rows.append(dict(sample_id=record["sample_id"],instance_id=g["instance_id"],grid_index=p["grid_index"],
                box_iou=iou(a,b),center_error_over_GT_diagonal=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])*.5)/diag),
                width_ratio=float(wh[0]/gt_wh[0]),height_ratio=float(wh[1]/gt_wh[1]),
                GT_full_bbox_fraction_inside_ROI=float(np.prod(intersection)/np.prod(gt_wh)),
                GT_full_bbox_area_over_ROI=float(np.prod(gt_wh)/np.prod(roi[2:]-roi[:2]))))
    keys=["box_iou","center_error_over_GT_diagonal","width_ratio","height_ratio","GT_full_bbox_fraction_inside_ROI","GT_full_bbox_area_over_ROI"]
    summary={k:dict(count=len(rows),mean=float(np.mean([r[k] for r in rows])) if rows else None,
        quantiles={str(q):float(np.percentile([r[k] for r in rows],q)) if rows else None for q in [0,5,50,95,100]}) for k in keys}
    return dict(scope="GT used only for diagnostics, never crop adjustment; bbox area is not visible-mask occupancy",
        images=len(records),matched=len(rows),missing_GT=missing,unmatched_predictions=unmatched,
        roi_expansion=expansion,summary=summary,rows=rows)
