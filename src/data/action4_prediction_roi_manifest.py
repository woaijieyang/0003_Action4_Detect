"""Frozen B0 crop ownership. GT assigns supervision only, never changes boxes."""
import math
import numpy as np
from .action4_dataset import is_ignored
from .action4_instance_assignment import iou,match_geometry

POLICY={"schema":"action4.frozen_prediction_roi_ownership.v1","score_threshold":0.3,"iou_threshold":0.5,
 "matching":["maximum_cardinality","maximum_total_IoU","stable_ID"],
 "multiple_GT_edges":"ambiguous_no_pose_supervision","GT_crop_fallback":False,
 "GT_box_correction":False,"online_augmentation":False}

def associate(record,predictions):
    for g in record["instances"]:
        b=np.asarray(g.get("bbox_full_xyxy"),float)
        if b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all():
            raise ValueError("Invalid GT full box; no association attempted")
    gt=[(i,g) for i,g in enumerate(record["instances"]) if not is_ignored(g)]
    ignored=[g for g in record["instances"] if is_ignored(g)]
    rows=[];eligible=[]
    grids=[p["grid_index"] for p in predictions]
    if any(type(g) is not int or g<0 for g in grids) or len(grids)!=len(set(grids)):raise ValueError("Duplicate native B0 grid")
    for pred in predictions:
        box=np.asarray(pred["bbox_full_xyxy"],float);score=pred["score"]
        if box.shape!=(4,) or not np.isfinite(box).all() or not (box[2:]>box[:2]).all() or isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError("Invalid B0 prediction")
        row={"sample_id":record["sample_id"],"prediction_id":f"b0:{record['sample_id']}:grid:{pred['grid_index']}",
             "grid_index":pred["grid_index"],"score":score,"bbox_for_a0":box.tolist(),
             "roi_source":"frozen_b0_prediction","GT_instance_id":None,"GT_instance_index":None,
             "pose_supervision_association":False,"crop_geometry_eligibility":"not_yet_evaluated"}
        edges=[j for j,(_,g) in enumerate(gt) if iou(box,np.asarray(g["bbox_full_xyxy"]))>=.5]
        if score<.3:row["status"]="below_operating_threshold"
        elif any(iou(box,np.asarray(g["bbox_full_xyxy"]))>=.5 for g in ignored):row["status"]="ignore_overlap"
        elif len(edges)>1:row["status"]="ambiguous_GT_overlap"
        elif not edges:row["status"]="unmatched"
        else:row["status"]="eligible";eligible.append(len(rows))
        rows.append(row)
    pairs=match_geometry([np.asarray(rows[i]["bbox_for_a0"]) for i in eligible],
        [np.asarray(g["bbox_full_xyxy"]) for _,g in gt],[rows[i]["prediction_id"] for i in eligible],[g["instance_id"] for _,g in gt])
    used=set()
    for pi,gi in pairs:
        i=eligible[pi];index,g=gt[gi];used.add(gi)
        rows[i].update(status="matched_unique",GT_instance_id=g["instance_id"],GT_instance_index=index,
                       pose_supervision_association=True)
    for i in eligible:
        if rows[i]["status"]=="eligible":rows[i]["status"]="duplicate_unselected"
    return rows,[g["instance_id"] for j,(_,g) in enumerate(gt) if j not in used]
