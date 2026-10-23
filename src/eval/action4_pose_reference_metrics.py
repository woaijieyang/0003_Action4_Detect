"""Coarse pose-envelope agreement with demo, never corner-ID ground truth accuracy."""
import numpy as np
from eval.action4_metrics import bbox_iou,maximum_matching

def compare_pose_envelopes(reference,predictions):
    refs=reference["rows"]
    if len(refs)!=len(predictions):raise ValueError("Full frame coverage required")
    scale=np.array([950/3248,720/2464]*2);xy_scale=scale[:2];rows=[]
    for fid,(ref,row) in enumerate(zip(refs,predictions)):
        if ref["frame_id"]!=fid or row["frame_id"]!=fid:raise ValueError("Frame identity mismatch")
        if not ref["reference_candidate"]:continue
        gt=[np.array(g["bbox_demo_px"]) for g in ref["components"]]
        ps=row["prediction"]["instances"]
        pairs=maximum_matching([[j for j,g in enumerate(gt) if bbox_iou(np.array(p["bbox_full_xyxy"])*scale,g)>=.5] for p in ps],2)
        poses=[]
        for pi,gi in pairs:
            p=ps[pi];valid=bool(p.get("pose_valid",False) and p.get("corners_px") is not None)
            iou=0.;distance=None;ratio=None
            if valid:
                corners=np.array(p["corners_px"])*xy_scale
                if corners.shape!=(8,2) or not np.isfinite(corners).all():raise ValueError("Invalid canonical8 output")
                envelope=np.r_[corners.min(0),corners.max(0)]
                iou=float(bbox_iou(envelope,gt[gi]))
                distance=float(np.linalg.norm((envelope[:2]+envelope[2:]-gt[gi][:2]-gt[gi][2:])*.5)/np.linalg.norm(gt[gi][2:]-gt[gi][:2]))
                ratio=float(np.prod(envelope[2:]-envelope[:2])/np.prod(gt[gi][2:]-gt[gi][:2]))
            poses.append({"prediction_index":pi,"reference_index":gi,"valid":valid,"envelope_iou":iou,
                "center_error_over_ref_diag":distance,"area_ratio":ratio,"envelope_matches":valid and iou>=.5})
        rows.append({"frame_id":fid,"both_boxes_match":len(pairs)==2,"both_pose_envelopes_match":len(poses)==2 and all(p["envelope_matches"] for p in poses),"poses":poses})
    return {"scope":"demo reference cuboid-envelope proxy only; not corner-ID or pose accuracy",
        "eligible_frames":len(rows),"unscorable_reference_frames":len(refs)-len(rows),
        "both_box_frames":sum(r["both_boxes_match"] for r in rows),
        "both_pose_envelope_frames":sum(r["both_pose_envelopes_match"] for r in rows),
        "failed_or_missed_are_not_removed":True,"rows":rows}
