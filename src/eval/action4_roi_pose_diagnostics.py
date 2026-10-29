"""Matched-only physical pose diagnostics; never used to rematch or select a pose."""
import numpy as np

def summarize(predictions,records,pose_report):
    predictions={(r["sample_id"],p["prediction_id"]):p for r in predictions for p in r["instances"]}
    truth={(r["sample_id"],g["instance_id"]):g for r in records for g in r["instances"]}
    rows=[];omitted=0
    for match in pose_report["per_gt_failure_reason"]:
        if match["prediction_id"] is None:continue
        p=predictions[(match["sample_id"],match["prediction_id"])]
        raw=p.get("raw_network_pose")
        if not raw or raw.get("finite") is not True:
            omitted+=1;continue
        if raw.get("coordinate_frame")!="object_to_capture_cv_m":raise ValueError("Pose coordinate frame differs")
        g=truth[(match["sample_id"],match["instance_id"])]
        T=np.asarray(g["canonical8_geometry"]["object_to_capture_cv_m"],dtype=float)
        rotation=np.asarray(raw["rotation"],dtype=float);translation=np.asarray(raw["translation_m"],dtype=float)
        if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Invalid finite raw pose")
        rotation_error=float(np.degrees(np.arccos(np.clip((np.trace(rotation@T[:3,:3].T)-1)/2,-1,1))))
        error=translation-T[:3,3]
        rows.append(dict(sample_id=match["sample_id"],instance_id=match["instance_id"],prediction_id=match["prediction_id"],
            rotation_error_deg=rotation_error,translation_center_error_m=float(np.linalg.norm(error)),
            depth_abs_error_m=float(abs(error[2])),depth_signed_error_m=float(error[2])))
    metrics={}
    for key in ("rotation_error_deg","translation_center_error_m","depth_abs_error_m","depth_signed_error_m"):
        values=[row[key] for row in rows]
        metrics[key]={"count":len(values),"median":float(np.median(values)) if values else None,
                      "p95":float(np.percentile(values,95)) if values else None}
    corners=[e for row in pose_report["per_gt_failure_reason"] if row["errors_px"] is not None for e in row["errors_px"]]
    return dict(scope="fixed pose-associated matching only; missing outputs remain in primary FP/FN",
        full_GT_count=pose_report["pose_gt_count"],pose_output_count=pose_report["pose_output_count"],
        matched_pose_with_raw_RT=len(rows),matched_pose_missing_raw_RT=omitted,
        corner_EPE_count=len(corners),corner_EPE_median_native_px=float(np.median(corners)) if corners else None,
        metrics=metrics,rows=rows)
