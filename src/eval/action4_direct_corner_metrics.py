"""Fixed-ID direct corner-head metrics; no PnP or pose validity gate."""
import numpy as np
from data import DataAdapter
from data.action4_dataset import is_ignored
from data.action4_instance_assignment import match_geometry

SCHEMA="action4.direct_corner_head_all8.v1"

def score_direct_head(payloads,records):
    byid={p["image_id"]:p for p in payloads}
    if len(byid)!=len(payloads) or set(byid)!={r["sample_id"] for r in records}:
        raise ValueError("Exactly one direct prediction payload per image required")
    rows=[];total=published=invalid=success2=success5=0
    for record in records:
        payload=byid[record["sample_id"]]
        observed=payload["corner_observations"]
        ids=[x["prediction_id"] for x in observed]
        if len(set(ids))!=len(ids):raise ValueError("Duplicate direct prediction identity")
        gt=[g for g in record["instances"] if not is_ignored(g)]
        boxes=[np.asarray(x["predicted_full_box"],float) for x in observed]
        pairs=match_geometry(boxes,[np.asarray(g["bbox_full_xyxy"],float) for g in gt],ids,[g["instance_id"] for g in gt])
        mapping={g:p for p,g in pairs};points=[]
        for ob in observed:
            obs=ob["observations"];uv=np.asarray(obs["uv224"],float);affine=np.asarray(ob["A224_from_native"],float)
            valid=obs["corner_ids"]==list(range(8)) and uv.shape==(8,2) and np.isfinite(uv).all() and affine.shape==(3,3) and np.isfinite(affine).all()
            try: native=DataAdapter.transform_points(uv,np.linalg.inv(affine)) if valid else None
            except np.linalg.LinAlgError:native=None
            if native is None or not np.isfinite(native).all():native=None;invalid+=1
            points.append(native)
        total+=len(gt);published+=len(observed)
        for j,g in enumerate(gt):
            pi=mapping.get(j);uv=None if pi is None else points[pi]
            target=np.asarray(g["corners_px"],float)
            if target.shape!=(8,2) or not np.isfinite(target).all():raise ValueError("Invalid fixed8 GT")
            diagonal=float(np.linalg.norm(target.max(0)-target.min(0)))
            if diagonal<=0:raise ValueError("Nonpositive GT corner diagonal")
            errors=None if uv is None else np.linalg.norm(uv-target,axis=1)
            ok2=bool(errors is not None and errors.max()<=.02*diagonal)
            ok5=bool(errors is not None and errors.max()<=.05*diagonal)
            success2+=ok2;success5+=ok5
            rows.append(dict(sample_id=record["sample_id"],instance_id=g["instance_id"],prediction_id=None if pi is None else ids[pi],
                errors_px=None if errors is None else errors.tolist(),all8_at_002=ok2,all8_at_005=ok5,
                failure_reason=None if ok2 else "missing_B0_box_match" if pi is None else "nonfinite_direct_head" if uv is None else "corner_error"))
    return dict(score_schema=SCHEMA,source="corner_head.uv224->inverse_A224->native",PnP_used=False,
        fixed_corner_ids=list(range(8)),confidence_filter=False,association="frozen_B0_full_box_IoU0.5_stable_ID",
        normalization="GT_corner_envelope_diagonal",images=len(records),gt_count=total,output_count=published,
        invalid_output_count=invalid,all8_at_002_success=success2,all8_at_005_success=success5,
        all8_at_002_rate=success2/total if total else None,all8_at_005_rate=success5/total if total else None,
        per_gt_failure_reason=rows)
