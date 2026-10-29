"""Seed-only positive-prompt component eligibility; raw native mask is preserved."""
import numpy as np
from scipy.ndimage import label
from data.action4_sam_roi_inputs import array_digest

def seed_component_evidence(raw_mask,points_native,point_labels):
    if not isinstance(raw_mask,np.ndarray) or raw_mask.dtype!=np.bool_ or raw_mask.ndim!=2:
        raise ValueError("Actual native bool mask required")
    points=np.asarray(points_native,float);labels=np.asarray(point_labels)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)!=len(labels) or not np.isfinite(points).all()
        or labels.ndim!=1 or not np.isin(labels,[0,1]).all() or not (labels==1).any()):
        raise ValueError("Actual positive/negative prompt identities required")
    pixels=np.rint(points).astype(int)
    if (pixels[:,0]<0).any() or (pixels[:,0]>=raw_mask.shape[1]).any() or (pixels[:,1]<0).any() or (pixels[:,1]>=raw_mask.shape[0]).any():
        raise ValueError("Native prompt outside image")
    components,n=label(raw_mask,structure=np.ones((3,3),int))
    at=components[pixels[:,1],pixels[:,0]]
    ids=np.unique(at[labels==1]);out=np.zeros_like(raw_mask)
    report=dict(status="unknown_seed_component",raw_mask_sha256=array_digest(raw_mask),
        raw_foreground_pixels=int(raw_mask.sum()),component_count=n,positive_component_ids=ids.tolist(),
        raw_mask_modified=False,native_quality_report_must_keep_all_components=True,
        full_face_verified=False,GT_used=False,propagation_admission=False)
    if len(ids)!=1 or ids[0]==0:
        report["reason"]="positive_points_missing_or_split_across_components"
    elif (at[labels==0]==ids[0]).any():
        report["reason"]="negative_prompt_in_selected_component"
    else:
        out=components==ids[0]
        report.update(status="positive_prompt_component_for_candidate_geometry_only",
            reason="unprompted_disconnected_components_retained_as_unknown",
            eligible_pixels=int(out.sum()),unknown_unprompted_pixels=int(raw_mask.sum()-out.sum()),
            selected_component_id=int(ids[0]))
    report["eligible_mask_sha256"]=array_digest(out)
    return out,report
