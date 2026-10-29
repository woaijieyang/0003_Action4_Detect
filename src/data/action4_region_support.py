"""Current-image component association for geometry, never native mask rewriting.

Every supplied current cue must support the same component; statistical
independence of those cues is not implied.
No largest-component heuristic, historical mask, GT, or full-visibility claim.
"""
import numpy as np
from scipy.ndimage import label
from .action4_sam_roi_inputs import array_digest


def associate_supported_component(raw_mask, supports, *, minimum_coverage=.9, minimum_margin=.15):
    if (not isinstance(raw_mask,np.ndarray) or raw_mask.dtype!=np.bool_ or raw_mask.ndim!=2
        or not isinstance(supports,dict) or len(supports)<2
        or not np.isfinite([minimum_coverage,minimum_margin]).all()
        or not 0<minimum_coverage<=1 or not 0<=minimum_margin<=1):
        raise ValueError("Actual native bool mask, at least two current support roles and finite gates required")
    for role,mask in supports.items():
        if not isinstance(role,str) or not role or not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=raw_mask.shape:
            raise ValueError("Distinct support roles must share the exact current pixel domain")
    components,count=label(raw_mask,np.ones((3,3),int));areas=np.bincount(components.ravel(),minlength=count+1)
    report=dict(status="unavailable",reason=None,raw_mask_sha256=array_digest(raw_mask),raw_pixels=int(raw_mask.sum()),
        component_count=count,component_areas={str(i):int(areas[i]) for i in range(1,count+1)},support={},
        selected_component=None,selected_pixels=0,unknown_unselected_pixels=int(raw_mask.sum()),
        native_mask_modified=False,native_quality_denominator_unchanged=True,full_visibility_verified=False,
        complete_physical_boundary_verified=False,GT_used=False,history_used=False)
    support_hashes={role:array_digest(mask) for role,mask in supports.items()}
    if len(set(support_hashes.values()))!=len(support_hashes):
        report.update(reason="duplicate_support_pixels_not_two_distinct_cues",support_hashes=support_hashes)
        return None,report
    chosen=[]
    for role,mask in supports.items():
        size=int(mask.sum());overlap=np.bincount(components[mask],minlength=count+1)
        coverage=overlap[1:]/max(1,size);order=np.argsort(-coverage,kind="stable")
        first=float(coverage[order[0]]) if len(order) else 0.;second=float(coverage[order[1]]) if len(order)>1 else 0.
        valid=size>0 and bool(len(order)) and first>=minimum_coverage and first-second>=minimum_margin
        report["support"][role]=dict(pixels=size,sha256=array_digest(mask),best_coverage=first,
            runner_up_coverage=second,component=int(order[0]+1) if valid else None,unique=valid)
        if valid:chosen.append(int(order[0]+1))
    if len(chosen)!=len(supports):report["reason"]="missing_or_nonunique_current_support"
    elif len(set(chosen))!=1:report["reason"]="current_support_roles_disagree_on_component"
    else:
        selected=components==chosen[0]
        report.update(status="current_supported_component_for_geometry_only",reason="all_current_roles_agree",
            selected_component=chosen[0],selected_pixels=int(selected.sum()),
            unknown_unselected_pixels=int(raw_mask.sum()-selected.sum()),selected_mask_sha256=array_digest(selected))
        return selected,report
    return None,report
