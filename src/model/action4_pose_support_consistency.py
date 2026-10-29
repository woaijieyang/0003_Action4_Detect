"""Necessary visible-support consistency; never a full-pose or corner-ID score."""
from dataclasses import dataclass
import numpy as np
from model.action4_video_motion_evidence import _pose

@dataclass(frozen=True)
class SupportPolicy:
    min_visible_coverage:float=.9
    min_visible_over_projection:float=.4
    tolerance_diagonal_fraction:float=.01
    min_mask_pixels:int=32
    def __post_init__(self):
        if (not 0<self.min_visible_coverage<=1 or not 0<self.min_visible_over_projection<=1
            or not 0<=self.tolerance_diagonal_fraction<=.05 or self.min_mask_pixels<1):
            raise ValueError("Explicit bounded support policy required")

def pose_support_consistency(mask,K,canonical,rotation,translation,*,policy,other_instance_mask=None):
    import cv2
    if not isinstance(policy,SupportPolicy):raise TypeError("Explicit support policy")
    m=np.asarray(mask)
    if m.dtype!=np.bool_ or m.ndim!=2:raise ValueError("Native bool mask required")
    m=m.copy()
    if other_instance_mask is not None:
        other=np.asarray(other_instance_mask)
        if other.dtype!=np.bool_ or other.shape!=m.shape:raise ValueError("Other-instance mask differs")
        m &= ~other
    r,t=_pose(rotation,translation)
    k=np.asarray(K,float);cad=np.asarray(canonical,float)
    if k.shape!=(3,3) or cad.shape!=(8,3) or not np.isfinite(k).all() or not np.isfinite(cad).all():
        raise ValueError("Native K/canonical8 required")
    count=int(m.sum())
    result=dict(available=False,reference_compatible=False,visible_coverage=None,IoU_diagnostic_only=None,
        visible_mask_pixels=count,canonical_ID_accuracy_measured=False,semantic_mask_accuracy_verified=False,
        geometry_source="fixed_canonical8_projected_convex_body_not_observed_faces",
        calibration_status="development_thresholds_not_deployment_calibrated",reasons=[])
    if count<policy.min_mask_pixels:
        return dict(result,reasons=["insufficient_owned_mask"])
    camera=cad@r.T+t
    if np.any(camera[:,2]<=1e-6):raise ValueError("Nonpositive projected geometry")
    h=camera@k.T;uv=h[:,:2]/h[:,2,None]
    if not np.isfinite(uv).all() or np.max(abs(uv))>1e6:raise ValueError("Invalid projected pixel magnitude")
    poly=cv2.convexHull(uv.astype(np.float32))
    raster=np.zeros(m.shape,np.uint8);cv2.fillConvexPoly(raster,np.rint(poly).astype(np.int32),1)
    area=int(raster.sum());intersection=int((raster.astype(bool)&m).sum())
    diagonal=float(np.linalg.norm(uv.max(0)-uv.min(0)))
    tolerance=max(1,int(np.ceil(policy.tolerance_diagonal_fraction*diagonal)))
    expanded=cv2.dilate(raster,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance+1,2*tolerance+1))).astype(bool)
    coverage=float((expanded&m).sum()/count);fraction=count/max(area,1)
    height,width=m.shape
    truncated=bool((uv[:,0]<-.5).any() or (uv[:,0]>width-.5).any() or (uv[:,1]<-.5).any() or (uv[:,1]>height-.5).any())
    reasons=[]
    if coverage<policy.min_visible_coverage:reasons.append("visible_support_outside_projected_body")
    if fraction<policy.min_visible_over_projection:reasons.append("insufficient_visible_shape_evidence")
    if truncated:reasons.append("projected_body_truncated_shape_unmeasured")
    return dict(result,available=True,reference_compatible=not reasons,visible_coverage=coverage,
        untoleranced_visible_coverage=intersection/count,projected_pixels=area,
        IoU_diagnostic_only=intersection/max(area+count-intersection,1),
        visible_over_projected=fraction,tolerance_native_px=tolerance,
        projected_corners_native=uv.tolist(),projection_truncated=truncated,reasons=reasons)
