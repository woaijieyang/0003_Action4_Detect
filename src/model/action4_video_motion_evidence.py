"""Current image correspondences can veto an unsupported A0 rotation jump.

The 3D binding is estimated from the previous pose, not CAD ground-truth material
identity. A low residual only supports temporal consistency, not true 6D accuracy.
No hidden corner is treated as an observed optical-flow point.
"""
import numpy as np

def _pose(rotation, translation):
    r=np.asarray(rotation,dtype=float);t=np.asarray(translation,dtype=float)
    if (r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(r).all()
        or not np.isfinite(t).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5,rtol=0)
        or abs(np.linalg.det(r)-1)>1e-5 or t[2]<=0):
        raise ValueError("Finite proper object-to-camera pose required")
    return r,t

def bind_visible_box_points(uv,K,canonical,rotation,translation):
    """Ray-box nearest entry intersection; unsupported rays are explicitly invalid."""
    uv=np.asarray(uv,float);k=np.asarray(K,float);cad=np.asarray(canonical,float)
    r,t=_pose(rotation,translation)
    if (uv.ndim!=2 or uv.shape[1]!=2 or k.shape!=(3,3) or cad.shape!=(8,3)
        or not np.isfinite(uv).all() or not np.isfinite(k).all()
        or not np.isfinite(cad).all() or abs(np.linalg.det(k))<1e-12):
        raise ValueError("Finite native pixel coordinates, K and canonical8 required")
    lo,hi=cad.min(0),cad.max(0)
    if np.any(hi-lo<=0) or not np.all(np.isclose(cad,lo,atol=1e-8)|np.isclose(cad,hi,atol=1e-8)):
        raise ValueError("Axis aligned object-frame canonical cuboid required")
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T
    direction=rays@r;origin=-r.T@t
    lower=np.full((len(uv),3),-np.inf);upper=np.full((len(uv),3),np.inf)
    possible=np.ones(len(uv),bool)
    for axis in range(3):
        moving=np.abs(direction[:,axis])>1e-12
        possible &= moving | ((origin[axis]>=lo[axis]) & (origin[axis]<=hi[axis]))
        a=(lo[axis]-origin[axis])/direction[moving,axis]
        b=(hi[axis]-origin[axis])/direction[moving,axis]
        lower[moving,axis]=np.minimum(a,b);upper[moving,axis]=np.maximum(a,b)
    near=lower.max(1);far=upper.min(1)
    valid=possible & np.isfinite(near) & (near>1e-6) & (far>=near)
    points=np.zeros((len(uv),3))
    points[valid]=origin+near[valid,None]*direction[valid]
    return points,valid

def score_pose_motion(previous_uv,current_uv,K,canonical,previous_rotation,previous_translation,
                      candidate_rotation,current_translation,object_diagonal,*,min_points=12,
                      hold_tolerance_fraction=.01,candidate_tolerance_fraction=.03):
    """Compare identical observed pixels under candidate vs previous rotation.

    Translation is the same current estimate in both branches. The caller must
    supply independently admitted same-instance, forward/backward-checked image
    correspondences; this function does not certify correspondence identity.
    """
    a=np.asarray(previous_uv,float);b=np.asarray(current_uv,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(b).all():
        raise ValueError("Finite paired image observations required")
    if not np.isfinite(object_diagonal) or object_diagonal<=0 or min_points<4:
        raise ValueError("Explicit valid observation scale and support count required")
    rr,tt=_pose(candidate_rotation,current_translation)
    rp,tp=_pose(previous_rotation,previous_translation)
    xyz,valid=bind_visible_box_points(a,K,canonical,rp,tp)
    result=dict(available=False,reject_candidate=False,bound_points=int(valid.sum()),
                XYZ_source="previous_estimated_pose_ray_cuboid_intersection",
                independent_pose_accuracy=False,hidden_corners_observed=False)
    if valid.sum()<min_points:
        return dict(result,reason="insufficient_ray_surface_support")
    points=xyz[valid];target=b[valid];k=np.asarray(K,float)
    errors={}
    for name,r in (("candidate",rr),("hold",rp)):
        camera=points@r.T+tt
        if not np.isfinite(camera).all() or np.any(camera[:,2]<=1e-6):
            return dict(result,reason="invalid_projected_depth")
        h=camera@k.T
        e=np.linalg.norm(h[:,:2]/h[:,2,None]-target,axis=1)
        errors[name+"_median_px"]=float(np.median(e))
        errors[name+"_p95_px"]=float(np.percentile(e,95))
    hold_gate=max(3.,hold_tolerance_fraction*object_diagonal)
    candidate_gate=max(6.,candidate_tolerance_fraction*object_diagonal)
    reject=(errors["hold_median_px"]<=hold_gate and
            errors["hold_p95_px"]<=2*hold_gate and
            errors["candidate_median_px"]>candidate_gate and
            errors["candidate_median_px"]>2*errors["hold_median_px"]+2)
    return dict(result,available=True,reject_candidate=bool(reject),
                reason="image_motion_contradicts_rotation_jump" if reject else "no_safe_veto",
                hold_tolerance_px=hold_gate,candidate_tolerance_px=candidate_gate,**errors)
