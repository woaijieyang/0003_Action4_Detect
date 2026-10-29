"""Prediction-only fixed-canonical8 single-frame refinement; no GT or file IO.

The caller must bind quality_ready to a whole-checkpoint calibration receipt.
diagnostic_candidate permits measuring a candidate while final remains raw.
No per-image target or ground-truth quality decision is accepted here.
"""
import copy
import math
import numpy as np

POLICY = {
    "schema": "action4.single_frame_corner_pose.v1",
    "projection_model": "explicit_pinhole_assumption_not_proven_zero_distortion",
    "pixel_space": "caller_registered_native_or_rectified",
    "se3_update": "right_object_frame_radians_metres",
    "p_valid_min": .8, "q_corner_min": .6, "minimum_ids": 6,
    "huber_delta": 1.5, "native_floor_px": 2.,
    "prior_weight": .1, "rotation_scale_deg": 5., "translation_scale_cad": .05,
    "total_lm_iterations": 10, "initial_damping": .001,
    "finite_difference_normalized_step": 1e-5,
    "minimum_depth_m": 1e-4, "minimum_corner_improvement": .1,
    "full_box_iou_min": .3, "coarse_support_coverage_min": .9,
    "maximum_rotation_change_deg": 30., "maximum_translation_change_cad": .5,
    "noncoplanar_singular_ratio_min": 1e-6,
    "multi_solution_relative_cost_gap": .05,
    "multi_solution_rotation_gap_deg": 5., "multi_solution_translation_gap_cad": .05,
    "multimodal_peak_ratio": .8, "peak_suppression_radius": 3,
}
IDS = tuple(range(8))

def _policy(config):
    if config != POLICY:
        raise ValueError("Explicit registered decoder policy required")
    return config

def _array(value,shape,name):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError("Invalid/nonfinite "+name)
    return result

def _rotation(value):
    r=_array(value,(3,3),"rotation")
    if not np.allclose(r.T@r,np.eye(3),atol=1e-4,rtol=0) or abs(np.linalg.det(r)-1)>1e-4:
        raise ValueError("Proper object-to-capture rotation required")
    return r

def _identity(value):
    fields=("image_id","instance_id","roi_id","pixel_space_id")
    if all(hasattr(value,k) for k in fields):
        key=tuple(getattr(value,k) for k in fields)
    elif isinstance(value,(tuple,list)):
        key=tuple(value)
    else:raise ValueError("Explicit image/instance/ROI identity required")
    if len(key)!=4 or any(not isinstance(v,str) or not v.strip() for v in key):
        raise ValueError("Invalid full ROI identity")
    return key

def _box(value):
    box=_array(value,(4,),"native box")
    if (box[2:]<=box[:2]).any():raise ValueError("Positive native box extent required")
    return box

def _iou(a,b):
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection))

def _project(cad,K,R,t,min_depth):
    camera=cad@R.T+t
    depth=float(camera[:,2].min())
    if not np.isfinite(camera).all() or depth<=min_depth:return None,depth
    hom=camera@K.T
    return hom[:,:2]/hom[:,2:],depth

def _exp(w,v):
    theta=np.linalg.norm(w)
    W=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0.]])
    if theta<1e-8:
        return np.eye(3)+W+.5*W@W,np.eye(3)+.5*W+(W@W)/6
    return (np.eye(3)+math.sin(theta)/theta*W+(1-math.cos(theta))/theta**2*(W@W),
            np.eye(3)+(1-math.cos(theta))/theta**2*W+(theta-math.sin(theta))/theta**3*(W@W))

def _plus(R,t,x,scales):
    w=x[:3]*scales[0];v=x[3:]*scales[1]
    dR,V=_exp(w,v)
    return R@dR,t+R@(V@v)

def _log(R):
    import cv2
    return cv2.Rodrigues(np.ascontiguousarray(R))[0].reshape(3)

def _pose_distance(R,t,rawR,rawt,diag):
    return float(np.rad2deg(np.linalg.norm(_log(R@rawR.T)))),float(np.linalg.norm(t-rawt)/diag)

def _canonical(value):
    cad=_array(value,(8,3),"canonical8 metres");lo=cad.min(0);hi=cad.max(0)
    if ((hi-lo)<=0).any():raise ValueError("Nonzero rigid cuboid dimensions required")
    x,y,z=lo;X,Y,Z=hi
    expected=np.array([[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]])
    if not np.array_equal(cad,expected):raise ValueError("Canonical corner IDs/order changed")
    return cad,float(np.linalg.norm(hi-lo))

def _optimize(initial,raw,cad,K,uv,whiten,q,diag,cfg,iterations):
    R,t=(x.copy() for x in initial);rawR,rawt=raw
    scales=(np.deg2rad(cfg["rotation_scale_deg"]),cfg["translation_scale_cad"]*diag)
    damping=cfg["initial_damping"];accepted=0
    def components(r,tr):
        pixels,_=_project(cad,K,r,tr,cfg["minimum_depth_m"])
        if pixels is None:return None
        residual=np.einsum("nij,nj->ni",whiten,pixels-uv)
        norm=np.linalg.norm(residual,axis=1);delta=cfg["huber_delta"]
        huber=np.where(norm<=delta,.5*norm**2,delta*(norm-.5*delta))
        corner=float(np.dot(q,huber)/q.sum())
        prior=np.r_[_log(r@rawR.T)/scales[0],(tr-rawt)/scales[1]]
        total=corner+.5*cfg["prior_weight"]*float(prior@prior)
        return residual,norm,prior,corner,total
    state=components(R,t)
    if state is None:return None
    for _ in range(iterations):
        residual,norm,prior,corner,total=state
        robust=np.minimum(1.,cfg["huber_delta"]/np.maximum(norm,1e-12))
        weights=np.sqrt(q/q.sum()*robust)[:,None]
        def fixed_residual(r,tr):
            data=components(r,tr)
            if data is None:return None
            return np.r_[(data[0]*weights).ravel(),np.sqrt(cfg["prior_weight"])*data[2]]
        y=fixed_residual(R,t);columns=[];eps=cfg["finite_difference_normalized_step"]
        for axis in range(6):
            direction=np.zeros(6);direction[axis]=eps
            rp,tp=_plus(R,t,direction,scales);rm,tm=_plus(R,t,-direction,scales)
            yp,ym=fixed_residual(rp,tp),fixed_residual(rm,tm)
            if yp is None or ym is None:return None
            columns.append((yp-ym)/(2*eps))
        J=np.stack(columns,axis=1)
        try:step=np.linalg.solve(J.T@J+damping*np.eye(6),-J.T@y)
        except np.linalg.LinAlgError:return None
        if not np.isfinite(step).all():return None
        r,tr=_plus(R,t,step,scales);candidate=components(r,tr)
        if candidate is not None and candidate[-1]<total:
            R,t,state=r,tr,candidate;accepted+=1;damping=max(damping/3,1e-12)
        else:damping=min(damping*10,1e12)
    return {"R":R,"t":t,"corner_objective":state[-2],"total_objective":state[-1],
            "iterations":iterations,"accepted_updates":accepted}

def decode_single_frame_pose(*,canonical_m,K_native,A224_from_native,raw_rotation,raw_translation_m,
                             observations,route,pixel_space_id,raw_identity,roi_identity,quality_ready,config,
                             diagnostic_candidate=False):
    cfg=_policy(config)
    if type(quality_ready) is not bool or type(diagnostic_candidate) is not bool:
        raise ValueError("Explicit checkpoint readiness and diagnostic flags required")
    if not isinstance(pixel_space_id,str) or not pixel_space_id:
        raise ValueError("Registered projection pixel space required")
    identity=_identity(raw_identity)
    if _identity(roi_identity)!=identity:raise ValueError("Raw pose/ROI identity mismatch")
    if _identity(route.get("identity"))!=identity:raise ValueError("Raw pose/route identity mismatch")
    cad,diag=_canonical(canonical_m);K=_array(K_native,(3,3),"K native")
    if (not np.array_equal(K[2],[0,0,1]) or K[0,0]<=0 or K[1,1]<=0 or K[0,1]!=0 or K[1,0]!=0):
        raise ValueError("Registered zero-skew pinhole K required; no implicit calibration")
    A=_array(A224_from_native,(3,3),"native-to-224 affine")
    if not np.array_equal(A[2],[0,0,1]) or np.linalg.det(A[:2,:2])<=0:
        raise ValueError("Positive invertible affine required")
    inv=np.linalg.inv(A);rawR=_rotation(raw_rotation);rawt=_array(raw_translation_m,(3,),"translation metres")
    expected={"corner_ids","uv224","p_valid","q_corner","b224","multimodal","pixel_space_id","identity"}
    if not isinstance(observations,dict) or set(observations)!=expected:
        raise ValueError("Explicit fixed-ID corner observation fields required")
    if _identity(observations["identity"])!=identity:raise ValueError("Corner observation/raw pose identity mismatch")
    corner_ids=np.asarray(observations["corner_ids"])
    if corner_ids.shape!=(8,) or corner_ids.dtype.kind not in "iu" or not np.array_equal(corner_ids,np.arange(8)) or observations["pixel_space_id"]!=pixel_space_id:
        raise ValueError("Fixed IDs and matching observation pixel space required")
    uv224=_array(observations["uv224"],(8,2),"224 observations")
    pv=_array(observations["p_valid"],(8,),"valid probability");q=_array(observations["q_corner"],(8,),"quality")
    b=_array(observations["b224"],(8,),"224 scale")
    multi=np.asarray(observations["multimodal"])
    if multi.shape!=(8,) or multi.dtype!=np.bool_ or ((pv<0)|(pv>1)|(q<0)|(q>1)|(b<.5)|(b>20)).any():
        raise ValueError("Invalid corner quality/scale/ambiguity")
    if route.get("pixel_space_id")!=pixel_space_id or route.get("kind") not in ("full","history","coarse"):
        raise ValueError("Explicit same-space prediction route required")
    raw_pixels,raw_depth=_project(cad,K,rawR,rawt,cfg["minimum_depth_m"])
    result={"schema":cfg["schema"],"identity":list(identity),"pixel_space_id":pixel_space_id,"quality_ready":quality_ready,
            "accepted":False,"geometric_candidate_accepted":False,"reason":None,
            "rotation":rawR.tolist(),"translation_m":rawt.tolist(),
            "corners_native":None if raw_pixels is None else raw_pixels.tolist(),"pose_valid":raw_pixels is not None,
            "raw_minimum_depth_m":raw_depth,"used_corner_ids":[],"candidates":[],
            "config":copy.deepcopy(cfg),"quality_calibration":"caller-bound whole-checkpoint receipt, not evaluated here"}
    def fallback(reason):result["reason"]=reason;return result
    if raw_pixels is None:return fallback("raw_pose_nonpositive_or_near_depth")
    if not quality_ready and not diagnostic_candidate:return fallback("checkpoint_quality_not_ready")
    keep=(pv>=cfg["p_valid_min"])&(q>=cfg["q_corner_min"])&~multi&((uv224>=0)&(uv224<=223)).all(1)
    ids=np.flatnonzero(keep);result["used_corner_ids"]=ids.tolist()
    if len(ids)<cfg["minimum_ids"]:return fallback("insufficient_reliable_corner_ids")
    points=cad[ids];sv=np.linalg.svd(points-points.mean(0),compute_uv=False)
    if sv[-1]<=cfg["noncoplanar_singular_ratio_min"]*sv[0]:return fallback("coplanar_or_degenerate_corner_ids")
    uv=(np.c_[uv224[ids],np.ones(len(ids))]@inv.T)[:,:2]
    linear=inv[:2,:2];S=np.stack([linear@(float(x)**2*np.eye(2))@linear.T+cfg["native_floor_px"]**2*np.eye(2) for x in b[ids]])
    whiten=np.stack([np.linalg.inv(np.linalg.cholesky(x)) for x in S])
    weights=q[ids]
    initial=[(rawR,rawt)]
    import cv2
    cv2.setNumThreads(1)
    try:
        answer=cv2.solvePnPGeneric(np.ascontiguousarray(points),np.ascontiguousarray(uv),K,None,flags=cv2.SOLVEPNP_SQPNP)
        if not answer[0]:return fallback("pnp_candidate_failed")
        if len(answer[1])!=1:return fallback("multiple_pnp_solutions")
        pr=cv2.Rodrigues(answer[1][0])[0];pt=np.asarray(answer[2][0]).reshape(3)
        if _project(cad,K,pr,pt,cfg["minimum_depth_m"])[0] is None:return fallback("pnp_nonpositive_depth")
        initial.append((pr,pt))
    except cv2.error:return fallback("pnp_numerical_failure")
    # Ten LM iterations TOTAL across the two registered starts.
    per_start=cfg["total_lm_iterations"]//len(initial)
    fits=[_optimize(x,(rawR,rawt),points,K,uv,whiten,weights,diag,cfg,per_start) for x in initial]
    fits=[x for x in fits if x is not None]
    if not fits:return fallback("lm_failed")
    fits.sort(key=lambda x:x["total_objective"])
    for fit in fits:
        result["candidates"].append({k:v for k,v in fit.items() if k not in ("R","t")})
    chosen=fits[0]
    if len(fits)>1:
        angle,shift=_pose_distance(fits[1]["R"],fits[1]["t"],chosen["R"],chosen["t"],diag)
        near=fits[1]["total_objective"]-chosen["total_objective"]<=cfg["multi_solution_relative_cost_gap"]*max(chosen["total_objective"],1e-6)
        if near and (angle>cfg["multi_solution_rotation_gap_deg"] or shift>cfg["multi_solution_translation_gap_cad"]):
            return fallback("ambiguous_refined_pose_solutions")
    raw_fit=_optimize((rawR,rawt),(rawR,rawt),points,K,uv,whiten,weights,diag,cfg,0)
    raw_cost=raw_fit["corner_objective"];cost=chosen["corner_objective"]
    result.update(raw_corner_objective=raw_cost,candidate_corner_objective=cost)
    if raw_cost<=1e-12 or cost>(1-cfg["minimum_corner_improvement"])*raw_cost:return fallback("insufficient_corner_objective_improvement")
    pixels,depth=_project(cad,K,chosen["R"],chosen["t"],cfg["minimum_depth_m"])
    if pixels is None:return fallback("candidate_nonpositive_depth")
    angle,shift=_pose_distance(chosen["R"],chosen["t"],rawR,rawt,diag)
    result.update(candidate_minimum_depth_m=depth,rotation_change_deg=angle,translation_change_cad=shift)
    if angle>cfg["maximum_rotation_change_deg"] or shift>cfg["maximum_translation_change_cad"]:
        return fallback("pose_change_exceeds_gate")
    box=np.r_[pixels.min(0),pixels.max(0)]
    if route["kind"] in ("full","history"):
        if route.get("full_box_quality_passed") is not True:return fallback("full_box_not_admitted")
        overlap=_iou(box,_box(route["full_box_native"]));result["full_box_iou"]=overlap
        if overlap<cfg["full_box_iou_min"]:return fallback("full_box_support_gate")
    else:
        support=np.asarray(route.get("support_mask_native"))
        if support.ndim!=2 or support.dtype!=np.bool_ or not support.any():return fallback("coarse_support_missing")
        bounds=_box(route["candidate_bounds_native"]);center=(box[:2]+box[2:])/2
        if not ((center>=bounds[:2])&(center<=bounds[2:])).all():return fallback("coarse_center_support_gate")
        y,x=np.nonzero(support)
        coverage=float(((x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3])).mean())
        result["coarse_support_coverage"]=coverage
        if coverage<cfg["coarse_support_coverage_min"]:return fallback("coarse_visible_support_gate")
    result.update(geometric_candidate_accepted=True,candidate_rotation=chosen["R"].tolist(),
                  candidate_translation_m=chosen["t"].tolist(),candidate_corners_native=pixels.tolist())
    if not quality_ready:return fallback("checkpoint_quality_not_ready_diagnostic_candidate_only")
    result.update(accepted=True,reason="accepted_corner_refinement",rotation=chosen["R"].tolist(),
                  translation_m=chosen["t"].tolist(),corners_native=pixels.tolist(),pose_valid=True)
    return result

def observations_from_head(output,index,*,identity,pixel_space_id,config):
    """Detach one typed head result; flag a second peak outside the local7 window."""
    cfg=_policy(config)
    from model.action4_corner_supervision import validate_output
    validate_output(output)
    if type(index) is not int or not 0<=index<len(output.identities) or output.identities[index]!=identity:
        raise ValueError("Head output/ROI identity mismatch")
    if not isinstance(pixel_space_id,str) or not pixel_space_id:
        raise ValueError("Explicit upstream projection pixel space required")
    heat=output.localization.logits[index].detach().cpu().double().numpy()
    multimodal=[]
    radius=cfg["peak_suppression_radius"]
    for h in heat:
        y,x=np.unravel_index(np.argmax(h),h.shape)
        other=h.copy()
        other[max(0,y-radius):min(224,y+radius+1),max(0,x-radius):min(224,x+radius+1)]=-np.inf
        ratio=float(np.exp(other.max()-h[y,x]))
        multimodal.append(ratio>=cfg["multimodal_peak_ratio"])
    array=lambda v:v[index].detach().cpu().double().numpy()
    return {"identity":_identity(identity),"corner_ids":list(IDS),"uv224":array(output.uv224),
            "p_valid":array(output.p_valid),"q_corner":array(output.q_corner),
            "b224":array(output.scale_b224),"multimodal":np.asarray(multimodal,dtype=bool),
            "pixel_space_id":pixel_space_id}
