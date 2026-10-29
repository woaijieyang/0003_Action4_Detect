"""Pilot RGB-edge single-SE3 candidate; never deployed or semantic pose truth."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.ndimage import sobel,gaussian_filter,distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

@dataclass(frozen=True)
class EdgePoseConfig:
    gradient_min: float
    gaussian_sigma: float
    body_margin_px: float
    lcd_margin_px: float
    correspondence_radius_px: float
    direction_weight_px: float
    max_rotation_rad: float
    max_translation_diag: float
    min_observations: int
    max_nfev: int
    max_mean_normal_residual_px: float
    jacobian_relative_tolerance: float
    max_condition_number: float
    ambiguity_residual_px: float
    ambiguity_rotation_rad: float
    ambiguity_translation_diag: float
    orientation_max_degrees: float | None = None
    correspondence_neighbors: int = 1
    face_border_max_distance_px: float | None = None
    correspondence_method: str = "knn"
    profile_step_px: float | None = None
    profile_tangent_half_width_px: float | None = None
    profile_min_side_support_px: float | None = None
    profile_min_prominence: float | None = None
    profile_max_peak_width_px: float | None = None
    profile_ambiguity_ratio: float | None = None
    profile_sampling_domain: str = "candidate_support"
    mixture_ambiguity_nll: float | None = None
    translation_chart: str = "legacy_left"
    def __post_init__(self):
        if self.translation_chart not in ("legacy_left","object_origin_delta"):
            raise ValueError("Explicit legacy_left or object_origin_delta translation chart required")
        if self.correspondence_method not in ("knn","normal_profile","normal_profile_mixture"):
            raise ValueError("Explicit knn or normal_profile correspondence method required")
        if self.profile_sampling_domain not in ("candidate_support","valid_rgb"):
            raise ValueError("Explicit profile sampling domain required")
        if self.profile_sampling_domain=="valid_rgb" and self.correspondence_method not in ("normal_profile","normal_profile_mixture"):
            raise ValueError("Separate RGB sampling is only defined for normal profiles")
        for k,v in asdict(self).items():
            if k in ("correspondence_method","profile_sampling_domain","translation_chart"):continue
            if (k.startswith("profile_") or k=="mixture_ambiguity_nll") and v is None:continue
            if k in ("orientation_max_degrees","face_border_max_distance_px") and v is None:continue
            if isinstance(v,bool) or not np.isfinite(v) or v<0:raise ValueError("Finite nonnegative explicit config required: "+k)
        for k in ("min_observations","max_nfev"):
            if type(getattr(self,k)) is not int or getattr(self,k)<1:raise ValueError("Positive integer count required")
        for k in ("gradient_min","correspondence_radius_px","max_rotation_rad","max_translation_diag",
                  "max_mean_normal_residual_px","jacobian_relative_tolerance","max_condition_number"):
            if getattr(self,k)<=0:raise ValueError("Positive config required: "+k)
        if type(self.correspondence_neighbors) is not int or not 1<=self.correspondence_neighbors<=128:
            raise ValueError("Explicit bounded1..128 correspondence neighbors required")
        if self.orientation_max_degrees is not None and not 0<=self.orientation_max_degrees<90:
            raise ValueError("Orientation gate must be below90 degrees")
        if self.face_border_max_distance_px is not None and self.face_border_max_distance_px<=self.lcd_margin_px:
            raise ValueError("Face border outer distance must exceed LCD exclusion margin")
        if self.correspondence_method in ("normal_profile","normal_profile_mixture"):
            values=[self.profile_step_px,self.profile_tangent_half_width_px,self.profile_min_side_support_px,
                    self.profile_min_prominence,self.profile_max_peak_width_px,self.profile_ambiguity_ratio]
            if any(v is None for v in values):raise ValueError("All profile thresholds must be explicit")
            if (not .25<=self.profile_step_px<=2 or not 0<=self.profile_tangent_half_width_px<=4
                or self.profile_min_side_support_px<=0 or self.profile_min_prominence<=0
                or self.profile_max_peak_width_px<=0 or not 0<self.profile_ambiguity_ratio<=1
                or 2*self.correspondence_radius_px/self.profile_step_px>512):
                raise ValueError("Bounded profile sampling and positive uncertainty gates required")
        if self.correspondence_method=="normal_profile_mixture" and self.mixture_ambiguity_nll is None:
            raise ValueError("Explicit mixture ambiguity NLL tolerance required")
        if self.max_rotation_rad>=np.pi or self.jacobian_relative_tolerance>=1:
            raise ValueError("Local rotation and relative Jacobian tolerance required")

def _pose(R,t):
    R=np.asarray(R,float);t=np.asarray(t,float)
    if (R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(R),1.,atol=1e-6)):
        raise ValueError("Finite proper object-to-camera SE3 required")
    return R.copy(),t.copy()

def pose_from_increment(x,baseR,baset,cad_diagonal_m,translation_chart="legacy_left"):
    """Rotation is left-multiplied; translation chart explicitly chooses origin."""
    delta=Rotation.from_rotvec(x[:3]).as_matrix()
    if translation_chart=="legacy_left":
        translation=delta@baset+x[3:]*cad_diagonal_m
    elif translation_chart=="object_origin_delta":
        translation=baset+x[3:]*cad_diagonal_m
    else:raise ValueError("Unknown translation chart")
    return delta@baseR,translation

def pose_increment_record(x,baseR,baset,cad_diagonal_m,translation_chart):
    delta=Rotation.from_rotvec(x[:3]).as_matrix()
    _,translation=pose_from_increment(x,baseR,baset,cad_diagonal_m,translation_chart)
    net=translation-baset
    equivalent_left=net+(np.eye(3)-delta)@baset
    return {"translation_chart":translation_chart,
        "net_object_origin_translation_m":net.tolist(),
        "net_object_origin_translation_CADdiag_units":(net/cad_diagonal_m).tolist(),
        "equivalent_left_translation_m":equivalent_left.tolist(),
        "equivalent_left_translation_CADdiag_units":(equivalent_left/cad_diagonal_m).tolist(),
        "translation_bound_semantics":"net_object_origin_delta" if translation_chart=="object_origin_delta" else "legacy_camera_origin_left_increment",
        "rotation_units":"radians","translation_scale_m":float(cad_diagonal_m)}

def observed_jacobian_diagnostics(fun,x,relative_tolerance,max_condition_number,*,translation_chart="legacy_left"):
    """Central differences of RAW image residual only, without prior/damping."""
    step=1e-5
    jac=np.column_stack([(fun(x+np.eye(6)[i]*step)-fun(x-np.eye(6)[i]*step))/(2*step) for i in range(6)])
    singular=np.linalg.svd(jac,compute_uv=False)
    rank=int(np.count_nonzero(singular>singular[0]*relative_tolerance)) if len(singular) and singular[0]>0 else 0
    condition=float(singular[0]/singular[-1]) if len(singular)==6 and singular[-1]>0 else None
    return {"rank":rank,"singular_values":singular.tolist(),"condition_number":condition,
            "degenerate":rank<6 or condition is None or condition>max_condition_number,
            "source":"raw_RGB_observation_residual_only_no_prior_no_damping",
            "translation_chart":translation_chart,"condition_comparable_only_with_same_chart":True,
            "parameter_scaling":("left-multiplied rotation radians; net object-origin translation in CAD diagonal units"
                if translation_chart=="object_origin_delta" else "camera-origin left rotation radians; left translation in CAD diagonal units")}

def select_edge_correspondences(tree,normals,uv,tangent2,*,radius,neighbors,orientation_max_degrees):
    """Nearest orientation-compatible RGB edge among bounded k; no invented edge."""
    k=min(neighbors,len(normals)) if orientation_max_degrees is not None else 1
    distance,index=tree.query(uv,k=k)
    distance=np.asarray(distance).reshape(len(uv),k);index=np.asarray(index).reshape(len(uv),k)
    tangent_length=np.linalg.norm(tangent2,axis=1)
    unit=tangent2/np.maximum(tangent_length[:,None],1e-12)
    angle=np.degrees(np.arcsin(np.clip(np.abs(np.sum(normals[index]*unit[:,None,:],axis=2)),0,1)))
    valid=distance<=radius
    if orientation_max_degrees is not None:
        valid&=(angle<=orientation_max_degrees)&(tangent_length[:,None]>1e-12)
    any_valid=valid.any(axis=1);choice=np.argmax(valid,axis=1);row=np.arange(len(uv))
    return distance[row,choice],index[row,choice],any_valid,{
        "nearest_angle_degrees":angle[:,0],"selected_angle_degrees":angle[row,choice],
        "selected_neighbor_rank":choice+1,"nearest_in_radius":distance[:,0]<=radius}


def mixture_profile_statistics(displacements, peaks_by_line, direction_errors=None):
    """Gaussian displacement mixture, retaining ALL legal peaks and width units.

    Curvature is the observed marginal displacement curvature, not the
    Gauss-Newton derivative of sqrt(NLL), nor a posterior hard-selected peak.
    """
    from scipy.special import logsumexp
    statistics=[]
    for i,(d,peaks) in enumerate(zip(displacements,peaks_by_line)):
        if not peaks:raise ValueError("Empty legal mixture cannot become an observation")
        mu=np.array([p["displacement_px"] for p in peaks],float)
        sigma=np.maximum(1.,np.array([p["width_px"] for p in peaks],float)/2.354820045)
        height=np.array([p["height"] for p in peaks],float)
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or not np.isfinite(height).all() or (height<=0).any():
            raise ValueError("Finite positive real peak evidence required")
        weight=height/height.sum();error=d-mu
        direction=np.zeros(len(peaks)) if direction_errors is None else np.asarray(direction_errors[i],float)
        log_component=np.log(weight)-np.log(sigma)-.5*np.log(2*np.pi)-.5*(error**2+direction**2)/sigma**2
        logp=logsumexp(log_component);posterior=np.exp(log_component-logp)
        slope=error/sigma**2
        curvature=float(np.sum(posterior/sigma**2)-(np.sum(posterior*slope**2)-np.sum(posterior*slope)**2))
        prior_mean=float(weight@mu)
        statistics.append({"query_displacement_px":float(d),"negative_log_likelihood":float(-logp),"posterior":posterior.tolist(),
            "prior_weights":weight.tolist(),"sigma_px":sigma.tolist(),
            "mixture_mean_px":prior_mean,"mixture_total_variance_px2":float(weight@(sigma**2+(mu-prior_mean)**2)),
            "posterior_entropy_nats":float(-np.sum(posterior*np.log(np.maximum(posterior,1e-300)))),
            "observed_displacement_curvature_per_px2":curvature,
            "expected_absolute_pixel_residual":float(posterior@np.abs(error)),
            "expected_squared_pixel_residual":float(posterior@(error**2)),
            "expected_squared_direction_residual_px2":float(posterior@(direction**2))})
    return statistics

def _fit_profile_mixture(points,tangents,uv_initial,tangent_initial,profiles,ids,groups,name,
                         baseR,baset,K,cad_diagonal_m,config,project,in_support,tree,matching_diagnostics):
    from scipy.optimize import minimize
    p=points[ids];tangent=tangents[ids];origin=uv_initial[ids]
    t2=tangent_initial[ids];t2=t2/np.maximum(np.linalg.norm(t2,axis=1)[:,None],1e-12)
    line_normal=np.c_[-t2[:,1],t2[:,0]]
    peaks=[]
    for original,center,normal in zip(ids,origin,line_normal):
        local=[]
        for peak in profiles[original]["peaks"]:
            if peak["reasons"]:continue
            local.append(dict(peak,displacement_px=float((np.asarray(peak["pixel_xy"])-center)@normal)))
        peaks.append(local)
    def transform(x):
        return pose_from_increment(x,baseR,baset,cad_diagonal_m,config.translation_chart)
    def measurements(x):
        R,t=transform(x);uv,depth=project(R,t,p)
        q=p@R.T+t;hom=q@K.T;dh=(tangent@R.T)@K.T
        direction=dh[:,:2]*hom[:,2,None]-hom[:,:2]*dh[:,2,None]
        unit=direction/np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-12)
        displacement=np.sum((uv-origin)*line_normal,axis=1)
        errors=[config.direction_weight_px*(np.array([pk["normal"] for pk in row])@u) for row,u in zip(peaks,unit)]
        return displacement,errors,unit
    def statistics(x):
        displacement,direction,_=measurements(x)
        return mixture_profile_statistics(displacement,peaks,direction)
    def objective(x):return sum(row["negative_log_likelihood"] for row in statistics(x))
    bound=np.r_[np.full(3,config.max_rotation_rad),np.full(3,config.max_translation_diag)]
    start=np.zeros(6)
    fit=minimize(objective,start,method="L-BFGS-B",bounds=list(zip(-bound,bound)),
                 options={"maxiter":config.max_nfev,"maxfun":config.max_nfev})
    rawstats=statistics(start);stats=statistics(fit.x)
    R,t=transform(fit.x);pred,depth=project(R,t,points)
    curvature=np.array([s["observed_displacement_curvature_per_px2"] for s in stats])
    # Only actual positive marginal information contributes to local information
    # rank. Report unweighted projection rank separately; no temporal/pose prior.
    diag=observed_jacobian_diagnostics(lambda x:np.sqrt(np.maximum(curvature,0))*measurements(x)[0],
        fit.x,config.jacobian_relative_tolerance,config.max_condition_number,translation_chart=config.translation_chart)
    geometry=observed_jacobian_diagnostics(lambda x:measurements(x)[0],fit.x,
        config.jacobian_relative_tolerance,config.max_condition_number,translation_chart=config.translation_chart)
    # Retain negative marginal contributions in the summed information matrix;
    # clipping them away could manufacture good conditioning from ambiguous lines.
    step=1e-5
    J=np.column_stack([(measurements(fit.x+np.eye(6)[i]*step)[0]-
                       measurements(fit.x-np.eye(6)[i]*step)[0])/(2*step) for i in range(6)])
    eigenvalues=np.linalg.eigvalsh(J.T@(curvature[:,None]*J))
    scale=max(float(eigenvalues[-1]),0.)
    rank=int(np.count_nonzero(eigenvalues>scale*config.jacobian_relative_tolerance**2)) if scale else 0
    condition=float(np.sqrt(eigenvalues[-1]/eigenvalues[0])) if eigenvalues[0]>0 else None
    diag.update(rank=rank,condition_number=condition,
        singular_values=np.sqrt(np.maximum(eigenvalues[::-1],0)).tolist(),
        signed_information_eigenvalues=eigenvalues.tolist(),
        degenerate=rank<6 or condition is None or condition>config.max_condition_number)
    diag["source"]="signed_RGB_marginal_displacement_curvature_projected_by_J_no_prior_no_damping"
    diag["scope"]="displacement observation information only; not full nonlinear objective Hessian"
    diag["nonpositive_curvature_lines"]=int(np.count_nonzero(curvature<=0))
    diag["geometry_only"]=geometry
    supported=in_support(pred[ids]);distance,_=tree.query(pred[ids]);reasons=[]
    if not fit.success:reasons.append("optimizer_not_converged")
    if np.linalg.norm(fit.x[:3])>config.max_rotation_rad or np.linalg.norm(fit.x[3:])>config.max_translation_diag:
        reasons.append("candidate_outside_local_SE3_trust_radius")
    if not (depth>1e-6).all():reasons.append("nonpositive_curve_depth")
    if not supported.all() or (distance>config.correspondence_radius_px).any():reasons.append("candidate_left_observation_support")
    mean=float(np.mean([s["expected_absolute_pixel_residual"] for s in stats]))
    if mean>config.max_mean_normal_residual_px:reasons.append("edge_residual_above_pilot_limit")
    if diag["degenerate"]:reasons.append("observation_jacobian_degenerate")
    _,_,unit=measurements(fit.x)
    if config.orientation_max_degrees is not None:
        # Posterior expected squared sine, never a hard-picked peak normal.
        angle=[np.degrees(np.arcsin(np.sqrt(np.clip(np.asarray(st["posterior"])@
            ((np.asarray([pk["normal"] for pk in row])@u)**2),0,1)))) for row,u,st in zip(peaks,unit,stats)]
        if np.any(np.asarray(angle)>config.orientation_max_degrees):reasons.append("candidate_direction_left_orientation_gate")
    matching_diagnostics.update(final_unsupported_count=int(np.count_nonzero(~supported)),
                               final_outside_radius_count=int(np.count_nonzero(distance>config.correspondence_radius_px)))
    return {"initializer":name,"usable":True,"R":R.tolist(),"t_m":t.tolist(),
        "matching_diagnostics":matching_diagnostics,
        "pose_increment":pose_increment_record(fit.x,baseR,baset,cad_diagonal_m,config.translation_chart),
        "local_perturbation_rotvec_rad":fit.x[:3].tolist(),
        "local_translation_CADdiag_units":fit.x[3:].tolist(),"observation_count":len(ids),
        "curve_observation_counts":{str(i):int(np.count_nonzero(groups[ids]==i)) for i in np.unique(groups)},
        "observed_sample_indices":ids.tolist(),"image_edge_matches_xy":None,
        "mixture_peak_sources":peaks,"mixture_line_diagnostics":stats,"raw_mixture_line_diagnostics":rawstats,
        "mean_mixture_nll":float(np.mean([s["negative_log_likelihood"] for s in stats])),
        "raw_mean_mixture_nll":float(np.mean([s["negative_log_likelihood"] for s in rawstats])),
        "objective_semantics":"sum normalized Gaussian displacement mixture NLL plus per-peak direction energy; no robust pixel objective",
        "pixel_residual_semantics":"posterior expectation over every legal source peak, not selected-peak residual",
        "raw_mean_normal_residual_px":float(np.mean([s["expected_absolute_pixel_residual"] for s in rawstats])),
        "mean_normal_residual_px":mean,
        "posterior_pixel_rms":float(np.sqrt(np.mean([s["expected_squared_pixel_residual"] for s in stats]))),
        "posterior_direction_rms_px":float(np.sqrt(np.mean([s["expected_squared_direction_residual_px2"] for s in stats]))),
        "raw_observation_rms":float(np.sqrt(np.mean([s["expected_squared_pixel_residual"]+
            s["expected_squared_direction_residual_px2"] for s in stats])/2)),
        "positive_depth":bool((depth>1e-6).all()),"jacobian":diag,"reasons":reasons,
        "geometric_candidate_eligible":not reasons}

def refine_face_edge_pose(rgb,body_support,lcd_exclusion,K,curves_object_m,initial_R,initial_t,*,
                          cad_diagonal_m,config,previous_trusted_pose=None,initial_pose_source="A0"):
    """Curves are actual caller-verified CAD polylines, NOT bbox-corner surrogates.

    Retains fixed image point-to-normal correspondences per initializer, avoiding
    dropping high residual observations inside the solver. Orientation residual
    uses projected curve tangent dot measured image gradient. A previous pose is
    only another initializer; no temporal prior enters residual or Jacobian.
    """
    if initial_pose_source not in ("A0","current_region_initializer"):
        raise ValueError("Explicit pose initializer source required")
    if not isinstance(config,EdgePoseConfig):raise TypeError("Explicit EdgePoseConfig required")
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError("Native HWC uint8 RGB required")
    h,w=rgb.shape[:2]
    if min(h,w)<3:raise ValueError("Nontrivial native image required")
    for mask in (body_support,lcd_exclusion):
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=(h,w):
            raise ValueError("Native boolean masks must match RGB")
    K=np.asarray(K,float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError("Finite native camera K required")
    mixture=config.correspondence_method=="normal_profile_mixture"
    R0,t0=_pose(initial_R,initial_t)
    if isinstance(cad_diagonal_m,bool) or not np.isfinite(cad_diagonal_m) or cad_diagonal_m<=0:
        raise ValueError("Actual positive CAD diagonal scale required")
    points=[];tangents=[];groups=[]
    if not isinstance(curves_object_m,(list,tuple)) or not curves_object_m:raise ValueError("Verified curve list required")
    for group,c in enumerate(curves_object_m):
        c=np.asarray(c,float)
        if c.ndim!=2 or c.shape[1]!=3 or len(c)<3 or not np.isfinite(c).all():
            raise ValueError("Each CAD curve needs finite Nx3 object metre samples")
        d=np.gradient(c,axis=0);length=np.linalg.norm(d,axis=1)
        if (length<=1e-12).any():raise ValueError("Repeated/undefined curve tangent")
        points.extend(c);tangents.extend(d/length[:,None]);groups.extend([group]*len(c))
    points=np.array(points);tangents=np.array(tangents);groups=np.array(groups)
    gray=np.asarray(rgb,float).mean(2)/255.
    gray=gaussian_filter(gray,config.gaussian_sigma) if config.gaussian_sigma else gray
    gx=sobel(gray,axis=1)/8.;gy=sobel(gray,axis=0)/8.;magnitude=np.hypot(gx,gy)
    # Padding makes an all-true body still have explicit image-border exclusion.
    interior=distance_transform_edt(np.pad(body_support,1))[1:-1,1:-1]
    lcd_distance=distance_transform_edt(~lcd_exclusion) if lcd_exclusion.any() else np.full((h,w),np.inf)
    allowed=body_support&(interior>config.body_margin_px)&(lcd_distance>config.lcd_margin_px)&~lcd_exclusion
    allowed_without_outer_band=allowed.copy()
    if config.face_border_max_distance_px is not None:
        allowed&=lcd_distance<=config.face_border_max_distance_px
    ys,xs=np.nonzero(allowed&(magnitude>=config.gradient_min))
    result={"status":"candidate_not_deployed","deployed":False,"accepted":False,
            "raw_pose":{"R":R0.tolist(),"t_m":t0.tolist()},"input_pose_source":initial_pose_source,"candidate_pose":None,
            "thresholds":asdict(config),"thresholds_calibrated":False,
            "translation_chart":config.translation_chart,
            "condition_comparable_only_with_same_chart":True,
            "rgb_edge_pixels":len(xs),"cad_samples":len(points),"solutions":[],
            "correspondence_policy":("normal_profile_mixture_RGB_gradient_not_region_likelihood" if mixture else
                                     "normal_profile_RGB_gradient_not_region_likelihood" if config.correspondence_method in ("normal_profile","normal_profile_mixture")
                                     else "nearest1" if config.orientation_max_degrees is None else "orientation_gated_k_nearest"),
            "mixture_enabled":mixture,
            "face_region_semantics":"LCD-relative candidate center/peak band only, not physical face truth",
            "profile_domain_semantics":config.profile_sampling_domain,
            "LCD_used_as_persistent_texture":False,
            "support_diagnostics":{"body_pixels":int(body_support.sum()),"LCD_pixels":int(lcd_exclusion.sum()),
                "allowed_pixels_before_outer_band":int(allowed_without_outer_band.sum()),
                "allowed_pixels":int(allowed.sum()),
                "RGB_edges_before_outer_band":int(np.count_nonzero(allowed_without_outer_band&(magnitude>=config.gradient_min)))},
            "single_face_ambiguity_resolved":False,"accuracy_verified":False,
            "reasons":[],"prior_residual_used":False,"geometric_candidate_eligible":False}
    if not len(xs):
        if config.correspondence_method in ("normal_profile","normal_profile_mixture"):
            from model.action4_edge_profiles import normal_profile_matches
            q=points@R0.T+t0;hom=q@K.T;dh=(tangents@R0.T)@K.T
            uv=hom[:,:2]/np.maximum(hom[:,2,None],1e-9)
            tangent2=dh[:,:2]*hom[:,2,None]-hom[:,:2]*dh[:,2,None]
            _,_,_,profiles=normal_profile_matches(gx,gy,allowed,uv,tangent2,
                radius=config.correspondence_radius_px,step_px=config.profile_step_px,
                tangent_half_width_px=config.profile_tangent_half_width_px,
                min_side_support_px=config.profile_min_side_support_px,min_gradient=config.gradient_min,
                min_prominence=config.profile_min_prominence,max_peak_width_px=config.profile_max_peak_width_px,
                ambiguity_ratio=config.profile_ambiguity_ratio,orientation_max_degrees=config.orientation_max_degrees,
                sampling_valid=np.ones((h,w),bool) if config.profile_sampling_domain=="valid_rgb" else None)
            result["solutions"].append({"initializer":"A0","usable":False,"observation_count":0,
                "reason":"no_supported_RGB_edges","matching_diagnostics":{"normal_profiles":profiles}})
        result["reasons"]=["no_supported_RGB_edges"];return result
    edge_xy=np.c_[xs,ys];normals=np.c_[gx[ys,xs],gy[ys,xs]]/magnitude[ys,xs,None]
    tree=cKDTree(edge_xy)
    def project(R,t,p):
        q=p@R.T+t;uv=q@K.T
        depth=q[:,2]
        return uv[:,:2]/np.maximum(depth[:,None],1e-9),depth
    def in_support(uv):
        xy=np.rint(uv).astype(np.int64);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        good=np.zeros(len(uv),bool);good[inside]=allowed[xy[inside,1],xy[inside,0]]
        return good
    initializers=[(initial_pose_source,R0,t0)]
    if previous_trusted_pose is not None:
        R,t=_pose(*previous_trusted_pose);initializers.append(("previous_trusted_initializer_only",R,t))
    for name,baseR,baset in initializers:
        uv,z=project(baseR,baset,points)
        q=points@baseR.T+baset;hom=q@K.T;dh=(tangents@baseR.T)@K.T
        tangent2=(dh[:,:2]*hom[:,2,None]-hom[:,:2]*dh[:,2,None])/np.maximum(hom[:,2,None]**2,1e-18)
        distance,index,matched,match_info=select_edge_correspondences(tree,normals,uv,tangent2,
            radius=config.correspondence_radius_px,neighbors=config.correspondence_neighbors,
            orientation_max_degrees=config.orientation_max_degrees)
        profile_diagnostics=None
        selected_edges=edge_xy[index];selected_normals=normals[index]
        if config.correspondence_method in ("normal_profile","normal_profile_mixture"):
            from model.action4_edge_profiles import normal_profile_matches
            selected_edges,selected_normals,matched,profile_diagnostics=normal_profile_matches(
                gx,gy,allowed,uv,tangent2,radius=config.correspondence_radius_px,
                step_px=config.profile_step_px,tangent_half_width_px=config.profile_tangent_half_width_px,
                min_side_support_px=config.profile_min_side_support_px,min_gradient=config.gradient_min,
                min_prominence=config.profile_min_prominence,max_peak_width_px=config.profile_max_peak_width_px,
                ambiguity_ratio=config.profile_ambiguity_ratio,orientation_max_degrees=config.orientation_max_degrees,
                sampling_valid=np.ones((h,w),bool) if config.profile_sampling_domain=="valid_rgb" else None)
            if mixture:
                matched=np.array([any(not peak["reasons"] for peak in item["peaks"]) for item in profile_diagnostics])
                for item in profile_diagnostics:
                    legal=[peak for peak in item["peaks"] if not peak["reasons"]]
                    item["mixture_legal_peak_count"]=len(legal)
                    item["mixture_retains_all_legal_peaks"]=True
                    # Never use the helper's optional single selected peak in mixture.
                    item["single_peak_helper_status"]=item["status"]
                    item["selected_peak"]=None
                    if legal:item.update(status="mixture_RGB_peak_distribution",reason=None)
            distance=np.linalg.norm(selected_edges-uv,axis=1)
            # kNN statistics remain an explicit baseline diagnostic, not selected profile evidence.
        base_support=(z>1e-6)&in_support(uv)
        keep=base_support&matched
        def angular_stats(values):
            return None if not len(values) else {"median":float(np.median(values)),"p90":float(np.percentile(values,90)),
                                                "over30_count":int(np.count_nonzero(values>30))}
        nearest_domain=base_support&match_info["nearest_in_radius"]
        matching_diagnostics={"projected_supported_samples":int(base_support.sum()),
            "nearest_in_radius_samples":int(nearest_domain.sum()),
            "orientation_rejected_without_alternative":int(np.count_nonzero(nearest_domain&~matched)),
            "selected_farther_neighbor_count":int(np.count_nonzero(keep&(match_info["selected_neighbor_rank"]>1))),
            "nearest_angle_degrees":angular_stats(match_info["nearest_angle_degrees"][nearest_domain]),
            "selected_angle_degrees":angular_stats(match_info["selected_angle_degrees"][keep]),
            "curve_sample_counts":{str(i):int(np.count_nonzero(groups==i)) for i in range(len(curves_object_m))},
            "curve_supported_counts":{str(i):int(np.count_nonzero(base_support&(groups==i))) for i in range(len(curves_object_m))},
            "curve_selected_counts":{str(i):int(np.count_nonzero(keep&(groups==i))) for i in range(len(curves_object_m))},
            "selected_quadrant_counts":{str(i):int(np.count_nonzero(keep&((uv[:,0]>=w/2).astype(int)+2*(uv[:,1]>=h/2).astype(int)==i))) for i in range(4)}}
        if profile_diagnostics is not None:
            counts={}
            for item in profile_diagnostics:
                key=item["reason"] or item["status"];counts[key]=counts.get(key,0)+1
            matching_diagnostics["profile_status_counts_all_CAD_samples"]=counts
            matching_diagnostics["normal_profiles"]=profile_diagnostics
            matching_diagnostics["selected_angle_degrees"]=(None if mixture else angular_stats(np.array([
                item["selected_peak"]["angle_degrees"] for use,item in zip(keep,profile_diagnostics) if use])))
            matching_diagnostics["selected_farther_neighbor_count"]=None
            matching_diagnostics["orientation_rejected_without_alternative"]=None
            matching_diagnostics["nearest_angle_degrees_semantics"]="knn baseline only, not profile selected correspondences"
        ids=np.flatnonzero(keep)
        if len(ids)<config.min_observations:
            result["solutions"].append({"initializer":name,"usable":False,"reason":"insufficient_supported_curve_observations",
                                        "observation_count":len(ids),"matching_diagnostics":matching_diagnostics});continue
        if mixture:
            result["solutions"].append(_fit_profile_mixture(points,tangents,uv,tangent2,profile_diagnostics,
                ids,groups,name,baseR,baset,K,cad_diagonal_m,config,project,in_support,tree,matching_diagnostics))
            continue
        p=points[ids];tangent=tangents[ids];edges=selected_edges[ids];n=selected_normals[ids]
        def transformed(x):
            return pose_from_increment(x,baseR,baset,cad_diagonal_m,config.translation_chart)
        def residual(x):
            R,t=transformed(x);pred,depth=project(R,t,p)
            q=p@R.T+t;direction=tangent@R.T
            hom=q@K.T;dh=direction@K.T
            tangent2=(dh[:,:2]*hom[:,2,None]-hom[:,:2]*dh[:,2,None])/np.maximum(hom[:,2,None]**2,1e-18)
            tangent2/=np.maximum(np.linalg.norm(tangent2,axis=1)[:,None],1e-12)
            return np.r_[np.sum((pred-edges)*n,axis=1),config.direction_weight_px*np.sum(tangent2*n,axis=1)]
        bounds=np.r_[np.full(3,config.max_rotation_rad),np.full(3,config.max_translation_diag)]
        fit=least_squares(residual,np.zeros(6),bounds=(-bounds,bounds),loss="soft_l1",f_scale=1.,
                          max_nfev=config.max_nfev)
        R,t=transformed(fit.x);pred,depth=project(R,t,points);res=residual(fit.x)
        diag=observed_jacobian_diagnostics(residual,fit.x,config.jacobian_relative_tolerance,config.max_condition_number,translation_chart=config.translation_chart)
        final_distance,_=tree.query(pred[ids]);supported=in_support(pred[ids])
        reasons=[]
        if not fit.success:reasons.append("optimizer_not_converged")
        if np.linalg.norm(fit.x[:3])>config.max_rotation_rad or np.linalg.norm(fit.x[3:])>config.max_translation_diag:
            reasons.append("candidate_outside_local_SE3_trust_radius")
        if not (depth>1e-6).all():reasons.append("nonpositive_curve_depth")
        if not supported.all() or (final_distance>config.correspondence_radius_px).any():
            reasons.append("candidate_left_observation_support")
        mean=float(np.mean(np.abs(res[:len(ids)])))
        if mean>config.max_mean_normal_residual_px:reasons.append("edge_residual_above_pilot_limit")
        if diag["degenerate"]:reasons.append("observation_jacobian_degenerate")
        if config.orientation_max_degrees is not None:
            q=p@R.T+t;hom=q@K.T;dh=(tangent@R.T)@K.T
            final_tangent=dh[:,:2]*hom[:,2,None]-hom[:,:2]*dh[:,2,None]
            unit=final_tangent/np.maximum(np.linalg.norm(final_tangent,axis=1)[:,None],1e-12)
            final_angle=np.degrees(np.arcsin(np.clip(np.abs(np.sum(unit*n,axis=1)),0,1)))
            if (final_angle>config.orientation_max_degrees).any():reasons.append("candidate_direction_left_orientation_gate")
        matching_diagnostics.update(final_unsupported_count=int(np.count_nonzero(~supported)),
                                    final_outside_radius_count=int(np.count_nonzero(final_distance>config.correspondence_radius_px)))

        result["solutions"].append({"initializer":name,"usable":True,"R":R.tolist(),"t_m":t.tolist(),
            "matching_diagnostics":matching_diagnostics,
            "pose_increment":pose_increment_record(fit.x,baseR,baset,cad_diagonal_m,config.translation_chart),
            "local_perturbation_rotvec_rad":fit.x[:3].tolist(),"local_translation_CADdiag_units":fit.x[3:].tolist(),
            "observation_count":len(ids),"curve_observation_counts":{str(i):int(np.count_nonzero(groups[ids]==i)) for i in range(len(curves_object_m))},
            "observed_sample_indices":ids.tolist(),"image_edge_matches_xy":edges.tolist(),
            "raw_mean_normal_residual_px":float(np.mean(np.abs(residual(np.zeros(6))[:len(ids)]))),
            "mean_normal_residual_px":mean,"raw_observation_rms":float(np.sqrt(np.mean(res**2))),
            "positive_depth":bool((depth>1e-6).all()),"jacobian":diag,"reasons":reasons,
            "geometric_candidate_eligible":not reasons})
    usable=[x for x in result["solutions"] if x.get("usable")]
    if not usable:result["reasons"]=["insufficient_supported_curve_observations"];return result
    best=min(usable,key=lambda x:x["mean_mixture_nll"] if mixture else x["raw_observation_rms"])
    result["candidate_pose"]={"R":best["R"],"t_m":best["t_m"]}
    result["reasons"]=list(best["reasons"])
    for other in usable:
        if other is best:continue
        if other["observed_sample_indices"]!=best["observed_sample_indices"]:
            result["reasons"].append("multistart_observation_domains_differ_not_comparable")
            continue
        if mixture:
            domains=lambda solution:[[p["pixel_xy"] for p in line] for line in solution["mixture_peak_sources"]]
            if domains(other)!=domains(best):
                result["reasons"].append("multistart_mixture_peak_domains_differ_not_comparable")
        angle=float(Rotation.from_matrix(np.array(other["R"])@np.array(best["R"]).T).magnitude())
        translation=float(np.linalg.norm(np.array(other["t_m"])-best["t_m"])/cad_diagonal_m)
        close_cost=(abs(other["mean_mixture_nll"]-best["mean_mixture_nll"])<=config.mixture_ambiguity_nll if mixture
                    else abs(other["raw_observation_rms"]-best["raw_observation_rms"])<=config.ambiguity_residual_px)
        if (close_cost
            and (angle>=config.ambiguity_rotation_rad or translation>=config.ambiguity_translation_diag)):
            result["reasons"].append("ambiguous_similar_mixture_likelihood_distinct_pose" if mixture else "ambiguous_similar_RGB_residual_distinct_pose")
    # Even a well-conditioned single-face candidate does not establish metric accuracy.
    result["geometric_candidate_eligible"]=not result["reasons"]
    result["reasons"].append("single_face_pose_accuracy_not_established")
    return result
