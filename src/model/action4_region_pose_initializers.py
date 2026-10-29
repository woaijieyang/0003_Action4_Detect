"""Finite current-region pose INITIALIZERS, never pose measurements or publication."""
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

def _project(points,R,t,K):
    xyz=points@R.T+t
    if not (xyz[:,2]>1e-6).all():return None
    uv=xyz@K.T
    return uv[:,:2]/uv[:,2:]

def outside_convex_envelope(points,hull):
    # Image hull orientation is normalized. This is a conservative model
    # envelope cost, NOT a fitted image quadrilateral or full-mask IoU.
    area=np.sum(hull[:,0]*np.roll(hull[:,1],-1)-hull[:,1]*np.roll(hull[:,0],-1))
    q=hull if area>0 else hull[::-1]
    edge=np.roll(q,-1,axis=0)-q
    length=np.linalg.norm(edge,axis=1)
    if (length<1e-8).any():return np.full(len(points),np.inf)
    d=points[:,None,:]-q[None,:,:]
    signed=(edge[None,:,0]*d[:,:,1]-edge[None,:,1]*d[:,:,0])/length
    return np.maximum(0.,-signed.min(1))

def propose_initializers(K,R0,t0,parts,*,rotation_grid_rad,translation_offsets_m,depth_factors,
                         max_points_per_part,rotation_prior_weight,translation_prior_weight,
                         top_k,min_facing_cos):
    import cv2
    from scipy.ndimage import distance_transform_edt,map_coordinates
    K=np.asarray(K,float);R0=np.asarray(R0,float);t0=np.asarray(t0,float)
    if (K.shape!=(3,3) or R0.shape!=(3,3) or t0.shape!=(3,) or not np.isfinite(K).all()
            or not np.isfinite(R0).all() or not np.isfinite(t0).all()
            or not np.allclose(R0.T@R0,np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(R0),1.,atol=1e-6)):
        raise ValueError("Finite proper raw pose and native K required")
    if len(rotation_grid_rad)!=3 or not parts or type(top_k) is not int or top_k<1 or max_points_per_part<4:
        raise ValueError("Explicit finite candidate budget required")
    grids=[list(g) for g in rotation_grid_rad]
    values=[x for g in grids for x in g]+list(translation_offsets_m)+list(depth_factors)+[rotation_prior_weight,translation_prior_weight,min_facing_cos]
    if not values or not np.isfinite(values).all() or not 0<=min_facing_cos<1 or rotation_prior_weight<0 or translation_prior_weight<0:
        raise ValueError("Finite explicit proposal parameters required")
    total=np.prod([len(g) for g in grids])*len(translation_offsets_m)**2*len(depth_factors)
    if not 1<=total<=5000 or any(x<=0 for x in depth_factors):raise ValueError("Bounded candidate grid required")
    prepared=[];shape=None;identities=set()
    for part in parts:
        identity=part.get("part_id")
        if not isinstance(identity,str) or not identity or identity in identities:raise ValueError("Unique part identity required")
        identities.add(identity)
        mask=part["mask"];points=np.asarray(part["vertices_m"],float);normal=None if part.get("outward_normal_object") is None else np.asarray(part["outward_normal_object"],float)
        if (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=2 or not mask.any()
            or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()
            or normal is not None and (normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8)):
            raise ValueError("Current part mask and fixed CAD part required")
        if shape is not None and mask.shape!=shape:raise ValueError("Different pixel domains")
        shape=mask.shape
        yy,xx=np.nonzero(mask);observed=np.c_[xx,yy].astype(float)
        if len(observed)>max_points_per_part:observed=observed[np.linspace(0,len(observed)-1,max_points_per_part,dtype=int)]
        symmetric=part.get("symmetric_seed_envelope",False)
        if type(symmetric) is not bool:raise ValueError("Explicit envelope policy")
        if symmetric and part.get("full_visibility_seed_review") is not True:
            raise ValueError("Symmetric mask initialization only at explicitly reviewed clear seed")
        outside=distance_transform_edt(~mask) if symmetric else None
        prepared.append((part["part_id"],points,None if normal is None else normal/np.linalg.norm(normal),observed,outside))
    candidates=[];rejected_depth=0;rejected_facing=0
    for angles in product(*grids):
        R=R0@Rotation.from_rotvec(np.asarray(angles)).as_matrix()
        for dx,dy,factor in product(translation_offsets_m,translation_offsets_m,depth_factors):
            t=t0*np.array([1.,1.,factor])+[dx,dy,0.]
            camera=-R.T@t;costs={};failed=None
            for identity,vertices,normal,observed,outside_map in prepared:
                direction=camera-vertices.mean(0)
                if normal is not None and np.dot(normal,direction)/max(np.linalg.norm(direction),1e-12)<min_facing_cos:
                    failed="facing";break
                uv=_project(vertices,R,t,K)
                if uv is None:failed="depth";break
                hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2).astype(float)
                error=outside_convex_envelope(observed,hull)
                if not np.isfinite(error).all():failed="depth";break
                reverse_error=None
                if outside_map is not None:
                    edges=np.roll(hull,-1,axis=0)-hull;length=np.linalg.norm(edges,axis=1)
                    cumulative=np.r_[0,np.cumsum(length)]
                    positions=np.linspace(0,cumulative[-1],128,endpoint=False)
                    indices=np.minimum(np.searchsorted(cumulative,positions,side="right")-1,len(hull)-1)
                    samples=hull[indices]+edges[indices]*((positions-cumulative[indices])/np.maximum(length[indices],1e-12))[:,None]
                    reverse_error=map_coordinates(outside_map,np.stack([samples[:,1],samples[:,0]]),order=1,mode="constant",cval=float(max(shape)))
                cost=float(np.mean(error**2))
                if reverse_error is not None:cost+=float(np.mean(reverse_error**2))
                costs[identity]=dict(mean_outside_px=float(error.mean()),p95_outside_px=float(np.percentile(error,95)),
                    mean_square_outside_px=cost,sampled_observed_pixels=len(observed),
                    symmetric_clear_seed_only=outside_map is not None,
                    reverse_mean_px=None if reverse_error is None else float(reverse_error.mean()))
            if failed:
                rejected_facing+=failed=="facing";rejected_depth+=failed=="depth";continue
            obs=sum(v["mean_square_outside_px"] for v in costs.values())/len(costs)
            prior=rotation_prior_weight*np.sum(np.asarray(angles)**2)+translation_prior_weight*(dx*dx+dy*dy+(t[2]-t0[2])**2)
            candidates.append(dict(R=R.tolist(),t_m=t.tolist(),rotation_offset_rad=list(angles),
                translation_offset_m=(t-t0).tolist(),observation_envelope_cost=obs,prior_cost=float(prior),
                proposal_score=float(obs+prior),parts=costs))
    candidates.sort(key=lambda x:(x["proposal_score"],x["rotation_offset_rad"],x["translation_offset_m"]))
    return dict(schema="action4.part_region_pose_initializers.v1",status="INITIALIZERS_REQUIRE_INDEPENDENT_RGB_GEOMETRY_VALIDATION",
        total_grid=int(total),facing_rejections=int(rejected_facing),depth_rejections=int(rejected_depth),
        retained=candidates[:top_k],observed_parts=[p[0] for p in prepared],
        final_pose_available=False,geometry_acceptance=False,mask_centroid_not_used=True,
        assumptions="Known CAD part projected convex envelopes for one-way visible support containment; not complete image masks or metric GT",
        final_corners_modified=False)


_TEXTURE_VERIFIED = object()

@dataclass(frozen=True)
class FixedTextureLandmarks:
    """Source-checked fixed keyframe XYZ, never independent geometry truth."""
    points_object_m: tuple
    pixels_working: tuple
    keypoint_ids: tuple
    binding_sha256: str
    track_id: str
    generation: int
    frame_id: int
    crop_origin: tuple
    pixel_space_id: str
    group_weight: float = 1.
    XYZ_source: str = "A0_estimated_keyframe_CAD_raycast"
    _verified: object = field(default=None,init=False,repr=False,compare=False)

    @classmethod
    def from_binding(cls,ref,*,track_id,generation,frame_id,keypoint_ids,pixels_native,crop_origin,group_weight=1.,pixel_space_id="raw_video_native_integer_centers"):
        import json,hashlib
        from data.action4_paths import PROJECT_ROOT,contained_path
        p=contained_path(PROJECT_ROOT,ref["path"],exists=True);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref["sha256"]:raise ValueError("Fixed texture binding SHA differs")
        doc=json.loads(raw)
        if doc.get("XYZ_source")!="A0_estimated_keyframe_CAD_raycast" or doc.get("rebinding_allowed") is not False or doc.get("units")!="m":
            raise ValueError("Fixed estimated ray binding required")
        rows=[r for r in doc["rows"] if r["seed"]["track_id"]==track_id]
        if len(rows)!=1 or rows[0]["seed"]["generation"]!=generation:raise ValueError("Texture body/generation differs")
        seed=rows[0]["seed"]
        if type(frame_id) is not int or frame_id<seed["seed_frame"]:raise ValueError("Texture frame predates keyframe")
        ids=tuple(keypoint_ids);pixels=np.asarray(pixels_native,float);origin=np.asarray(crop_origin,float)
        if (len(ids)<6 or len(set(ids))!=len(ids) or any(type(k) is not int for k in ids)
            or pixels.shape!=(len(ids),2) or origin.shape!=(2,) or not np.isfinite(pixels).all() or not np.isfinite(origin).all()
            or pixel_space_id!="raw_video_native_integer_centers" or type(generation) is not int
            or isinstance(group_weight,bool) or not np.isfinite(group_weight) or group_weight<=0):
            raise ValueError("Unique6+ texture IDs and finite same-native-pixel observations required")
        lookup={r["keypoint_id"]:r for r in rows[0]["points"] if r["status"]=="provisional_fixed_binding"}
        if any(k not in lookup for k in ids):raise ValueError("Texture ID has no accepted fixed CAD ray")
        points=np.asarray([lookup[k]["X_cad_m"] for k in ids],float)
        if points.shape!=(len(ids),3) or not np.isfinite(points).all():raise ValueError("Nonfinite fixed texture XYZ")
        result=cls(tuple(map(tuple,points)),tuple(map(tuple,pixels-origin)),ids,ref["sha256"],track_id,generation,frame_id,tuple(origin),pixel_space_id,float(group_weight))
        object.__setattr__(result,"_verified",_TEXTURE_VERIFIED)
        return result

    def validate_context(self,context):
        expected=dict(track_id=self.track_id,generation=self.generation,frame_id=self.frame_id,crop_origin=list(self.crop_origin),pixel_space_id=self.pixel_space_id,binding_sha256=self.binding_sha256)
        if self._verified is not _TEXTURE_VERIFIED or context!=expected:raise ValueError("Fixed texture/current observation identity differs")
        points=np.asarray(self.points_object_m,float);pixels=np.asarray(self.pixels_working,float)
        if points.shape!=(len(self.keypoint_ids),3) or pixels.shape!=(len(self.keypoint_ids),2) or not np.isfinite(points).all() or not np.isfinite(pixels).all():raise ValueError("Texture array changed/nonfinite")
        return points,pixels,self.group_weight


def texture_group_residual(errors_px,sigma_px,samples_per_part,weight):
    """Exact mean soft-L1 contribution, independent of the number of points.

    Invert the outer solver's soft-L1 so each coordinate contributes mass/N
    times the same robust native error, not a different robustness threshold.
    """
    errors=np.asarray(errors_px,float)/sigma_px
    mass=samples_per_part*weight/errors.size
    # Algebraically equivalent forms avoid subtracting nearly equal numbers
    # around zero, where the local optimizer still needs a nonzero derivative.
    robust=errors*(errors/(np.hypot(1.,errors)+1.))
    scaled=mass*robust
    return (np.sign(errors)*np.sqrt(scaled*(2.+scaled))).reshape(-1)


def refine_clear_seed_regions(K, R0, t0, parts, *, cad_diagonal_m,
                              samples_per_part=96, sigma_px=3.0,
                              rotation_limit_rad=0.8, translation_limit_cad=0.3,
                              max_nfev=60, landmark_groups=(), initial_increment=None, texture_context=None):
    """Bounded joint region INITIALIZER at reviewed clear seeds only.

    All parts share one SE(3). Convex envelopes approximate CAD silhouettes;
    region edges are not certified physical correspondences. No pose prior is
    included in the reported observation Jacobian, no acceptance is implied.
    """
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull, cKDTree
    K = np.asarray(K, float); R0 = np.asarray(R0, float); t0 = np.asarray(t0, float)
    if (K.shape != (3, 3) or R0.shape != (3, 3) or t0.shape != (3,)
        or not all(np.isfinite(v).all() for v in (K, R0, t0))
        or not np.allclose(R0.T @ R0, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(R0), 1., atol=1e-6)
        or not np.allclose(K[2], [0, 0, 1]) or min(K[0, 0], K[1, 1]) <= 0):
        raise ValueError("Proper pose and calibrated-or-explicit-assumed native K required")
    if (not parts or type(samples_per_part) is not int or not 16 <= samples_per_part <= 256
        or type(max_nfev) is not int or not 1 <= max_nfev <= 200
        or not np.isfinite([cad_diagonal_m, sigma_px, rotation_limit_rad, translation_limit_cad]).all()
        or min(cad_diagonal_m, sigma_px, rotation_limit_rad, translation_limit_cad) <= 0):
        raise ValueError("Explicit positive bounded diagnostic parameters required")

    def sample(poly):
        edge = np.roll(poly, -1, axis=0) - poly
        length = np.linalg.norm(edge, axis=1); cumulative = np.r_[0., np.cumsum(length)]
        if cumulative[-1] < 1e-8: raise ValueError("Degenerate region perimeter")
        pos = np.linspace(0., cumulative[-1], samples_per_part, endpoint=False)
        idx = np.minimum(np.searchsorted(cumulative, pos, side="right") - 1, len(poly)-1)
        return poly[idx] + edge[idx] * ((pos-cumulative[idx]) / np.maximum(length[idx], 1e-12))[:, None]

    prepared = []; names = set(); shape = None
    for part in parts:
        name = part.get("part_id"); mask = part["mask"]; vertices = np.asarray(part["vertices_m"], float)
        partial=part.get("partial_boundary",False)
        if type(partial) is not bool:raise ValueError("Explicit boundary-direction policy required")
        if not partial and part.get("full_visibility_seed_review") is not True:
            raise ValueError("Symmetric region refinement requires explicit clear-seed review")
        eligibility=part.get("observed_boundary_eligibility")
        if partial and (not isinstance(eligibility,np.ndarray) or eligibility.dtype!=np.bool_ or eligibility.shape!=mask.shape):
            raise ValueError("Partial boundary needs explicit current observation eligibility domain")
        if (not isinstance(name, str) or not name or name in names
            or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2
            or not mask.any() or vertices.ndim != 2 or vertices.shape[1] != 3
            or len(vertices) < 4 or not np.isfinite(vertices).all()):
            raise ValueError("Unique part, actual mask and fixed CAD vertices required")
        if shape is not None and mask.shape != shape: raise ValueError("Mixed pixel domains")
        shape = mask.shape; names.add(name)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Keep all components; no silent largest-component selection.
        observed = np.concatenate([c.reshape(-1, 2) for c in contours if len(c) >= 3]).astype(float)
        if partial:
            pixels=observed.astype(int);observed=observed[eligibility[pixels[:,1],pixels[:,0]]]
            if len(observed)<16:raise ValueError("Insufficient current eligible boundary pixels")
        boundary_count=len(observed)
        observed = observed[np.linspace(0, len(observed)-1, samples_per_part, dtype=int)]
        edge_mask = np.zeros(shape, np.uint8); cv2.drawContours(edge_mask, contours, -1, 1, 1)
        by,bx=np.nonzero(edge_mask)
        # Distance to the SAME observed pixel centres is defined over the
        # entire image plane. A computational crop must not create a constant
        # exterior penalty or an artificial zero-gradient region.
        distance = cKDTree(np.c_[bx,by].astype(float))
        prepared.append((name, vertices, observed, distance, partial, boundary_count))

    landmarks = []
    for group in landmark_groups:
        if isinstance(group,FixedTextureLandmarks):
            points,pixels,weight=group.validate_context(texture_context)
            landmarks.append((points,pixels,weight,"fixed_texture"))
            continue
        points = np.asarray(group["points_object_m"], float); pixels = np.asarray(group["pixels_native"], float)
        weight = group["group_weight"]
        if (group.get("XYZ_source") != "existing_appearance_texture_mapping_estimate"
            or group.get("real_metric_calibration") is not False
            or points.shape != (4, 3) or pixels.shape != (4, 2)
            or not np.isfinite(points).all() or not np.isfinite(pixels).all()
            or isinstance(weight, bool) or not np.isfinite(weight) or weight <= 0):
            raise ValueError("Explicit four-point estimated binding and positive group weight required")
        landmarks.append((points, pixels, float(weight),"legacy_marker"))

    def pose(x):
        return Rotation.from_rotvec(x[:3]).as_matrix() @ R0, t0 + x[3:] * cad_diagonal_m

    def residual(x):
        R, t = pose(x); values = []
        for _, vertices, observed, distance, partial, _ in prepared:
            uv = _project(vertices, R, t, K)
            if uv is None:
                values.append(np.full(samples_per_part if partial else 2*samples_per_part, 1e4)); continue
            hull = uv[ConvexHull(uv).vertices]
            forward = None if partial else distance.query(sample(hull),k=1)[0]
            a = hull; e = np.roll(hull, -1, axis=0) - a
            delta = observed[:, None, :] - a[None, :, :]
            along = np.clip(np.sum(delta * e[None, :, :], axis=2) / np.maximum(np.sum(e*e, axis=1), 1e-12), 0., 1.)
            nearest = a[None, :, :] + along[:, :, None] * e[None, :, :]
            backward = np.linalg.norm(observed[:, None, :] - nearest, axis=2).min(axis=1)
            values.append((backward if partial else np.r_[forward, backward]) / sigma_px)
        for points, pixels, weight, kind in landmarks:
            uv = _project(points, R, t, K)
            # Pre-robust scaling only; soft-L1 influence still depends on sampling and weight.
            scale = np.sqrt(samples_per_part * weight / len(points)) / sigma_px
            if kind=="fixed_texture":
                error=np.full((len(points),2),1e4) if uv is None else uv-pixels
                values.append(texture_group_residual(error,sigma_px,samples_per_part,weight))
            else:
                values.append(np.full(2*len(points),1e4) if uv is None else ((uv-pixels)*scale).reshape(-1))
        return np.concatenate(values)

    def jacobian(x):
        epsilon = 1e-4
        return np.column_stack([(residual(x + np.eye(6)[k]*epsilon) -
                                 residual(x - np.eye(6)[k]*epsilon))/(2*epsilon) for k in range(6)])

    bounds = np.r_[np.full(3, rotation_limit_rad), np.full(3, translation_limit_cad)]
    x0 = np.zeros(6) if initial_increment is None else np.asarray(initial_increment, float)
    if x0.shape != (6,) or not np.isfinite(x0).all() or (np.abs(x0) >= bounds).any():
        raise ValueError("Initializer must lie strictly within the SAME current-observation search domain")
    initial = residual(x0)
    fit = least_squares(residual, x0, jac=jacobian, bounds=(-bounds, bounds),
                        max_nfev=max_nfev, loss="soft_l1", f_scale=1., ftol=1e-7, xtol=1e-7, gtol=1e-7)
    final = residual(fit.x); R, t = pose(fit.x)
    singular = np.linalg.svd(jacobian(fit.x), compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0]*1e-6))
    diagnostics = {};offset=0
    for name, _, _, _, partial, boundary_count in prepared:
        count=samples_per_part if partial else 2*samples_per_part
        section = slice(offset, offset+count);offset+=count
        before = initial[section]*sigma_px; after = final[section]*sigma_px
        diagnostics[name] = dict(observation_direction="observed_to_model_only" if partial else "symmetric_clear_region_assumption",
            eligible_boundary_pixels=boundary_count,residual_count=count,
            initial_median_px=float(np.median(before)), final_median_px=float(np.median(after)),
            initial_p95_px=float(np.percentile(before, 95)), final_p95_px=float(np.percentile(after, 95)))
    return dict(status="JOINT_REGION_INITIALIZER_NOT_GEOMETRIC_ACCEPTANCE", R=R.tolist(), t_m=t.tolist(),
        optimizer_success=bool(fit.success), optimizer_message=str(fit.message), nfev=int(fit.nfev),
        initial_cost=float(np.sum(2*(np.sqrt(1+initial**2)-1))), final_cost=float(np.sum(2*(np.sqrt(1+final**2)-1))),
        parts=diagnostics, residual_count=len(final), landmark_residual_counts=[2*len(x[0]) for x in landmarks], observation_jacobian_rank=rank,
        observation_jacobian_condition=None if rank < 6 else float(singular[0]/singular[-1]),
        translation_chart="object_origin_delta_CAD_diagonal_scaled", rotation_units="rad",
        pixel_space="caller_working_coordinates_at_native_pixel_scale", residual_units="native_image_pixels",
        geometric_accuracy_verified=False, final_prediction_replaced=False,
        boundary_assumption="explicit per-part direction; symmetric parts require clear-region assumption; partial parts constrain original observed contour only; no exact surface GT",
        bounds_semantics="per_component_box_not_vector_norm_ball", estimated_landmark_groups=len(landmarks), increment=fit.x.tolist(), at_component_bound=bool((np.abs(fit.x) >= bounds*.99).any()))


def recent_pose_initializer(observations, *, track_id, generation, target_pts,
                            time_base, max_horizon_seconds=0.25, max_gap_seconds=0.25):
    """Last five current-image candidate poses propose INITIALIZATION only.

    The extrapolation is not a measurement, loss prior, or published output.
    Callers must refit current observations in the unchanged search domain.
    """
    from fractions import Fraction
    if (type(target_pts) is not int or len(time_base)!=2 or any(type(x) is not int or x<=0 for x in time_base)
        or not np.isfinite([max_horizon_seconds,max_gap_seconds]).all()
        or min(max_horizon_seconds,max_gap_seconds)<=0):
        raise ValueError("Exact PTS/time base and positive bounded horizon required")
    rows=list(observations)[-5:]
    if not rows:return dict(status="unavailable",reason="no_current_image_history")
    pts=[];rotations=[];translations=[];ids=[]
    for row in rows:
        if (row["track_id"]!=track_id or row["generation"]!=generation
            or row.get("current_image_supported") is not True or row.get("prediction_only") is not False
            or type(row["pts"]) is not int or row["pts"]>=target_pts
            or row["observation_id"] in ids):
            raise ValueError("History must contain distinct earlier current-image evidence of this body generation")
        R=np.asarray(row["R"],float);t=np.asarray(row["t_m"],float)
        if (R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
            or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1.,atol=1e-6)):
            raise ValueError("Invalid history rigid pose")
        pts.append(row["pts"]);rotations.append(R);translations.append(t);ids.append(row["observation_id"])
    if any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError("History PTS must strictly increase")
    unit=Fraction(time_base[0],time_base[1]);h=float((target_pts-pts[-1])*unit)
    gaps=[float((b-a)*unit) for a,b in zip(pts,pts[1:])]
    if h>max_horizon_seconds or any(d>max_gap_seconds for d in gaps):
        return dict(status="unavailable",reason="history_time_gap")
    velocity=np.zeros(3);omega=np.zeros(3)
    if gaps:
        velocity=np.median([(b-a)/d for a,b,d in zip(translations,translations[1:],gaps)],axis=0)
        omega=np.median([Rotation.from_matrix(b@a.T).as_rotvec()/d for a,b,d in zip(rotations,rotations[1:],gaps)],axis=0)
    R=Rotation.from_rotvec(omega*h).as_matrix()@rotations[-1];t=translations[-1]+velocity*h
    if not np.isfinite(R).all() or not np.isfinite(t).all():return dict(status="unavailable",reason="nonfinite_extrapolation")
    return dict(status="initializer_only",R=R.tolist(),t_m=t.tolist(),history_count=len(rows),
        source_observation_ids=ids,source_pts=pts,horizon_seconds=h,prediction_only=True,
        measurement=False,published=False,correlated_image_pose_history=True)


def select_clear_seed_region_fit(fresh, alternative, canonical8_m, button_vertices_m, *, observed_normal_object=(0.,0.,1.)):
    """Apply the same numeric/facing gate BEFORE comparing current-image cost.

    This is only candidate eligibility, never semantic/metric acceptance.
    """
    canonical=np.asarray(canonical8_m,float);button=np.asarray(button_vertices_m,float)
    normal=np.asarray(observed_normal_object,float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9:
        raise ValueError("Explicit finite observed-part outward normal required")
    normal=normal/np.linalg.norm(normal)
    if canonical.shape!=(8,3) or button.ndim!=2 or button.shape[1]!=3 or not len(button) or not np.isfinite(canonical).all() or not np.isfinite(button).all():
        raise ValueError("Finite fixed CAD canonical8 and current button component required")
    def gate(fit):
        if fit is None:return dict(eligible=False,reasons=["absent_candidate"])
        R=np.asarray(fit.get("R"),float);t=np.asarray(fit.get("t_m"),float);reasons=[]
        if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            return dict(eligible=False,reasons=["invalid_pose"])
        if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1.,atol=1e-6):reasons.append("invalid_rotation")
        if fit.get("optimizer_success") is not True:reasons.append("optimizer_failure")
        if fit.get("at_component_bound") is not False:reasons.append("numeric_search_boundary")
        if not np.isfinite(fit.get("final_cost",np.nan)):reasons.append("nonfinite_current_cost")
        if not ((canonical@R.T+t)[:,2]>1e-6).all():reasons.append("nonpositive_depth")
        if np.dot(-R.T@t-button.mean(0),normal)<=0:
            reasons.append("observed_top_part_backfacing" if np.array_equal(normal,[0.,0.,1.]) else "observed_part_backfacing")
        return dict(eligible=not reasons,reasons=reasons)
    fg,ag=gate(fresh),gate(alternative)
    use_alt=ag["eligible"] and (not fg["eligible"] or alternative["final_cost"]<fresh["final_cost"]-1e-8)
    return dict(chosen=alternative if use_alt else fresh,history_selected=bool(use_alt),
        fresh_gate=fg,history_gate=ag,any_eligible=bool(fg["eligible"] or ag["eligible"]),
        gate_scope="numeric_and_current_part_facing_only_not_accuracy")
