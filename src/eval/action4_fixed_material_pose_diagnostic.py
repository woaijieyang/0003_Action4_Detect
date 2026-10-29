"""Bounded fixed-estimated-binding diagnostic, never a deployment pose publisher.

Fits only an existing FitPacket, without heldout inputs or an A0 pose prior.
The initialization is a numerical starting point; its error is inherited by CAD
bindings and cannot be certified by low heldout 2D reprojection error.
"""
import numpy as np
from model.action4_phase2_tracking_contracts import FitPacket


def project(xyz, K, R, t):
    camera = np.asarray(xyz) @ R.T + t
    if not np.isfinite(camera).all() or (camera[:, 2] <= 1e-4).any():
        raise ValueError("Nonpositive/nonfinite geometry")
    h = camera @ K.T
    return h[:, :2] / h[:, 2:]


def solve_diagnostic(packet, *, K, initial_R, initial_t, canonical, threshold_px):
    import cv2
    cv2.setNumThreads(1)
    if not isinstance(packet, FitPacket):
        raise TypeError("Use the existing heldout-filtered FitPacket")
    result = dict(status="not_solved", geometry_accepted=False, pose_modified=False,
                  binding_confirmed=False, fit_count=packet.count,
                  fit_grid_count=packet.grid_count,
                  solver_anchor_ids=[k.anchor_id for k in packet.ids])
    if not packet.eligible:
        return dict(result, reason="fewer_than12_fit_or4_grids")
    K = np.asarray(K, float); R = np.asarray(initial_R, float)
    t = np.asarray(initial_t, float); cad = np.asarray(canonical, float)
    if K.shape != (3,3) or R.shape != (3,3) or t.shape != (3,) or cad.shape != (8,3):
        raise ValueError("Explicit K/R/t/canonical required")
    if not all(np.isfinite(x).all() for x in (K,R,t,cad)) or not np.allclose(R.T@R,np.eye(3),atol=1e-5):
        raise ValueError("Finite metric pose required")
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("Positive prediction-derived pixel threshold required")
    xyz = np.ascontiguousarray(packet.xyz_m, dtype=np.float64)
    uv = np.ascontiguousarray(packet.predicted_uv, dtype=np.float64)
    singular = np.linalg.svd(xyz-xyz.mean(0), compute_uv=False)
    result["xyz_singular_values_m"] = singular.tolist()
    if singular[0] <= 0 or singular[-1]/singular[0] < 1e-6:
        return dict(result, reason="coplanar_or_degenerate_diagnostic_not_disambiguated")
    cv2.setRNGSeed(0)
    try:
        ok, rv, tv, inlier = cv2.solvePnPRansac(xyz,uv,K,None,
            rvec=cv2.Rodrigues(R)[0], tvec=t.reshape(3,1).copy(),
            useExtrinsicGuess=True, iterationsCount=100, reprojectionError=float(threshold_px),
            confidence=.99, flags=cv2.SOLVEPNP_ITERATIVE)
        ids = [] if inlier is None else inlier.reshape(-1).tolist()
        result["inlier_count"] = len(ids)
        result["inlier_grid_count"] = len({packet.grid_ids[i] for i in ids})
        if not ok or len(ids)<12 or result["inlier_grid_count"]<4:
            return dict(result, reason="insufficient_ransac_inliers")
        rv,tv=cv2.solvePnPRefineLM(xyz[ids],uv[ids],K,None,rv,tv)
        rot=cv2.Rodrigues(rv)[0]; trans=tv.reshape(3)
        projected=project(xyz,K,rot,trans); corners=project(cad,K,rot,trans)
        errors=np.linalg.norm(projected-uv,axis=1)
        # Observation-only right-perturbation Jacobian. No prior or damping.
        d=float(np.linalg.norm(cad.max(0)-cad.min(0))); scales=np.r_[np.full(3,np.deg2rad(5.)),np.full(3,.05*d)]
        columns=[]; eps=1e-5
        for axis in range(6):
            delta=np.zeros(6);delta[axis]=eps*scales[axis]
            rp=rot@cv2.Rodrigues(delta[:3])[0];rm=rot@cv2.Rodrigues(-delta[:3])[0]
            yp=project(xyz[ids],K,rp,trans+rot@delta[3:])
            ym=project(xyz[ids],K,rm,trans-rot@delta[3:])
            columns.append(((yp-ym)/(2*eps*threshold_px)).ravel())
        jac=np.stack(columns,axis=1);sv=np.linalg.svd(jac,compute_uv=False)
        cond=float((sv[0]/sv[-1])**2) if sv[-1]>0 else None
        return dict(result,status="diagnostic_candidate_only",reason="unconfirmed_estimated_binding_not_deployment",
            rotation=rot.tolist(),translation_m=trans.tolist(),corners_native=corners.tolist(),
            inlier_anchor_ids=[packet.ids[i].anchor_id for i in ids],
            fit_median_px=float(np.median(errors[ids])),fit_p95_px=float(np.percentile(errors[ids],95)),
            observation_jtj_condition=cond,observation_condition_pass=cond is not None and cond<=1e8,
            jacobian_convention="right rotation5deg translation0.05CAD; residual/pixel_threshold; observations only")
    except (cv2.error,ValueError,np.linalg.LinAlgError) as exc:
        return dict(result,reason="numerical_failure",error=str(exc))



def validate_domain(cfg,item,source,cycle,arrays,metadata):
    seed=source['seed_record']['seeds'];identity=source['identity'];tid=source['track_id']
    ids=[s['key']['anchor_id'] for s in seed]
    expected=sorted(ids)[:3]
    if (cfg['heldout_rule']!='lexicographic_anchor_id_first3_before_any_solver'
        or len(set(ids))!=len(ids) or len(ids)<3 or item['heldout_anchor_ids']!=expected):
        raise ValueError("Heldout registry violates preregistered rule")
    if cycle['source_track']!=item['track'] or cycle['track_id']!=tid:
        raise ValueError("Cycle belongs to a different track snapshot")
    if source['seed_record']['source_surface']!=item['surface']:
        raise ValueError("Initial surface differs from fixed seed source")
    if metadata['models']!=source['seed_record']['source_models']:
        raise ValueError("Model identities differ")
    frame=source['seed_record']['frame_id']
    if metadata['prediction_id']!=f"video:{tid}:frame:{frame}" or metadata['video_context']['frame_id']!=frame:
        raise ValueError("Surface frame/track mismatch")
    if identity['anchor_ids']!=[s['key'] for s in seed]:
        raise ValueError("Point ordering differs")
    n=len(seed);T=len(identity['frame_ids'])
    if arrays['uv_native'].shape!=(T,n,2) or arrays['is_query'].shape!=(T,n) or len(source['frame_observations'])!=T:
        raise ValueError("Frame/point count differs")
    if len(identity['frame_PTS'])!=T:raise ValueError("Missing frame timing")
    for i,obs in enumerate(source['frame_observations']):
        if obs['frame_id']!=identity['frame_ids'][i] or obs['pts']!=identity['frame_PTS'][i]:
            raise ValueError("Observation frame ordering differs")
    if set(r['anchor_id'] for r in cycle['records'])!=set(ids) or len(cycle['records'])!=n:
        raise ValueError("Cycle anchor domain differs")


def run(config_path):
    import json, hashlib
    from pathlib import Path
    from model.action4_phase2_tracking_contracts import AnchorKey,Binding,AnchorRegistry,partition_geometry
    root=Path.cwd().resolve()
    if str(root)!="/data/zy/vln/a-test": raise ValueError("Project root required")
    def path(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root): raise ValueError("Outside project")
        return p
    cfg=json.loads(path(config_path).read_text())
    def read(ref):
        p=path(ref['path']);data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError("Input identity changed")
        return json.loads(data)
    out=path(cfg['output']);out.mkdir(exist_ok=False)
    rows=[];summaries=[]
    for item in cfg['tracks']:
        source=read(item['track']);cycle=read(item['cycle']);seed=source['seed_record']['seeds']
        p=path(source['arrays']['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=source['arrays']['sha256']:raise ValueError("Track changed")
        a=np.load(p,allow_pickle=False);held=set(item['heldout_anchor_ids'])
        keys=tuple(AnchorKey(**s['key']) for s in seed)
        registry=AnchorRegistry([Binding(k,tuple(s['X_cad_m']),s['XYZ_source'],s['query_frame_id'],s['binding_quality'],k.anchor_id in held) for k,s in zip(keys,seed)])
        surface_path=path(item['surface']['path'])
        if hashlib.sha256(surface_path.read_bytes()).hexdigest()!=item['surface']['sha256']:raise ValueError("Surface changed")
        surf=np.load(surface_path,allow_pickle=False);K=surf['K_native'];cad=surf['canonical_m'];R=surf['rotation'].astype(float);t=surf['translation_m'].astype(float)
        validate_domain(cfg,item,source,cycle,a,json.loads(str(surf['metadata'])))
        passes={r['anchor_id'] for r in cycle['records'] if r['cycle_passed']}
        threshold=max(2.,.01*cycle['d_gate']);success=[];helderrors=[]
        for i,f in enumerate(source['identity']['frame_ids']):
            obs=source['frame_observations'][i]
            mask=np.asarray(obs['partial_supported_candidate'],bool)&np.array([k.anchor_id in passes for k in keys])
            interval=cycle.get('validated_frame_interval',[source['identity']['frame_ids'][0],source['identity']['frame_ids'][-1]+1])
            if not interval[0]<=f<interval[1]:mask[:]=False
            packet,score=partition_geometry(registry,keys,a['uv_native'][i],mask,a['is_query'][i],np.ones(len(keys)),tuple(s['spatial_grid_id'] for s in seed))
            candidate=solve_diagnostic(packet,K=K,initial_R=R,initial_t=t,canonical=cad,threshold_px=threshold)
            result=dict(frame_id=f,track_id=source['track_id'],pts=obs['pts'],candidate=candidate,
                heldout_anchor_ids=item['heldout_anchor_ids'],interpretation="fixed_estimated_binding_2d_consistency_not_independent_pose_truth")
            if candidate['status']=='diagnostic_candidate_only':
                rr=np.asarray(candidate['rotation']);tt=np.asarray(candidate['translation_m']);success.append(candidate)
                # Heldout enters only this separate diagnostic scorer AFTER solver completion.
                held_indices=[j for j,k in enumerate(keys) if k.anchor_id in held and mask[j] and not a['is_query'][i,j]]
                if held_indices:
                    hx=np.array([seed[j]['X_cad_m'] for j in held_indices]);hu=a['uv_native'][i,held_indices]
                    er=np.linalg.norm(project(hx,K,rr,tt)-hu,axis=1);helderrors.extend(er.tolist())
                    result['heldout_score']=dict(count=len(er),errors_native_px=er.tolist(),median_px=float(np.median(er)))
            rows.append(result)
        summaries.append(dict(track_id=source['track_id'],frames=len(source['identity']['frame_ids']),candidate_frames=len(success),
             fit_median_px=None if not success else float(np.median([s['fit_median_px'] for s in success])),
             heldout_pointframes=len(helderrors),heldout_median_px=None if not helderrors else float(np.median(helderrors)),
             heldout_p95_px=None if not helderrors else float(np.percentile(helderrors,95)),
             condition_pass_frames=sum(s['observation_condition_pass'] for s in success),geometry_accepted=False))
    report=dict(status="FIXED_BINDING_POSE_DIAGNOSTIC_COMPLETE_NOT_DEPLOYED",configuration=cfg,results=rows,summary=summaries,
                new_model_calls=0,GT_used=False,pose_modified=False,geometry_accepted=False)
    (out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));return summaries
