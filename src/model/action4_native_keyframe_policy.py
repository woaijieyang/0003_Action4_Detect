"""Pure CPU engineering guard for host-committed native keyframes, not pose truth."""
import numpy as np

DEFAULT_POLICY = dict(schema="action4.native_keyframe_host.dev_v1", minimum_unique_observations=12,
    grid_side=8, minimum_grid_cells=4, reprojection_floor_px=2.0, reprojection_bbox_fraction=.01,
    body_coverage_min=.9, rotation_change_max_deg=30.0, translation_change_max_cad=.5,
    so3_tolerance=1e-5, minimum_depth_m=1e-6,
    duplicate_rule="native_integer_pixel_worst_final_pose_residual")


def validate_policy(policy):
    if not isinstance(policy, dict) or policy != DEFAULT_POLICY:
        raise ValueError("Exact explicitly registered development keyframe policy required")
    return dict(policy)


def decide_keyframe(*, policy, track_id, frame_id, pending_frame, pending_support_frame,
        native_ok, pose, before_pose, canonical, K, observations, owned_mask, body_mask,
        support_audit, expected_mask_sha256, current_sam, predicted_box):
    p=validate_policy(policy)
    result=dict(commit_recommended=False, engineering_guard_only=True, geometry_accepted=False,
        pose_accuracy_verified=False, track_id=track_id, frame_id=frame_id,
        unique_observations=0, grid_cells=0, residual_median_px=None,
        residual_threshold_px=None, body_coverage=None, rotation_change_deg=None,
        translation_change_cad=None, duplicate_rule=p['duplicate_rule'])
    def reject(reason):return dict(result,reason=reason)
    if (type(frame_id) is not int or not isinstance(track_id,str) or not track_id or
        pending_frame!=frame_id or pending_support_frame!=frame_id):return reject('pending_frame_or_support_identity')
    if native_ok is not True:return reject('native_failed')
    if current_sam is not True:return reject('current_SAM_unknown_or_absent')
    if (not isinstance(support_audit,dict) or support_audit.get('frame_id')!=frame_id or
        support_audit.get('track_id')!=track_id or support_audit.get('status')!='current_support' or
        not expected_mask_sha256 or support_audit.get('native_mask_sha256')!=expected_mask_sha256):
        return reject('current_support_identity')
    try:
        T=np.asarray(pose,float);B=np.asarray(before_pose,float);cad=np.asarray(canonical,float);k=np.asarray(K,float)
        box=np.asarray(predicted_box,float);obs=np.asarray(observations,float)
        owned=np.asarray(owned_mask);body=np.asarray(body_mask)
    except (ValueError,TypeError):return reject('malformed_input')
    if (T.shape!=(4,4) or B.shape!=(4,4) or cad.shape!=(8,3) or k.shape!=(3,3) or
        box.shape!=(4,) or obs.ndim!=2 or obs.shape[1]!=10 or
        owned.ndim!=2 or body.shape!=owned.shape or not owned.size or
        any(not np.isfinite(x).all() for x in (T,B,cad,k,box,obs))):return reject('nonfinite_or_wrong_shape')
    if not np.isin(owned,[0,1,255]).all() or not np.isin(body,[0,1,255]).all():return reject('unknown_mask_values')
    if not (owned>0).any() or not (body>0).any():return reject('empty_current_support')
    tol=p['so3_tolerance']
    for matrix in (T,B):
        r=matrix[:3,:3]
        if (not np.array_equal(matrix[3],[0,0,0,1]) or
            not np.allclose(r.T@r,np.eye(3),atol=tol,rtol=0) or abs(np.linalg.det(r)-1)>tol):
            return reject('nonrigid_pose')
    if (not np.array_equal(k[2],[0,0,1]) or min(k[0,0],k[1,1])<=0 or (box[2:]<=box[:2]).any()):
        return reject('invalid_camera_or_prediction_box')
    d=float(np.linalg.norm(np.ptp(cad,axis=0)))
    if d<=0:return reject('invalid_canonical_extent')
    rotation=float(np.rad2deg(np.arccos(np.clip((np.trace(T[:3,:3]@B[:3,:3].T)-1)/2,-1,1))))
    translation=float(np.linalg.norm(T[:3,3]-B[:3,3])/d)
    result.update(rotation_change_deg=rotation,translation_change_cad=translation)
    corners=cad@T[:3,:3].T+T[:3,3]
    if (corners[:,2]<=p['minimum_depth_m']).any():return reject('nonpositive_canonical_depth')
    projected=corners@k.T;uv=projected[:,:2]/projected[:,2:];bounds=np.r_[uv.min(0),uv.max(0)]
    ys,xs=np.nonzero(body);coverage=float(((xs>=bounds[0])&(xs<=bounds[2])&(ys>=bounds[1])&(ys<=bounds[3])).mean())
    result['body_coverage']=coverage
    threshold=max(p['reprojection_floor_px'],p['reprojection_bbox_fraction']*float(np.linalg.norm(box[2:]-box[:2])))
    result['residual_threshold_px']=threshold
    # Reproject raw native observations with the FINAL pose, not stale linearization residuals.
    camera=obs[:,7:10]@T[:3,:3].T+T[:3,3]
    positive=camera[:,2]>p['minimum_depth_m'];xy=obs[:,:2];pix=np.rint(xy).astype(np.int64)
    h,w=owned.shape;inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<w)&(xy[:,1]<h)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    valid=positive&inside&(obs[:,5]>0)
    ids=np.flatnonzero(valid);valid[ids]&=owned[pix[ids,1],pix[ids,0]]>0
    valid&=(xy[:,0]>=box[0])&(xy[:,0]<=box[2])&(xy[:,1]>=box[1])&(xy[:,1]<=box[3])
    unique={}
    for j in np.flatnonzero(valid):
        proj=k@camera[j];error=float(np.linalg.norm(proj[:2]/proj[2]-xy[j]));key=tuple(pix[j])
        # Worst residual avoids a duplicated target UV voting optimistically.
        if key not in unique or error>unique[key][0]:unique[key]=(error,xy[j])
    residuals=[v[0] for v in unique.values()];grids=set()
    for _,point in unique.values():
        grid=np.floor((point-box[:2])/(box[2:]-box[:2])*p['grid_side']).astype(int)
        grids.add(tuple(np.minimum(grid,p['grid_side']-1)))
    result.update(unique_observations=len(unique),grid_cells=len(grids),residual_median_px=float(np.median(residuals)) if residuals else None,
        raw_observation_count=len(obs),current_owned_prediction_box_rows=int(valid.sum()))
    failed=[]
    if len(unique)<p['minimum_unique_observations']:failed.append('insufficient_unique_observations')
    if len(grids)<p['minimum_grid_cells']:failed.append('insufficient_grid_coverage')
    if not residuals or result['residual_median_px']>threshold:failed.append('reprojection_residual')
    if coverage<p['body_coverage_min']:failed.append('body_coverage')
    if rotation>p['rotation_change_max_deg']:failed.append('rotation_change')
    if translation>p['translation_change_max_cad']:failed.append('translation_change')
    return dict(result,commit_recommended=not failed,reason='engineering_guards_pass_not_accuracy' if not failed else failed[0],failed_gates=failed)