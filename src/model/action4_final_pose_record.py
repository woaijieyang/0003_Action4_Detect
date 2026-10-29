"""Authoritative selected/final pose contract, including explicit legacy adaptation.

Numeric validity is not pose accuracy. Never falls back to stale top-level R/t.
"""
import copy
import numpy as np

CONTRACT='action4.video_final_pose.v2'


def resolve_final_pose(instance,canonical8,K_native=None):
    pose=instance.get('final_pose');kind=instance.get('pose_output_contract')
    if kind not in (None,CONTRACT):raise ValueError('Unknown final-pose contract')
    if not isinstance(pose,dict) or pose.get('coordinate_frame')!='object_to_capture_cv_m':
        raise ValueError('Explicit final object-to-capture metric pose required')
    legacy=kind is None and instance.get('post_A0_geometry')=='video_pose_stabilization_v1'
    if pose.get('finite') is not True and not (legacy and 'finite' not in pose):
        raise ValueError('Selected pose is invalid or lacks an admitted numeric contract')
    try:r=np.asarray(pose['rotation'],float);t=np.asarray(pose['translation_m'],float);c=np.asarray(canonical8,float)
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('Malformed final pose') from exc
    if r.shape!=(3,3) or t.shape!=(3,) or c.shape!=(8,3) or not all(np.isfinite(x).all() for x in (r,t,c)):
        raise ValueError('Nonfinite/wrong-shape final geometry')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(r)-1)>1e-5:
        raise ValueError('Final rotation is not SO3')
    ids=instance.get('corner_ids')
    if not isinstance(ids,list) or ids!=list(range(8)) or any(type(x) is not int for x in ids):
        raise ValueError('Fixed canonical IDs0..7 required')
    camera=c@r.T+t;minimum=float(camera[:,2].min())
    if minimum<=1e-6:raise ValueError('Nonpositive final canonical depth')
    has_aliases='rotation' in instance and 'translation_m' in instance
    disagree=has_aliases and (not np.array_equal(np.asarray(instance['rotation']),r) or not np.array_equal(np.asarray(instance['translation_m']),t))
    if kind==CONTRACT and (not has_aliases or disagree):raise ValueError('Final aliases contradict v2 contract')
    if disagree and not legacy:raise ValueError('Unversioned conflicting pose aliases')
    pixel_error=None
    if K_native is not None:
        k=np.asarray(K_native,float);uv=np.asarray(instance.get('corners_final'),float)
        if k.shape!=(3,3) or uv.shape!=(8,2) or not np.isfinite(k).all() or not np.isfinite(uv).all():raise ValueError('Finite native K and final corners required')
        if not np.array_equal(k[2],[0,0,1]) or k[0,0]<=0 or k[1,1]<=0:raise ValueError('Invalid pinhole K')
        h=camera@k.T;projected=h[:,:2]/h[:,2:];pixel_error=float(np.max(np.abs(projected-uv)))
        if pixel_error>1e-3:raise ValueError('Final corners do not match authoritative final pose')
    resolved=copy.deepcopy(pose);resolved['finite']=True
    receipt=dict(source_field='final_pose',source_contract=kind or ('legacy_video_stabilization_v1' if legacy else 'selected_A0'),
        legacy_aliases_ignored=bool(disagree),finite_inferred_from_numeric_checks='finite' not in pose,
        min_canonical_depth_m=minimum,projection_max_error_px=pixel_error,geometry_changed=False,
        pose_accuracy_admitted=False,raw_source_preserved=True)
    return resolved,receipt
