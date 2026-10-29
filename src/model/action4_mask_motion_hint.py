"""Pure adjacent authenticated-mask motion hint; no I/O, orientation hold or pose estimate."""
from fractions import Fraction
import hashlib
import json
import numpy as np

DEFAULT_POLICY=dict(schema='action4.mask_motion_hint.v1',centroid_displacement_fraction_max=.015,
    translation_mask_iou_min=.95,absolute_log_area_ratio_max=.03,
    pixel_space_id='raw_video_native_integer_centers',warp='translation_only_native_nearest_zero_border')


def policy_identity(policy):
    if not isinstance(policy,dict) or policy!=DEFAULT_POLICY:raise ValueError('Exact preregistered mask-motion policy required')
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def rational(value):
    try:
        fraction=Fraction(*value) if isinstance(value,(list,tuple)) else Fraction(value)
        return str(fraction) if fraction>0 else None
    except (ValueError,TypeError,ZeroDivisionError,OverflowError):return None


def mask_digest(mask):
    a=np.ascontiguousarray(mask);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes());return h.hexdigest()


def scalar_pass(stats,policy=DEFAULT_POLICY):
    policy_identity(policy)
    values=[stats.get(k) for k in ('centroid_displacement_fraction','warp_nearest_IoU','absolute_log_area_ratio')]
    if any(isinstance(x,bool) or not isinstance(x,(float,int,np.floating,np.integer)) or not np.isfinite(x) for x in values):return False
    displacement,iou,area=values
    return bool(0<=displacement<=policy['centroid_displacement_fraction_max'] and
        policy['translation_mask_iou_min']<=iou<=1 and 0<=area<=policy['absolute_log_area_ratio_max'])


def measure_mask_motion(previous_mask,current_mask):
    """Native boolean masks; image axes x/y and metric denominator use integer centers."""
    import cv2
    a=np.asarray(previous_mask);b=np.asarray(current_mask)
    if a.dtype!=np.bool_ or b.dtype!=np.bool_ or a.ndim!=2 or b.shape!=a.shape or not a.any() or not b.any():
        raise ValueError('Same nonempty native boolean masks required')
    aa=a.astype(np.uint8);bb=b.astype(np.uint8);ma=cv2.moments(aa,True);mb=cv2.moments(bb,True)
    ca=np.array([ma['m10']/ma['m00'],ma['m01']/ma['m00']]);cb=np.array([mb['m10']/mb['m00'],mb['m01']/mb['m00']]);delta=cb-ca
    ax,ay,aw,ah=cv2.boundingRect(aa);bx,by,bw,bh=cv2.boundingRect(bb);diag=min(float(np.hypot(aw,ah)),float(np.hypot(bw,bh)))
    transform=np.array([[1.,0.,delta[0]],[0.,1.,delta[1]]]);h,w=a.shape
    warped=cv2.warpAffine(aa,transform,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    intersection=int((warped&b).sum());union=int((warped|b).sum());areas=[int(ma['m00']),int(mb['m00'])]
    return dict(reference_centroid_native=ca.tolist(),current_centroid_native=cb.tolist(),centroid_translation_native_px=delta.tolist(),
        reference_body_bbox=[ax,ay,ax+aw,ay+ah],current_body_bbox=[bx,by,bx+bw,by+bh],
        smaller_body_bbox_diag_native_px=diag,centroid_displacement_fraction=float(np.linalg.norm(delta)/diag),
        native_translation_affine=transform.tolist(),warp_nearest_IoU=intersection/union,
        absolute_log_area_ratio=abs(float(np.log(areas[1]/areas[0]))),reference_current_area_pixels=areas,
        intersection_pixels=intersection,union_pixels=union)


def evaluate_mask_motion_hint(previous_mask,current_mask,*,track_id,previous_identity,current_identity,boundary_evidence,policy=DEFAULT_POLICY):
    pid=policy_identity(policy);old=previous_identity if isinstance(previous_identity,dict) else {};now=current_identity if isinstance(current_identity,dict) else {}
    tb=rational(now.get('time_base'))
    result=dict(track_id=track_id,frame_id=now.get('frame_id'),pts=now.get('pts'),time_base=tb,
        current_RGB_sha256=now.get('RGB_sha256'),previous_frame_id=old.get('frame_id'),previous_pts=old.get('pts'),
        policy_id=pid,eligible_for_gain=False,weak_low_motion=False,current_mask_authenticated=False,
        scalar_available=False,reason=None,stats={},current_mask_sha256=now.get('mask_sha256'),previous_mask_sha256=old.get('mask_sha256'),
        current_session_id=now.get('session_id'),previous_session_id=old.get('session_id'),
        current_generation=now.get('generation'),previous_generation=old.get('generation'),
        source_receipts={'previous':old.get('source_receipt'),'current':now.get('source_receipt')},
        boundary_policy_id=boundary_evidence.get('policy_id') if isinstance(boundary_evidence,dict) else None,
        boundary_available=boundary_evidence.get('available') if isinstance(boundary_evidence,dict) else None,
        semantics='weak adjacent-mask motion hint for gain reduction only')
    def fail(reason):return dict(result,reason=reason)
    def identity_ok(item):
        return (item.get('track_id')==track_id and type(item.get('frame_id')) is int and item['frame_id']>=0 and
            type(item.get('pts')) is int and rational(item.get('time_base')) is not None and
            isinstance(item.get('RGB_sha256'),str) and len(item['RGB_sha256'])==64 and
            item.get('pixel_space_id')==policy['pixel_space_id'] and item.get('mask_authenticated') is True and
            isinstance(item.get('session_id'),str) and bool(item['session_id']) and
            type(item.get('generation')) is int and item['generation']>=0 and
            isinstance(item.get('mask_sha256'),str) and len(item['mask_sha256'])==64)
    if not identity_ok(now):return fail('current_mask_identity_unknown')
    if not isinstance(current_mask,np.ndarray) or current_mask.dtype!=np.bool_ or current_mask.ndim!=2 or not current_mask.any():return fail('current_mask_missing_empty_or_invalid')
    if mask_digest(current_mask)!=now['mask_sha256']:return fail('current_mask_sha_mismatch')
    result['current_mask_authenticated']=True
    if not identity_ok(old):return fail('previous_mask_identity_unknown')
    if not isinstance(previous_mask,np.ndarray) or previous_mask.dtype!=np.bool_ or previous_mask.shape!=current_mask.shape or not previous_mask.any():return fail('previous_mask_missing_empty_or_invalid')
    if mask_digest(previous_mask)!=old['mask_sha256']:return fail('previous_mask_sha_mismatch')
    if now['frame_id']!=old['frame_id']+1 or now['pts']<=old['pts']:return fail('nonconsecutive_frame_or_time')
    if any(now[k]!=old[k] for k in ('track_id','session_id','generation','pixel_space_id')) or tb!=rational(old['time_base']):return fail('track_session_generation_timebase_changed')
    e=boundary_evidence
    if (not isinstance(e,dict) or e.get('track_id')!=track_id or e.get('frame_id')!=now['frame_id'] or
        e.get('pts')!=now['pts'] or rational(e.get('time_base'))!=tb or e.get('current_RGB_sha256')!=now['RGB_sha256'] or
        type(e.get('available')) is not bool or type(e.get('hold_rotation')) is not bool):return fail('boundary_evidence_identity_unknown')
    stats=measure_mask_motion(previous_mask,current_mask);quiet=scalar_pass(stats,policy)
    result.update(scalar_available=True,stats=stats,scalar_thresholds_pass=quiet)
    if e['available'] is True:return dict(result,reason='boundary_known_never_override')
    if e['hold_rotation'] is True:return dict(result,reason='boundary_hold_unknown_contradiction')
    if e.get('current_mask_authenticated') is not True:return dict(result,reason='boundary_current_mask_not_authenticated')
    return dict(result,weak_low_motion=quiet,eligible_for_gain=quiet,reason='unknown_boundary_weak_mask_motion' if quiet else 'mask_motion_threshold_rejected')