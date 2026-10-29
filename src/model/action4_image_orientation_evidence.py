"""Bounded boundary-image stationarity evidence; no I/O, pose estimation or CAD truth."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import numpy as np
from model.action4_video_motion_observation import measure_motion

DEFAULT_POLICY=dict(schema='action4.image_orientation_evidence.v1',quiet_frames_to_hold=5,
    boundary_exclude_px=3.,boundary_fraction_short_side=.12,minimum_points=12,
    minimum_body_hull_coverage=.1,grid_side=8,minimum_grid_cells=4,
    affine_angle_max_deg=1.5,absolute_log_scale_max=.025,
    translation_removed_median_fraction=.003,translation_removed_p95_fraction=.01,
    pixel_space_id='raw_video_native_integer_centers',working_resolution='native_no_downsampling')


def policy_identity(policy):
    if not isinstance(policy,dict) or policy!=DEFAULT_POLICY:raise ValueError('Exact registered stationarity policy required')
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def body_boundary(mask,other,policy=DEFAULT_POLICY):
    """Interior boundary band, not a semantic static-material mask."""
    import cv2
    policy_identity(policy);mask=np.asarray(mask);other=np.asarray(other)
    if mask.dtype!=np.bool_ or other.dtype!=np.bool_ or mask.ndim!=2 or other.shape!=mask.shape:
        raise ValueError('Same native boolean body/other masks required')
    y,x=np.nonzero(mask)
    if not len(x):return None,None
    box=np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)
    short=float(min(box[2:]-box[:2]))
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    band=(distance>policy['boundary_exclude_px'])&(distance<=policy['boundary_fraction_short_side']*short)&mask&~other
    return band,box


def boundary_stationarity(reference,current,policy=DEFAULT_POLICY):
    """Use existing bidirectional/inlier gates, then native full-body geometry gates."""
    import cv2
    policy_identity(policy)
    rb,cb=reference.box,current.box;patch=np.r_[np.minimum(rb[:2],cb[:2]),np.maximum(rb[2:],cb[2:])]
    result=measure_motion(reference.gray,current.gray,reference.band,current.band,patch,
        max_side=max(reference.gray.shape),min_points=policy['minimum_points'])
    stats={'motion_reason':result.get('reason'),'measured_6D_pose':False,'material_identity_certified':False,
        'dynamic_LCD_semantics_certified':False,'pixel_space_id':policy['pixel_space_id'],
        'legacy_ring_coverage_not_fraction':result.get('coverage'),'points':result.get('points',0)}
    if not result.get('available',False):return False,False,'motion_unknown:'+str(result.get('reason')),stats
    pp=np.asarray(result['previous_uv'],float);qq=np.asarray(result['current_uv'],float)
    affine=np.asarray(result['affine_patch'],float)
    if pp.ndim!=2 or pp.shape[1]!=2 or qq.shape!=pp.shape or affine.shape!=(2,3) or not all(np.isfinite(a).all() for a in (pp,qq,affine)):
        return False,False,'nonfinite_or_malformed_motion',stats
    l,t,r,b=result['patch_native'];ow,oh=result['patch_size'];sx,sy=ow/(r-l),oh/(b-t)
    n2w=np.array([[sx,0,-sx*l+.5*sx-.5],[0,sy,-sy*t+.5*sy-.5],[0,0,1.]])
    stats.update(native_to_working_affine=n2w.tolist(),working_to_native_affine=np.linalg.inv(n2w).tolist(),
        patch_native=result['patch_native'],patch_size=result['patch_size'],inlier_ratio=result['inlier_ratio'])
    if (ow,oh)!=(r-l,b-t):return False,False,'unexpected_downsampling',stats
    coverage=[];grids=[]
    for points,box in [(pp,rb),(qq,cb)]:
        coverage.append(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))/float(np.prod(box[2:]-box[:2])) if len(points)>=3 else 0.)
        inside=((points>=box[:2])&(points<box[2:])).all(1)
        grid=np.floor((points[inside]-box[:2])/(box[2:]-box[:2])*policy['grid_side']).astype(int)
        grids.append(len({tuple(g) for g in grid}))
    angle=float(np.rad2deg(np.arctan2(affine[1,0],affine[0,0])))
    scale=float(np.hypot(affine[0,0],affine[1,0]))
    if scale<=0:return False,False,'invalid_affine_scale',stats
    logscale=float(np.log(scale));flow=qq-pp;median_translation=np.median(flow,axis=0)
    residual=np.linalg.norm(flow-median_translation,axis=1);diag=float(min(np.linalg.norm(rb[2:]-rb[:2]),np.linalg.norm(cb[2:]-cb[:2])))
    median=float(np.median(residual));p95=float(np.percentile(residual,95))
    stats.update(points=len(pp),reference_body_bbox=rb.tolist(),current_body_bbox=cb.tolist(),
        body_hull_coverage=coverage,grid_cells=grids,affine_angle_deg=angle,absolute_log_scale=abs(logscale),
        median_translation_native_px=median_translation.tolist(),translation_removed_median_native_px=median,
        translation_removed_p95_native_px=p95,smaller_body_diag_native_px=diag,
        median_limit_native_px=policy['translation_removed_median_fraction']*diag,
        p95_limit_native_px=policy['translation_removed_p95_fraction']*diag)
    if len(pp)<policy['minimum_points'] or min(coverage)<policy['minimum_body_hull_coverage'] or min(grids)<policy['minimum_grid_cells']:
        return False,False,'insufficient_full_body_spatial_support',stats
    quiet=(abs(angle)<=policy['affine_angle_max_deg'] and abs(logscale)<=policy['absolute_log_scale_max'] and
        median<=policy['translation_removed_median_fraction']*diag and p95<=policy['translation_removed_p95_fraction']*diag)
    return True,bool(quiet),'boundary_image_quiet' if quiet else 'boundary_image_motion',stats


@dataclass(frozen=True)
class _Frame:
    gray: np.ndarray
    body: np.ndarray
    band: np.ndarray
    box: np.ndarray
    frame_id: int
    pts: int
    time_base: str
    rgb_sha256: str
    identity: tuple
    source_refs: dict


class ImageOrientationEvidence:
    """One shared previous grayscale and at most one fixed HOLD reference per track."""
    def __init__(self,track_ids=('track_00','track_01'),*,policy=None):
        self.policy=dict(DEFAULT_POLICY if policy is None else policy);self.policy_id=policy_identity(self.policy)
        self.track_ids=tuple(track_ids)
        if not self.track_ids or len(set(self.track_ids))!=len(self.track_ids) or any(not isinstance(t,str) or not t for t in self.track_ids):
            raise ValueError('Fixed unique track domain required')
        self.states={t:{'mode':'MOTION','quiet':0,'reference':None} for t in self.track_ids}
        self.previous={};self.last=None

    def clear(self):
        self.previous={};self.last=None
        for state in self.states.values():state.update(mode='MOTION',quiet=0,reference=None)

    def cache_stats(self):
        refs=[s['reference'] for s in self.states.values() if s['reference'] is not None]
        allframes=list(self.previous.values())+refs
        return dict(previous_track_slots=len(self.previous),reference_slots=len(refs),
            retained_gray_images=len({id(f.gray) for f in allframes}),maximum_gray_images=1+len(self.track_ids))

    def observe(self,gray,*,frame_id,pts,time_base,RGB_sha256,track_inputs):
        image=np.asarray(gray)
        if image.dtype!=np.uint8 or image.ndim!=2 or type(frame_id) is not int or type(pts) is not int:
            raise ValueError('Explicit native grayscale/frame/PTS required')
        tb=str(Fraction(*time_base)) if isinstance(time_base,(list,tuple)) else str(Fraction(time_base))
        if Fraction(tb)<=0 or not isinstance(RGB_sha256,str) or len(RGB_sha256)!=64:raise ValueError('Positive timebase and RGB SHA required')
        if not isinstance(track_inputs,dict) or not set(track_inputs)<=set(self.track_ids):raise ValueError('Unknown track domain')
        gap=self.last is not None and (frame_id!=self.last[0]+1 or pts<=self.last[1] or tb!=self.last[2])
        # Exactly one owned immutable grayscale for the current frame, shared by all tracks.
        image=np.array(image,copy=True,order='C');image.setflags(write=False)
        current={};reasons={};authenticated={}
        for tid in self.track_ids:
            value=track_inputs.get(tid,{})
            mask=value.get('body_mask');other=value.get('other_mask')
            auth=(value.get('current_mask_authenticated') is True and value.get('track_id')==tid and
                value.get('frame_id')==frame_id and value.get('pts')==pts and value.get('time_base')==tb and value.get('RGB_sha256')==RGB_sha256 and
                value.get('pixel_space_id')==self.policy['pixel_space_id'] and isinstance(value.get('session_id'),str) and bool(value.get('session_id')) and
                type(value.get('generation')) is int and value['generation']>=0)
            authenticated[tid]=bool(auth)
            if not auth:reasons[tid]='current_mask_identity_unknown';continue
            if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=image.shape or not mask.any():
                reasons[tid]='current_mask_missing';authenticated[tid]=False;continue
            if not isinstance(other,np.ndarray) or other.dtype!=np.bool_ or other.shape!=image.shape:
                reasons[tid]='other_body_support_unknown';authenticated[tid]=False;continue
            band,box=body_boundary(mask,other,self.policy)
            body=np.array(mask,copy=True);body.setflags(write=False);band.setflags(write=False);box.setflags(write=False)
            identity=(tid,value['session_id'],value['generation'],value['pixel_space_id'],tb,image.shape)
            current[tid]=_Frame(image,body,band,box,frame_id,pts,tb,RGB_sha256,identity,dict(value.get('source_refs',{})))
        output={}
        for tid,state in self.states.items():
            previous=self.previous.get(tid);now=current.get(tid);ref=state['reference']
            result=dict(track_id=tid,frame_id=frame_id,pts=pts,time_base=tb,current_RGB_sha256=RGB_sha256,
                current_mask_authenticated=authenticated[tid],available=False,hold_rotation=False,reason=None,
                reference_frame_id=None if ref is None else ref.frame_id,previous_frame_id=None if previous is None else previous.frame_id,
                policy_id=self.policy_id,mode=state['mode'],stats={},source_refs={'current':{} if now is None else now.source_refs},
                material_identity_certified=False,measured_6D_pose=False)
            reason=None
            if gap:reason='frame_or_time_gap_reset'
            elif now is None:reason=reasons[tid]
            elif previous is None:reason='awaiting_previous_authenticated_frame'
            elif previous.identity!=now.identity:reason='track_session_generation_reset'
            if reason is not None:
                state.update(mode='MOTION',quiet=0,reference=None);result.update(reason=reason,reference_frame_id=None)
            else:
                base=ref if state['mode']=='HOLD' else previous
                if base is None or base.identity!=now.identity:
                    state.update(mode='MOTION',quiet=0,reference=None);result.update(reason='reference_identity_reset',reference_frame_id=None)
                else:
                    available,quiet,reason,stats=boundary_stationarity(base,now,self.policy)
                    result.update(available=available,reason=reason,stats=stats,
                        reference_frame_id=base.frame_id if state['mode']=='HOLD' else None)
                    result['source_refs']['comparison_reference']=base.source_refs
                    result['stats']['comparison_reference_frame_id']=base.frame_id
                    if state['mode']=='HOLD':
                        if available and quiet:result['hold_rotation']=True
                        else:state.update(mode='MOTION',quiet=0,reference=None)
                    elif available and quiet:
                        state['quiet']+=1
                        if state['quiet']>=self.policy['quiet_frames_to_hold']:
                            state.update(mode='HOLD',quiet=0,reference=now)
                            result.update(reason='entered_HOLD_after_five_quiet_no_hold_this_frame',reference_frame_id=frame_id)
                    else:state.update(mode='MOTION',quiet=0,reference=None)
            result.update(mode=state['mode'],quiet_streak=state['quiet']);output[tid]=result
        self.previous=current;self.last=(frame_id,pts,tb)
        counts=self.cache_stats()
        assert counts['retained_gray_images']<=counts['maximum_gray_images'] and counts['reference_slots']<=len(self.track_ids)
        for result in output.values():result['cache']=dict(counts)
        return output