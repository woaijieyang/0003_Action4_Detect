"""Causal fixed-ID 2D corner tracking. No pose solver or cuboid projection.

Current B0/SAM full ROI transports global image motion; detector observations
alone drive the eight independent point states. Raw observations are preserved.
"""
from collections import deque
import copy,math
import numpy as np

def direct_native_observations(instance,record,spec):
    pid=instance['prediction_id'];tid=instance['track_id']
    if any(x.get('prediction_id')!=pid or x.get('track_id')!=tid for x in (record,spec)):
        raise ValueError('Cross-instance direct corner/ROI identity')
    obs=record['observations']
    if obs['corner_ids']!=list(range(8)) or obs['identity']!=record['roi_identity'] or obs['identity'][1]!=pid:
        raise ValueError('Fixed detector ID and ROI identity required')
    if record['frame_id']!=instance['frame_id'] or record['pts']!=instance['pts'] or record['roi_decision_id']!=instance['roi_decision_id']:
        raise ValueError('Different frame/PTS/actual ROI decision')
    if instance['bbox_for_a0']!=record['bbox_for_a0'] or spec['full_body_bbox']!=instance['bbox_for_a0']:
        raise ValueError('Actual consumed full ROI differs')
    A=np.asarray(record['A224_from_native'],float);uv=np.asarray(obs['uv224'],float)
    if A.shape!=(3,3) or uv.shape!=(8,2) or not np.isfinite(A).all() or not np.isfinite(uv).all() or not np.array_equal(A,np.asarray(spec['A224'])) or not np.array_equal(A[2],[0,0,1]) or abs(np.linalg.det(A))<1e-12:
        raise ValueError('Finite actual A224 and direct224 coordinates required')
    bounds=np.asarray(spec['continuous_bounds'],float);L=bounds[2]-bounds[0]
    box=np.asarray(instance['bbox_for_a0'],float)
    if spec['output_size']!=448 or spec['pixel_convention']!='native_integer_centres_continuous_crop_edges' or not np.isclose(L,spec['expansion']*max(box[2:]-box[:2]),atol=1e-7,rtol=0):raise ValueError('ROI expansion or pixel convention mismatch')
    expected=np.array([[224/L,0,-224*bounds[0]/L-.5],[0,224/L,-224*bounds[1]/L-.5],[0,0,1]])
    if L<=0 or not np.isclose(L,bounds[3]-bounds[1],atol=1e-7) or not np.allclose(A,expected,rtol=0,atol=1e-7):raise ValueError('ROI half-pixel sampling convention mismatch')
    h=np.c_[uv,np.ones(8)]@np.linalg.inv(A).T;native=h[:,:2]/h[:,2:]
    back=np.c_[native,np.ones(8)]@A.T
    if not np.allclose(back[:,:2]/back[:,2:],uv,rtol=0,atol=1e-8):raise ValueError('Direct coordinate inverse inconsistency')
    for key in ('p_valid','q_corner','b224'):
        v=np.asarray(obs[key],float)
        if v.shape!=(8,) or not np.isfinite(v).all():raise ValueError('Malformed direct-head metadata')
    return dict(corner_ids=list(range(8)),corners_detector_native=native.tolist(),uv224=uv.tolist(),
        p_valid=copy.deepcopy(obs['p_valid']),q_corner=copy.deepcopy(obs['q_corner']),b224=copy.deepcopy(obs['b224']),
        multimodal=copy.deepcopy(obs['multimodal']),A224_from_native=A.tolist(),roi_identity=copy.deepcopy(record['roi_identity']),
        source='saved_corner_head_uv224_inverse_actual_A224',pose_projection_used=False,PnP_used=False,
        quality_threshold_applied=False,coordinates_clamped=False)


def _alpha(f,dt):return 1./(1.+1./(2*math.pi*f*dt))

class DirectCornerTracker:
    def __init__(self,*,history_frames=5,min_cutoff_hz=2.,max_cutoff_hz=3.,beta=.1,max_speed_diagonals_s=1.):
        if history_frames!=5 or not 0<min_cutoff_hz<=max_cutoff_hz or beta<0 or max_speed_diagonals_s<=0:raise ValueError('Explicit bounded five-frame point filter required')
        self.history_frames=history_frames;self.min_cutoff=min_cutoff_hz;self.max_cutoff=max_cutoff_hz;self.beta=beta;self.max_speed=max_speed_diagonals_s;self.states={}
    def update(self,*,track_id,frame_id,time_seconds,points_native,full_box,corner_ids):
        if not isinstance(track_id,str) or not track_id or type(frame_id) is not int or not math.isfinite(time_seconds):raise ValueError("Explicit track/frame/finite time required")
        u=np.asarray(points_native,float);b=np.asarray(full_box,float)
        if corner_ids!=list(range(8)) or u.shape!=(8,2) or b.shape!=(4,) or not np.isfinite(u).all() or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():raise ValueError('Finite fixed-ID direct detector points and current ROI required')
        center=(b[:2]+b[2:])/2;diag=float(np.linalg.norm(b[2:]-b[:2]));q=(u-center)/diag;old=self.states.get(track_id)
        if old is not None and (frame_id<=old['frame'] or time_seconds<=old['time']):raise ValueError('Nonincreasing per-instance timeline')
        reset=old is None or frame_id!=old['frame']+1 or time_seconds-old['time']>.25
        if reset:old=dict(frame=frame_id,time=time_seconds,history=deque(maxlen=5),q=q.copy(),target=q.copy(),velocity=np.zeros((8,2)))
        history=list(old['history'])+[(frame_id,q.copy())];weights=.8**np.arange(len(history)-1,-1,-1);targets=[];sources=[]
        for k in range(8):
            costs=[sum(w*min(float(np.linalg.norm(a[1][k]-x[1][k])),.5) for w,x in zip(weights,history)) for a in history]
            chosen=min(range(len(history)),key=lambda i:(costs[i],-history[i][0]));targets.append(history[chosen][1][k]);sources.append(history[chosen][0])
        target=np.stack(targets);limited=[False]*8;cutoffs=[self.min_cutoff]*8
        if not reset:
            dt=time_seconds-old['time'];a=_alpha(1.,dt);old['velocity']=a*(target-old['target'])/dt+(1-a)*old['velocity']
            for k in range(8):
                fc=min(self.max_cutoff,self.min_cutoff+self.beta*float(np.linalg.norm(old['velocity'][k])));cutoffs[k]=fc
                delta=_alpha(fc,dt)*(target[k]-old['q'][k]);length=float(np.linalg.norm(delta));limit=self.max_speed*dt
                if length>limit:delta*=limit/length;limited[k]=True
                old['q'][k]+=delta
        else:old['q']=q.copy()
        result=center+old['q']*diag;old.update(frame=frame_id,time=time_seconds,target=target.copy());old['history'].append((frame_id,q.copy()));self.states[track_id]=old
        return dict(corners_tracked_native=result.tolist(),corner_ids=list(range(8)),status='initialized' if reset else 'tracked_and_smoothed',
            source_frame_per_corner=sources,rate_limited=limited,cutoff_hz=cutoffs,point_identity='track_id_and_fixed_corner_id_no_reordering',
            global_motion_source='current_B0_SAM_full_ROI',observation_source='direct_corner_head',PnP_used=False,pose_projection_used=False)
