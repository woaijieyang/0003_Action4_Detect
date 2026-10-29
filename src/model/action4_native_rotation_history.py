"""Bounded camera-relative rotational motion prior from qualified native observations."""
from collections import deque
import numpy as np




def rotation_exp(vector):
    v=np.asarray(vector,float);theta=float(np.linalg.norm(v))
    x,y,z=v
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    t2=theta*theta
    a=1-t2/6+t2*t2/120 if theta<1e-6 else np.sin(theta)/theta
    b=.5-t2/24+t2*t2/720 if theta<1e-6 else (1-np.cos(theta))/t2
    return np.eye(3)+a*skew+b*(skew@skew)

def rotation_log(matrix):
    r=np.asarray(matrix,float)
    cosine=np.clip((np.trace(r)-1)/2,-1.,1.);theta=float(np.arccos(cosine))
    vee=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    if theta<1e-6:return .5*(1+theta*theta/6)*vee
    if np.pi-theta<1e-5:
        diagonal=np.maximum((np.diag(r)+1)/2,0);i=int(np.argmax(diagonal))
        axis=np.zeros(3);axis[i]=np.sqrt(diagonal[i])
        for j in range(3):
            if j!=i:axis[j]=(r[j,i]+r[i,j])/(4*axis[i])
        axis/=np.linalg.norm(axis)
        if np.dot(axis,vee)<0:axis=-axis
        return axis*theta
    return theta/(2*np.sin(theta))*vee

class NativeRotationHistory:
    def __init__(self,track_id,window=5,max_gap_seconds=.5,max_speed_rad_s=8.,min_matches=12):
        if not isinstance(track_id,str) or not track_id:raise ValueError("Explicit body identity")
        if window!=5 or max_gap_seconds<=0 or max_speed_rad_s<=0 or min_matches<4:raise ValueError("Invalid history policy")
        self.track_id=track_id;self.rows=deque(maxlen=window);self.max_gap=max_gap_seconds
        self.max_speed=max_speed_rad_s;self.min_matches=min_matches;self.seen=-1;self.seen_time=None
    def _time(self,track_id,frame_id,pts,time_base):
        if track_id!=self.track_id:raise ValueError("Cross-body history rejected")
        if type(frame_id) is not int or type(pts) is not int or len(time_base)!=2 or any(type(v) is not int or v<=0 for v in time_base):
            raise ValueError("Explicit frame/PTS domain required")
        return pts*time_base[0]/time_base[1]
    def predict(self,track_id,frame_id,pts,time_base):
        now=self._time(track_id,frame_id,pts,time_base)
        if frame_id<=self.seen or (self.seen_time is not None and now<=self.seen_time):raise ValueError("Prediction must follow previous frame and PTS")
        if not self.rows:return dict(available=False,reason="no_qualified_history")
        dt=now-self.rows[-1][1]
        if not 0<dt<=self.max_gap:return dict(available=False,reason="history_expired_or_time_invalid")
        velocities=[]
        for a,b in zip(list(self.rows)[:-1],list(self.rows)[1:]):
            delta=b[1]-a[1]
            if delta<=0:raise ValueError("Nonincreasing observed PTS")
            velocities.append(rotation_log(b[2]@a[2].T)/delta)
        omega=np.median(velocities,axis=0) if velocities else np.zeros(3)
        if np.linalg.norm(omega)>self.max_speed:return dict(available=False,reason="fast_rotation_unmodeled")
        rotation=rotation_exp(omega*dt)@self.rows[-1][2]
        return dict(available=True,reason="qualified_image_history_prior_not_GT",rotation=rotation.tolist(),
                    source_frames=[r[0] for r in self.rows],prediction_dt_seconds=dt,
                    omega_rad_s=omega.tolist(),independent_accuracy_evidence=False)
    def observe(self,track_id,frame_id,pts,time_base,rotation,*,matches,current_support,native_ok,reseeded=False):
        now=self._time(track_id,frame_id,pts,time_base)
        if frame_id<=self.seen or (self.seen_time is not None and now<=self.seen_time):raise ValueError("Duplicate/reversed observation")
        if type(matches) is not int or matches<0 or any(type(x) is not bool for x in (current_support,native_ok,reseeded)):raise ValueError("Explicit support eligibility required")
        self.seen=frame_id;self.seen_time=now
        if not native_ok or not current_support or reseeded or matches<self.min_matches:
            return dict(accepted=False,reason="unqualified_current_native_observation",matches=matches)
        matrix=np.asarray(rotation,float)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(matrix)-1)>1e-5:
            raise ValueError("Invalid native rotation")
        if self.rows and now<=self.rows[-1][1]:raise ValueError("Nonincreasing observation time")
        if self.rows and now-self.rows[-1][1]>self.max_gap:self.rows.clear()
        self.rows.append((frame_id,now,matrix.copy()))
        return dict(accepted=True,reason="native_support_proxy_not_certified_pose",matches=matches,history_frames=[r[0] for r in self.rows])
