"""Rectangle-marker orientation plus predicted full-body extent: hypotheses only.
No measured marker size/position is invented. Translation is estimated separately
from the current predicted body box; this is not independent metric pose truth.
"""
import numpy as np
from scipy.optimize import least_squares

def candidates(quad,K,canonical,full_box,*,marker_aspect):
 import cv2
 q=np.asarray(quad,np.float64);K=np.asarray(K,float);C=np.asarray(canonical,float);box=np.asarray(full_box,float)
 if q.shape!=(4,2) or K.shape!=(3,3) or C.shape!=(8,3) or box.shape!=(4,) or not all(np.isfinite(a).all() for a in [q,K,C,box]) or (box[2:]<=box[:2]).any():raise ValueError('Finite same-native geometry required')
 # UV u increases toward -X; image v increases toward -Z in the fixed rear-display mapping.
 marker_to_object=np.array([[-1.,0,0],[0,0,-1],[0,-1,0]])
 if isinstance(marker_aspect,bool) or not np.isfinite(marker_aspect) or marker_aspect<=0:raise ValueError("Explicit rectangle aspect hypothesis required")
 unit=np.array([[0,0,0],[marker_aspect,0,0],[marker_aspect,1,0],[0,1,0]],np.float64)
 response=cv2.solvePnPGeneric(unit,q,K,None,flags=cv2.SOLVEPNP_IPPE)
 center=(box[:2]+box[2:])/2;diag=float(np.linalg.norm(box[2:]-box[:2]));out=[]
 for i,(rv,tv) in enumerate(zip(response[1],response[2])):
  Rt=cv2.Rodrigues(rv)[0];tt=tv.reshape(3);u=(unit@Rt.T+tt)@K.T
  if not (u[:,2]>0).all():continue
  u=u[:,:2]/u[:,2:];rms=float(np.sqrt(np.mean(np.sum((u-q)**2,axis=1))));R=Rt@marker_to_object.T;rot=C@R.T
  z0=max(K[0,0]*np.ptp(rot[:,0])/(box[2]-box[0]),K[1,1]*np.ptp(rot[:,1])/(box[3]-box[1]),.05)
  def pose(x):
   z=np.exp(x[2]);ray=np.linalg.solve(K,[x[0],x[1],1.]);return ray*z-rot.mean(0)
  def residual(x):
   T=pose(x);cam=rot+T
   if not (cam[:,2]>1e-5).all():return np.ones(4)*100
   uv=cam@K.T;uv=uv[:,:2]/uv[:,2:];bb=np.r_[uv.min(0),uv.max(0)];return (bb-box)/diag
  fit=least_squares(residual,[*center,np.log(np.clip(z0,.051,4.9))],bounds=([center[0]-diag,center[1]-diag,np.log(.05)],[center[0]+diag,center[1]+diag,np.log(5)]),max_nfev=60)
  t=pose(fit.x);cam=rot+t
  if not (cam[:,2]>1e-5).all():continue
  uv=cam@K.T;uv=uv[:,:2]/uv[:,2:]
  out.append(dict(solution_id=i,rotation=R.tolist(),translation_m=t.tolist(),corners_native=uv.tolist(),corner_ids=list(range(8)),marker_reprojection_rms_px=rms,body_box_edge_rms_fraction=float(np.sqrt(np.mean(residual(fit.x)**2))),fit_success=bool(fit.success),marker_metric_size_used=False,marker_aspect_hypothesis=float(marker_aspect),marker_to_object_axes_assumed_from_UV_mapping=True,geometry_accepted=False))
 return out
