"""Independent analytic-Jacobian candidate, same starts/budget/loss. Offline candidate: fixed metric cuboid + predicted ordered8 -> optional rigid projection.
No truth pose/corners/visibility inputs. Raw predictions always retained.
Not installed in production; real-camera intrinsics must be separately qualified.
"""
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def skew(v):
 x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def projection_jacobian(parameters,points,radius,K,weights):
 r=parameters[:3];rr=Rotation.from_rotvec(r).as_matrix();theta=np.linalg.norm(r);s=skew(r)
 if theta<1e-5:
  a=.5-theta**2/24+theta**4/720;b=1/6-theta**2/120+theta**4/5040
 else:a=(1-np.cos(theta))/theta**2;b=(theta-np.sin(theta))/theta**3
 jr=np.eye(3)-a*s+b*(s@s)
 t=np.r_[parameters[3:5],radius+np.exp(parameters[5])]
 camera=points@rr.T+t;hom=camera@K.T
 dp=np.zeros((len(points),3,6));dp[:,:,:3]=np.array([-rr@skew(x)@jr for x in points]);dp[:,0,3]=1;dp[:,1,4]=1;dp[:,2,5]=np.exp(parameters[5])
 perspective=(K[:2][None]*hom[:,2,None,None]-hom[:,:2,None]*K[2][None,None])/hom[:,2,None,None]**2
 return (np.einsum("nij,njk->nik",perspective,dp)*weights[:,:,None]).reshape(-1,6)


def _array(x,shape,name):
 a=np.asarray(x,dtype=np.float64)
 if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite '+name+' shape required')
 return a

def project(points_m,rotation,translation,K):
 p=np.asarray(points_m)@np.asarray(rotation).T+np.asarray(translation)
 if np.min(p[:,2])<=0:raise ValueError('All corners must have positive camera depth')
 h=p@np.asarray(K).T
 return h[:,:2]/h[:,2,None]

def fit_rigid_cuboid(predicted_px,canonical_m,K,confidence=None,*,max_nfev=100):
 uv=_array(predicted_px,(8,2),'predicted ordered8');xyz=_array(canonical_m,(8,3),'canonical8');K=_array(K,(3,3),'K')
 if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]) or abs(np.linalg.det(K))<1e-12:raise ValueError('Valid calibrated pinhole K required')
 if type(max_nfev) is not int or not 1<=max_nfev<=500:raise ValueError('Bounded solver budget required')
 centre=xyz.mean(0);x=xyz-centre;radius=float(np.linalg.norm(x,axis=1).max())
 if radius<=0 or np.linalg.matrix_rank(x)<3 or len(np.unique(xyz,axis=0))!=8:raise ValueError('Nondegenerate fixed3D canonical8 required')
 w=np.ones(8) if confidence is None else _array(confidence,(8,),'confidence')
 if np.any(w<0) or np.any(w>1):raise ValueError('Confidence must be in0..1')
 raw=uv.tolist();result={'status':'REJECTED','raw_corners_px':raw,'corners_px':raw,'fitted_corners_px':None,'accepted':False,'corner_ids':list(range(8)),'scope':'candidate geometry consistency; not accuracy proof'}
 good=w>=max(.05,float(w.max())*.25)
 if good.sum()<4:result['reason']='insufficient_confident_corners';return result
 span=float(np.linalg.norm(np.ptp(uv[good],axis=0)));sing=np.linalg.svd(uv[good]-uv[good].mean(0),compute_uv=False)
 if span<8 or sing[1]<.01*sing[0]:result['reason']='collapsed_or_collinear_prediction';return result
 weights=np.clip(w,.001,1);weights/=weights.max();weights=np.sqrt(weights)[:,None]
 norm=np.c_[uv,np.ones(8)]@np.linalg.inv(K).T;norm=norm[:,:2]/norm[:,2,None]
 X=np.c_[x/radius,np.ones(8)];A=[]
 for Xr,(u,v),ww in zip(X,norm,weights[:,0]):A.extend([np.r_[Xr,np.zeros(4),-u*Xr]*ww,np.r_[np.zeros(4),Xr,-v*Xr]*ww])
 _,_,vh=np.linalg.svd(np.array(A));P=vh[-1].reshape(3,4)
 if np.linalg.det(P[:,:3])<0:P=-P
 U,S,V=np.linalg.svd(P[:,:3]);d=np.eye(3);d[2,2]=np.linalg.det(U@V);rot=U@d@V;trans=P[:,3]/max(float(S.mean()),1e-12)*radius
 starts=[]
 def pack(rot,t):
  if t[2]<=radius+1e-6:return None
  return np.r_[Rotation.from_matrix(rot).as_rotvec(),t[:2],np.log(t[2]-radius)]
 dlt=pack(rot,trans)
 if dlt is not None:starts.append(dlt)
 z=max(radius*2,np.sqrt(K[0,0]*K[1,1])*2*radius/span);uvcentre=np.average(norm,axis=0,weights=np.maximum(w,.001));t=np.r_[uvcentre*z,z]
 for perm in itertools.permutations(range(3)):
  for signs in itertools.product((-1,1),repeat=3):
   rr=np.eye(3)[list(perm)]*np.array(signs)[:,None]
   if np.linalg.det(rr)>.5:starts.append(pack(rr,t))
 def unpack(p):return Rotation.from_rotvec(p[:3]).as_matrix(),np.r_[p[3:5],radius+np.exp(p[5])]
 def residual(p):
  rr,tt=unpack(p);return ((project(x,rr,tt,K)-uv)*weights).ravel()
 best=None;attempts=0
 for start in starts:
  attempts+=1
  fit=least_squares(residual,start,jac=lambda p:projection_jacobian(p,x,radius,K,weights),method='trf',loss='huber',f_scale=2.,x_scale='jac',max_nfev=max_nfev,bounds=([-np.inf]*5+[-15],[np.inf]*5+[10]))
  if np.isfinite(fit.cost) and (best is None or fit.cost<best.cost):best=fit
  if best is not None and best.cost<1e-10:break
 if best is None:result['reason']='solver_no_finite_solution';return result
 rr,tt=unpack(best.x);fitted=project(x,rr,tt,K);errors=np.linalg.norm(fitted-uv,axis=1);rms=float(np.sqrt(np.average(errors**2,weights=np.maximum(w,.001))));move=float(errors[good].max());actual_t=tt-rr@centre
 result.update(fitted_corners_px=fitted.tolist(),rotation=rr.tolist(),translation_m=actual_t.tolist(),weighted_rms_px=rms,maximum_confident_correction_px=move,raw_confident_span_px=span,starts=attempts,solver_success=bool(best.success),rigid_distance_max_error_m=float(np.max(np.abs(np.linalg.norm((xyz@rr.T)[:,None,:]-(xyz@rr.T)[None,:,:],axis=-1)-np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=-1)))))
 if not best.success:result['reason']='solver_not_converged'
 elif rms>max(4.,.06*span) or move>max(12.,.3*span):result['reason']='predictions_inconsistent_with_fixed_cuboid'
 else:result.update(status='ACCEPTED_RIGID_CANDIDATE',accepted=True,corners_px=fitted.tolist(),reason='bounded_residual_positive_depth_fixed_metric_geometry')
 return result


def fit_job(payload):
    return fit_rigid_cuboid(payload[0],payload[1],payload[2],payload[3])
