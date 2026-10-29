"""Read-only intrinsic hypothesis from decoded planar rectangular marker axes.
No metric square size, real pose label, rectification, or deployment is inferred.
"""
import numpy as np
from scipy.optimize import least_squares

def marker_homographies(quads):
    import cv2
    unit=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
    q=np.asarray(quads,float)
    if q.ndim!=3 or q.shape[1:]!=(4,2) or not np.isfinite(q).all():raise ValueError("Finite ordered rectangle observations required")
    return np.stack([cv2.getPerspectiveTransform(unit,u.astype(np.float32)) for u in q])

def orthogonality_residual(H,parameters):
    cx,cy,logf=parameters;f=np.exp(logf)
    A=H[:,:3,0].copy();B=H[:,:3,1].copy()
    for x in (A,B):x[:,0]=(x[:,0]-cx*x[:,2])/f;x[:,1]=(x[:,1]-cy*x[:,2])/f
    return np.sum(A*B,axis=1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1))

def fit_hypothesis(H,groups,*,width,height):
    groups=np.asarray(groups);_,inv,counts=np.unique(groups,return_inverse=True,return_counts=True);weights=1/np.sqrt(counts[inv])
    def residual(x):return orthogonality_residual(H,[x[0]*width,x[1]*height,x[2]])*weights
    fits=[]
    for focal in (1000.,1600.,2639.):
        r=least_squares(residual,[.5,.5,np.log(focal)],bounds=([.25,.25,np.log(500.)],[.75,.75,np.log(6000.)]),loss='soft_l1',f_scale=.02,max_nfev=200)
        fits.append(r)
    best=min(fits,key=lambda r:r.cost);cx,cy,f=best.x[0]*width,best.x[1]*height,np.exp(best.x[2]);sv=np.linalg.svd(best.jac,compute_uv=False)
    return dict(K=[[float(f),0,float(cx)],[0,float(f),float(cy)],[0,0,1]],parameters=[float(cx),float(cy),float(np.log(f))],cost=float(best.cost),success=bool(best.success),evaluations=int(best.nfev),active_bounds=best.active_mask.tolist(),jacobian_condition=float(sv[0]/max(sv[-1],1e-15)),all_start_K=[[float(np.exp(r.x[2])),float(r.x[0]*width),float(r.x[1]*height)] for r in fits],calibration_status='planar_marker_hypothesis_not_verified_camera_calibration')
