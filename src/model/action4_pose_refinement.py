"""Network-initialized iterative PnP, following GDR-Net evaluation option.

Adaptation uses the existing112-grid correspondences and fixed canonical8.
Only predicted XYZ/mask/R/t are inputs. Numerical fallback is explicit.
"""
import numpy as np

def refine_network_pose(xyz_normalized,visible_probability,K,canonical,rotation,translation,max_points=512):
    import cv2
    cv2.setNumThreads(1)
    xyz=np.asarray(xyz_normalized,dtype=float);prob=np.asarray(visible_probability,dtype=float)
    K=np.asarray(K,dtype=float);cad=np.asarray(canonical,dtype=float)
    rot=np.asarray(rotation,dtype=float);t=np.asarray(translation,dtype=float).reshape(3,1)
    if xyz.ndim!=3 or xyz.shape[0]!=3 or prob.shape!=xyz.shape[1:] or K.shape!=(3,3) or cad.shape!=(8,3) or rot.shape!=(3,3):
        raise ValueError("Invalid pose/refinement dimensions")
    def pack(r,tv,refined,reason):
        camera=cad@r.T+tv.reshape(3);valid=bool(np.isfinite(camera).all() and (camera[:,2]>0).all())
        hom=camera@K.T
        return {"valid":valid,"corners_hm":(hom[:,:2]/hom[:,2:]).tolist() if valid else None,
                "refined":refined,"reason":reason,"rotation":r.tolist(),"translation_m":tv.reshape(3).tolist()}
    if not np.isfinite(rot).all() or not np.isfinite(t).all() or not np.allclose(rot.T@rot,np.eye(3),atol=1e-3):
        return {"valid":False,"corners_hm":None,"refined":False,"reason":"invalid_network_pose"}
    mask=(prob>=.5)&np.isfinite(prob)&np.isfinite(xyz).all(0)&(xyz.min(0)>=-.05)&(xyz.max(0)<=1.05)
    y,x=np.where(mask)
    if len(x)<6:return pack(rot,t,False,"network_fallback_insufficient_correspondences")
    ids=np.linspace(0,len(x)-1,min(max_points,len(x)),dtype=int)
    points=np.ascontiguousarray(xyz[:,y[ids],x[ids]].T*(cad.max(0)-cad.min(0))+cad.min(0))
    pixels=np.ascontiguousarray(np.c_[x[ids],y[ids]].astype(np.float64))
    if np.linalg.svd(points-points.mean(0),compute_uv=False)[1]<1e-5:
        return pack(rot,t,False,"network_fallback_degenerate_correspondences")
    try:
        r0=cv2.Rodrigues(rot)[0]
        ok,r,tnew=cv2.solvePnP(points,pixels,K,None,rvec=r0.copy(),tvec=t.copy(),
                               useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:return pack(rot,t,False,"network_fallback_solver_failed")
        result=pack(cv2.Rodrigues(r)[0],tnew,True,None)
        if not result["valid"]:return pack(rot,t,False,"network_fallback_invalid_refinement")
        return result
    except cv2.error:
        return pack(rot,t,False,"network_fallback_opencv_error")
