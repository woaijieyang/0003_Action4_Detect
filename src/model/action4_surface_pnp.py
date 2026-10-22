"""Predicted visible surface -> robust PnP -> all fixed-ID cuboid corners.
No target pose, target mask or target coordinates are accepted by this API.
"""
import numpy as np

def decode_surface_pose(xyz_normalized,visible_probability,K_hm,canonical_m,*,seed=23,max_points=512):
    import cv2
    cv2.setNumThreads(1);cv2.setRNGSeed(seed)
    xyz=np.asarray(xyz_normalized,dtype=float);prob=np.asarray(visible_probability,dtype=float);K=np.asarray(K_hm,dtype=float);canonical=np.asarray(canonical_m,dtype=float)
    if xyz.ndim!=3 or xyz.shape[0]!=3 or prob.shape!=xyz.shape[1:] or K.shape!=(3,3) or canonical.shape!=(8,3):raise ValueError('Surface pose dimensions invalid')
    if not np.isfinite(K).all() or not np.isfinite(canonical).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Invalid pinhole/CAD')
    if not np.allclose(K[2],[0,0,1]) or abs(K[0,1])>1e-8:raise ValueError('This OpenCV adapter requires zero-skew pinhole K')
    def rejected(reason):return {'valid':False,'reason':reason,'corners_hm':None,'scope':'numerical pose validity, not calibrated accuracy'}
    mask=(prob>=.5)&np.isfinite(prob)&np.isfinite(xyz).all(0)&(xyz.min(0)>=-.05)&(xyz.max(0)<=1.05)
    y,x=np.where(mask)
    if len(x)<6:return rejected('insufficient_predicted_surface')
    indices=np.linspace(0,len(x)-1,min(max_points,len(x)),dtype=int);pixels=np.c_[x[indices],y[indices]].astype(np.float64)
    low=canonical.min(0);extent=canonical.max(0)-low;points=xyz[:,y[indices],x[indices]].T*extent+low;points=np.ascontiguousarray(points)
    if np.linalg.svd(points-points.mean(0),compute_uv=False)[1]<1e-5 or np.linalg.svd(pixels-pixels.mean(0),compute_uv=False)[1]<1:return rejected('collapsed_or_collinear_correspondences')
    try:
        ok,r,t,inliers=cv2.solvePnPRansac(points,pixels,K,None,iterationsCount=100,reprojectionError=2.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<6:return rejected('pnp_no_consensus')
        keep=inliers.ravel();r,t=cv2.solvePnPRefineLM(points[keep],pixels[keep],K,None,r,t)
        rotation=cv2.Rodrigues(r)[0];camera=canonical@rotation.T+t.reshape(3)
        if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):return rejected('nonpositive_cuboid_depth')
        projected=cv2.projectPoints(points[keep],r,t,K,None)[0].reshape(-1,2);errors=np.linalg.norm(projected-pixels[keep],axis=1)
        if not np.isfinite(errors).all() or np.median(errors)>2:return rejected('unstable_reprojection')
        hom=camera@K.T;corners=hom[:,:2]/hom[:,2,None]
        return {'valid':True,'corners_hm':corners.tolist(),'corner_ids':list(range(8)),'rotation':rotation.tolist(),'translation_m':t.reshape(3).tolist(),'inliers':len(keep),'sampled_points':len(points),'median_reprojection_px':float(np.median(errors)),'scope':'predicted correspondences only; numerical validity not accuracy guarantee'}
    except cv2.error as error:return rejected('opencv_solver_error:'+str(error).splitlines()[0])
