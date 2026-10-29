"""Raw planar pose hypotheses from fixed estimated CAD/image correspondences."""
import numpy as np

def planar_pose_candidates(object_points_m,image_points_native,K_native,canonical8_m):
    import cv2
    x=np.asarray(object_points_m,np.float64);u=np.asarray(image_points_native,np.float64)
    K=np.asarray(K_native,np.float64);c=np.asarray(canonical8_m,np.float64)
    if (x.shape!=(4,3) or u.shape!=(4,2) or K.shape!=(3,3) or c.shape!=(8,3)
        or not all(np.isfinite(a).all() for a in (x,u,K,c)) or not np.allclose(K[2],[0,0,1])
        or K[0,0]<=0 or K[1,1]<=0):
        raise ValueError("Four ordered finite CAD/image points, native K and canonical8 required")
    singular=np.linalg.svd(x-x.mean(0),compute_uv=False)
    if singular[1]<1e-8 or singular[2]>1e-5*singular[0]:
        raise ValueError("Nondegenerate planar estimated correspondence required")
    if abs(cv2.contourArea(u.astype(np.float32)))<4:
        raise ValueError("Image correspondence degeneracy")
    response=cv2.solvePnPGeneric(np.ascontiguousarray(x),np.ascontiguousarray(u),K,None,flags=cv2.SOLVEPNP_IPPE)
    candidates=[]
    for index,(rv,tv) in enumerate(zip(response[1],response[2])):
        R,_=cv2.Rodrigues(rv);t=np.asarray(tv).reshape(3)
        xc=x@R.T+t;cc=c@R.T+t
        if not np.isfinite(R).all() or not np.isfinite(t).all() or not (xc[:,2]>1e-6).all() or not (cc[:,2]>1e-6).all():
            candidates.append(dict(solution_id=index,valid=False,reason="nonfinite_or_nonpositive_depth"));continue
        pixel=xc@K.T;pixel=pixel[:,:2]/pixel[:,2:]
        corners=cc@K.T;corners=corners[:,:2]/corners[:,2:]
        candidates.append(dict(solution_id=index,valid=True,R=R.tolist(),t_m=t.tolist(),
            correspondence_errors_native_px=np.linalg.norm(pixel-u,axis=1).tolist(),
            correspondence_rms_native_px=float(np.sqrt(np.mean(np.sum((pixel-u)**2,axis=1)))),
            corners_native=corners.tolist(),corner_ids=list(range(8))))
    return dict(schema="action4.planar_landmark_pose_candidates.v1",status="CANDIDATES_NOT_SELECTED_OR_DEPLOYED",
        candidates=candidates,solver="OpenCV_SOLVEPNP_IPPE",object_units="m",pixel_domain="native",
        camera_assumption="project_bound_uncalibrated_pinhole; no proven_zero_distortion",
        full_pose_accuracy_verified=False,GT_used=False,final_prediction_replaced=False)
