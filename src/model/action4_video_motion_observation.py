"""SAM-owned image texture motion, for a video pose innovation check only."""
import numpy as np

def measure_motion(previous_gray,current_gray,previous_mask,current_mask,box,
                   *,previous_other=None,current_other=None,previous_dynamic=None,current_dynamic=None,
                   require_dynamic_exclusion=False,max_side=512,min_points=12):
    import cv2
    images=[np.asarray(x) for x in (previous_gray,current_gray)]
    masks=[np.asarray(x,dtype=bool).copy() for x in (previous_mask,current_mask)]
    if images[0].ndim!=2 or images[0].shape!=images[1].shape or any(x.dtype!=np.uint8 for x in images):
        raise ValueError("Same native uint8 grayscale pixel space required")
    if any(m.shape!=images[0].shape for m in masks):
        raise ValueError("Mask/native image shape differs")
    dynamic_known=previous_dynamic is not None and current_dynamic is not None
    if require_dynamic_exclusion and not dynamic_known:
        return dict(available=False,rigid_support_qualified=False,reason="dynamic_display_exclusion_unknown")
    if dynamic_known:
        for i,dynamic in enumerate((previous_dynamic,current_dynamic)):
            d=np.asarray(dynamic)
            if d.dtype!=np.bool_ or d.shape!=masks[i].shape:raise ValueError("Dynamic region pixel space differs")
            # Reuse the existing conservative display-hull exclusion; holes
            # between displayed patches must not re-enter as rigid texture.
            if not d.any():
                if require_dynamic_exclusion:
                    return dict(available=False,rigid_support_qualified=False,reason="dynamic_region_absent_or_unresolved")
                continue
            from model.action4_face_motion_evidence import border_support,FaceMotionConfig
            _,exclusion=border_support(masks[i],d,FaceMotionConfig())
            masks[i] &= ~exclusion
    for i,other in enumerate((previous_other,current_other)):
        if other is not None:
            other=np.asarray(other,bool)
            if other.shape!=masks[i].shape:raise ValueError("Other instance pixel space differs")
            masks[i] &= ~other
    bb=np.asarray(box,float)
    if bb.shape!=(4,) or not np.isfinite(bb).all() or np.any(bb[2:]<=bb[:2]):
        raise ValueError("Valid same-coordinate enclosing patch required")
    h,w=images[0].shape
    l,t=np.maximum(np.floor(bb[:2]).astype(int),0)
    right,bottom=np.minimum(np.ceil(bb[2:]).astype(int),[w,h])
    if right<=l or bottom<=t:
        return dict(available=False,reason="empty_image_patch")
    width,height=right-l,bottom-t
    scale=min(1.,max_side/max(width,height));ow,oh=max(1,round(width*scale)),max(1,round(height*scale))
    sx,sy=ow/width,oh/height
    ims=[cv2.resize(x[t:bottom,l:right],(ow,oh),interpolation=cv2.INTER_AREA) for x in images]
    ms=[cv2.resize(x[t:bottom,l:right].astype(np.uint8),(ow,oh),interpolation=cv2.INTER_NEAREST) for x in masks]
    eroded=cv2.erode(ms[0],np.ones((3,3),np.uint8))
    p=cv2.goodFeaturesToTrack(ims[0],maxCorners=160,qualityLevel=.01,minDistance=5,mask=eroded,blockSize=5)
    if p is None or len(p)<min_points:return dict(available=False,reason="insufficient_texture")
    q,st,_=cv2.calcOpticalFlowPyrLK(ims[0],ims[1],p,None,winSize=(21,21),maxLevel=3)
    if q is None:return dict(available=False,reason="forward_LK_failed")
    back,sb,_=cv2.calcOpticalFlowPyrLK(ims[1],ims[0],q,None,winSize=(21,21),maxLevel=3)
    if back is None:return dict(available=False,reason="backward_LK_failed")
    p=p.reshape(-1,2);q=q.reshape(-1,2);back=back.reshape(-1,2)
    valid=st.ravel().astype(bool)&sb.ravel().astype(bool)&np.isfinite(q).all(1)&np.isfinite(back).all(1)
    valid &= np.linalg.norm(back-p,axis=1)<=1.
    qi=np.floor(np.where(np.isfinite(q),q,0)+.5).astype(int)
    inside=(qi[:,0]>=0)&(qi[:,0]<ow)&(qi[:,1]>=0)&(qi[:,1]<oh)
    owner=np.zeros(len(p),bool);owner[inside]=ms[1][qi[inside,1],qi[inside,0]]>0
    valid &= inside&owner
    p,q=p[valid],q[valid]
    if len(p)<min_points:return dict(available=False,reason="insufficient_bidirectional_owned_points",points=len(p))
    cv2.setRNGSeed(0)
    affine,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=500,confidence=.99,refineIters=10)
    if affine is None or inliers is None:return dict(available=False,reason="motion_fit_failed")
    inliers=inliers.ravel().astype(bool);ratio=float(inliers.mean());p,q=p[inliers],q[inliers]
    coverage=min(float(cv2.contourArea(cv2.convexHull(x.astype(np.float32))))/max(int(m.sum()),1) for x,m in zip((p,q),ms)) if len(p)>=3 else 0.
    if len(p)<min_points or ratio<.6 or coverage<.1:
        return dict(available=False,reason="insufficient_spatial_motion_support",points=len(p),inlier_ratio=ratio,coverage=coverage)
    # OpenCV resize integer-center inverse, shared for both frames.
    def native(x):return (x+.5)/np.array([sx,sy])-.5+np.array([l,t])
    pp,qq=native(p),native(q)
    return dict(available=True,rigid_support_qualified=dynamic_known,
                dynamic_exclusion_applied=dynamic_known,reason="same_instance_bidirectional_texture",
                previous_uv=pp.tolist(),current_uv=qq.tolist(),points=len(p),
                inlier_ratio=ratio,coverage=coverage,patch_native=[int(l),int(t),int(right),int(bottom)],
                patch_size=[ow,oh],affine_patch=affine.tolist(),
                displacement_native_median_px=float(np.median(np.linalg.norm(qq-pp,axis=1))),
                measured_6D_pose=False)
