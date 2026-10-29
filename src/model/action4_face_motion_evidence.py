"""Current-image border correspondences; no CAD pose or final corner publication.

Uses OpenCV pyramidal LK + backward check (official lk_homography sample
pattern). LCD convex hull is EXCLUDED conservatively, never used as a fitted
face or a replacement SAM mask. H is an image-registration diagnostic only.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class FaceMotionConfig:
    border_radius_px:int=32
    dynamic_margin_px:int=3
    max_features:int=300
    min_feature_distance_px:float=8.
    quality_level:float=.01
    lk_window:int=21
    lk_levels:int=3
    forward_backward_px:float=1.5
    ransac_reprojection_px:float=2.
    min_inliers:int=12
    min_inlier_fraction:float=.65
    min_spread_fraction:float=.06
    min_covariance_ratio:float=.01
    max_gap_seconds:float=.15
    dynamic_hull_mode:str="global"

    def __post_init__(self):
        if self.dynamic_hull_mode not in ("global","per_component"):raise ValueError("Explicit dynamic exclusion geometry required")
        for key in ("border_radius_px","dynamic_margin_px","max_features","lk_window","lk_levels","min_inliers"):
            v=getattr(self,key)
            if type(v) is not int or v<0:raise ValueError("Invalid integer setting: "+key)
        if self.border_radius_px<=self.dynamic_margin_px or self.lk_window<3 or self.lk_window%2!=1 or self.min_inliers<4:
            raise ValueError("Invalid border/LK/inlier policy")
        for key in ("min_feature_distance_px","quality_level","forward_backward_px","ransac_reprojection_px",
                    "min_inlier_fraction","min_spread_fraction","min_covariance_ratio","max_gap_seconds"):
            v=getattr(self,key)
            if isinstance(v,bool) or not np.isfinite(v) or v<=0:raise ValueError("Invalid finite setting: "+key)
        if not 0<self.quality_level<=1 or not 0<self.min_inlier_fraction<=1 or not 0<self.min_spread_fraction<=1:
            raise ValueError("Invalid fraction setting")

def _inputs(rgb,body,dynamic):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError("Native HWC uint8 RGB required")
    for mask in (body,dynamic):
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=rgb.shape[:2]:
            raise ValueError("Native unmodified bool support required")
    if not body.any() or not dynamic.any():raise ValueError("Current body and dynamic-region evidence required")

def border_support(body,dynamic,cfg):
    import cv2
    exclusion=np.zeros(body.shape,np.uint8)
    if cfg.dynamic_hull_mode=="global":
        ys,xs=np.nonzero(dynamic)
        hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
        cv2.fillConvexPoly(exclusion,hull,1)
    else:
        # Keep every native component, including one-pixel islands. Only avoid
        # inventing exclusion pixels between disconnected SAM components.
        count,labels,stats,_=cv2.connectedComponentsWithStats(np.asarray(dynamic,np.uint8),connectivity=8)
        for index in range(1,count):
            x,y,w,h,_=stats[index]
            yy,xx=np.nonzero(labels[y:y+h,x:x+w]==index)
            hull=cv2.convexHull(np.column_stack((xx+x,yy+y)).astype(np.int32))
            cv2.fillConvexPoly(exclusion,hull,1)
        if not np.all(exclusion[np.asarray(dynamic,bool)]):raise ValueError("Native dynamic pixel lost")
    def dilate(radius):
        return cv2.dilate(exclusion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)
    outer=dilate(cfg.border_radius_px);inner=dilate(cfg.dynamic_margin_px)
    return body&outer&~inner,inner

def _sample(mask,points):
    ij=np.rint(points).astype(np.int64)
    valid=(ij[:,0]>=0)&(ij[:,0]<mask.shape[1])&(ij[:,1]>=0)&(ij[:,1]<mask.shape[0])
    out=np.zeros(len(ij),bool);out[valid]=mask[ij[valid,1],ij[valid,0]]
    return out

def measure_border_motion(previous_rgb,current_rgb,*,previous_body,current_body,previous_dynamic,current_dynamic,
                          previous_identity,current_identity,config):
    import cv2
    if not isinstance(config,FaceMotionConfig):raise TypeError("Explicit immutable motion configuration required")
    _inputs(previous_rgb,previous_body,previous_dynamic);_inputs(current_rgb,current_body,current_dynamic)
    if previous_rgb.shape!=current_rgb.shape:raise ValueError("Pixel geometry changed")
    keys=("body_id","body_generation","face_id","face_generation","pixel_space_id","time_base")
    if any(k not in previous_identity or k not in current_identity or previous_identity[k]!=current_identity[k] for k in keys):
        raise ValueError("Cross-instance, generation or pixel-space motion refused")
    if (type(previous_identity["frame_id"]) is not int or current_identity["frame_id"]!=previous_identity["frame_id"]+1):
        raise ValueError("Explicit adjacent current image pair required")
    tb=previous_identity["time_base"]
    if len(tb)!=2 or any(type(x) is not int or x<=0 for x in tb):raise ValueError("Positive rational PTS scale required")
    dt=(current_identity["pts"]-previous_identity["pts"])*tb[0]/tb[1]
    if not 0<dt<=config.max_gap_seconds:raise ValueError("Invalid or excessive image time gap")
    for identity in (previous_identity,current_identity):
        if type(identity["face_id"]) is not int or not 0<=identity["face_id"]<6:raise ValueError("Fixed FaceID0..5 required")
        if any(not isinstance(identity.get(k),str) or len(identity[k])!=64 for k in ("rgb_sha256","body_mask_sha256","dynamic_mask_sha256")):
            raise ValueError("Bound source evidence required")
    from data.action4_sam_roi_inputs import array_digest
    for identity,rgb,body,dynamic in [(previous_identity,previous_rgb,previous_body,previous_dynamic),
                                    (current_identity,current_rgb,current_body,current_dynamic)]:
        for key,value in [("rgb_tensor_sha256",rgb),("body_mask_sha256",body),("dynamic_mask_sha256",dynamic)]:
            if identity.get(key)!=array_digest(value):raise ValueError("Actual image/support bytes differ: "+key)
    support0,excluded0=border_support(previous_body,previous_dynamic,config)
    support1,excluded1=border_support(current_body,current_dynamic,config)
    # One pair-local crop in the SAME native pixel scale for both images.
    yy,xx=np.nonzero(previous_body|current_body);pad=config.lk_window*(2**config.lk_levels)
    l=max(0,int(xx.min())-pad);r=min(previous_rgb.shape[1],int(xx.max())+pad+1)
    t=max(0,int(yy.min())-pad);b=min(previous_rgb.shape[0],int(yy.max())+pad+1)
    gray0=cv2.cvtColor(previous_rgb[t:b,l:r],cv2.COLOR_RGB2GRAY)
    gray1=cv2.cvtColor(current_rgb[t:b,l:r],cv2.COLOR_RGB2GRAY)
    result=dict(schema="action4.face_border_motion.v1",status="insufficient_image_support",accepted_image_registration=False,
        previous_identity=previous_identity,current_identity=current_identity,dt_seconds=dt,config=asdict(config),
        crop_native_xyxy=[l,t,r,b],source="current_RGB_border_correspondences",
        dynamic_region_used_as_texture=False,exclusion_policy="conservative_convex_hull_plus_margin_for_exclusion_only",
        previous_support_pixels=int(support0.sum()),current_support_pixels=int(support1.sum()),
        H_native=None,tracks=[],geometry_fit_eligible=False,CAD_XYZ_available=False,pose_accuracy_verified=False)
    p0=cv2.goodFeaturesToTrack(gray0,mask=support0[t:b,l:r].astype(np.uint8)*255,
        maxCorners=config.max_features,qualityLevel=config.quality_level,minDistance=config.min_feature_distance_px,blockSize=7)
    if p0 is None:return result
    lk=dict(winSize=(config.lk_window,config.lk_window),maxLevel=config.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    p1,s1,e1=cv2.calcOpticalFlowPyrLK(gray0,gray1,p0,None,**lk)
    if p1 is None:return result
    backward_input=p1.copy()
    invalid_forward=~np.isfinite(p1).all(axis=(1,2))|(s1[:,0]==0)
    backward_input[invalid_forward]=p0[invalid_forward]
    pback,sback,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,backward_input,p0.copy(),**lk)
    if pback is None:return result
    src=p0[:,0]+[l,t];dst=p1[:,0]+[l,t]
    fb=np.linalg.norm(pback[:,0]-p0[:,0],axis=1)
    finite=np.isfinite(dst).all(1)&np.isfinite(fb)&np.isfinite(e1[:,0])
    safe=dst.copy();safe[~np.isfinite(safe)]=0
    current_support=_sample(support1,safe)
    good=finite&(s1[:,0]!=0)&(sback[:,0]!=0)&(fb<=config.forward_backward_px)&current_support
    candidate=np.flatnonzero(good);inlier=np.zeros(len(src),bool);H=None
    if len(candidate)>=config.min_inliers:
        H,m=cv2.findHomography(src[candidate].astype(np.float64),dst[candidate].astype(np.float64),
            cv2.RANSAC,config.ransac_reprojection_px,maxIters=2000,confidence=.995)
        if H is not None and m is not None and np.isfinite(H).all():inlier[candidate[m[:,0].astype(bool)]]=True
    result["detected_features"]=len(src);result["valid_bidirectional_features"]=len(candidate);result["inliers"]=int(inlier.sum())
    fraction=float(inlier.sum()/max(len(candidate),1))
    spread=0.;cov_ratio=0.;errors=[]
    if inlier.sum()>=4:
        pts=src[inlier];area=float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
        sy,sx=np.nonzero(support0);support_box_area=max((sx.max()-sx.min())*(sy.max()-sy.min()),1)
        spread=area/support_box_area
        eigen=np.linalg.eigvalsh(np.cov(pts.T));cov_ratio=float(eigen[0]/max(eigen[-1],1e-12))
        projected=cv2.perspectiveTransform(pts.reshape(1,-1,2).astype(np.float64),H)[0]
        errors=np.linalg.norm(projected-dst[inlier],axis=1).tolist()
    result.update(inlier_fraction=fraction,spatial_hull_fraction=spread,covariance_ratio=cov_ratio,
        reprojection_median_px=float(np.median(errors)) if errors else None,
        reprojection_p95_px=float(np.percentile(errors,95)) if errors else None)
    accepted=(inlier.sum()>=config.min_inliers and fraction>=config.min_inlier_fraction
        and spread>=config.min_spread_fraction and cov_ratio>=config.min_covariance_ratio)
    if accepted:
        result.update(status="image_registration_observed_not_CAD_pose",accepted_image_registration=True,H_native=H.tolist())
    for i in range(len(src)):
        result["tracks"].append(dict(local_feature_id=i,previous_native=src[i].tolist(),
            current_native=dst[i].tolist() if finite[i] else None,forward_backward_px=float(fb[i]) if finite[i] else None,
            forward_ok=bool(s1[i,0]),backward_ok=bool(sback[i,0]),current_support=bool(current_support[i]),
            inlier=bool(inlier[i]),accepted=bool(accepted and inlier[i])))
    return result
