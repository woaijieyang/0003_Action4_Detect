"""Instance point prompts from uniquely bound predicted mask and B0 support only."""
import hashlib,json
import numpy as np
from scipy.ndimage import distance_transform_edt

def predicted_instance_points(mask,target_box,other_boxes=()):
    mask=np.asarray(mask);box=np.asarray(target_box,float)
    if mask.dtype!=np.bool_ or mask.ndim!=2 or not mask.any() or box.shape!=(4,) or not np.isfinite(box).all() or not (box[2:]>box[:2]).all():
        raise ValueError("Native predicted mask and finite B0 box required")
    h,w=mask.shape
    x0,y0=np.maximum(np.floor(box[:2]),[0,0]).astype(int);x1,y1=np.minimum(np.ceil(box[2:]),[w,h]).astype(int)
    if x1<=x0 or y1<=y0:raise ValueError("No in-image prediction support")
    support=mask[y0:y1,x0:x1].copy()
    yy,xx=np.indices(support.shape);xx=xx+x0;yy=yy+y0
    others=[]
    for value in other_boxes:
        b=np.asarray(value,float)
        if b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all():raise ValueError("Invalid competing predicted box")
        others.append(b);support&=~((xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]))
    distance=distance_transform_edt(np.pad(support,1))[1:-1,1:-1]
    points=[];radius=max(1.,.15*min(x1-x0,y1-y0))
    for _ in range(2):
        iy,ix=np.unravel_index(np.argmax(distance),distance.shape)
        if distance[iy,ix]<=0:raise ValueError("Insufficient unique predicted foreground")
        points.append([float(ix+x0),float(iy+y0)])
        distance[(xx-(ix+x0))**2+(yy-(iy+y0))**2<=radius**2]=0
    extent=box[2:]-box[:2]
    lo=np.maximum(0,box[:2]-.25*extent);hi=np.minimum([w-1,h-1],box[2:]+.25*extent)
    negatives=[]
    for x,y in [(lo[0],lo[1]),(hi[0],lo[1]),(lo[0],hi[1]),(hi[0],hi[1])]:
        ix,iy=int(round(x)),int(round(y))
        if mask[iy,ix] or any(b[0]<=x<b[2] and b[1]<=y<b[3] for b in [box]+others):continue
        if [float(ix),float(iy)] not in negatives:negatives.append([float(ix),float(iy)])
    if len(negatives)<2:raise ValueError("Insufficient independent predicted background")
    first=negatives[0];second=max(negatives[1:],key=lambda v:np.linalg.norm(np.asarray(v)-first))
    points.extend([first,second]);labels=[1,1,0,0]
    normalized=(np.asarray(points)/[w,h]).tolist()
    evidence={"mask_shape":[h,w],"mask_sha256":hashlib.sha256(mask.tobytes()).hexdigest(),
              "B0_box":box.tolist(),"other_boxes":[b.tolist() for b in others],"points_native":points,
              "points_xy01":normalized,"labels":labels,"GT_used":False,
              "source":"predicted_SAM_mask_intersect_B0_box_excluding_competing_boxes"}
    identity=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return normalized,labels,dict(evidence,identity=identity)
