"""Geometry-only component recovery inside a current predicted body window."""
import numpy as np
from scipy.ndimage import label

def coherent_extent(mask,body_box,*,margin=.25,minimum_overlap=.2,minimum_component_fraction=.005):
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=2:
        raise ValueError("Native boolean mask required")
    b=np.asarray(body_box,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all():
        raise ValueError("Finite predicted body box required")
    if not (np.isfinite([margin,minimum_overlap,minimum_component_fraction]).all()
            and 0<=margin<=1 and 0<minimum_overlap<=1 and 0<=minimum_component_fraction<=.05):
        raise ValueError("Registered geometric limits required")
    proof={"policy":"predicted_body_window_components_v1","semantic_body_certified":False,
           "raw_mask_modified":False,"connected_attachments_separated":False,
           "margin":margin,"minimum_overlap":minimum_overlap}
    yy=np.flatnonzero(mask.any(1));xx=np.flatnonzero(mask.any(0))
    if not len(xx):return None,dict(proof,retained_components=0,reason="empty")
    y0,y1=int(yy[0]),int(yy[-1])+1;x0,x1=int(xx[0]),int(xx[-1])+1
    labels,n=label(mask[y0:y1,x0:x1],np.ones((3,3),bool))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    size=b[2:]-b[:2];lo=b[:2]-margin*size;hi=b[2:]+margin*size
    wx0=max(x0,int(np.floor(lo[0])));wy0=max(y0,int(np.floor(lo[1])))
    wx1=min(x1,int(np.ceil(hi[0])));wy1=min(y1,int(np.ceil(hi[1])))
    overlaps=np.zeros_like(sizes)
    if wx1>wx0 and wy1>wy0:
        overlaps=np.bincount(labels[wy0-y0:wy1-y0,wx0-x0:wx1-x0].ravel(),minlength=len(sizes))
    overlaps[0]=0
    keep=(sizes>=max(1,sizes.max()*minimum_component_fraction)) & (overlaps>=minimum_overlap*sizes)
    keep[0]=False
    proof.update(raw_components=int(n),retained_components=int(keep.sum()),
                 selected_pixels=int(sizes[keep].sum()),excluded_pixels=int(sizes[~keep].sum()),
                 search_window=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])])
    if not keep.any():return None,dict(proof,reason="no_component_supported_by_current_body_window")
    retained=keep[labels];ys,xs=np.nonzero(retained)
    return [int(xs.min()+x0),int(ys.min()+y0),int(xs.max()+x0+1),int(ys.max()+y0+1)],proof
