"""Surface supervision sampled at transformed native CAD correspondences."""
import torch
from torch.nn import functional as F

def sample_points(feature,xy):
    """Deterministic bilinear feature sampling; labels/locations never need gradients."""
    if feature.ndim!=3 or xy.ndim!=2 or xy.shape[1]!=2:raise ValueError("C,H,W and Nx2 required")
    if xy.requires_grad:raise ValueError("Target point locations must not be trainable")
    channels,height,width=feature.shape
    if not torch.isfinite(xy).all() or not ((xy>=0).all() and (xy[:,0]<=width-1).all() and (xy[:,1]<=height-1).all()):
        raise ValueError("Finite interior pixel centres required")
    xy=xy.to(device=feature.device,dtype=feature.dtype);x,y=xy.unbind(-1)
    x0=x.floor().long();y0=y.floor().long();dx=x-x0;dy=y-y0
    values=feature.new_zeros((channels,len(xy)))
    for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx=x0+ox;yy=y0+oy;inside=(xx>=0)&(yy>=0)&(xx<width)&(yy<height)
        index=yy.clamp(0,height-1)*width+xx.clamp(0,width-1)
        values=values+feature.flatten(1).index_select(1,index)*(weight*inside)[None]
    return values.T

def sparse_surface_losses(output,target):
    if len(output["xyz"])!=len(target["sparse"]) or output["masks"].shape!=target["masks"].shape:
        raise ValueError("Surface target batch/grid mismatch")
    xyz=[];regions=[]
    for index,labels in enumerate(target["sparse"]):
        points=labels["xy_hm"]
        if not len(points):raise ValueError("Cannot silently train on zero correspondences")
        predicted=sample_points(output["xyz"][index],points)
        logits=sample_points(output["regions"][index],points)
        xyz.append((predicted-labels["xyz"]).abs().sum(-1).mean())
        regions.append(F.cross_entropy(logits,labels["regions"]))
    return {"xyz":torch.stack(xyz).mean(),"region":torch.stack(regions).mean(),
            "mask":F.binary_cross_entropy_with_logits(output["masks"],target["masks"])}
