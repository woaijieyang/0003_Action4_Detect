"""Box-conditioned IACS adapter, not a full VarifocalNet reproduction.
VFL formula: https://arxiv.org/abs/2008.13367; box sampling uses actual predicted boxes.
"""
import torch
from torch import nn
from torch.nn import functional as F

def box_features(feature,boxes,canvas_size):
    if feature.ndim!=4 or feature.shape[0]!=1 or boxes.ndim!=2 or boxes.shape[1]!=4:raise ValueError('One image and Nx4 boxes required')
    width,height=canvas_size
    if len(boxes)==0:return feature.new_empty((0,feature.shape[1]*9)),feature.new_empty((0,4))
    boxes=boxes.to(device=feature.device,dtype=feature.dtype);extent=boxes[:,2:]-boxes[:,:2]
    if not torch.isfinite(boxes).all() or not (extent>0).all():raise ValueError('Finite positive boxes required')
    fraction=feature.new_tensor([0.,.5,1.]);x=boxes[:,0,None]+extent[:,0,None]*fraction;y=boxes[:,1,None]+extent[:,1,None]*fraction
    grid=torch.stack((x[:,None,:].expand(-1,3,-1),y[:,:,None].expand(-1,-1,3)),-1);grid=2*(grid+.5)/feature.new_tensor([width,height])-1
    pooled=F.grid_sample(feature.expand(len(boxes),-1,-1,-1),grid,mode='bilinear',padding_mode='zeros',align_corners=False).flatten(1)
    size=extent/feature.new_tensor([width,height]);geometry=torch.stack((size[:,0],size[:,1],torch.log(size[:,0]/size[:,1]),torch.log(size.prod(-1))),-1)
    return pooled,geometry

class BoxAwareQuality(nn.Module):
    def __init__(self,channels=256):
        super().__init__();self.appearance=nn.Sequential(nn.LayerNorm(channels*9),nn.Linear(channels*9,128),nn.ReLU());self.geometry=nn.Sequential(nn.Linear(4,16),nn.ReLU());self.score=nn.Linear(144,1)
        nn.init.constant_(self.score.bias,-4.59511985013459)
    def forward(self,pooled,geometry):return self.score(torch.cat([self.appearance(pooled),self.geometry(geometry)],-1)).squeeze(-1)

def varifocal_loss(logits,target,alpha=.75,gamma=2.):
    if logits.shape!=target.shape or not torch.isfinite(target).all() or ((target<0)|(target>1)).any():raise ValueError('Continuous IACS targets must be0..1')
    probability=logits.sigmoid();weight=torch.where(target>0,target,alpha*(probability-target).abs().pow(gamma))
    return (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).sum()/(target>0).sum().clamp_min(1)
