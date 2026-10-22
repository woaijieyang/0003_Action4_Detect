"""GDR-derived surface head with explicit visible-only XYZ supervision."""
from types import SimpleNamespace as NS
import torch
from torch import nn
from torch.nn import functional as F
from model.heads.gdr_region_head import RotWithRegionHead

class SurfaceHead(nn.Module):
    def __init__(self):
        super().__init__();cfg=NS(MODEL=NS(CDPN=NS(ROT_HEAD=NS(ROT_CONCAT=False))))
        self.head=RotWithRegionHead(cfg,in_channels=2048,num_filters=64,num_layers=3,mask_output_dim=2,num_regions=64,norm="GN",num_gn_groups=8)
    def forward(self,features):
        masks,x,y,z,regions=self.head(features)
        return {"xyz":torch.cat([x,y,z],1),"masks":masks,"regions":regions}

def surface_losses(output,target):
    valid=target["valid"].bool();counts=valid.sum((-1,-2)).clamp_min(1)
    if output["xyz"].shape!=target["xyz"].shape:raise ValueError("XYZ grid mismatch")
    xyz=((output["xyz"]-target["xyz"]).abs()*valid[:,None]).sum((1,2,3))/counts
    region=F.cross_entropy(output["regions"],target["regions"],ignore_index=-100,reduction="none").sum((-1,-2))/counts
    masks=F.binary_cross_entropy_with_logits(output["masks"],target["masks"].float())
    return {"xyz":xyz.mean(),"region":region.mean(),"mask":masks}
