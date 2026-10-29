"""Stage4 opt-in C5-to-corner localization extension; no quality or pose admission.

The existing JointSurfaceRGB owns the only backbone/SAM/pose. This module owns
only the planned128-channel FPN and eight heatmap logits. It is not yet a complete
stage4 train/eval/bundle variant and must not replace an admitted A0 output.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_sam_feature_adapter import _identities


def backbone_stages_once(backbone, rgb):
    """Capture C2/C3/C4 from the existing named ResNet Sequential's one call."""
    if not isinstance(backbone, nn.Sequential):
        raise ValueError("Existing named ResNet backbone required")
    stages = {}
    handles = []
    for layer, key in (("layer1","c2"),("layer2","c3"),("layer3","c4")):
        if layer not in backbone._modules:raise ValueError("Missing ResNet stage "+layer)
    try:
        for layer,key in (("layer1","c2"),("layer2","c3"),("layer3","c4")):
            def capture(module,args,output,key=key):
                if key in stages:raise ValueError("Backbone stage executed twice")
                stages[key]=output
            handles.append(backbone._modules[layer].register_forward_hook(capture))
        stages["c5"]=backbone(rgb)
    finally:
        for handle in handles:handle.remove()
    return stages


@dataclass(frozen=True)
class SamCornerFeatures:
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    c5_sam: torch.Tensor
    identities: tuple

    def validate(self):
        if not isinstance(self.c5_sam,torch.Tensor):raise ValueError("Enhanced C5 tensor required")
        n=len(self.c5_sam)
        _identities(self.identities,n)
        for name,channels,size in (("c2",256,112),("c3",512,56),("c4",1024,28),("c5_sam",2048,14)):
            value=getattr(self,name)
            if (not isinstance(value,torch.Tensor) or value.shape!=(n,channels,size,size)
                    or not value.is_floating_point() or value.dtype!=self.c5_sam.dtype
                    or value.device!=self.c5_sam.device or not torch.isfinite(value).all()):
                raise ValueError("Corner feature shape/dtype/device/finiteness mismatch: "+name)


@dataclass(frozen=True)
class CornerLocalizationOutput:
    logits: torch.Tensor
    identities: tuple
    grid: int = 224
    protocol_id: str = "action4.corner_localization_fpn.v1"


class CornerLocalizationHead(nn.Module):
    """Enhanced C5->P5; C4/C3/C2 lateral sums->P2->8x224 logits.

    No mask hard multiplication. No second RGB backbone. Corner quality,
    validity/scale losses and admitted final-pose refinement are NOT supplied.
    """
    def __init__(self):
        super().__init__()
        self.p5=nn.Conv2d(2048,128,1)
        self.lateral4=nn.Conv2d(1024,128,1)
        self.lateral3=nn.Conv2d(512,128,1)
        self.lateral2=nn.Conv2d(256,128,1)
        self.localization=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU(),
                                        nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU())
        self.heatmap=nn.Conv2d(128,8,1)

    def forward(self,features,*,identities,return_features=False):
        if type(return_features) is not bool:raise ValueError("Explicit feature-return flag required")
        if not isinstance(features,SamCornerFeatures):raise ValueError("Typed enhanced corner features required")
        features.validate()
        if _identities(identities,len(features.c5_sam))!=features.identities:
            raise ValueError("Corner instance order/identity mismatch")
        if features.c5_sam.device!=self.p5.weight.device or features.c5_sam.dtype!=self.p5.weight.dtype:
            raise ValueError("Corner head dtype/device mismatch")
        p=self.p5(features.c5_sam)
        for lower,lateral in ((features.c4,self.lateral4),(features.c3,self.lateral3),(features.c2,self.lateral2)):
            p=F.interpolate(p,size=lower.shape[-2:],mode="bilinear",align_corners=False)+lateral(lower)
        p=F.interpolate(p,size=(224,224),mode="bilinear",align_corners=False)
        spatial=self.localization(p)
        logits=self.heatmap(spatial)
        if not torch.isfinite(logits).all():raise ValueError("Nonfinite corner logits")
        output=CornerLocalizationOutput(logits,features.identities)
        return (output,spatial) if return_features else output


def local_corner_uv(output):
    """7x7 peak-local soft argmax in224 pixel centers; diagnostic, not pose admission."""
    if not isinstance(output,CornerLocalizationOutput) or output.logits.shape[1:]!=(8,224,224):
        raise ValueError("Fixed8 corner logits224 required")
    logits=output.logits
    if not torch.isfinite(logits).all():raise ValueError("Nonfinite corner logits")
    peaks=logits.flatten(2).argmax(-1)
    y,x=torch.meshgrid(torch.arange(224,device=logits.device,dtype=logits.dtype),
                       torch.arange(224,device=logits.device,dtype=logits.dtype),indexing="ij")
    valid=((x[None,None]-peaks.remainder(224)[...,None,None]).abs()<=3)&(
           (y[None,None]-peaks.div(224,rounding_mode="floor")[...,None,None]).abs()<=3)
    probabilities=logits.masked_fill(~valid,float("-inf")).flatten(2).softmax(-1).view_as(logits)
    return torch.stack(((probabilities*x).sum((-2,-1)),(probabilities*y).sum((-2,-1))),-1)
