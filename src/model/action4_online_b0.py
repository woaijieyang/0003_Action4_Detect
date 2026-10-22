"""Joint B0 detector and box-quality refinement; independent of A0 weights."""
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_component_experiments import Candidate
from model.action4_box_refinement import BoxRefinement

class OnlineB0(nn.Module):
    def __init__(self,network):
        super().__init__()
        self.detector=Candidate("B0",network,"giou_quality")
        self.refiner=BoxRefinement()
    def forward(self,images):
        self.detector._check(images)
        f=self.detector.decoder(self.detector.backbone(images))
        dense={"center_logits":self.detector.center_head(f),
               "offset":self.detector.offset_head(f),
               "ltrb":F.softplus(self.detector.box_head(f)),
               "quality_logits":self.detector.quality_head(f)}
        return dense,f

def proposal_features(feature,boxes,canvas_size):
    """Bilinear9point box sampling with fixed proposal locations.

    Coordinates use the same pixel-centre convention as existing box_features.
    Selection/box coordinates are detached; feature gradients remain live.
    """
    width,height=canvas_size
    if feature.shape[0]!=1:raise ValueError("One image per proposal pool")
    boxes=boxes.detach().to(feature);extent=boxes[:,2:]-boxes[:,:2]
    if not len(boxes):return feature.new_empty((0,feature.shape[1]*9)),feature.new_empty((0,4))
    if not torch.isfinite(boxes).all() or not (extent>0).all():raise ValueError("Invalid candidate boxes")
    q=feature.new_tensor([0.,.5,1.]);x=boxes[:,0,None]+extent[:,0,None]*q;y=boxes[:,1,None]+extent[:,1,None]*q
    x=x[:,None,:].expand(-1,3,-1).reshape(-1);y=y[:,:,None].expand(-1,-1,3).reshape(-1)
    fh,fw=feature.shape[-2:];x=(x+.5)*(fw/width)-.5;y=(y+.5)*(fh/height)-.5
    x0=x.floor().long();y0=y.floor().long();dx=x-x0;dy=y-y0
    pooled=feature.new_zeros((feature.shape[1],len(x)))
    for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx=x0+ox;yy=y0+oy;inside=(xx>=0)&(yy>=0)&(xx<fw)&(yy<fh)
        index=yy.clamp(0,fh-1)*fw+xx.clamp(0,fw-1)
        pooled=pooled+feature[0].flatten(1).index_select(1,index)*(weight*inside)[None]
    pooled=pooled.reshape(feature.shape[1],len(boxes),9).permute(1,0,2).flatten(1)
    size=extent/feature.new_tensor([width,height])
    geometry=torch.stack((size[:,0],size[:,1],torch.log(size[:,0]/size[:,1]),torch.log(size.prod(-1))),-1)
    return pooled,geometry

from model.action4_proposal_pool import candidate_union

def proposals(model,dense,feature,affine,cfg):
    # Discrete candidate selection intentionally carries no box/score gradient.
    # Detector localization still trains through its own dense losses.
    with torch.no_grad():
        rows,canvas=candidate_union({k:v.detach() for k,v in dense.items()},affine,cfg)
    f,g=proposal_features(feature,canvas,tuple(cfg["preprocess"]["full_size"]))
    boxes=f.new_tensor([p["bbox_full_xyxy"] for p in rows]).reshape(-1,4)
    priors=f.new_tensor([[p["center_score"],p["old_quality_score"],p["old_product_score"]] for p in rows]).reshape(-1,3)
    return rows,boxes,model.refiner(f,g,priors,boxes)
