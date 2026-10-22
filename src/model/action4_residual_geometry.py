import torch
from torch import nn
from torch.nn import functional as F
class ResidualGeometry(nn.Module):
 def __init__(self):
  super().__init__();self.image=nn.Sequential(nn.Linear(4096,256),nn.LayerNorm(256),nn.ReLU());self.visibility=nn.Linear(256,8);self.delta=nn.Sequential(nn.Linear(256+32,128),nn.ReLU(),nn.Linear(128,9));nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
 def forward(self,features,observed_uv,peak_scores,K_roi,canonical_m,initial_rotation,initial_translation):
  h=self.image(features);v=self.visibility(h);w=(.05+.95*v.sigmoid())*(.05+.95*peak_scores);x=torch.cat([h,(observed_uv*w[...,None]).flatten(1),peak_scores,v.sigmoid()],1);d=self.delta(x);identity=d.new_tensor([1,0,0,0,1,0])[None];six=identity+.3*torch.tanh(d[:,:6]);a=F.normalize(six[:,:3],dim=1);b=F.normalize(six[:,3:]-(six[:,3:]*a).sum(1,keepdim=True)*a,dim=1);c=torch.cross(a,b,dim=1);rot=torch.stack([a,b,c],-1)@initial_rotation
  radius=canonical_m.norm(dim=-1).max(-1).values;z=(initial_translation[:,2]*torch.exp(.35*torch.tanh(d[:,8]))).clamp_min(radius+.01);hom=torch.einsum('bij,bj->bi',K_roi,initial_translation);center=hom[:,:2]/hom[:,2:];center=center+448*.1*torch.tanh(d[:,6:8]);ray=torch.linalg.solve(K_roi,torch.cat([center,torch.ones_like(center[:,:1])],1)[...,None]).squeeze(-1);t=ray/ray[:,2:]*z[:,None];cam=canonical_m@rot.transpose(1,2)+t[:,None];pix=cam@K_roi.transpose(1,2);uv=pix[:,:,:2]/pix[:,:,2:]/448
  return {'corners_uv':uv,'visibility_logits':v,'rotation':rot,'translation':t,'camera_corners':cam,'delta':d}
def loss(out,b):
 err=F.smooth_l1_loss((out['corners_uv']-b['corners_uv'])/b['diagonal'][:,None,None],torch.zeros_like(out['corners_uv']),reduction='none',beta=.05).mean(-1);vis=b['visible'].bool();a=err[vis].mean();h=err[~vis].mean();v=F.binary_cross_entropy_with_logits(out['visibility_logits'],b['visible']);reg=out['delta'].square().mean();return {'total':a+h+.02*v+.001*reg,'visible':a,'hidden':h,'visibility':v,'residual_regularization':reg}
