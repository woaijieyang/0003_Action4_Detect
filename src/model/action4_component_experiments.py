"""Incremental independent A0/B0 candidates. No legacy checkpoint mutation."""
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_components import Action4Component
from model.action4_decode import decode_corners
from model.action4_losses import detection_losses,gather_at_indices,corner_heatmap_loss

class Candidate(Action4Component):
 def __init__(self,role,network,variant):
  super().__init__(role,network);self.variant=variant
  legal={'B0':('baseline','giou','giou_quality'),'A0':('baseline','joint','joint_noedges')}
  if variant not in legal[role]:raise ValueError('Invalid role/variant')
  c=network['decoder_channels']
  if variant=='giou_quality':
   self.quality_head=nn.Sequential(nn.Conv2d(c,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,1,1))
   nn.init.zeros_(self.quality_head[-1].weight);nn.init.zeros_(self.quality_head[-1].bias)
  if variant.startswith('joint'):
   self.token_projection=nn.Linear(c*2+2,64);self.corner_identity=nn.Parameter(torch.randn(1,8,64)*.01)
   self.visibility_head=nn.Linear(64,1)
   layer=nn.TransformerEncoderLayer(64,4,128,dropout=0.,batch_first=True,norm_first=False)
   self.corner_relation=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
   self.correction_head=nn.Linear(64,2)
   nn.init.zeros_(self.correction_head.weight);nn.init.zeros_(self.correction_head.bias)
 def forward(self,images):
  self._check(images);f=self.decoder(self.backbone(images))
  if self.role=='B0':
   out={'center_logits':self.center_head(f),'offset':self.offset_head(f),'ltrb':F.softplus(self.box_head(f))}
   if self.variant=='giou_quality':out['quality_logits']=self.quality_head(f)
   return out
  logits=self.corner_head(f);out={'corner_logits':logits}
  if self.variant.startswith('joint'):
   # Legacy indexed channel decoding anchors; corrections jointly learn all8 IDs.
   uv=decode_corners(logits).detach();h,w=logits.shape[-2:]
   grid=2*(uv+0.5)/uv.new_tensor([w,h])-1
   # Gather nearest feature token avoids nondeterministic CUDA grid-sample backward.
   xy=uv.round().long();indices=xy[...,1].clamp(0,h-1)*w+xy[...,0].clamp(0,w-1)
   local=gather_at_indices(f,indices)
   globalf=f.mean((-2,-1))[:,None].expand(-1,8,-1)
   tokens=self.token_projection(torch.cat([local,globalf,uv/uv.new_tensor([w,h])],-1))+self.corner_identity
   visibility=self.visibility_head(tokens).squeeze(-1)
   # No GT visibility used in inference; floor avoids erasing uncertain evidence.
   fused=self.corner_relation(tokens*(.25+.75*visibility.sigmoid()[...,None]))
   out.update(corners_hm=uv+.25*uv.new_tensor([w,h])*self.correction_head(fused).tanh(),visibility_logits=visibility)
  return out
 @staticmethod
 def _check(x):
  from model.action4_network import Action4Net
  Action4Net._validate_images(x)

def aligned_iou_giou(a,b):
 lo=torch.maximum(a[...,:2],b[...,:2]);hi=torch.minimum(a[...,2:],b[...,2:]);inter=(hi-lo).clamp_min(0).prod(-1)
 area=lambda x:(x[...,2:]-x[...,:2]).clamp_min(0).prod(-1)
 union=(area(a)+area(b)-inter).clamp_min(1e-9);iou=inter/union
 enclosure=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1).clamp_min(1e-9)
 return iou,iou-(enclosure-union)/enclosure

def box_loss(outputs,targets,variant):
 losses=detection_losses(outputs,targets)
 if variant=='baseline':return losses
 mask=targets['instance_mask'];idx=targets['indices'].clamp_min(0);h,w=outputs['center_logits'].shape[-2:]
 anchor=torch.stack([idx%w,idx//w],-1).float();scale=anchor.new_tensor([w,h])
 pc=(anchor+gather_at_indices(outputs['offset'],idx)+.5)/scale
 gc=(anchor+targets['offset']+.5)/scale
 pd=gather_at_indices(outputs['ltrb'],idx);gd=targets['ltrb']
 pb=torch.cat([pc-pd[...,:2],pc+pd[...,2:]],-1);gb=torch.cat([gc-gd[...,:2],gc+gd[...,2:]],-1)
 iou,giou=aligned_iou_giou(pb,gb)
 losses['giou']=((1-giou)*mask).sum()/mask.sum().clamp_min(1)
 if variant=='giou_quality':
  # QFL-inspired continuous targets at positives, hard predicted candidates as negatives.
  # IoU targets detached: quality cannot reduce its target by damaging box regression.
  logit=outputs['quality_logits'];q=gather_at_indices(logit,idx)[...,0];t=iou.detach().clamp(0,1)
  pos=F.binary_cross_entropy_with_logits(q,t,reduction='none')*(q.sigmoid()-t).abs().square()
  pos=(pos*mask).sum()/mask.sum().clamp_min(1)
  heat=outputs['center_logits'].detach().sigmoid();peaks=heat*heat.eq(F.max_pool2d(heat,3,1,1));hard=peaks.flatten(1).topk(20,dim=1).indices
  qn=gather_at_indices(logit,hard)[...,0]
  # Dynamic hard-candidate IoU labels: a good off-centre box is not a background label.
  ha=torch.stack([hard%w,hard//w],-1).float();hc=(ha+gather_at_indices(outputs['offset'],hard)+.5)/scale
  hd=gather_at_indices(outputs['ltrb'],hard);hb=torch.cat([hc-hd[...,:2],hc+hd[...,2:]],-1)
  hiou,_=aligned_iou_giou(hb[:,:,None],gb[:,None]);ht=(hiou.detach()*mask[:,None]).amax(-1).clamp(0,1)
  bg=torch.ones_like(ht,dtype=torch.bool)
  if 'negative_mask' in targets:bg=gather_at_indices(targets['negative_mask'].float(),hard)[...,0].bool() | (ht>0)
  neg=(F.binary_cross_entropy_with_logits(qn,ht,reduction='none')*(qn.sigmoid()-ht).abs().square()*bg).sum()/bg.sum().clamp_min(1)
  losses['quality']=pos+neg
 return losses

def corner_loss(outputs,meta,variant):
 loss={'heatmap':corner_heatmap_loss(outputs['corner_logits'],meta['targets'],meta['valid'],meta['roi_batch_index'],meta['batch_size'])}
 if variant.startswith('joint'):
  valid=meta['valid'];visible=meta['visible'];err=F.smooth_l1_loss(outputs['corners_hm']/meta['diagonal'][:,None,None],meta['points_hm']/meta['diagonal'][:,None,None],beta=.05,reduction='none').mean(-1)
  # Separate normalizers: numerous visible points cannot drown hidden supervision.
  for name,mask in [('visible',visible&valid),('hidden',~visible&valid)]:loss[name]=(err*mask).sum()/mask.sum().clamp_min(1)
  loss['visibility']=.05*F.binary_cross_entropy_with_logits(outputs['visibility_logits'],visible.float())
  # Edge displacement consistency, no false parallel-edge assumption under perspective.
  edges=torch.tensor([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],device=valid.device)
  p=outputs['corners_hm'];t=meta['points_hm'];e=F.smooth_l1_loss((p[:,edges[:,0]]-p[:,edges[:,1]])/meta['diagonal'][:,None,None],(t[:,edges[:,0]]-t[:,edges[:,1]])/meta['diagonal'][:,None,None],beta=.05,reduction='none').mean(-1)
  em=valid[:,edges[:,0]]&valid[:,edges[:,1]];loss['edges']=(.25 if variant=='joint' else 0.)*(e*em).sum()/em.sum().clamp_min(1)
 return loss

def quality_decode(outputs,affines,cfg,variant):
 from eval.action4_runner import decode_full_boxes
 if variant!='giou_quality':return decode_full_boxes(outputs,affines,cfg)
 # Re-rank BEFORE top-k selection, so third candidates can enter final two.
 out=dict(outputs);s=outputs['center_logits'].sigmoid()*outputs['quality_logits'].sigmoid()
 out['center_logits']=torch.logit(s.clamp(1e-7,1-1e-7))
 return decode_full_boxes(out,affines,cfg)
