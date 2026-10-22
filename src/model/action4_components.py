"""Independent, trainable A0/B0 components with role-checked weight artifacts.

Legacy combined checkpoints remain untouched. Exported artifacts are weights
only, not optimizer-resume checkpoints. No parameter sharing between roles.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_network import Action4Net

PREFIXES={'A0':('backbone','decoder','corner_head'),
          'B0':('backbone','decoder','center_head','offset_head','box_head')}

class Action4Component(nn.Module):
 def __init__(self,role,network):
  super().__init__()
  if role not in PREFIXES:raise ValueError('Expected A0 or B0 role')
  self.role=role;self.network=dict(network)
  base=Action4Net(weights=None,**network)
  self.freeze_bn=base.freeze_bn
  for name in PREFIXES[role]:self.add_module(name,getattr(base,name))
  self.train(True)
 def train(self,mode=True):
  super().train(mode)
  if self.freeze_bn:
   for m in self.backbone.modules():
    if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
  return self
 def forward(self,images):
  Action4Net._validate_images(images)
  features=self.decoder(self.backbone(images))
  if self.role=='A0':return {'corner_logits':self.corner_head(features)}
  return {'center_logits':self.center_head(features),'offset':self.offset_head(features),
          'ltrb':F.softplus(self.box_head(features))}

def component_state(legacy,role):
 if role not in PREFIXES:raise ValueError('Unknown component role')
 embedded=role=='A0' and any(k.startswith('roi_branch.') for k in legacy['model'])
 source={k.removeprefix('roi_branch.'):v for k,v in legacy['model'].items() if k.startswith('roi_branch.')} if embedded else legacy['model']
 return {k:v.detach().cpu().clone() for k,v in source.items() if k.split('.')[0] in PREFIXES[role]}

def save_component(path,model,*,preprocess,provenance):
 from pathlib import Path
 from model.training.action4_runtime import save_checkpoint_atomic
 from data.action4_paths import PROJECT_ROOT,contained_path
 path=contained_path(PROJECT_ROOT,Path(path),exists=False)
 if path.exists():raise FileExistsError(path)
 save_checkpoint_atomic(path,dict(schema_version='action4.component_weights.v1',role=model.role,
  network=model.network,preprocess=preprocess,model=model.state_dict(),provenance=provenance,
  resumable=False,purpose='independent component weights; optimizer state not included'))

def load_component(path,*,role,device='cpu'):
 from pathlib import Path
 from data.action4_paths import PROJECT_ROOT,contained_path
 state=torch.load(contained_path(PROJECT_ROOT,Path(path),exists=True),map_location='cpu',weights_only=True)
 if state.get('schema_version')!='action4.component_weights.v1' or state.get('role')!=role:raise ValueError('Checkpoint component role/schema mismatch')
 model=Action4Component(role,state['network']);model.load_state_dict(state['model'],strict=True)
 return model.to(device),state

class IndependentPipeline(nn.Module):
 """Compatibility bridge for predict_images; routing only, never joint optimization."""
 def __init__(self,detector,corners):
  super().__init__()
  if detector.role!='B0' or corners.role!='A0':raise ValueError('B0 detector and A0 corners required')
  if {p.data_ptr() for p in detector.parameters()} & {p.data_ptr() for p in corners.parameters()}:raise ValueError('Components must not share parameter storage')
  self.detector=detector;self.corners=corners
 def forward(self,images,roi_images=None):
  result={} if images is None else self.detector(images)
  if roi_images is not None:result.update(self.corners(roi_images))
  return result
