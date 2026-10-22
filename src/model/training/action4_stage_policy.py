"""Explicit optional B0 trainability policy; preserve A0 features while learning detector heads."""
def configure_stage_trainability(model,stage,stage_config):
 policy=stage_config.get('shared_feature_policy','joint')
 if policy=='joint':
  if hasattr(model,'roi_branch'):raise ValueError('Independent ROI topology requires its explicit stage policy')
  return {'policy':'joint','frozen_parameters':[]}
 if policy=='independent_A0_roi':
  if stage!='B0':raise ValueError('Independent ROI policy currently applies only to B0')
  model.enable_independent_roi_features()
  if not all(p.requires_grad for name in ('backbone','decoder','center_head','offset_head','box_head') for p in getattr(model,name).parameters()):raise ValueError('Detector feature branch must remain trainable')
  names=[name for name,p in model.named_parameters() if name.split('.')[0] in ('roi_branch','corner_head')]
  if any(p.requires_grad for name,p in model.named_parameters() if name in names):raise ValueError('ROI branch must remain frozen')
  return {'policy':policy,'frozen_parameters':names,'frozen_state_prefixes':['roi_branch','corner_head'],'scope':'Detector features trainable, copied A0 ROI features frozen; accuracy unproven'}
 if policy!='frozen_A0_features':raise ValueError('Unknown explicit shared-feature policy')
 if stage!='B0':raise ValueError('Frozen A0 feature policy applies only to B0')
 if getattr(model,'freeze_bn',None) is not True:raise ValueError('Frozen backbone BatchNorm statistics required')
 required=('backbone','decoder','corner_head','center_head','offset_head','box_head')
 if any(not hasattr(model,name) for name in required):raise ValueError('Expected original Action4 module interfaces')
 names=[]
 for prefix in ('backbone','decoder','corner_head'):
  for name,p in getattr(model,prefix).named_parameters():p.requires_grad_(False);names.append(prefix+'.'+name)
 for prefix in ('center_head','offset_head','box_head'):
  if not all(p.requires_grad for p in getattr(model,prefix).parameters()):raise ValueError('Detector heads must remain trainable')
 return {'policy':policy,'frozen_parameters':names,'scope':'A0 shared features/corners preserved; detector head optimization only. Does not establish accuracy.'}


def frozen_state_digest(model,receipt):
 if receipt['policy']=='joint':return None
 import hashlib
 h=hashlib.sha256()
 for name,value in model.state_dict().items():
  if name.split('.')[0] in receipt.get('frozen_state_prefixes',('backbone','decoder','corner_head')):
   tensor=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.numpy().tobytes())
 return h.hexdigest()


def prepare_checkpoint_topology(model,state,*,expected_policy=None):
 """Explicit checkpoint policy creates modules before strict state loading."""
 receipt=state.get('identity',{}).get('trainability_policy',{})
 policy=receipt.get('policy','joint')
 if policy not in ('joint','frozen_A0_features','independent_A0_roi'):raise ValueError('Unknown checkpoint trainability policy')
 if expected_policy is not None and policy!=expected_policy:raise ValueError('Resume trainability policy changed')
 has_extra=any(name.startswith('roi_branch.') for name in state['model'])
 if policy=='independent_A0_roi':
  if not has_extra:raise ValueError('Independent checkpoint lacks ROI branch weights')
  model.enable_independent_roi_features()
 elif has_extra:raise ValueError('ROI weights present without explicit independent policy')
 return policy
