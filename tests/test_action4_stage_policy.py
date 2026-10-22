import unittest
import torch
from torch import nn
from model.training.action4_stage_policy import configure_stage_trainability, frozen_state_digest
class Toy(nn.Module):
 def __init__(self):
  super().__init__();self.freeze_bn=True
  for n in ['backbone','decoder','corner_head','center_head','offset_head','box_head']:setattr(self,n,nn.Linear(3,3))
class Tests(unittest.TestCase):
 def test_default_preserves_joint(self):
  m=Toy();r=configure_stage_trainability(m,'B0',{});self.assertEqual(r['policy'],'joint');self.assertTrue(all(p.requires_grad for p in m.parameters()))
 def test_only_detector_heads_receive_updates(self):
  m=Toy();configure_stage_trainability(m,'B0',{'shared_feature_policy':'frozen_A0_features'});before={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=.01);x=torch.ones(2,3);h=m.decoder(m.backbone(x));loss=sum(getattr(m,n)(h).square().mean() for n in ['center_head','offset_head','box_head'])+m.corner_head(h).square().mean();loss.backward();opt.step()
  for n,p in m.named_parameters():
   if n.split('.')[0] in ('backbone','decoder','corner_head'):self.assertTrue(torch.equal(p,before[n]));self.assertIsNone(p.grad)
   else:self.assertFalse(torch.equal(p,before[n]));self.assertIsNotNone(p.grad)
 def test_wrong_stage_rejected(self):
  with self.assertRaises(ValueError):configure_stage_trainability(Toy(),'A0',{'shared_feature_policy':'frozen_A0_features'})
 def test_unknown_policy_rejected(self):
  with self.assertRaises(ValueError):configure_stage_trainability(Toy(),'B0',{'shared_feature_policy':'guess'})
 def test_mutable_BN_rejected(self):
  m=Toy();m.freeze_bn=False
  with self.assertRaises(ValueError):configure_stage_trainability(m,'B0',{'shared_feature_policy':'frozen_A0_features'})

import tempfile,copy
from pathlib import Path
from unittest.mock import patch
from test_action4_training_eval import TinyControlNet,fixture
from model.training.action4_trainer import train
from model.training.action4_runtime import load_checkpoint
O=Path(__file__).resolve().parents[1]/"outputs/action4_stage_policy_test_fixtures"
O.mkdir(exist_ok=True)
class FrozenToy(TinyControlNet):
 def __init__(self,**kw):super().__init__(**kw);self.decoder=nn.Identity();self.freeze_bn=True
class FrozenIntegration(unittest.TestCase):
 def test_save_resume_and_exact_frozen_state(self):
  with tempfile.TemporaryDirectory(dir=O) as temp,patch('model.action4_network.Action4Net',FrozenToy):
   root=Path(temp);cfg=fixture(root);train(root,cfg,stage='A0',run_id='initial',allow_diagnostic=True,component_smoke=True);init=root/'models/action4/initial/best.pt';cfg['stages']['B0']['shared_feature_policy']='frozen_A0_features'
   whole=train(root,cfg,stage='B0',run_id='full',init_weights=init,allow_diagnostic=True,component_smoke=True)
   train(root,cfg,stage='B0',run_id='part',init_weights=init,allow_diagnostic=True,component_smoke=True,max_updates=1)
   latest=root/'models/action4/part/last.pt';resumed=train(root,cfg,stage='B0',run_id='part',resume=latest,allow_diagnostic=True,component_smoke=True,max_updates=2)
   a=load_checkpoint(root/'models/action4/full/last.pt');b=load_checkpoint(latest);original=load_checkpoint(init)
   self.assertTrue(whole['frozen_features_unchanged']);self.assertTrue(resumed['frozen_features_unchanged']);self.assertFalse(whole['backbone_parameter_changed']);self.assertEqual(a['identity']['trainability_policy']['policy'],'frozen_A0_features')
   for name,value in a['model'].items():
    self.assertTrue(torch.equal(value,b['model'][name]),name)
    if name.split('.')[0] in ('backbone','decoder','corner_head'):self.assertTrue(torch.equal(value,original['model'][name]),name)
   bad=copy.deepcopy(cfg);bad['stages']['B0']['shared_feature_policy']='joint'
   with self.assertRaises(ValueError):train(root,bad,stage='B0',run_id='part',resume=latest,allow_diagnostic=True,component_smoke=True,max_updates=2)
 def test_frozen_digest_detects_parameter_mutation(self):
  m=FrozenToy();r=configure_stage_trainability(m,'B0',{'shared_feature_policy':'frozen_A0_features'});before=frozen_state_digest(m,r)
  with torch.no_grad():next(m.corner_head.parameters()).add_(1)
  self.assertNotEqual(before,frozen_state_digest(m,r))

if __name__=="__main__":torch.set_num_threads(2);unittest.main()
