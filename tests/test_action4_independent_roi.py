import copy,tempfile,unittest,torch
from pathlib import Path
from unittest.mock import patch
from torch import nn
from test_action4_training_eval import TinyControlNet,fixture
from model.action4_network import Action4Net
from model.training.action4_trainer import train
from model.training.action4_runtime import load_checkpoint
from model.training.action4_stage_policy import prepare_checkpoint_topology
from eval.action4_runner import build_model
R=Path('/data/zy/vln/a-test');F=R/'outputs/action4_independent_roi_test_fixtures';F.mkdir(exist_ok=True)
class TinySeparated(TinyControlNet):
 def __init__(self,**kw):super().__init__(**kw);self.decoder=nn.Identity();self.freeze_bn=True;self.configuration={}
 enable_independent_roi_features=Action4Net.enable_independent_roi_features
 def train(self,mode=True):
  super().train(mode)
  if hasattr(self,'roi_branch'):self.roi_branch.eval()
  return self
 def forward(self,images,roi_images=None,*,roi_builder=None):
  result=super().forward(images,roi_images=None) if images is not None else {}
  if roi_builder is not None:roi_images,metadata=roi_builder({k:v.detach() for k,v in result.items() if k!='corner_logits'});result['roi_metadata']=metadata
  if roi_images is not None:
   if len(roi_images):
    result['corner_logits']=self.roi_branch(roi_images) if hasattr(self,'roi_branch') else self.corner_head(self.backbone(roi_images))
   else:result['corner_logits']=roi_images.new_empty((0,8,roi_images.shape[-2]//4,roi_images.shape[-1]//4))+self.corner_head.weight.sum()*0
  return result
class Tests(unittest.TestCase):
 def test_training_resume_optimizer_and_eval_load(self):
  with tempfile.TemporaryDirectory(dir=F) as temp,patch('model.action4_network.Action4Net',TinySeparated):
   root=Path(temp);cfg=fixture(root);train(root,cfg,stage='A0',run_id='a0',component_smoke=True,allow_diagnostic=True);init=root/'models/action4/a0/best.pt';original=load_checkpoint(init);cfg['stages']['B0']['shared_feature_policy']='independent_A0_roi'
   full=train(root,cfg,stage='B0',run_id='full',init_weights=init,component_smoke=True,allow_diagnostic=True)
   train(root,cfg,stage='B0',run_id='resume',init_weights=init,component_smoke=True,allow_diagnostic=True,max_updates=1);last=root/'models/action4/resume/last.pt'
   resumed=train(root,cfg,stage='B0',run_id='resume',resume=last,component_smoke=True,allow_diagnostic=True,max_updates=2);a=load_checkpoint(root/'models/action4/full/last.pt');b=load_checkpoint(last)
   self.assertTrue(full['frozen_features_unchanged']);self.assertTrue(full['backbone_parameter_changed']);self.assertTrue(resumed['frozen_features_unchanged'])
   for name,value in a['model'].items():
    self.assertTrue(torch.equal(value,b['model'][name]),name)
    if name.startswith('roi_branch.'):self.assertTrue(torch.equal(value,original['model'][name.removeprefix('roi_branch.')]),name)
   self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
   for i,values in a['optimizer']['state'].items():
    for k,v in values.items():
     if isinstance(v,torch.Tensor):self.assertTrue(torch.equal(v,b['optimizer']['state'][i][k]))
     else:self.assertEqual(v,b['optimizer']['state'][i][k])
   loaded=build_model(cfg,checkpoint=last,device='cpu');self.assertTrue(hasattr(loaded,'roi_branch'));self.assertEqual(set(loaded.state_dict()),set(b['model']))
   for k,v in loaded.state_dict().items():self.assertTrue(torch.equal(v,b['model'][k]))
   bad=copy.deepcopy(cfg);bad['stages']['B0']['shared_feature_policy']='joint'
   with self.assertRaises(ValueError):train(root,bad,stage='B0',run_id='resume',resume=last,component_smoke=True,allow_diagnostic=True,max_updates=2)
 def test_policy_weight_mismatch_rejected(self):
  m=TinySeparated();m.enable_independent_roi_features();state={'model':m.state_dict(),'identity':{'trainability_policy':{'policy':'joint'}}}
  with self.assertRaises(ValueError):prepare_checkpoint_topology(TinySeparated(),state)
  state={'model':TinySeparated().state_dict(),'identity':{'trainability_policy':{'policy':'independent_A0_roi'}}}
  with self.assertRaises(ValueError):prepare_checkpoint_topology(TinySeparated(),state)

import os,socket,json

def independent_ddp_worker(rank,root_text,port):
 root=Path(root_text);os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port));torch.set_num_threads(2);cfg=json.loads((root/'ddp_config.json').read_text())
 with patch('model.action4_network.Action4Net',TinySeparated):result=train(root,cfg,stage='B0',run_id='ddp',init_weights=root/'models/action4/a0/best.pt',allow_diagnostic=True,component_smoke=True,max_updates=1)
 (root/f'rank{rank}.json').write_text(json.dumps(result))

class IndependentDDP(unittest.TestCase):
 def test_independent_tail_empty_and_frozen_state(self):
  with tempfile.TemporaryDirectory(dir=F) as tmp:
   root=Path(tmp);cfg=fixture(root);path=Path(cfg['data']['train_manifest']);records=[json.loads(x) for x in path.read_text().splitlines()];empty=copy.deepcopy(records[0]);empty.update(sample_id='empty',group_id='empty',background_source_id='empty',instances=[]);records.append(empty);path.write_text(''.join(json.dumps(x)+'\n' for x in records))
   with patch('model.action4_network.Action4Net',TinySeparated):train(root,cfg,stage='A0',run_id='a0',allow_diagnostic=True,component_smoke=True,max_updates=1)
   original=load_checkpoint(root/'models/action4/a0/best.pt');cfg['stages']['B0'].update(shared_feature_policy='independent_A0_roi',global_batch=4);(root/'ddp_config.json').write_text(json.dumps(cfg))
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   torch.multiprocessing.spawn(independent_ddp_worker,args=(str(root),port),nprocs=2,join=True);results=[json.loads((root/f'rank{i}.json').read_text()) for i in range(2)];self.assertTrue(all(x['global_step']==1 and x['all_rank_model_states_equal'] and x['frozen_features_unchanged'] for x in results));self.assertEqual(results[0]['final_model_state_sha256'],results[1]['final_model_state_sha256']);state=load_checkpoint(root/'models/action4/ddp/last.pt');self.assertEqual(len(state['rng_by_rank']),2)
   for name,value in state['model'].items():
    if name.startswith('roi_branch.'):self.assertTrue(torch.equal(value,original['model'][name.removeprefix('roi_branch.')]))
