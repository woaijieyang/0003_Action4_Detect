import copy,tempfile,unittest,torch,numpy as np
from pathlib import Path
from unittest.mock import patch
from test_action4_training_eval import TinyControlNet,fixture
from model.training.action4_trainer import train,_jitter_box,_roi_builder
from model.training.action4_initialization import validate_weights_initialization
from model.training.action4_runtime import load_checkpoint
R=Path('/data/zy/vln/a-test');F=R/'outputs/action4_roi_adaptation_test_fixtures';F.mkdir(exist_ok=True)
class Tests(unittest.TestCase):
 def test_explicit_migration_only_and_no_detector_change(self):
  with tempfile.TemporaryDirectory(dir=F) as tmp:
   cfg=fixture(Path(tmp));state={'stage':'A0','config':copy.deepcopy(cfg)}
   with self.assertRaises(ValueError):validate_weights_initialization('A0',cfg,state)
   cfg['stages']['A0']['weights_only_roi_migration']=True;cfg['preprocess'].update(roi_size=64,roi_expansion=2.25);result=validate_weights_initialization('A0',cfg,state);self.assertEqual(result['changed_preprocess_fields'],['roi_expansion','roi_size']);self.assertEqual(state['config']['preprocess']['roi_size'],32)
   with self.assertRaises(ValueError):validate_weights_initialization('B0',cfg,state)
   bad=copy.deepcopy(cfg);bad['preprocess']['full_size']=[128,64]
   with self.assertRaises(ValueError):validate_weights_initialization('A0',bad,state)
   bad=copy.deepcopy(state);bad['stage']='B0'
   with self.assertRaises(ValueError):validate_weights_initialization('A0',cfg,bad)
 def test_legacy_jitter_exact_and_declared_bounds(self):
  box=np.array([10.,20.,50.,40.]);a=np.random.default_rng(17);b=np.random.default_rng(17);size=box[2:]-box[:2];center=(box[:2]+box[2:])/2+b.uniform(-.05,.05,2)*size;sz=size*b.uniform(.9,1.1);self.assertTrue(np.array_equal(_jitter_box(box,a),np.r_[center-sz/2,center+sz/2]))
  rng=np.random.default_rng(3)
  for _ in range(100):
   out=np.asarray(_jitter_box(box,rng,center_fraction=.15,scale_range=[.65,1.35]));delta=((out[:2]+out[2:])-(box[:2]+box[2:]))/(2*size);ratio=(out[2:]-out[:2])/size;self.assertTrue((np.abs(delta)<=.15+1e-12).all());self.assertTrue(((ratio>=.65)&(ratio<=1.35)).all())
  with self.assertRaises(ValueError):_jitter_box(box,rng,scale_range=[0,1])
 def test_warmstart_jitter_exact_resume_and_metadata(self):
  with tempfile.TemporaryDirectory(dir=F) as tmp,patch('model.action4_network.Action4Net',TinyControlNet):
   root=Path(tmp);cfg=fixture(root);train(root,cfg,stage='A0',run_id='old',component_smoke=True,allow_diagnostic=True);init=root/'models/action4/old/best.pt';cfg['preprocess'].update(roi_size=64,roi_expansion=2.25);cfg['stages']['A0'].update(weights_only_roi_migration=True,roi_jitter=True,roi_jitter_parameters={'center_fraction':.15,'scale_range':[.65,1.35]})
   train(root,cfg,stage='A0',run_id='full',init_weights=init,component_smoke=True,allow_diagnostic=True);train(root,cfg,stage='A0',run_id='part',init_weights=init,component_smoke=True,allow_diagnostic=True,max_updates=1);last=root/'models/action4/part/last.pt';train(root,cfg,stage='A0',run_id='part',resume=last,component_smoke=True,allow_diagnostic=True,max_updates=2);a=load_checkpoint(root/'models/action4/full/last.pt');b=load_checkpoint(last)
   self.assertEqual(a['initialization']['mode'],'weights_only_A0_roi_migration_new_optimizer');self.assertEqual(a['initialization']['source_preprocess']['roi_size'],32);self.assertEqual(a['initialization']['target_preprocess']['roi_size'],64);self.assertEqual(a['roi_rng_by_rank'],b['roi_rng_by_rank'])
   for name,v in a['model'].items():self.assertTrue(torch.equal(v,b['model'][name]),name)
   with self.assertRaises(ValueError):train(root,cfg,stage='A0',run_id='missing',component_smoke=True,allow_diagnostic=True)
 def test_fixed_D0_rejects_jitter(self):
  with tempfile.TemporaryDirectory(dir=F) as tmp:
   root=Path(tmp);cfg=fixture(root);cfg['stages']['D0']['roi_jitter']=True
   with self.assertRaises(ValueError):train(root,cfg,stage='D0',run_id='bad',component_smoke=True,allow_diagnostic=True)
