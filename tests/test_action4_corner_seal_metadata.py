"""Corner seal metadata fixtures only; never real model/seal acceptance."""
import unittest,copy
class CornerSealMetadataTests(unittest.TestCase):
 def setUp(self):
  from test_action4_role_seal_metadata import RoleSealMetadataTests,seal
  self.fixture=RoleSealMetadataTests();self.fixture.setUp();self.seal=seal;f=self.fixture
  f.cfg.update(role='A0',training_protocol=seal.CORNER_NATIVE_PROTOCOL,a0_variant='a0_corner_image',
   corner_softmax_bias_frozen=True,train_backbone=False,formal_training_units=1,train_indices=None,eval_every=25,
   score_schema='action4.pose_projection_all8_002.v1')
  f.write('snapshot/resolved.json',f.cfg);f.spec['files']['resolved.json']=seal.digest(f.code/'resolved.json');f.write('spec.json',f.spec);f.cid=seal.canonical(f.cfg)
  for p in (f.out/'eval').iterdir():p.unlink()
  f.train.update(role='A0',config_id=f.cid);f.controller.update(role='A0',config_id=f.cid,best_step=50)
  f.evaluation.update(config_id=f.cid,all_steps=[0,25,50,75,80],best_step=50)
  for step in f.evaluation['all_steps']:
   raw=dict(score_schema=f.cfg['score_schema'],score=2. if step==0 else .5,evaluation_images=650,evaluation_wall_seconds=.1,
    fixture_only=True)
   cal_path=f'output/eval_runs/step_{step:08d}_fixture_0/CALIBRATION.json'
   image_ids=[f'fixture_{i}' for i in range(650)]
   f.write(cal_path,{'raw_report':{'image_ids':image_ids},'candidate_report':{'image_ids':image_ids}})
   raw['calibration_receipt']={'path':cal_path,'sha256':seal.digest(f.root/cal_path)}
   path=f'output/eval_runs/step_{step:08d}_fixture_0/REPORT.json';f.write(path,raw)
   request=dict(step=step,config_id=f.cid,role='A0',snapshot=f'output/eval_step_{step:08d}.pt')
   done=dict(step=step,config_id=f.cid,role='A0',training_mode='full_training',report=raw,raw_report=path,checkpoint='output/best.pt',deployment_eligible=step==50)
   f.write(f'output/eval/step_{step:08d}.json',request);f.write(f'output/eval/step_{step:08d}.done.json',done)
   if step==50:f.best=done
  f.terminals()
 def tearDown(self):self.fixture.tearDown()
 def rank(self,row):
  if row['deployment_eligible'] is not True:raise AssertionError('Unready corner passed to rank')
  return (row['report']['score'],-row['step'])
 def test_only_admitted_candidates_ranked_and_full_payload_retained(self):
  called=[]
  def validator(cfg,reports,best):called.append((len(reports),best));return {'fixture_only':True,'quality_ready':True}
  state=self.seal.inspect('spec.json',rank_fn=self.rank,corner_validator=validator)[3]
  self.assertEqual(state['best_step'],50);self.assertEqual(called,[(5,50)])
 def test_no_admitted_corner_not_sealable(self):
  f=self.fixture;f.best['deployment_eligible']=False;f.write('output/eval/step_00000050.done.json',f.best);f.terminals()
  with self.assertRaisesRegex(ValueError,'No admitted'):self.seal.inspect('spec.json',rank_fn=self.rank)
 def test_timing_stripped_corner_report_rejected(self):
  f=self.fixture;r=f.read(f.best['raw_report']);r.pop('evaluation_wall_seconds');f.write(f.best['raw_report'],r)
  with self.assertRaisesRegex(ValueError,'Raw/report'):self.seal.inspect('spec.json',rank_fn=self.rank)

class ImmutableLoaderCompatibilityTests(unittest.TestCase):
 def test_old_three_argument_loader_remains_callable(self):
  from test_action4_role_seal_metadata import seal
  def old(path,role,device):return path,role,device
  self.assertEqual(seal.strict_load_selected(old,'model.pt',{'role':'A0'}),('model.pt','A0','cpu'))
 def test_corner_permission_is_explicit_only_on_corner_path(self):
  from test_action4_role_seal_metadata import seal
  def loader(path,role,device,*,allow_corner_candidate=False):return allow_corner_candidate
  self.assertTrue(seal.strict_load_selected(loader,'model.pt',{'role':'A0','training_protocol':seal.CORNER_NATIVE_PROTOCOL}))
  self.assertFalse(seal.strict_load_selected(loader,'model.pt',{'role':'A0','training_protocol':seal.SAM_NATIVE_PROTOCOL}))
