"""CPU control-plane fixtures only; no real plan/scene/label acceptance or GPU jobs."""
from pathlib import Path
import copy,hashlib,json,tempfile,unittest
from data import action4_execution_authority as authority
ROOT=Path(__file__).resolve().parents[1]
class ExecutionAuthorityTests(unittest.TestCase):
 def setUp(self):
  base=ROOT/'outputs/action4_execution_authority_test_fixtures';base.mkdir(exist_ok=True);self.root=Path(tempfile.mkdtemp(prefix='case_',dir=base));(self.root/'configs/execution').mkdir(parents=True);(self.root/'inputs').mkdir()
  self.source=self.write('inputs/source.json',{'fixture_only':True});self.rows=[{'entry':{'index':i,'sample_id':f'fixture_{i}'}} for i in range(2000)];self.params=self.write('inputs/params.json',{'entries':self.rows});self.geometry=self.write('inputs/geometry.json',{'status':'128_GEOMETRY_SELECTED_MASKS_PENDING','selected':128,'selected_rows':self.rows[:128]})
  plan={'count':2000,'entries':[x['entry'] for x in self.rows],'recipe':{'modular_backend':{'source_context_ref':self.source,'parameters_release_ref':self.params}}};self.plan=self.write('inputs/plan.json',plan)
  self.a={'schema':'action4.execution_authority.v1','status':'ACTIVE','phase':'render128','source_context_ref':self.source,'parameters_release_ref':self.params,'render':{'enabled':True,'entrypoint':'scripts/action4_render_cached.py','plan_ref':self.plan,'index_range':[0,128],'output_parent':'data/candidates','output_name_prefix':'preview_','geometry_ref':self.geometry},'training':{'enabled':False}};self.publish()
 def write(self,name,value):
  p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return authority.ref(self.root,name)
 def publish(self):
  r=self.write('configs/execution/revision.json',self.a);self.write(authority.POINTER,{'schema':'action4.active_execution_pointer.v1','active_ref':r})
 def check(self,**kw):
  args=dict(root=self.root,entrypoint='scripts/action4_render_cached.py',plan_path=self.plan['path'],output='data/candidates/preview_shard_00',shard_index=0,num_shards=36,entry_count=4);args.update(kw);return authority.authorize_render(**args)
 def test_current_exact_phase(self):self.assertGreaterEqual(len(self.check()),5)
 def test_old_plan_rejected(self):
  other=self.write('inputs/old_plan.json',authority.readref(self.root,self.plan))
  with self.assertRaisesRegex(ValueError,'Old or unapproved'):self.check(plan_path=other['path'])
 def test_legacy_entry_rejected(self):
  with self.assertRaisesRegex(ValueError,'Legacy'):self.check(entrypoint='scripts/action4_synthesis.py')
 def test_preparation_blocks_render(self):
  self.a['status']='PREPARATION_ONLY';self.publish()
  with self.assertRaisesRegex(ValueError,'does not permit'):self.check()
 def test_bulk_indices_cannot_enter128(self):
  with self.assertRaisesRegex(ValueError,'range crosses'):self.check(entry_start=4,entry_count=1)
 def test_output_namespace(self):
  with self.assertRaisesRegex(ValueError,'namespace'):self.check(output='data/old/preview_shard_00')
 def test_mixed_source_ref_rejected(self):
  self.a['source_context_ref']=self.write('inputs/other_source.json',{'fixture_only':'other'});self.publish()
  with self.assertRaisesRegex(ValueError,'source/parameter'):self.check()
 def test_selected_parameters_must_match(self):
  self.geometry=self.write('inputs/bad_geometry.json',{'status':'128_GEOMETRY_SELECTED_MASKS_PENDING','selected':128,'selected_rows':self.rows[1:129]});self.a['render']['geometry_ref']=self.geometry;self.publish()
  with self.assertRaisesRegex(ValueError,'selected rows'):self.check()
 def test_pointer_digest_mismatch(self):
  self.write(authority.POINTER,{'schema':'action4.active_execution_pointer.v1','active_ref':{'path':'configs/execution/revision.json','sha256':'0'*64}})
  with self.assertRaisesRegex(ValueError,'reference changed'):self.check()
 def test_training_disabled(self):
  with self.assertRaisesRegex(ValueError,'does not permit training'):authority.spec_references(self.root,{'kind':'train','argv':['.venv/bin/python','scripts/action4_train.py','--stage','D0','--config','config.json']})

 def test_phase_cannot_broaden_its_range(self):
  self.a['render']['index_range']=[0,2000];self.publish()
  with self.assertRaisesRegex(ValueError,'Phase range'):self.check(entry_start=4,entry_count=1)
 def train_fixture(self,old_source=False):
  plan=authority.readref(self.root,self.plan)
  if old_source:plan['recipe']['modular_backend']['source_context_ref']=self.write('inputs/legacy_source.json',{'fixture':'old'})
  pr=self.write('inputs/training_plan.json',plan);cfg=self.write('inputs/training_config.json',{'data':{}});ds=self.write('inputs/dataset.json',{'training_accepted':True,'count':2000,'plan_sha256':pr['sha256'],'split_counts':{'train':1600,'eval':200,'test':200}})
  self.a.update(status='ACTIVE',phase='training',training={'enabled':True,'stages':['D0'],'config_ref':cfg,'dataset_manifest_ref':ds,'plan_ref':pr,'split_refs':{}});self.publish()
  return {'kind':'train','argv':['.venv/bin/python','scripts/action4_train.py','--stage','D0','--config',cfg['path']]}
 def test_training_old_source_rejected(self):
  spec=self.train_fixture(old_source=True)
  with self.assertRaisesRegex(ValueError,'Training plan source'):authority.spec_references(self.root,spec)
 def test_training_missing_splits_rejected(self):
  spec=self.train_fixture()
  with self.assertRaisesRegex(ValueError,'Complete current'):authority.spec_references(self.root,spec)

if __name__=='__main__':unittest.main()
