"""CPU admission fixtures, never actual dataset approval or GPU training."""
from pathlib import Path
import copy,json,tempfile,unittest,hashlib
from data import action4_execution_authority as a
ROOT=Path(__file__).resolve().parents[1]
class TemplateAuthorityTests(unittest.TestCase):
 def setUp(self):
  parent=Path('/data/zy/vln/a-test/outputs/action4_execution_authority_test_fixtures');parent.mkdir(exist_ok=True)
  self.root=Path(tempfile.mkdtemp(prefix='template_',dir=parent))
  self.splits={'train':1600,'eval':200,'test':200};self.layout={s:'data/'+s for s in self.splits};self.layout['commit']='data/action4_dataset_commit.json'
  self.plan={'schema':'action4.fixed_template_batch_plan.v1','count':2000,'entries':[{} for _ in range(2000)],'split_counts':self.splits};plan=self.write('inputs/plan.json',self.plan)
  protocol=self.write('inputs/protocol.json',{'mode':'independent_samples_shared_scene_templates','plan_ref':plan,'output_layout':self.layout})
  self.binding={'plan_ref':plan,'output_layout':self.layout,'split_protocol_ref':protocol,'ledger_ref':self.write('inputs/ledger.json',{'fixture':True}),'relocation_ledger_ref':self.write('inputs/relocation.json',{'fixture':True})}
  bs=hashlib.sha256(json.dumps(self.binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  self.reviews={role:self.write('inputs/'+role+'.json',{'role':role,'decision':'approve_formal_2000_release','binding_sha256':bs}) for role in ('root','spec')}
  release=self.write('inputs/release.json',{'schema':'action4.template_generation_release.v1','status':'FORMAL_2000_RELEASE_APPROVED','binding':self.binding,'reviews':self.reviews})
  splitrefs={s:self.write('data/'+s+'/annotations.jsonl',{'synthetic_fixture_only':s}) for s in self.splits}
  self.dataset={'schema':'action4.template_dataset_commit.v1','status':'COMPLETE','complete':True,'training_accepted':True,'count':2000,'split_counts':self.splits,'output_layout':self.layout,'plan_ref':plan,'release_ref':release,'split_protocol_ref':protocol,'ledger_ref':self.binding['ledger_ref'],'relocation_ledger_ref':self.binding['relocation_ledger_ref'],'manifest_refs':splitrefs}
  commit=self.write('data/action4_dataset_commit.json',self.dataset)
  self.config={'data':{'dataset_commit_ref':commit,'dataset_release_ref':release,'split_protocol_ref':protocol,**{s+'_manifest':r['path'] for s,r in splitrefs.items()}}}
  self.t={'dataset_manifest_ref':commit,'plan_ref':plan,'release_ref':release,'split_protocol_ref':protocol,'split_refs':splitrefs}
 def write(self,name,obj):
  p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return a.ref(self.root,name)
 def check(self):return a.template_training_references(self.root,self.t,self.dataset,self.config,self.plan)
 def test_complete_bound_commit(self):self.assertEqual(len(self.check()),7)
 def test_incomplete_commit(self):
  self.dataset['complete']=False
  with self.assertRaisesRegex(ValueError,'Complete current'):self.check()
 def test_config_commit_missing(self):
  self.config['data'].pop('dataset_commit_ref')
  with self.assertRaisesRegex(ValueError,'commit/config'):self.check()
 def test_wrong_output(self):
  self.dataset['output_layout']={**self.layout,'test':'data/old_test'}
  with self.assertRaisesRegex(ValueError,'Complete current'):self.check()
 def test_missing_test(self):
  self.t['split_refs']={k:v for k,v in self.t['split_refs'].items() if k!='test'}
  with self.assertRaisesRegex(ValueError,'Committed train'):self.check()
 def test_old_split_bytes(self):
  (self.root/self.t['split_refs']['train']['path']).write_text('changed')
  with self.assertRaisesRegex(ValueError,'Old or mixed'):self.check()
 def test_foreign_ledger(self):
  self.dataset['ledger_ref']=self.write('inputs/foreign.json',{})
  with self.assertRaisesRegex(ValueError,'Release/commit'):self.check()
 def test_wrong_plan(self):
  self.plan['count']=128
  with self.assertRaisesRegex(ValueError,'fixed-template2000'):self.check()
 def test_review_mutation(self):
  (self.root/self.reviews['spec']['path']).write_text('{}')
  with self.assertRaisesRegex(ValueError,'reference changed'):self.check()
 def test_shared_protocol_removed(self):
  self.config['data'].pop('split_protocol_ref')
  with self.assertRaisesRegex(ValueError,'shared-template'):self.check()
if __name__=='__main__':unittest.main()
