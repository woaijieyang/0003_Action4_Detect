"""Synthetic file/selection fixtures only; no real run mutation or model evaluation."""
import unittest,tempfile,json,hashlib,copy,types,sys
from unittest.mock import patch
from pathlib import Path
import torch
from eval import action4_role_continuation_eval as m
from model.training.action4_role_continuation_observer import LineageObserver

FIELDS=('score_schema','pose_gt_count','pose_output_count','pose_all8_at_002_success','pose_all8_at_002_rate','pose_fn','pose_fp','pose_corner_p95_valid_matched_only','pose_all8_at_002_precision','pose_instance_precision','pose_groups')
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p
class LineageTests(unittest.TestCase):
 def setUp(self):
  base=m.R/'outputs/action4_a0_resource_continuation_v001/eval_lineage';base.mkdir(parents=True,exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=base);self.d=Path(self.tmp.name);self.old=self.d/'old';self.old.mkdir();self.obs=self.d/'old_observer';self.obs.mkdir()
  self.cfg,self.spec=self.config('old',self.old)
  self.rows={}
  for s in [0,10]:
   row=self.report(self.old,self.cfg,s,2,True);self.rows[s]=row
   rec={'step':s,'config_id':m.canonical(self.cfg),'metrics':{k:row['report'][k] for k in FIELDS},'source_done':m.reference(self.old/'eval'/f'step_{s:08d}.done.json'),'deployment_guard_passed':True,'weight_available_at_processing':True}
   put(self.obs/'evaluated'/f'step_{s:08d}.json',rec)
  weight=self.old/'selected_step_00000000.pt';weight.write_bytes(b'fixture legacy weights')
  best=dict(self.rows[0],checkpoint=str(weight));put(self.old/'BEST.json',best);put(self.old/'REFERENCE_REPORT.json',self.rows[0]['report'])
  selected=self.obs/'selected/step_00000000.pt';selected.parent.mkdir();selected.write_bytes(b'fixture training weights')
  rec=json.loads((self.obs/'evaluated/step_00000000.json').read_text());self.parent_training=dict(rec,checkpoint=str(selected),checkpoint_sha256=m.reference(selected)['sha256'],policy='fixture',sealed=False)
  put(self.obs/'TRAINING_BEST.json',self.parent_training);put(self.obs/'OBSERVER_IDENTITY.json',{'config_id':m.canonical(self.cfg),'run':str(self.old.relative_to(m.R))})
  self.cp=self.d/'boundary.pt';torch.save({'step':10,'config_id':m.canonical(self.cfg),'world_size':1,'resumable':True,'training_model':{},'optimizer':{},'scheduler':{},'rng_by_rank':[],'epoch':0,'cursor':10},self.cp)
 def tearDown(self):self.tmp.cleanup()
 def config(self,name,out):
  code=self.d/(name+'_source');code.mkdir()
  cfg={'role':'A0','mode':'full_training','eval_every':10,'expected_eval_images':650,'score_schema':'action4.pose_projection_all8_002.v1','world_size':1,'epochs':1,'expected_units':20,'batch':1,'output':str(out.relative_to(m.R))}
  cp=put(code/'cfg.json',cfg);files={'cfg.json':m.reference(cp)['sha256']}
  for n in ['src/eval/action4_stage2_eval.py','src/eval/action4_pose_protocol.py']:
   p=code/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((m.R/n).read_bytes());files[n]=m.reference(p)['sha256']
  spec=put(self.d/(name+'_RUN_SPEC.json'),{'code_root':str(code.relative_to(m.R)),'config':str(cp.relative_to(m.R)),'files':files})
  return cfg,spec
 def report(self,out,cfg,s,tp,eligible):
  r={'score_schema':cfg['score_schema'],'evaluation_images':650,'pose_gt_count':4,'pose_output_count':4,'pose_all8_at_002_success':tp,'pose_all8_at_002_rate':tp/4,'pose_fn':4-tp,'pose_fp':4-tp,'pose_corner_p95_valid_matched_only':10.,'pose_all8_at_002_precision':tp/4,'pose_instance_precision':1.,'pose_groups':{'hard':{'gt':4,'recall':0.}}}
  raw=put(out/'reports'/f'{s}.json',r);snapshot=out/f'eval_step_{s:08d}.pt';snapshot.write_bytes(b'fixture suffix weights')
  row={'role':'A0','config_id':m.canonical(cfg),'step':s,'training_mode':cfg['mode'],'deployment_eligible':eligible,'report':r,'raw_report':str(raw.relative_to(m.R))}
  put(out/'eval'/f'step_{s:08d}.json',{'role':'A0','config_id':m.canonical(cfg),'step':s,'snapshot':str(snapshot.relative_to(m.R))});put(out/'eval'/f'step_{s:08d}.done.json',row)
  return row
 def seal(self):
  ref=m.create_prefix_ledger(self.d/'prefix',old_run_spec=self.spec,old_run_spec_sha256=m.reference(self.spec)['sha256'],checkpoint_path=self.cp,boundary_step=10,observer_output=self.obs)
  return ref,m.load_prefix_reports(ref['path'],ref['sha256'])
 def test_prefix_exact_reference_and_original_best_config(self):
  ref,p=self.seal();self.assertIs(m.load_prefix_reports(ref['path'],ref['sha256']),p)
  new=self.d/'new';new.mkdir();m.initialize_successor(new,p)
  self.assertEqual((new/'BEST.json').read_bytes(),(self.old/'BEST.json').read_bytes())
  self.assertEqual((new/'REFERENCE_REPORT.json').read_bytes(),(self.old/'REFERENCE_REPORT.json').read_bytes())
  self.assertEqual(json.loads((new/'BEST.json').read_text())['config_id'],m.canonical(self.cfg))
  init=json.loads((new/'PREFIX_INITIALIZATION.json').read_text());self.assertEqual(init['boundary_step'],10);self.assertEqual(init['prefix_config_id'],m.canonical(self.cfg));self.assertEqual(init['prefix_ledger'],ref)
  with self.assertRaises(FileExistsError):m.initialize_successor(new,p)
 def test_changed_hash_and_missing_boundary(self):
  ref,p=self.seal()
  with self.assertRaises(ValueError):m.load_prefix_reports(ref['path'],'0'*64)
  (self.old/'eval/step_00000010.done.json').write_text('{}')
  with self.assertRaises(ValueError):p.assert_unchanged()
 def test_creator_missing_step_fails_before_publication(self):
  (self.old/'eval/step_00000010.done.json').unlink()
  with self.assertRaises(FileNotFoundError):self.seal()
  self.assertFalse((self.d/'prefix/PREFIX.json').exists())
 def test_best_mismatch(self):
  best=json.loads((self.old/'BEST.json').read_text());best['step']=10;put(self.old/'BEST.json',best)
  with self.assertRaisesRegex(ValueError,'BEST'):self.seal()
 def test_reference_and_full650_guards(self):
  put(self.old/'REFERENCE_REPORT.json',{})
  with self.assertRaises(ValueError):self.seal()
  put(self.old/'REFERENCE_REPORT.json',self.rows[0]['report'])
  p=self.old/'eval/step_00000010.done.json';r=json.loads(p.read_text());r['report']['evaluation_images']=649;put(p,r)
  with self.assertRaisesRegex(ValueError,'650'):self.seal()
 def test_source_and_observer_identity_guards(self):
  spec=json.loads(self.spec.read_text());spec['files']['src/eval/action4_pose_protocol.py']='0'*64;put(self.spec,spec)
  with self.assertRaisesRegex(ValueError,'source'):self.seal()
 def test_suffix_duplicate_missing_and_no_false_complete(self):
  ref,p=self.seal();new=self.d/'new';new.mkdir();cfg,spec=self.config('new',new);m.initialize_successor(new,p)
  with self.assertRaises(ValueError):m.combine_reports(p,new,cfg,final_step=20)
  self.report(new,cfg,10,3,True)
  with self.assertRaisesRegex(ValueError,'overlapping'):m.combine_reports(p,new,cfg)
  (new/'eval/step_00000010.json').unlink();(new/'eval/step_00000010.done.json').unlink()
  self.report(new,cfg,20,3,False)
  with self.assertRaises(FileNotFoundError):m.finalize_lineage(p,new,cfg)
  self.assertFalse((new/'ONELINEAGE_EVAL_COMPLETE.json').exists())
 def test_training_global_selection_is_independent_and_preserves_old_cfg(self):
  ref,p=self.seal();new=self.d/'new';new.mkdir();cfg,spec=self.config('new',new);m.initialize_successor(new,p)
  row=self.report(new,cfg,20,3,False)
  request=new/'eval/step_00000020.json';m.commit_lineage_result(new,request,new/'eval_step_00000020.pt',row,m.legacy_rank)
  self.assertEqual(json.loads((new/'BEST.json').read_text())['config_id'],m.canonical(self.cfg))
  observer=LineageObserver(run_spec=spec,run_spec_sha256=m.reference(spec)['sha256'],output=self.d/'new_observer',prefix_ledger=ref['path'],prefix_ledger_sha256=ref['sha256'])
  _,pointer=observer.sweep();self.assertEqual(pointer['winner']['record']['step'],20);self.assertEqual(pointer['winner']['record'],observer.original.best);self.assertFalse(pointer['deployment_guard_veto_used'])
  self.assertEqual(pointer['winner']['record']['config_id'],m.canonical(cfg));self.assertEqual(json.loads((self.obs/'TRAINING_BEST.json').read_text()),self.parent_training)
  self.assertFalse(observer.complete_if_terminal())
  put(new/'TRAIN_COMPLETE.json',{'step':20,'config_id':m.canonical(cfg),'formal_training_complete':True,'attempt_id':'new','no_supervision_update_count':0})
  result=m.finalize_lineage(p,new,cfg);self.assertEqual(result['legacy_best_step'],0)
  put(new/'CONTROLLER_COMPLETE.json',{'config_id':m.canonical(cfg),'role':'A0','mode':'full_training','attempt_id':'new'})
  self.assertTrue(observer.complete_if_terminal());self.assertTrue((observer.out/'ONELINEAGE_COMPLETE.json').exists())
 def test_foreign_ledger_observer_source_and_eval_semantics(self):
  ref,p=self.seal();ledger=p.ledger;ledger['observer_source']['sha256']='0'*64
  bad=put(self.d/'foreign_ledger.json',ledger)
  with self.assertRaises(ValueError):m.load_prefix_reports(bad,m.reference(bad)['sha256'])
  new=self.d/'new';new.mkdir();cfg,_=self.config('new',new)
  cfg['evaluation_counterpart_sha256']='different'
  with self.assertRaisesRegex(ValueError,'semantics'):m.combine_reports(p,new,cfg)
 def test_global_best_alias_mismatch_blocks_completion(self):
  ref,p=self.seal();new=self.d/'new';new.mkdir();cfg,_=self.config('new',new);m.initialize_successor(new,p);self.report(new,cfg,20,2,False)
  put(new/'TRAIN_COMPLETE.json',{'step':20,'config_id':m.canonical(cfg),'formal_training_complete':True,'attempt_id':'new'})
  (new/'best.pt').write_bytes(b'wrong alias')
  with self.assertRaisesRegex(ValueError,'alias'):m.finalize_lineage(p,new,cfg)
  self.assertFalse((new/'ONELINEAGE_EVAL_COMPLETE.json').exists())
 def test_worker_hook_receipt_and_config_guards_without_gpu_worker(self):
  import eval as eval_package
  ref,p=self.seal();new=self.d/'new';new.mkdir();cfg,spec=self.config('new',new);m.initialize_successor(new,p)
  row=self.report(new,cfg,20,3,False);fake=types.ModuleType('eval.action4_stage2_eval')
  def fake_worker(config,worker_id):
   self.assertEqual(worker_id,'0');self.assertEqual(json.loads(m.inside(config).read_text()),cfg)
   wrong=dict(row,config_id=m.canonical(self.cfg))
   with self.assertRaises(ValueError):fake.commit_result(new,new/'eval/step_00000020.json',new/'eval_step_00000020.pt',wrong,m.legacy_rank)
   fake.commit_result(new,new/'eval/step_00000020.json',new/'eval_step_00000020.pt',row,m.legacy_rank)
   self.assertEqual(fake.prune_old_selected(new),[])
   with self.assertRaises(ValueError):fake.prune_old_selected(self.old)
   return 'stub_worker_only'
  fake.worker=fake_worker
  config=json.loads(spec.read_text())['config']
  with patch.dict(sys.modules,{'eval.action4_stage2_eval':fake}),patch.object(eval_package,'action4_stage2_eval',fake,create=True):
   self.assertEqual(m.worker(config,'0',prefix_ledger=ref['path'],prefix_sha256=ref['sha256']),'stub_worker_only')
  receipt=json.loads((new/'LINEAGE_WORKER_0.json').read_text());self.assertEqual(receipt['config_id'],m.canonical(cfg));self.assertFalse(receipt['pruning_enabled'])
  p.assert_unchanged()
 def test_rollback_parent_pointer_mutation_and_pruning_keep_archived_prefix_valid(self):
  old_best_bytes=(self.old/'BEST.json').read_bytes();old_training_bytes=(self.obs/'TRAINING_BEST.json').read_bytes()
  ref,p=self.seal();ledger=p.ledger
  original_legacy=Path(json.loads(old_best_bytes)['checkpoint']);original_training=Path(json.loads(old_training_bytes)['checkpoint'])
  put(self.old/'BEST.json',{'step':999,'config_id':'later-parent-branch'})
  put(self.obs/'TRAINING_BEST.json',{'step':999,'config_id':'later-parent-branch'})
  original_legacy.unlink();original_training.unlink()
  p.assert_unchanged();m._CACHE.clear();p=m.load_prefix_reports(ref['path'],ref['sha256'])
  self.assertEqual(m.read_ref(p.ledger['legacy_best']),old_best_bytes)
  self.assertEqual(m.read_ref(p.ledger['training_best']),old_training_bytes)
  new=self.d/'new';new.mkdir();cfg,spec=self.config('new',new);m.initialize_successor(new,p)
  self.assertEqual((new/'BEST.json').read_bytes(),old_best_bytes)
  self.report(new,cfg,20,2,False)
  put(new/'TRAIN_COMPLETE.json',{'step':20,'config_id':m.canonical(cfg),'formal_training_complete':True,'attempt_id':'new'})
  final=m.finalize_lineage(p,new,cfg);self.assertEqual(final['legacy_weight'],p.ledger['retained_legacy_weight'])
  observer=LineageObserver(run_spec=spec,run_spec_sha256=m.reference(spec)['sha256'],output=self.d/'new_observer',prefix_ledger=ref['path'],prefix_ledger_sha256=ref['sha256'])
  _,pointer=observer.sweep();self.assertEqual(pointer['winner']['record']['config_id'],m.canonical(self.cfg))
  self.assertEqual(pointer['winner']['record']['checkpoint'],str(original_training))
  self.assertTrue(m.inside(pointer['winner']['retained_checkpoint']['path']).exists())
  (self.old/'eval/step_00000000.done.json').write_text('{}')
  with self.assertRaises(ValueError):p.assert_unchanged()
 def test_captured_pointer_tamper_still_fails(self):
  ref,p=self.seal();captured=m.inside(p.ledger['legacy_best']['path']);captured.chmod(0o644);captured.write_text('{}')
  with self.assertRaises(ValueError):p.assert_unchanged()
 def test_old_vs_new_training_tie_keeps_earlier_parent(self):
  ref,p=self.seal();new=self.d/'new';new.mkdir();cfg,spec=self.config('new',new);m.initialize_successor(new,p);self.report(new,cfg,20,2,True)
  observer=LineageObserver(run_spec=spec,run_spec_sha256=m.reference(spec)['sha256'],output=self.d/'new_observer',prefix_ledger=ref['path'],prefix_ledger_sha256=ref['sha256'])
  _,pointer=observer.sweep();self.assertEqual(pointer['winner']['record']['step'],0);self.assertEqual(pointer['winner']['record']['config_id'],m.canonical(self.cfg))

if __name__=='__main__':unittest.main()
