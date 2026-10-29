"""Metadata contract tests; fixtures never load models or seal artifacts."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest import mock
from eval import action4_corner_research_seal as s

def fixture():
 cfg={'mode':'full_training','role':'A0','training_protocol':s.helpers().CORNER_NATIVE_PROTOCOL,'epochs':80,'formal_epochs':80,'train_indices':None,'expected_units':1,'formal_training_units':1,'batch':1,'eval_every':25,'expected_eval_images':650,'train_backbone':False,'corner_softmax_bias_frozen':True,'a0_variant':'a0_corner_image','score_schema':s.SCORE}
 cid=s.helpers().canonical(cfg);steps=[0,25,50,75,80]
 train={'config_id':cid,'attempt_id':'attempt','mode':'full_training','role':'A0','epochs':80,'units_per_epoch':1,'step':80,'no_supervision_update_count':0,'data_update_attempt':80,'planned_update_attempts':80,'formal_training_complete':True}
 ev={'config_id':cid,'attempt_id':'attempt','all_steps':steps,'evaluations':5,'status':'NO_DEPLOYABLE_CHECKPOINT','best_step':None}
 ctrl={'config_id':cid,'attempt_id':'attempt','mode':'full_training','role':'A0','status':'NO_DEPLOYABLE_CHECKPOINT','best_step':None}
 return cfg,dict(zip(s.TERMINALS,[train,ev,ctrl])),set(steps)

def route(step=25,kind='corner_candidate',success=1100,p95=12,quality=False):
 return {'step':step,'route':kind,'quality_ready':quality,'metrics':{'pose_all8_at_002_rate':success/1300,'pose_fp':1300-success,'pose_fn':1300-success,'pose_corner_p95_valid_matched_only':p95}}

class ResearchSealTests(unittest.TestCase):
 def test_missing_each_terminal_waits_without_subprocess_or_model_write(self):
  base=s.ROOT/'outputs/action4_corner_research_seal_preparation_v001'
  for missing in s.TERMINALS:
   with tempfile.TemporaryDirectory(dir=base) as tmp:
    root=Path(tmp);code=root/'source';code.mkdir();out=root/'run';out.mkdir();specpath=root/'RUN_SPEC.json';cfgpath=code/'cfg.json'
    cfg={'require_immutable_source':True,'run_spec':str(specpath),'output':str(out)}
    cfgpath.write_text(json.dumps(cfg));specpath.write_text(json.dumps({'schema':'action4.immutable_run.v1','project_root':str(s.ROOT),'code_root':str(code),'config':str(cfgpath)}))
    for name in s.TERMINALS:
     if name!=missing:(out/name).write_text('{}')
    with mock.patch.object(s.subprocess,'run',side_effect=AssertionError('must not dispatch')) as call:
     result=s.launch(str(specpath),str(s.ROOT/'models/candidates/never-created-test'))
    self.assertEqual(result['status'],'WAITING_FOR_COMPLETION');self.assertEqual(result['missing'],[missing]);self.assertFalse(result['model_loaded']);call.assert_not_called()
 def test_valid_terminal_metadata(self):
  cfg,rs,expected=fixture();self.assertEqual(s.validate_terminal(cfg,rs,expected),'attempt')
 def test_missing_eval_rejected(self):
  cfg,rs,expected=fixture();rs['EVAL_COMPLETE.json']['all_steps'].pop();rs['EVAL_COMPLETE.json']['evaluations']=4
  with self.assertRaisesRegex(ValueError,'evaluation domain'):s.validate_terminal(cfg,rs,expected)
 def test_duplicate_eval_step_rejected(self):
  cfg,rs,expected=fixture();rs['EVAL_COMPLETE.json']['all_steps']=[0,25,50,75,75]
  with self.assertRaisesRegex(ValueError,'evaluation domain'):s.validate_terminal(cfg,rs,expected)
 def test_config_mismatch_rejected(self):
  cfg,rs,expected=fixture();rs['EVAL_COMPLETE.json']['config_id']='other'
  with self.assertRaisesRegex(ValueError,'config'):s.validate_terminal(cfg,rs,expected)
 def test_attempt_mismatch_rejected(self):
  cfg,rs,expected=fixture();rs['CONTROLLER_COMPLETE.json']['attempt_id']='other'
  with self.assertRaisesRegex(ValueError,'attempt'):s.validate_terminal(cfg,rs,expected)
 def test_false80_completion_rejected(self):
  for field,value in [('epochs',79),('step',79),('units_per_epoch',2),('data_update_attempt',79),('planned_update_attempts',79),('formal_training_complete',False)]:
   cfg,rs,expected=fixture();rs['TRAIN_COMPLETE.json'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):s.validate_terminal(cfg,rs,expected)
 def test_zero_supervision_accounting_preserved(self):
  cfg,rs,expected=fixture();rs['TRAIN_COMPLETE.json'].update(step=79,no_supervision_update_count=1);rs['EVAL_COMPLETE.json']['all_steps']=[0,25,50,75,79]
  self.assertEqual(s.validate_terminal(cfg,rs,{0,25,50,75,79}),'attempt')
 def test_quality_false_high_all8_wins_without_upgrade(self):
  rows=[route(kind='raw',success=1040),route(success=1115)];original=copy.deepcopy(rows);chosen=s.choose(rows)
  self.assertEqual(chosen['route'],'corner_candidate');self.assertIs(chosen['quality_ready'],False);self.assertEqual(rows,original)
 def test_rank_earlier_then_raw_exact_tie_and_null_p95(self):
  self.assertEqual(s.choose([route(step=50),route(step=25)])['step'],25)
  self.assertEqual(s.choose([route(step=25,p95=None),route(step=50,p95=99)])['step'],50)
  self.assertEqual(s.choose([route(kind='corner_candidate'),route(kind='raw')])['route'],'raw')
 def test_duplicate_step_route_rejected(self):
  with self.assertRaises(ValueError):s.choose([route(),route()])
 def test_wrong_route_gt_count_and_full_domain_rejected(self):
  ids={f'image{i}' for i in range(650)};units={(i,str(j)) for i in ids for j in [1,2]};rows=[{'sample_id':i,'instance_id':j,'all8_at_002':False,'all8_at_005':False} for i,j in sorted(units)]
  report={'score_schema':s.SCORE,'pose_gt_count':1300,'pose_output_count':1300,'per_gt_failure_reason':rows,'pose_all8_at_002_success':0,'pose_all8_at_002_rate':0.,'pose_all8_at_005_rate':0.,'pose_fp':1300,'pose_fn':1300,'pose_corner_p95_valid_matched_only':None}
  s.validate_report(report,ids,units)
  bad=copy.deepcopy(report);bad['pose_gt_count']=1299
  with self.assertRaises(ValueError):s.validate_report(bad,ids,units)
  bad=copy.deepcopy(report);bad['per_gt_failure_reason'][-1]=bad['per_gt_failure_reason'][0]
  with self.assertRaises(ValueError):s.validate_report(bad,ids,units)
  bad=copy.deepcopy(report);bad['pose_all8_at_002_success']=1
  with self.assertRaises(ValueError):s.validate_report(bad,ids,units)

class UserStoppedTests(unittest.TestCase):
 def stopped(self):
  cfg,_,_=fixture();a={'role':'A0','mode':'full_training','step':62,'epoch':62,'cursor':1,'data_update_attempt':62,'no_supervision_update_count':0};req={'status':'USER_REQUESTED_STOP_KEEP_BEST_AND_DRAIN_EVAL','formal80_completed':False,'train_status_before':copy.deepcopy(a)};stop={'status':'OWNED_TRAINER_AND_CONTROLLER_EXIT_CONFIRMED_EVAL_DRAINING','formal_training_complete':False,'no_new_training':True,'actual_status':a,'submitted_eval_steps':[0,25,50]};process={'role':'A0','units':1,'attempt_id':'attempt'}
  return cfg,req,stop,process
 def test_stop_preserves80_plan_but_records62_actual(self):
  cfg,req,stop,process=self.stopped();proof=s.validate_user_stop(cfg,req,stop,process)
  self.assertEqual(cfg['epochs'],80);self.assertEqual(proof['completed_epoch'],62);self.assertEqual(proof['evaluated_boundary'],50);self.assertEqual(proof['discarded_unevaluated_updates'],12);self.assertFalse(proof['formal_training_complete'])
 def test_stopped_bad_actual_counts_future_step_and_duplicates(self):
  for mutation in ('fake_count','future','duplicate','omitted_cadence','fake_complete'):
   cfg,req,stop,process=self.stopped()
   if mutation=='fake_count':stop['actual_status']['data_update_attempt']=61
   elif mutation=='future':stop['submitted_eval_steps'].append(75)
   elif mutation=='duplicate':stop['submitted_eval_steps'].append(50)
   elif mutation=='omitted_cadence':stop['submitted_eval_steps']=[0,50]
   else:stop['formal_training_complete']=True
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):s.validate_user_stop(cfg,req,stop,process)
 def test_drain_requires_matching_stop_refs_and_exited_workers(self):
  cfg,req,stop,process=self.stopped();proof=s.validate_user_stop(cfg,req,stop,process);refs={k:{'path':k,'sha256':'a'*64} for k in ('run_spec','user_stop_request','user_stopped','train_process')};cid=s.helpers().canonical(cfg)
  drain={'schema':'action4.user_stopped_eval_drained.v1','status':'ALL_SUBMITTED_EVALUATIONS_DRAINED','config_id':cid,'attempt_id':'attempt',**refs,'submitted_eval_steps':[0,25,50],'done_steps':[0,25,50],'request_count':3,'done_count':3,'failed_count':0,'eval_workers_stopped':True,'formal_training_complete':False,'no_new_training':True}
  s.validate_drain_receipt(drain,proof,refs,cid,[0,25,50])
  for key,value in [('done_count',2),('eval_workers_stopped',False),('failed_count',1),('config_id','wrong'),('attempt_id','wrong'),('user_stopped',{'path':'wrong','sha256':'a'*64})]:
   bad=copy.deepcopy(drain);bad[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):s.validate_drain_receipt(bad,proof,refs,cid,[0,25,50])
 def test_pending_done_returns_waiting_without_worker(self):
  base=s.ROOT/'outputs/action4_corner_research_seal_preparation_v001'
  with tempfile.TemporaryDirectory(dir=base) as tmp:
   root=Path(tmp);code=root/'source';code.mkdir();out=root/'run';out.mkdir();queue=out/'eval';queue.mkdir();specpath=root/'RUN_SPEC.json';cfgpath=code/'cfg.json';cfg,req,stop,process=self.stopped();cfg.update(require_immutable_source=True,run_spec=str(specpath),output=str(out),corner_eval_source={'path':'source-fixture','sha256':'a'*64});cid=s.helpers().canonical(cfg)
   cfgpath.write_text(json.dumps(cfg));specpath.write_text(json.dumps({'schema':'action4.immutable_run.v1','project_root':str(s.ROOT),'code_root':str(code),'config':str(cfgpath)}))
   for name,value in [('USER_STOP_REQUEST.json',req),('USER_STOPPED.json',stop),('TRAIN_PROCESS.json',process),('CORNER_EVAL_IDENTITY.json',{'config_id':cid,'corner_eval_source':cfg['corner_eval_source']})]:(out/name).write_text(json.dumps(value))
   for step in stop['submitted_eval_steps']:
    snapshot=out/f'eval_step_{step:08d}.pt';snapshot.write_bytes(b'explicit metadata fixture, not a model');(queue/f'step_{step:08d}.json').write_text(json.dumps({'step':step,'role':'A0','config_id':cid,'snapshot':str(snapshot)}))
   (queue/'step_00000000.done.json').write_text('{}')
   with mock.patch.object(s.subprocess,'run',side_effect=AssertionError('must not dispatch')) as call:result=s.launch(str(specpath),user_stopped=True)
   self.assertEqual(result['status'],'WAITING_EVAL_DRAIN');self.assertEqual(result['pending_steps'],[25,50]);self.assertFalse(result['model_loaded']);call.assert_not_called()
   with self.assertRaises(ValueError):
    (queue/'step_00000075.json').write_text('{}');s.precheck(str(specpath),True)

if __name__=='__main__':unittest.main()
