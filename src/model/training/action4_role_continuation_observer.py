"""Successor-only original observer plus independently authenticated lineage choice."""
from pathlib import Path
import json,shutil,math,os,fcntl
from eval.action4_role_continuation_eval import (R,inside,reference,read_ref,json_ref,canonical,atomic,pinned_observer,
 load_prefix_reports,combine_reports,_source_config)

class LineageObserver:
 def __init__(self,*,run_spec,run_spec_sha256,output,prefix_ledger,prefix_ledger_sha256):
  self.prefix=load_prefix_reports(prefix_ledger,prefix_ledger_sha256)
  self.prefix_ref={'path':str(inside(prefix_ledger).relative_to(R)),'sha256':prefix_ledger_sha256}
  spec_ref={'path':str(inside(run_spec).relative_to(R)),'sha256':run_spec_sha256}
  self.cfg,_,_=_source_config(spec_ref);self.run=inside(self.cfg['output']);self.out=inside(output,False)
  self.original=pinned_observer().CandidateObserver(R,self.run,self.out,self.cfg)
  (self.out/'lineage_retained').mkdir(exist_ok=True)
  self.parent=json_ref(self.prefix.ledger['training_best'])
  self.identity={'schema':'action4.lineage_training_observer.v1','prefix':self.prefix_ref,'successor_run_spec':spec_ref,
   'successor_config_id':canonical(self.cfg),'original_observer_source':self.prefix.ledger['observer_source'],
   'selection':'fullAll8_without_deployment_group_veto','parent_records_rewritten':False,'defaults_changed':False}
  path=self.out/'LINEAGE_OBSERVER_IDENTITY.json'
  if path.exists() and json.loads(path.read_bytes())!=self.identity:raise ValueError('Lineage observer identity changed')
  if not path.exists():atomic(path,self.identity)

 def _retain(self,record,weight):
  # Immutable copies keep all winning weights; never prune parent or active assets.
  name=f"{record['config_id'][:12]}_step_{record['step']:08d}_{weight['sha256'][:12]}.pt"
  target=self.out/'lineage_retained'/name
  if not target.exists():shutil.copyfile(inside(weight['path']),target)
  held=reference(target)
  if held['sha256']!=weight['sha256']:raise ValueError('Held training candidate changed')
  return {'record':record,'checkpoint_source':weight,'retained_checkpoint':held}

 def sweep(self):
  self.prefix.assert_unchanged()
  changed=self.original.sweep() # UNCHANGED observer; successor requests only.
  candidates=[self._retain(self.parent,self.prefix.ledger['retained_training_weight'])]
  for row in self.prefix.ledger['training_candidates']:
   candidates.append(self._retain(json_ref(row['record']),row['retained']))
  if self.original.best is not None:
   candidates.append(self._retain(self.original.best,reference(self.original.best['checkpoint'])))
  available=[]
  for record in self.original.records.values():
   path=self.out/'selected'/f"step_{record['step']:08d}.pt"
   if path.exists():available.append((pinned_observer().rank(record['metrics'],record['step']),record,path))
  for _,record,path in sorted(available,key=lambda x:x[0],reverse=True)[:4]:candidates.append(self._retain(record,reference(path)))
  unique={}
  for row in candidates:unique.setdefault((row['record']['config_id'],row['record']['step']),row)
  candidates=sorted(unique.values(),key=lambda z:pinned_observer().rank(z['record']['metrics'],z['record']['step']),reverse=True)
  winner=candidates[0]
  pointer={'schema':'action4.onelineage_training_best.v1','prefix':self.prefix_ref,'winner':winner,
   'selection':'fullAll8_then_total_failures_then_p95_then_earlier_step','deployment_guard_veto_used':False,
   'retained_top_candidates':candidates[:4],'all_prior_winners_retained':True,'defaults_changed':False,'sealed':False}
  atomic(self.out/'ONELINEAGE_TRAINING_BEST.json',pointer)
  return changed,pointer

 def complete_if_terminal(self):
  # Never call original.complete_if_terminal: it incorrectly requires successor step0.
  terminal_path=self.run/'CONTROLLER_COMPLETE.json'
  if not terminal_path.exists():return False
  controller=json.loads(terminal_path.read_bytes());train=json.loads((self.run/'TRAIN_COMPLETE.json').read_bytes())
  if (controller.get('config_id'),controller.get('role'),controller.get('mode'))!=(canonical(self.cfg),'A0',self.cfg['mode']):raise ValueError('Foreign controller terminal')
  if self.cfg.get('mode')!='full_training' or train.get('config_id')!=canonical(self.cfg) or train.get('attempt_id')!=controller.get('attempt_id') or train.get('formal_training_complete') is not True:raise ValueError('Invalid training terminal')
  total=self.cfg['epochs']*math.ceil(self.cfg['expected_units']/self.cfg['batch'])
  if train['step']+train.get('no_supervision_update_count',0)!=total:raise ValueError('Incomplete traversal')
  reports=combine_reports(self.prefix,self.run,self.cfg,final_step=train['step'])
  expected_suffix={r['step'] for r in reports if r['step']>self.prefix.ledger['boundary_step']}
  if set(self.original.records)!=expected_suffix:raise ValueError('Successor observer coverage incomplete')
  receipt=json.loads((self.run/'ONELINEAGE_EVAL_COMPLETE.json').read_bytes())
  if receipt['steps']!=sorted(r['step'] for r in reports) or receipt['suffix_config_id']!=canonical(self.cfg):raise ValueError('Evaluation lineage not closed')
  _,pointer=self.sweep();metric_winner=max(reports,key=lambda z:pinned_observer().rank(z['report'],z['step']))
  winner=pointer['winner']['record']
  if (winner['step'],winner['config_id'])!=(metric_winner['step'],metric_winner['config_id']):raise ValueError('Full lineage training winner weight unavailable/mismatched')
  pointer=dict(pointer,sealed=True);atomic(self.out/'ONELINEAGE_TRAINING_BEST.json',pointer)
  atomic(self.out/'ONELINEAGE_COMPLETE.json',{'status':'COMPLETE_TRAINING_SELECTION_NOT_DEPLOYMENT','reports':len(reports),'winner':pointer['winner'],'prefix':self.prefix_ref,'parent_assets_pruned':False})
  return True

if __name__=='__main__':
 import argparse,time
 p=argparse.ArgumentParser()
 for name in ['run-spec','run-spec-sha256','output','prefix-ledger','prefix-ledger-sha256']:p.add_argument('--'+name,required=True)
 a=p.parse_args();observer=LineageObserver(run_spec=a.run_spec,run_spec_sha256=a.run_spec_sha256,output=a.output,prefix_ledger=a.prefix_ledger,prefix_ledger_sha256=a.prefix_ledger_sha256)
 with (observer.out/'lineage_observer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   observer.sweep()
   if observer.complete_if_terminal():break
   time.sleep(2)
