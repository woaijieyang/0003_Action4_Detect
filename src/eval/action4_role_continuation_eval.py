"""Authenticated immutable prefix + successor evaluation lineage. No process control."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,os,shutil,importlib.util,math
from data.action4_paths import PROJECT_ROOT as R,contained_path

OBSERVER_PATH='outputs/action4_training_candidate_selection_v001/source/observer.py'
OBSERVER_SHA='23bec9d4a870fd93416492dcfd29dc609096638ebb4212cdf10dab0b63832827'
_CACHE={}
_REF_CACHE={}
_OBSERVER=None

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def inside(path,exists=True):return contained_path(R,path,exists=exists)
def stamp(path):
 s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def reference(path):
 p=inside(path);before=stamp(p);key=str(p)
 if key in _REF_CACHE and _REF_CACHE[key][0]==before:return dict(_REF_CACHE[key][1])
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 if stamp(p)!=before:raise ValueError('Source changed while authenticating')
 ref={'path':str(p.relative_to(R)),'sha256':sha};_REF_CACHE[key]=(before,ref);return dict(ref)
def verify_ref(ref):
 if reference(ref['path'])!=ref:raise ValueError('Reference SHA changed: '+ref['path'])
def read_ref(ref):
 verify_ref(ref);return inside(ref['path']).read_bytes()
def json_ref(ref):return json.loads(read_ref(ref))
def atomic(path,value):
 p=inside(path,False);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def pinned_observer():
 global _OBSERVER
 p=inside(OBSERVER_PATH)
 if _OBSERVER is None:
  if reference(p)['sha256']!=OBSERVER_SHA:raise ValueError('Original observer source changed')
  spec=importlib.util.spec_from_file_location('action4_original_training_observer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  _OBSERVER=(m,stamp(p))
 if stamp(p)!=_OBSERVER[1]:raise ValueError('Observer changed after import')
 return _OBSERVER[0]
def legacy_rank(row):
 # Exact existing A0 rank_result for the registered all8 schema. The original
 # training rank validates counts; deployment eligibility is legacy's first key.
 return (row['deployment_eligible'],)+pinned_observer().rank(row['report'],row['step'])
def validate_report(row,cfg,step):
 if (row.get('role'),row.get('config_id'),row.get('step'))!=('A0',canonical(cfg),step):raise ValueError('Foreign report identity')
 if row.get('training_mode')!=cfg['mode'] or type(row.get('deployment_eligible')) is not bool:raise ValueError('Report mode/deployment eligibility missing')
 report=row['report']
 if cfg.get('expected_eval_images')!=650 or report.get('evaluation_images')!=650 or report.get('score_schema')!=cfg['score_schema']:raise ValueError('Full650/current score schema required')
 legacy_rank(row)
 return row

def _source_config(spec_ref):
 spec=json_ref(spec_ref);code=inside(spec['code_root']);cp=inside(spec['config']);rel=str(cp.relative_to(code))
 cr=reference(cp)
 if cr['sha256']!=spec['files'][rel]:raise ValueError('RunSpec config source mismatch')
 cfg=json.loads(cp.read_bytes())
 if cfg['role']!='A0' or cfg['eval_every']!=10 or cfg['expected_eval_images']!=650:raise ValueError('Registered A0 eval10/full650 required')
 semantics=[]
 for name in ['src/eval/action4_stage2_eval.py','src/eval/action4_pose_protocol.py']:
  ref=reference(code/name)
  if ref['sha256']!=spec['files'][name]:raise ValueError('Frozen scoring source mismatch')
  semantics.append(ref)
 return cfg,cr,semantics

def create_prefix_ledger(directory,*,old_run_spec,old_run_spec_sha256,checkpoint_path,boundary_step,observer_output):
 """Root calls only AFTER pause/drain. Never writes parent artifacts."""
 if type(boundary_step) is not int or boundary_step<=0 or boundary_step%10:raise ValueError('Positive eval10 checkpoint boundary required')
 spec_ref={'path':str(inside(old_run_spec).relative_to(R)),'sha256':old_run_spec_sha256}
 cfg,config_ref,semantics=_source_config(spec_ref);run=inside(cfg['output']);obs=inside(observer_output)
 cp_ref=reference(checkpoint_path)
 import torch
 state=torch.load(inside(checkpoint_path),map_location='cpu',weights_only=True,mmap=True)
 if (state.get('step'),state.get('config_id'),state.get('world_size'),state.get('resumable'))!=(boundary_step,canonical(cfg),cfg['world_size'],True):raise ValueError('Boundary checkpoint identity mismatch')
 if not {'training_model','optimizer','scheduler','rng_by_rank','epoch','cursor'}<=set(state):raise ValueError('Incomplete recovery state')
 del state
 expected=list(range(0,boundary_step+1,10));entries=[];reports=[]
 for step in expected:
  q=run/'eval'/f'step_{step:08d}.json';done=q.with_suffix('.done.json')
  req=json.loads(q.read_bytes())
  if q.with_suffix('.failed.json').exists():raise ValueError('Prefix evaluation failed')
  if (req.get('role'),req.get('config_id'),req.get('step'))!=('A0',canonical(cfg),step):raise ValueError('Prefix request identity mismatch')
  dr=reference(done);row=validate_report(json_ref(dr),cfg,step);raw_ref=reference(row['raw_report']);raw=json_ref(raw_ref)
  if any(row['report'].get(k)!=v for k,v in raw.items()):raise ValueError('Raw report was overwritten in done result')
  entries.append({'step':step,'request':reference(q),'done':dr,'raw_report':raw_ref});reports.append(row)
 ref=reference(run/'REFERENCE_REPORT.json')
 if json_ref(ref)!=reports[0]['report']:raise ValueError('Parent reference differs from original step0 report')
 best_ref=reference(run/'BEST.json');best=json_ref(best_ref);winner=max(reports,key=legacy_rank)
 if best['step']!=winner['step'] or best['config_id']!=canonical(cfg) or best['report']!=winner['report']:raise ValueError('Legacy BEST does not match prefix rank_result')
 best_weight=reference(best['checkpoint'])
 training_ref=reference(obs/'TRAINING_BEST.json');training=json_ref(training_ref)
 if training['config_id']!=canonical(cfg) or training['step'] not in expected:raise ValueError('Foreign parent training best')
 observer_identity_ref=reference(obs/'OBSERVER_IDENTITY.json');oi=json_ref(observer_identity_ref)
 if oi['config_id']!=canonical(cfg) or oi['run']!=str(run.relative_to(R)):raise ValueError('Observer identity mismatch')
 records=[];available=[]
 for entry in entries:
  rr=reference(obs/'evaluated'/f"step_{entry['step']:08d}.json");record=json_ref(rr);row=reports[expected.index(entry['step'])]
  if record['config_id']!=canonical(cfg) or record['step']!=entry['step'] or record['source_done']!=entry['done'] or any(row['report'].get(k)!=v for k,v in record['metrics'].items()):raise ValueError('Observer record differs from authenticated evaluation')
  records.append(rr);weight=obs/'selected'/f"step_{entry['step']:08d}.pt"
  if weight.exists():available.append({'record':rr,'checkpoint_path':weight,'rank':pinned_observer().rank(record['metrics'],record['step'])})
 training_winner=max((json_ref(x) for x in records),key=lambda z:pinned_observer().rank(z['metrics'],z['step']))
 if training['step']!=training_winner['step'] or training['metrics']!=training_winner['metrics']:raise ValueError('Parent training best is not complete fullAll8 winner')
 training_weight=reference(training['checkpoint'])
 if training_weight['sha256']!=training['checkpoint_sha256']:raise ValueError('Parent training weight changed')
 available.sort(key=lambda z:z['rank'],reverse=True)
 target=inside(directory,False)
 if target.exists():raise FileExistsError('Prefix destination must be new')
 target.mkdir(parents=True);(target/'retained').mkdir()
 def preserve(ref,name):
  dest=target/'retained'/name;shutil.copyfile(inside(ref['path']),dest);dest.chmod(0o444)
  held=reference(dest)
  if held['sha256']!=ref['sha256']:raise ValueError('Retained weight copy mismatch')
  return held
 held_legacy=preserve(best_weight,'legacy_best.pt');held_training=preserve(training_weight,'training_best.pt')
 captured_best=preserve(best_ref,'legacy_BEST.json');captured_training=preserve(training_ref,'TRAINING_BEST.json')
 provenance=lambda ref:{'original_path':ref['path'],'original_sha256':ref['sha256']}
 candidates=[]
 for item in available[:4]:
  record=json_ref(item['record']);wr=reference(item['checkpoint_path']);held=preserve(wr,f"candidate_{record['step']:08d}.pt");candidates.append({'record':item['record'],'checkpoint':held,'retained':held,'checkpoint_provenance':provenance(wr)})
 ledger={'schema':'action4.role_continuation_prefix.v1','boundary_step':boundary_step,'run_spec':spec_ref,'config':config_ref,'scoring_sources':semantics,
  'checkpoint':cp_ref,'entries':entries,'reference':ref,'legacy_best':captured_best,'legacy_weight':held_legacy,'retained_legacy_weight':held_legacy,
  'observer_source':{'path':OBSERVER_PATH,'sha256':OBSERVER_SHA},'observer_identity':observer_identity_ref,'observer_records':records,
  'training_best':captured_training,'training_weight':held_training,'retained_training_weight':held_training,'training_candidates':candidates,
  'mutable_parent_provenance':{'legacy_best':provenance(best_ref),'training_best':provenance(training_ref),'legacy_weight':provenance(best_weight),'training_weight':provenance(training_weight)},
  'reports_keep_original_config':True,'parent_assets_pruned':False}
 atomic(target/'PREFIX.json',ledger);return reference(target/'PREFIX.json')

@dataclass(frozen=True)
class PrefixReports:
 identity_json:bytes
 ledger_json:bytes
 reports_json:tuple
 tracked:tuple
 @property
 def identity(self):return json.loads(self.identity_json)
 @property
 def ledger(self):return json.loads(self.ledger_json)
 @property
 def reports(self):return tuple(json.loads(x) for x in self.reports_json)
 def assert_unchanged(self):
  if any(stamp(inside(path))!=sig for path,sig in self.tracked):raise ValueError('Authenticated prefix changed after admission')

def load_prefix_reports(path,sha256):
 """Verify once per process; keep returned object. Cache reuse checks cheap stats."""
 key=(str(inside(path)),sha256)
 if key in _CACHE:_CACHE[key].assert_unchanged();return _CACHE[key]
 ledger_ref={'path':str(inside(path).relative_to(R)),'sha256':sha256};ledger=json_ref(ledger_ref)
 if ledger.get('schema')!='action4.role_continuation_prefix.v1':raise ValueError('Foreign prefix schema')
 refs=[ledger_ref]
 def walk(x):
  if isinstance(x,dict):
   if set(x)=={'path','sha256'}:verify_ref(x);refs.append(x)
   else:
    for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(ledger);cfg,cr,semantics=_source_config(ledger['run_spec']);S=ledger['boundary_step']
 if ledger['config']!=cr or ledger['scoring_sources']!=semantics:raise ValueError('Ledger source/config identity mismatch')
 if type(S) is not int or S<=0 or S%10 or [e['step'] for e in ledger['entries']]!=list(range(0,S+1,10)):raise ValueError('Incomplete/duplicate prefix steps')
 reports=tuple(validate_report(json_ref(e['done']),cfg,e['step']) for e in ledger['entries'])
 for entry,row in zip(ledger['entries'],reports):
  if any(row['report'].get(k)!=v for k,v in json_ref(entry['raw_report']).items()):raise ValueError('Raw/done source mismatch')
 if json_ref(ledger['reference'])!=reports[0]['report']:raise ValueError('Reference mismatch')
 best=json_ref(ledger['legacy_best']);winner=max(reports,key=legacy_rank)
 if (best['step'],best['config_id'],best['report'])!=(winner['step'],winner['config_id'],winner['report']):raise ValueError('Prefix legacy best mismatch')
 provenance=ledger.get('mutable_parent_provenance')
 if not isinstance(provenance,dict):raise ValueError('Captured mutable-parent provenance required')
 for name in ('legacy_best','training_best','legacy_weight','training_weight'):
  if provenance[name]['original_sha256']!=ledger[name]['sha256']:raise ValueError('Captured bytes differ from original provenance')
  inside(provenance[name]['original_path'],False) # containment only; original may be retired.
 if str(inside(best['checkpoint'],False).relative_to(R))!=provenance['legacy_weight']['original_path']:raise ValueError('Captured legacy checkpoint provenance mismatch')
 for candidate in ledger['training_candidates']:
  if candidate['checkpoint_provenance']['original_sha256']!=candidate['retained']['sha256']:raise ValueError('Candidate retained SHA differs from provenance')
 if ledger['observer_source']!={'path':OBSERVER_PATH,'sha256':OBSERVER_SHA}:raise ValueError('Unapproved training observer source')
 records=[json_ref(x) for x in ledger['observer_records']]
 if [x['step'] for x in records]!=list(range(0,S+1,10)):raise ValueError('Observer prefix coverage mismatch')
 for record,entry,row in zip(records,ledger['entries'],reports):
  if record['config_id']!=canonical(cfg) or record['source_done']!=entry['done'] or any(row['report'].get(k)!=v for k,v in record['metrics'].items()):raise ValueError('Observer metric/source mismatch')
 tb=json_ref(ledger['training_best'])
 if str(inside(tb['checkpoint'],False).relative_to(R))!=provenance['training_weight']['original_path']:raise ValueError('Captured training checkpoint provenance mismatch')
 tw=max(records,key=lambda x:pinned_observer().rank(x['metrics'],x['step']))
 if (tb['step'],tb['metrics'],tb['config_id'],tb['checkpoint_sha256'])!=(tw['step'],tw['metrics'],canonical(cfg),ledger['training_weight']['sha256']):raise ValueError('Training winner mismatch')
 import torch
 cp=torch.load(inside(ledger['checkpoint']['path']),map_location='cpu',weights_only=True,mmap=True)
 if (cp.get('step'),cp.get('config_id'),cp.get('resumable'))!=(S,canonical(cfg),True):raise ValueError('Prefix checkpoint boundary mismatch')
 del cp
 result=PrefixReports(json.dumps(ledger_ref).encode(),json.dumps(ledger).encode(),tuple(json.dumps(x).encode() for x in reports),tuple((r['path'],stamp(inside(r['path']))) for r in refs))
 _CACHE[key]=result;return result

def initialize_successor(out,prefix):
 prefix.assert_unchanged();out=inside(out);ledger=prefix.ledger
 targets=[out/'REFERENCE_REPORT.json',out/'BEST.json',out/'best.pt',out/'PREFIX_INITIALIZATION.json']
 if any(p.exists() for p in targets):raise FileExistsError('Successor initialization is one-time')
 (out/'REFERENCE_REPORT.json').write_bytes(read_ref(ledger['reference']))
 (out/'BEST.json').write_bytes(read_ref(ledger['legacy_best']))
 # Copy avoids mutating parent retained-weight inode/ctime while prefix is sealed.
 shutil.copyfile(inside(ledger['retained_legacy_weight']['path']),out/'best.pt')
 atomic(out/'PREFIX_INITIALIZATION.json',{'schema':'action4.prefix_initialization.v1','prefix_ledger':prefix.identity,'boundary_step':ledger['boundary_step'],'prefix_config_id':canonical(json_ref(ledger['config'])),'prefix_run_spec':ledger['run_spec'],'prefix_config':ledger['config'],'prefix_checkpoint':ledger['checkpoint'],'prefix_legacy_best':ledger['legacy_best'],'best_retains_original_config':True,'best_alias_sha256':ledger['retained_legacy_weight']['sha256'],'reference':ledger['reference'],'no_parent_mutation':True})

def combine_reports(prefix,successor_out,successor_config,*,final_step=None):
 prefix.assert_unchanged();S=prefix.ledger['boundary_step'];reports=list(prefix.reports);out=inside(successor_out);seen={r['step'] for r in reports}
 if successor_config['output']!=str(out.relative_to(R)):raise ValueError('Successor output/config mismatch')
 old_cfg=json_ref(prefix.ledger['config'])
 for name in ('role','score_schema','expected_eval_images','validation_receipt','validation_receipt_sha256','evaluation_counterpart_sha256','evaluation_scoring_policy','b0_scoring_policy','dataset_root'):
  if old_cfg.get(name)!=successor_config.get(name):raise ValueError('Evaluation semantics changed: '+name)
 for q in sorted((out/'eval').glob('step_*.json')):
  if q.name.count('.')!=1:continue
  req=json.loads(q.read_bytes());step=req['step']
  if type(step) is not int or q.stem!=f'step_{step:08d}':raise ValueError('Suffix request filename/step mismatch')
  if step<=S or step in seen:raise ValueError('Duplicate/overlapping lineage step')
  if (req.get('role'),req.get('config_id'))!=('A0',canonical(successor_config)):raise ValueError('Foreign successor request')
  done=q.with_suffix('.done.json')
  if not done.exists():
   if final_step is not None:raise ValueError('Suffix evaluation incomplete')
   continue
  if q.with_suffix('.failed.json').exists():raise ValueError('Suffix evaluation failed')
  row=validate_report(json.loads(done.read_bytes()),successor_config,step)
  raw=json.loads(inside(row['raw_report']).read_bytes())
  if any(row['report'].get(k)!=v for k,v in raw.items()):raise ValueError('Suffix raw report mismatch')
  seen.add(step);reports.append(row)
 if final_step is not None:
  expected={0,final_step}|set(range(10,final_step+1,10))
  if final_step<S or seen!=expected:raise ValueError('Missing/extra full-lineage evaluation steps')
 return tuple(reports)

def finalize_lineage(prefix,out,cfg):
 """Call after actual train terminal and suffix drain; no pruning/default updates."""
 out=inside(out);terminal=json.loads((out/'TRAIN_COMPLETE.json').read_bytes());step=terminal['step']
 if cfg.get('mode')!='full_training' or terminal.get('config_id')!=canonical(cfg) or terminal.get('formal_training_complete') is not True or step+terminal.get('no_supervision_update_count',0)!=cfg['epochs']*math.ceil(cfg['expected_units']/cfg['batch']):raise ValueError('No valid full-training completion')
 reports=combine_reports(prefix,out,cfg,final_step=step);winner=max(reports,key=legacy_rank);best=json.loads((out/'BEST.json').read_bytes())
 if (best['step'],best['config_id'],best['report'])!=(winner['step'],winner['config_id'],winner['report']):raise ValueError('Global legacy BEST mismatch')
 parent_best=json_ref(prefix.ledger['legacy_best'])
 if (winner['config_id'],winner['step'])==(parent_best['config_id'],parent_best['step']):
  if best['checkpoint']!=parent_best['checkpoint']:raise ValueError('Parent BEST checkpoint provenance changed')
  weight=prefix.ledger['retained_legacy_weight'];verify_ref(weight)
 else:weight=reference(best['checkpoint'])
 alias=reference(out/'best.pt')
 if alias['sha256']!=weight['sha256']:raise ValueError('Legacy best alias bytes mismatch')
 result={'schema':'action4.complete_role_lineage.v1','status':'COMPLETE_EVAL_LINEAGE_NOT_ACCURACY_ADOPTION','steps':sorted(r['step'] for r in reports),'prefix_boundary':prefix.ledger['boundary_step'],'suffix_config_id':canonical(cfg),'winning_original_config_id':winner['config_id'],'legacy_best_step':winner['step'],'legacy_weight':weight,'legacy_alias':alias,'parent_assets_pruned':False}
 atomic(out/'ONELINEAGE_EVAL_COMPLETE.json',result);return result


def commit_lineage_result(out,request,snapshot,result,rank):
 """Worker already holds selection.lock. Preserve parent record and never prune."""
 out=inside(out);best_path=out/'BEST.json';best=json.loads(best_path.read_bytes()) if best_path.exists() else None
 if best is None or rank(result)>rank(best):
  selected=out/f"selected_step_{result['step']:08d}.pt"
  if not selected.exists():os.link(snapshot,selected)
  elif not os.path.samefile(snapshot,selected):raise ValueError('Successor selected weight collision')
  best=dict(result,checkpoint=str(selected));atomic(best_path,best)
  temp=out/f'best_alias_{os.getpid()}.tmp'
  os.link(selected,temp);os.replace(temp,out/'best.pt')
 # Losing prefix BEST remains byte-for-byte unchanged, including old config/path.
 atomic(request.with_suffix('.done.json'),result)
 # Do not unlink pending snapshots here: unchanged CandidateObserver must pin
 # them even when it sweeps after a very fast completion. Separate retention later.


def worker(config,worker_id,*,prefix_ledger,prefix_sha256):
 prefix=load_prefix_reports(prefix_ledger,prefix_sha256);cfg=json.loads(inside(config).read_bytes());out=inside(cfg['output'])
 init=json.loads((out/'PREFIX_INITIALIZATION.json').read_bytes())
 if init.get('prefix_ledger')!=prefix.identity or init['prefix_legacy_best']!=prefix.ledger['legacy_best']:raise ValueError('Wrong prefix initialization')
 if (out/'REFERENCE_REPORT.json').read_bytes()!=read_ref(prefix.ledger['reference']):raise ValueError('Successor reference changed')
 from eval import action4_stage2_eval as base
 def commit(output,request,snapshot,result,rank):
  if inside(output)!=out or inside(request).parent!=out/'eval' or inside(snapshot).parent!=out:raise ValueError('Continuation worker output escaped')
  validate_report(result,cfg,result['step'])
  if result['step']<=prefix.ledger['boundary_step']:raise ValueError('Continuation worker cannot rewrite prefix')
  return commit_lineage_result(output,request,snapshot,result,rank)
 base.commit_result=commit
 def no_prune(output):
  if inside(output)!=out:raise ValueError('Foreign retention target')
  return []
 base.prune_old_selected=no_prune
 if not str(worker_id).replace('_','').isalnum():raise ValueError('Invalid worker identity')
 atomic(out/f'LINEAGE_WORKER_{worker_id}.json',{'prefix_ledger':{'path':str(inside(prefix_ledger).relative_to(R)),'sha256':prefix_sha256},'config_id':canonical(cfg),'pid':os.getpid(),'hooks_scope':'this successor process only','original_inference_and_scoring_unchanged':True,'pruning_enabled':False})
 return base.worker(config,worker_id)


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker-id',required=True)
 p.add_argument('--prefix-ledger',required=True);p.add_argument('--prefix-ledger-sha256',required=True);a=p.parse_args()
 worker(a.config,a.worker_id,prefix_ledger=a.prefix_ledger,prefix_sha256=a.prefix_ledger_sha256)
