"""Explicit evaluation-only repair continuation; original checkpoint is unchanged.

Only metadata/source-control relocation is allowed. Model, optimizer, scheduler,
RNG, data cursor and numerical training recipe are loaded from the original state.
"""
import json,math,fcntl
from contextlib import contextmanager,ExitStack
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_read,canonical

SCHEMA='action4.roi115_scoring_repair_continuation.v1'
ALLOWED_CODE={
 'src/eval/action4_corner_scoring_inputs.py',
 'src/model/training/action4_stage2_train.py',
 'src/model/training/action4_stage2_dispatch.py',
 'src/model/training/action4_scoring_repair_continuation.py',
 'src/model/training/action4_eval_publication.py'}
META={'output','run_spec','source_identity','project_code_sha256','tested_project_code_sha256',
      'stage4_execution_gate','integration_or_admission_receipt','corner_eval_source','scoring_repair'}

def _require(ok,message):
 if not ok:raise ValueError(message)

def _normalized(value,code_root,output):
 if isinstance(value,dict):return {k:_normalized(v,code_root,output) for k,v in value.items()}
 if isinstance(value,list):return [_normalized(v,code_root,output) for v in value]
 if isinstance(value,str):return value.replace(code_root,'<CODE>').replace(output,'<OUTPUT>')
 return value

def validate_numerical_recipe(old,new,oldspec,newspec):
 _require(set(new)==set(old)|{'scoring_repair'},'Repair must not add unrelated config fields')
 for cfg in (old,new):
  for key,value in dict(role='A0',mode='full_training',epochs=75,batch=48,micro=2,world_size=4,eval_every=10,roi_expansion=1.15,expected_units=10400,expected_eval_images=650).items():
   _require(type(cfg.get(key)) is type(value) and cfg[key]==value,'Changed repair numerical protocol: '+key)
 for key in set(old)-META:
  a=_normalized(old[key],oldspec['code_root'],old['output']);b=_normalized(new[key],newspec['code_root'],new['output'])
  _require(a==b,'Changed numerical/data/dependency field: '+key)
 _require(old['output']!=new['output'] and oldspec['code_root']!=newspec['code_root'],'Separate immutable repair output/source required')
 def executable(files):return {k:v for k,v in files.items() if k.startswith(('src/','scripts/','runtime/surface_deps/'))}
 a=executable(oldspec['files']);b=executable(newspec['files']);delta={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
 _require(delta<=ALLOWED_CODE and bool(delta & {'src/eval/action4_corner_scoring_inputs.py','src/model/training/action4_stage2_train.py'}),'Unreviewed source change in repair')
 return {k:dict(before=a.get(k),after=b.get(k)) for k in sorted(delta)}

def validate_state(state,old):
 _require(state.get('resumable') is True and state.get('role')=='A0' and state.get('config_id')==canonical(old) and state.get('world_size')==4,'Not exact full predecessor checkpoint')
 for key in ['epoch','cursor','step','data_update_attempt','no_supervision_update_count']:_require(type(state.get(key)) is int and state[key]>=0,'Invalid recovery counter '+key)
 ep,cur,step=state['epoch'],state['cursor'],state['step'];N=old['expected_units'];batch=old['batch']
 _require(ep<75 and cur<N and cur%batch==0 and step>0,'Invalid checkpoint data boundary')
 attempts=ep*math.ceil(N/batch)+cur//batch
 _require(state['data_update_attempt']==attempts and step+state['no_supervision_update_count']==attempts,'Recovery cursor/update ledger mismatch')
 _require(isinstance(state.get('training_model'),dict) and state['training_model'],'Missing full training state')
 opt=state['optimizer'];sched=state['scheduler'];_require(isinstance(opt.get('state'),dict) and opt['state'] and isinstance(opt.get('param_groups'),list),'Missing optimizer moments')
 _require(sched['last_epoch']==step and sched['_last_lr']==[g['lr'] for g in opt['param_groups']] and sched['base_lrs']==[g['initial_lr'] for g in opt['param_groups']],'Scheduler/optimizer state mismatch')
 _require(isinstance(state.get('rng_by_rank'),list) and len(state['rng_by_rank'])==4 and all(set(x)=={'torch','cuda','python','numpy_global','roi'} for x in state['rng_by_rank']),'Four-rank RNG state required')
 import torch
 def finite(value):
  if torch.is_tensor(value):
   _require(value.device.type=='cpu','Recovery tensor must be loaded on CPU before validation')
   if value.is_floating_point():_require(bool(torch.isfinite(value).all()),'Nonfinite recovery tensor')
  elif isinstance(value,dict):
   for x in value.values():finite(x)
  elif isinstance(value,(list,tuple)):
   for x in value:finite(x)
  elif isinstance(value,float):_require(math.isfinite(value),'Nonfinite recovery scalar')
 finite(state)

 return dict(step=step,epoch=ep,cursor=cur,data_update_attempt=attempts,no_supervision_update_count=state['no_supervision_update_count'])

@contextmanager
def predecessor_reader_locks(output):
 with ExitStack() as stack:
  for name in ['controller.lock','train.lock']:
   f=stack.enter_context((contained_path(R,output,exists=True)/name).open('r'))
   fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
  yield


def load_scoring_repair_state(cfg,resume):
 import torch
 ref=cfg.get('scoring_repair');_require(isinstance(ref,dict) and ref.get('schema')==SCHEMA,'Explicit scoring repair contract required')
 oldspec=_v2_checked(ref['predecessor_run_spec']);oldcfg=_v2_checked(dict(path=oldspec['config'],sha256=oldspec['files']['resolved_config.json']))
 spec=json.loads(contained_path(R,cfg['run_spec'],exists=True).read_text());delta=validate_numerical_recipe(oldcfg,cfg,oldspec,spec)
 _require(delta==ref['approved_source_deltas'],'Actual source deltas differ from reviewed repair')
 gate=_v2_checked(cfg['stage4_execution_gate']);_require(gate.get('scoring_repair')==ref,'Execution gate did not bind this repair')
 stopped=_v2_checked(ref['predecessor_stopped']);_require(stopped.get('status')=='OWNED_FAILED_RUN_ALL_RECORDED_PIDS_EXITED' and stopped.get('recoverable_step')==ref['boundary_step'],'Predecessor shutdown/boundary missing')
 _require(contained_path(R,resume,exists=True)==contained_path(R,ref['checkpoint']['path'],exists=True),'Unexpected recovery checkpoint')
 # Shared reader locks allow all DDP ranks and still reject an old writer.
 with predecessor_reader_locks(oldcfg['output']):
  state=_v2_read(ref['checkpoint'],lambda f:torch.load(f,map_location='cpu',weights_only=True));boundary=validate_state(state,oldcfg)
  _require(boundary['step']==ref['boundary_step'],'Different recovery step')
 lineage=dict(schema=SCHEMA,predecessor=ref['predecessor_run_spec'],checkpoint=ref['checkpoint'],predecessor_config_id=state['config_id'],successor_config_id=canonical(cfg),boundary=boundary,model_optimizer_scheduler_rng_unchanged=True,original_checkpoint_metadata_rewritten=False,reason='scoring/publication control repair only; training arithmetic unchanged')
 return state,lineage
