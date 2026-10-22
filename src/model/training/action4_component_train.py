"""Independent component training with exact update-boundary recovery and async eval."""
import json,os,math,random,hashlib,time,fcntl
from pathlib import Path
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_component_data import ComponentData,roi_batch
from data.action4_dataset import prepare_full_images,build_detection_targets
from model.action4_component_experiments import Candidate,box_loss,corner_loss
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic

def bounded(p,exists=False):return contained_path(R,Path(p),exists=exists)
def config_id(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
def source_id():
 paths=['src/model/action4_component_experiments.py','src/model/training/action4_component_train.py','src/eval/action4_component_eval.py','src/data/action4_component_data.py','src/model/action4_components.py','src/model/action4_losses.py','src/model/action4_decode.py']
 return {p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths}
def seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)

def construct(cfg,device):
 model=Candidate(cfg['role'],cfg['network'],cfg['variant']).to(device)
 init=torch.load(bounded(cfg['initialization'],True),map_location='cpu',weights_only=True)
 if init['role']!=cfg['role']:raise ValueError('Initialization role mismatch')
 missing,unexpected=model.load_state_dict(init['model'],strict=False)
 allowed=('quality_head.','token_projection.','corner_identity','visibility_head.','corner_relation.','correction_head.')
 if unexpected or any(not k.startswith(allowed) for k in missing):raise ValueError((missing,unexpected))
 return model

def optimizer_for(model,cfg):
 backbone=[];other=[]
 for n,p in model.named_parameters():(backbone if n.startswith('backbone.') else other).append(p)
 opt=torch.optim.AdamW([{'params':backbone,'lr':cfg['lr_backbone']},{'params':other,'lr':cfg['lr_new']}],weight_decay=.0001)
 total=cfg['epochs']*math.ceil(cfg['units']/cfg['batch'])
 def factor(step):
  if step<20:return max(.05,(step+1)/20)
  progress=min(1.,(step-20)/max(1,total-20));return .01+.99*(1+math.cos(math.pi*progress))/2
 sched=torch.optim.lr_scheduler.LambdaLR(opt,factor)
 return opt,sched

def rng_state(rng):return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy_global':json.dumps(np.random.get_state(),default=lambda x:x.tolist() if hasattr(x,'tolist') else x),'roi':rng.bit_generator.state}
def restore_rng(s,rng):
 torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);random.setstate(s['python'])
 n=json.loads(s['numpy_global']);np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]));rng.bit_generator.state=s['roi']

def state_for(model,opt,sched,cfg,rng,epoch,cursor,step,source):
 return dict(schema_version='action4.component_training.v1',role=cfg['role'],variant=cfg['variant'],config=cfg,config_id=config_id(cfg),source=source,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(rng),epoch=epoch,cursor=cursor,step=step,resumable=True,precision='float32_no_scaler',update_boundary=True)
def restore(state,model,opt,sched,cfg,rng,source):
 if state.get('schema_version')!='action4.component_training.v1' or not state.get('resumable') or not state.get('update_boundary'):raise ValueError('Recovery checkpoint required')
 if state['config_id']!=config_id(cfg) or state['source']!=source:raise ValueError('Resume config/source identity mismatch')
 if state['role']!=cfg['role'] or state['variant']!=cfg['variant']:raise ValueError('Resume role mismatch')
 model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);restore_rng(state['rng'],rng)
 return state['epoch'],state['cursor'],state['step']

def batch_loss(model,items,cfg,device,rng):
 if cfg['role']=='A0':
  x,meta=roi_batch(items,cfg,device,rng,True);out=model(x);return corner_loss(out,meta,cfg['variant'])
 x,aff=prepare_full_images([i['image'] for i in items],size=tuple(cfg['preprocess']['full_size']),device=device)
 targets=build_detection_targets([i['record'] for i in items],aff,full_size=tuple(cfg['preprocess']['full_size']),stride=4,device=device)
 return box_loss(model(x),targets,cfg['variant'])

def update(model,opt,sched,items,cfg,device,rng):
 model.train();opt.zero_grad(set_to_none=True);logged={}
 for start in range(0,len(items),cfg['micro']):
  mini=items[start:start+cfg['micro']];losses=batch_loss(model,mini,cfg,device,rng);total=sum(losses.values());weight=len(mini)/len(items)
  if not torch.isfinite(total):raise FloatingPointError('Nonfinite training loss')
  (total*weight).backward()
  for k,v in losses.items():logged[k]=logged.get(k,0)+float(v.detach())*weight
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
 return dict(losses=logged,gradient_norm=float(norm),lr=[g['lr'] for g in opt.param_groups])

def publish(model,cfg,out,step):
 q=out/'eval';q.mkdir(exist_ok=True);request=q/f'step_{step:08d}.json'
 if request.exists():return
 # Bound pending snapshots; do not overwrite/delete unevaluated checkpoints.
 while len([p for p in q.glob('step_*.json') if p.name.count('.')==1 and not p.with_suffix('.done.json').exists()])>=16:
  if list(q.glob('*.failed.json')):raise RuntimeError('Async eval failed')
  time.sleep(5)
 snap=out/f'eval_step_{step:08d}.pt'
 save_checkpoint_atomic(snap,dict(schema_version='action4.component_eval.v1',role=cfg['role'],variant=cfg['variant'],config=cfg,config_id=config_id(cfg),model=model.state_dict(),step=step,resumable=False))
 atomic_json(request,dict(snapshot=str(snap.relative_to(R)),step=step,config_id=config_id(cfg)))

def train(config_path,resume=None,stop_after=None):
 cfg=json.loads(bounded(config_path,True).read_text());
 if hashlib.sha256(bounded(cfg['validation_receipt'],True).read_bytes()).hexdigest()!=cfg['validation_sha256']:raise ValueError('Validation receipt changed')
 out=bounded(cfg['output']);out.mkdir(parents=True,exist_ok=True)
 guard=(out/'training.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if resume and (out/'FAILURE.json').exists():(out/'FAILURE.json').rename(out/f'FAILURE_before_resume_{time.time_ns()}.json')
 torch.set_num_threads(2);seed(cfg['seed']);device='cuda:0';source=source_id();rng=np.random.default_rng(cfg['seed']+91)
 data=ComponentData(cfg['validation_receipt'],'train',cfg['role'])
 if len(data)!=cfg['units']:raise ValueError('Dataset count mismatch')
 model=construct(cfg,device);opt,sched=optimizer_for(model,cfg);epoch=cursor=step=0
 if resume:
  state=torch.load(bounded(resume,True),map_location='cpu',weights_only=True);epoch,cursor,step=restore(state,model,opt,sched,cfg,rng,source);del state
 elif (out/'last.pt').exists():raise FileExistsError('Existing recovery requires explicit --resume')
 atomic_json(out/'PROCESS.json',dict(pid=os.getpid(),config=config_path,role=cfg['role'],variant=cfg['variant'],source=source,resumed_from=resume))
 publish(model,cfg,out,step)
 from torch.utils.data import DataLoader
 for ep in range(epoch,cfg['epochs']):
  order=np.random.default_rng(cfg['seed']+ep*1000003).permutation(len(data)).tolist();start_cursor=cursor if ep==epoch else 0
  batches=[order[i:i+cfg['batch']] for i in range(start_cursor,len(order),cfg['batch'])]
  loader=DataLoader(data,batch_sampler=batches,collate_fn=list,num_workers=cfg['loader_workers'],prefetch_factor=2 if cfg['loader_workers'] else None,generator=torch.Generator().manual_seed(cfg['seed']+ep))
  seen=start_cursor
  for items in loader:
   begin=time.monotonic();metrics=update(model,opt,sched,items,cfg,device,rng);seen+=len(items);step+=1
   next_ep,next_cursor=(ep+1,0) if seen==len(data) else (ep,seen)
   record=dict(step=step,epoch=ep+1,cursor=seen,seconds=time.monotonic()-begin,peak_memory_mib=torch.cuda.max_memory_allocated()/1048576,**metrics)
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   atomic_json(out/'STATUS.json',record)
   if step%cfg['eval_every']==0 or next_cursor==0 or (stop_after and step>=stop_after):
    save_checkpoint_atomic(out/'last.pt',state_for(model,opt,sched,cfg,rng,next_ep,next_cursor,step,source))
   if step%cfg['eval_every']==0:publish(model,cfg,out,step)
   if list((out/'eval').glob('*.failed.json')):raise RuntimeError('Async evaluation failed; recovery retained')
   if stop_after and step>=stop_after:return
  cursor=0
 publish(model,cfg,out,step);atomic_json(out/'TRAIN_COMPLETE.json',dict(step=step,epochs=cfg['epochs']))

if __name__=='__main__':
 import argparse,traceback
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--stop-after',type=int);a=p.parse_args()
 try:train(a.config,a.resume,a.stop_after)
 except BlockingIOError:raise
 except Exception:
  cfg=json.loads(bounded(a.config,True).read_text());out=bounded(cfg['output']);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'FAILURE.json',{'traceback':traceback.format_exc()});raise
