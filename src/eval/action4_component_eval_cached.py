"""Cached revision of eval execution only; frozen scoring/model code is reused."""
import json,os,time,fcntl,traceback
import torch
from model.training.action4_component_train import bounded,config_id
from model.training.action4_runtime import atomic_json
from model.action4_component_experiments import Candidate
from data.action4_component_data import ComponentData
from data.action4_dataset import Action4Dataset
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_component_eval import evaluate,rank

class CachedComponentData(ComponentData):
 """Cache validated raw RGB; retain per-read file identity check and ROI identity."""
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.images={}
 def __getitem__(self,index):
  ri,gi=self.units[index];record=self.records[ri]
  if ri not in self.images:self.images[ri]=super().__getitem__(index)['image']
  p=contained_path(R,record['image'],exists=True)
  if Action4Dataset._file_identity(p)!=tuple(self.stats[record['image']]):raise ValueError('Cached RGB source changed')
  return {'image':self.images[ri],'record':record,'instance':gi}

def worker(suite):
 torch.set_num_threads(1);settings=json.loads(bounded(suite,True).read_text());configs=[json.loads(bounded(p,True).read_text()) for p in settings['configs']];data={};model=None;last_key=None
 while True:
  claimed=None
  # Oldest unclaimed request first within this worker's assigned experiment(s).
  available=[]
  for cfg in configs:
   out=bounded(cfg['output']);q=out/'eval'
   for req in q.glob('step_*.json'):
    if req.name.count('.')!=1 or req.with_suffix('.claim').exists() or req.with_suffix('.done.json').exists() or req.with_suffix('.failed.json').exists():continue
    available.append((req.stat().st_mtime_ns,cfg,out,req))
  for _,cfg,out,req in sorted(available,key=lambda x:x[0]):
   try:fd=os.open(req.with_suffix('.claim'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
   except FileExistsError:continue
   with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'time':time.time(),'engine':'rgb_cache_v2'},f)
   claimed=(cfg,out,req);break
  if claimed is None:
   if all((bounded(c['output'])/'TRAIN_COMPLETE.json').exists() or (bounded(c['output'])/'FAILURE.json').exists() for c in configs):return
   time.sleep(3);continue
  cfg,out,req=claimed
  try:
   begin=time.monotonic();request=json.loads(req.read_text());state=torch.load(bounded(request['snapshot'],True),map_location='cpu',weights_only=True)
   if state['schema_version']!='action4.component_eval.v1' or state['config_id']!=config_id(cfg) or state['step']!=request['step']:raise ValueError('Snapshot identity mismatch')
   key=(cfg['role'],cfg['variant'])
   if key!=last_key:
    del model;model=Candidate(cfg['role'],cfg['network'],cfg['variant']).to('cuda:0');last_key=key
   model.load_state_dict(state['model'],strict=True);del state
   if cfg['role'] not in data:data[cfg['role']]=CachedComponentData(cfg['validation_receipt'],'eval',cfg['role'])
   report=evaluate(model,data[cfg['role']],cfg,'cuda:0');result={'step':request['step'],'snapshot':request['snapshot'],'report':report,'eval_seconds':time.monotonic()-begin,'execution_engine':'rgb_cache_v2'}
   atomic_json(req.with_suffix('.done.json'),result)
   with (out/'selection.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);bp=out/'BEST.json';best=json.loads(bp.read_text()) if bp.exists() else None
    if best is None or rank(report,cfg['role'])>rank(best['report'],cfg['role']):
     link=out/f'best.{os.getpid()}.tmp';os.link(bounded(request['snapshot'],True),link);os.replace(link,out/'best.pt');atomic_json(bp,result)
    bounded(request['snapshot'],True).unlink();fcntl.flock(lock,fcntl.LOCK_UN)
  except Exception:
   atomic_json(req.with_suffix('.failed.json'),{'error':traceback.format_exc()});raise

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();worker(a.suite)
