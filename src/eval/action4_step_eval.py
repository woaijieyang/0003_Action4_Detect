"""Candidate asynchronous A0 evaluation queue. Snapshots are inference-only."""
import json,os,time,fcntl
from pathlib import Path

def bounded(root,path):
 p=(Path(root)/path).resolve()
 if not p.is_relative_to(Path(root).resolve()):raise ValueError('Outside project')
 return p

def publish(root,queue,snapshot,step,cumulative_step):
 from model.training.action4_runtime import atomic_json
 q=bounded(root,queue);q.mkdir(parents=True,exist_ok=True);p=bounded(root,snapshot)
 if not p.is_file() or type(step) is not int or step<1:raise ValueError('Completed snapshot and positive optimizer step required')
 target=q/f'step_{step:08d}.json'
 if target.exists():raise FileExistsError(target)
 atomic_json(target,{'schema':'action4.step_eval_request.v1','snapshot':str(p.relative_to(Path(root))),'step':step,'cumulative_step':cumulative_step,'status':'pending'})
 return target

def claim(queue):
 q=Path(queue)
 for path in sorted(q.glob('step_*.json')):
  if not __import__('re').fullmatch(r'step_[0-9]{8}\.json',path.name):continue
  if path.with_suffix('.done.json').exists() or path.with_suffix('.failed.json').exists():continue
  lock=path.with_suffix('.claim')
  try:fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  except FileExistsError:continue
  with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'time':time.time()},f)
  return path,json.loads(path.read_text())
 return None

def record_result(queue,request,report):
 from model.training.action4_runtime import atomic_json
 q=Path(queue);path=Path(request);value=report['mAP50_95'] if 'mAP50_95' in report else report['thresholds']['0.05']['f1'] if 'thresholds' in report else report['epe_p95_heatmap_px']
 if not isinstance(value,(int,float)) or not __import__('math').isfinite(value):raise ValueError('Finite validation score required')
 result={**json.loads(path.read_text()),'status':'complete','report':report}
 atomic_json(path.with_suffix('.done.json'),result)
 with (q/'selection.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  candidates=[json.loads(p.read_text()) for p in q.glob('step_*.done.json')]
  def rank(x):
   r=x['report']
   if 'mAP50_95' in r:return (-r['mAP50_95'],-r['overall']['f1'],x['step'])
   if 'thresholds' in r:return (-r['thresholds']['0.05']['f1'],-r['thresholds']['0.02']['f1'],x['step'])
   return (r['epe_p95_heatmap_px'],r['epe_mean_heatmap_px'],x['step'])
  candidates.sort(key=rank)
  atomic_json(q/'selection.json',{'schema':'action4.step_eval_selection.v1','metric':('B0 complete-body mAP50:95, tie boxF1; development selection only' if 'mAP50_95' in report else 'fullRGB F1@5%, tie F1@2%, higher is better' if 'thresholds' in report else 'GT ROI P95 heatmap px, lower is better'),'completed':len(candidates),'top3':candidates[:3],'best_snapshot':candidates[0]['snapshot'],'resumable':False,'scope':'model selection only; epoch last checkpoint is separate recovery state'})
  if (q/'retention.json').exists():
   from data.action4_paths import PROJECT_ROOT
   retained={x['snapshot'] for x in candidates[:3]}
   for item in candidates[3:]:
    target=bounded(PROJECT_ROOT,item['snapshot'])
    if item['snapshot'] in retained:continue
    if not __import__('re').fullmatch(r'eval_step_[0-9]{8}\.pt',target.name):raise ValueError('Unsafe snapshot retention path')
    if target.parent.name!=q.parent.name:raise ValueError('Snapshot belongs to another run')
    if target.exists():target.unlink()
   atomic_json(q/'retention_status.json',{'retained_top3':sorted(retained),'completed_evals':len(candidates),'pending_snapshots_untouched':True,'all_metrics_preserved':True})
  fcntl.flock(f,fcntl.LOCK_UN)

def worker(root,queue,device,cpu_threads=2,parent_pid=None):
 import torch
 from model.training.action4_runtime import load_checkpoint,input_path,atomic_json,seed_all
 from eval.action4_runner import build_model,evaluate_gt_rois,evaluate_records
 from data.action4_dataset import Action4Dataset
 torch.set_num_threads(cpu_threads);q=bounded(root,queue);dataset=None;identity=None;model=None
 while True:
  if parent_pid is not None:
   try:os.kill(parent_pid,0)
   except ProcessLookupError:return
  item=claim(q)
  if item is None:
   if (q/'producer_complete.json').exists():return
   time.sleep(5);continue
  path,request=item
  try:
   state=load_checkpoint(input_path(root,request['snapshot']))
   if state.get('stage') not in ('A0','B0') or state.get('claim_scope')!='accepted_data_model_training':raise ValueError('Formal A0 snapshot required')
   if state['global_step']!=request['step']:raise ValueError('Snapshot step mismatch')
   stage=state['stage'];cfg=state['config'];current=(state['identity']['data_sha256']['eval'],json.dumps(cfg['network'],sort_keys=True),json.dumps(cfg['preprocess'],sort_keys=True))
   if identity is not None and current!=identity:raise ValueError('Worker input contract changed')
   if dataset is None:
    from model.training.action4_runtime import sha256_file
    if sha256_file(input_path(root,cfg['data']['eval_manifest']))!=state['identity']['data_sha256']['eval']:raise ValueError('Eval manifest changed')
    dataset=Action4Dataset(root,input_path(root,cfg['data']['eval_manifest']),split='eval',validation_workers=cpu_threads);identity=current
    model=build_model(cfg,device=device)
   from model.training.action4_stage_policy import prepare_checkpoint_topology
   prepare_checkpoint_topology(model,state)
   model.load_state_dict(state['model'],strict=True);del state
   seed_all(cfg['runtime']['seed'],deterministic=True)
   if stage=='B0':
    import copy
    from data.action4_dataset import prepare_full_images
    from eval.action4_runner import decode_full_boxes
    from eval.action4_bbox_metrics import score_boxes
    decode_cfg=copy.deepcopy(cfg);decode_cfg['decode']['score_threshold']=cfg.get('bbox_evaluation',{}).get('AP_candidate_threshold',.001)
    predictions=[]
    with torch.inference_mode():
     model.eval()
     for index in range(len(dataset)):
      item=dataset[index];full,aff=prepare_full_images([item['image']],size=tuple(cfg['preprocess']['full_size']),device=device)
      boxes=decode_full_boxes(model(full),aff,decode_cfg)[0]
      predictions.append({'sample_id':item['record']['sample_id'],'instances':boxes})
    report=score_boxes(predictions,dataset.records,score_threshold=cfg['decode']['score_threshold'])
    report['AP_candidate_threshold']=decode_cfg['decode']['score_threshold'];report['max_detections_per_image']=cfg['decode']['max_detections'];report['dataset_role']='development_selection_not_independent_acceptance'

   else:report=evaluate_gt_rois(model,dataset,cfg,device,batch_size=cfg['eval'].get('batch_size',8))
   record_result(q,path,report)
  except Exception as exc:
   atomic_json(path.with_suffix('.failed.json'),{**request,'status':'failed','error':str(exc)})
   raise

class AsyncStepEvaluator:
 def __init__(self,root,output_dir,model_dir,config):
  import subprocess,sys
  self.root=Path(root);self.queue=Path(output_dir)/'step_eval';self.queue.mkdir(exist_ok=True);self.models=Path(model_dir);self.workers=[];self.handles=[]
  settings=config['stages'][config.get('async_eval_stage','A0')]['async_eval'];self.settings=settings;self.offset=settings.get('parent_optimizer_steps',0)
  free=int(subprocess.check_output(['nvidia-smi','-i',settings['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free<settings.get('minimum_free_mib',16384):raise RuntimeError('Dedicated eval GPU has insufficient free memory; no workers started')
  if settings.get('retain_completed_top3_only',False):
   (self.queue/'retention.json').write_text(json.dumps({'policy':'top3 plus all pending; completed non-top3 own-run snapshots removed'}))
  for i in range(settings['workers']):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=settings['gpu_uuid'],OMP_NUM_THREADS=str(settings['cpu_threads']),MKL_NUM_THREADS=str(settings['cpu_threads']),OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(self.root/'src'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
   for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
   f=(self.queue/f'worker_{i}_{time.time_ns()}.log').open('x');self.handles.append(f)
   process=subprocess.Popen([sys.executable,'-m','eval.action4_step_eval','--root',str(root),'--queue',str(self.queue),'--device','cuda:0','--threads',str(settings['cpu_threads']),'--parent-pid',str(os.getpid())],cwd=root,env=env,stdout=f,stderr=f)
   self.workers.append(process)
  from model.training.action4_runtime import atomic_json
  atomic_json(self.queue/'workers.json',{'pids':[p.pid for p in self.workers],'gpu_uuid':settings['gpu_uuid'],'cpu_total':settings['workers']*settings['cpu_threads']})
 def check(self):
  if list(self.queue.glob('step_*.failed.json')):raise RuntimeError('Asynchronous eval failed; preserve failure reports')
  if any(p.poll() is not None for p in self.workers):raise RuntimeError('Evaluation worker exited before producer completion')
 def submit(self,state):
  from model.training.action4_runtime import save_checkpoint_atomic
  self.check()
  while self.settings.get('retain_completed_top3_only',False) and sum(1 for p in self.queue.glob('step_*.json') if __import__('re').fullmatch(r'step_[0-9]{8}\.json',p.name) and not p.with_suffix('.done.json').exists())>=64:
   self.check();time.sleep(5)
  step=state['global_step'];path=self.models/f'eval_step_{step:08d}.pt'
  if path.exists():raise FileExistsError(path)
  state={k:v for k,v in state.items() if k not in ('optimizer','scheduler','scaler','rng_by_rank','roi_rng_by_rank','resume_integrity')}
  state.update(schema_version='action4.checkpoint.v1',resumable=False,epoch_boundary=False,snapshot_purpose='asynchronous_eval_only')
  save_checkpoint_atomic(path,state);publish(self.root,self.queue,path,step,self.offset+step)
 def mark_complete(self):
  import subprocess,sys,re
  from model.training.action4_runtime import atomic_json
  pending=sum(1 for p in self.queue.glob('step_*.json') if re.fullmatch(r'step_[0-9]{8}\.json',p.name) and not p.with_suffix('.claim').exists())
  capacities=[]
  for gpu in self.settings.get('after_training_gpu_uuids',[]):
   free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   capacities.append((gpu,min(8,max(0,(free-8192)//4096))))
  extra=[]
  for slot in range(8):
   for gpu,capacity in capacities:
    if not pending:break
    if slot>=capacity:continue
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(self.root/'src'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
    f=(self.queue/f'after_train_{len(extra)}_{time.time_ns()}.log').open('x');self.handles.append(f)
    process=subprocess.Popen([sys.executable,'-m','eval.action4_step_eval','--root',str(self.root),'--queue',str(self.queue),'--device','cuda:0','--threads','1','--parent-pid',str(os.getpid())],cwd=self.root,env=env,stdout=f,stderr=f)
    self.workers.append(process);extra.append({'pid':process.pid,'gpu_uuid':gpu});pending-=1
  atomic_json(self.queue/'after_training_workers.json',{'workers':extra,'rule':'up to8 per releasedGPU, oneCPU each, bounded by unclaimed requests and freeVRAM'})
  atomic_json(self.queue/'producer_complete.json',{'time':time.time()})
 def poll_finished(self):
  if list(self.queue.glob('step_*.failed.json')) or any(p.poll() not in (None,0) for p in self.workers):return 'Evaluation worker failed'
  return 'complete' if all(p.poll()==0 for p in self.workers) else None
 def finish(self):
  from model.training.action4_runtime import atomic_json,link_checkpoint_atomic
  atomic_json(self.queue/'producer_complete.json',{'time':time.time()})
  for p in self.workers:
   if p.wait()!=0:raise RuntimeError('Evaluation worker failed while draining queue')
  requests=[p for p in self.queue.glob('step_*.json') if __import__('re').fullmatch(r'step_[0-9]{8}\.json',p.name)]
  if len(list(self.queue.glob('step_*.done.json')))!=len(requests) or list(self.queue.glob('step_*.failed.json')):raise RuntimeError('Evaluation coverage incomplete')
  selected=json.loads((self.queue/'selection.json').read_text());link_checkpoint_atomic(bounded(self.root,selected['best_snapshot']),self.models/'best.pt')
  for f in self.handles:f.close()
  return selected
 def stop(self):
  for p in self.workers:
   if p.poll() is None:p.terminate()
  for p in self.workers:
   try:p.wait(timeout=15)
   except __import__('subprocess').TimeoutExpired:p.kill();p.wait()
  for f in self.handles:f.close()

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--queue',required=True);p.add_argument('--device',required=True);p.add_argument('--threads',type=int,required=True);p.add_argument('--parent-pid',type=int);a=p.parse_args();worker(a.root,a.queue,a.device,a.threads,a.parent_pid)
