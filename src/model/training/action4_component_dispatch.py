"""Explicit suite dispatcher; no process starts at import."""
from pathlib import Path
import json,os,subprocess,time,fcntl

def main():
 R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_independent_trials_v001'
 lock=(O/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (O/'PROCESSES.json').exists():raise RuntimeError('Suite already launched; inspect actual state before explicit resume')
 assert json.loads((O/'RECOVERY_TEST.json').read_text())['status']=='PASS'
 suite=json.loads((O/'SUITE.json').read_text());base=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(R/'runtime'))
 gpus={1:'GPU-a77a9d4c-7642-7112-698c-b02673d254cb',2:'GPU-8dfd78d1-6a27-4292-3338-9d33fffee9ef',3:'GPU-510dc520-5a9d-1317-019d-de5ed02c1b37',4:'GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f',5:'GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8',6:'GPU-b1ae5be6-1dba-7343-e233-59df41402f33',7:'GPU-a68477c3-1c2a-d557-58f5-405d80e5383e'}
 free=subprocess.check_output(['nvidia-smi','-i','1,2,3,4,5,6,7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 if any(int(x)<40000 for x in free.splitlines()):raise RuntimeError('GPU headroom below suite admission; no process started')
 processes=[];handles=[]
 for cp in suite['configs']:
  cfg=json.loads((R/cp).read_text());out=R/cfg['output'];out.mkdir(exist_ok=True);f=(out/'process.log').open('x');handles.append(f)
  cmd=[str(R/'.venv/bin/python'),'-B','-m','model.training.action4_component_train','--config',cp]
  p=subprocess.Popen(cmd,cwd=R,env=dict(base,CUDA_VISIBLE_DEVICES=gpus[cfg['gpu']]),stdout=f,stderr=f);processes.append((p,dict(pid=p.pid,kind='train',config=cp,gpu=cfg['gpu'],command=cmd)))
 for i in range(8):
  f=(O/f'eval_worker_{i}.log').open('x');handles.append(f);cmd=[str(R/'.venv/bin/python'),'-B','-m','eval.action4_component_eval','--suite',str((O/'SUITE.json').relative_to(R))]
  p=subprocess.Popen(cmd,cwd=R,env=dict(base,CUDA_VISIBLE_DEVICES=gpus[1],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=f,stderr=f);processes.append((p,dict(pid=p.pid,kind='eval',gpu=1,command=cmd)))
 (O/'PROCESSES.json').write_text(json.dumps({'controller_pid':os.getpid(),'processes':[r for p,r in processes],'cpu_reserved':68},indent=2))
 while any(p.poll() is None for p,r in processes):
  (O/'PROCESS_STATUS.json').write_text(json.dumps([{**r,'returncode':p.poll()} for p,r in processes],indent=2));time.sleep(30)
 (O/'PROCESS_STATUS.json').write_text(json.dumps([{**r,'returncode':p.returncode} for p,r in processes],indent=2))


def replace_eval_workers():
 """One controlled executor migration. No training PID is signalled."""
 import signal
 from eval.action4_component_eval import rank
 R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_independent_trials_v001'
 if (O/'EVAL_REPLACEMENT.json').exists():raise RuntimeError('Replacement already recorded')
 processes=json.loads((O/'PROCESSES.json').read_text())['processes'];old=[r['pid'] for r in processes if r['kind']=='eval']
 if len(old)!=8:raise ValueError('Expected8 original eval workers')
 for pid in old:
  command=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
  if '-m eval.action4_component_eval --suite outputs/action4_independent_trials_v001/SUITE.json' not in command:raise ValueError('Eval PID identity changed')
 for pid in old:os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  alive=[]
  for pid in old:
   result=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
   if result.returncode==0 and not result.stdout.strip().startswith('Z'):alive.append(pid)
  if not alive:break
  time.sleep(.2)
 if alive:raise RuntimeError('Old workers still alive, no new worker launched')
 configs=json.loads((O/'SUITE.json').read_text())['configs'];interrupted=[]
 for cp in configs:
  cfg=json.loads((R/cp).read_text());out=R/cfg['output'];q=out/'eval'
  for claim in q.glob('*.claim'):
   if claim.with_suffix('.done.json').exists():continue
   if json.loads(claim.read_text())['pid'] not in old:raise ValueError('Unexpected claim owner')
   interrupted.append(str(claim));claim.rename(claim.with_name(claim.name+'.interrupted_v1'))
  done=[json.loads(x.read_text()) for x in q.glob('*.done.json')]
  if done:
   top=max(done,key=lambda x:rank(x['report'],cfg['role']));current=json.loads((out/'BEST.json').read_text()) if (out/'BEST.json').exists() else None
   if current is None or rank(top['report'],cfg['role'])>rank(current['report'],cfg['role']):
    target=R/top['snapshot'];assert target.exists();link=out/'best.reconcile.tmp';os.link(target,link);os.replace(link,out/'best.pt');(out/'BEST.json').write_text(json.dumps(top,indent=2))
 C=R/'configs/eval/component_trials_v001';C.mkdir(exist_ok=True)
 assignments=[[cp] for cp in configs]+[configs[:3],configs[3:]];workers=[]
 env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='GPU-a77a9d4c-7642-7112-698c-b02673d254cb',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(R/'runtime'))
 for slot,assigned in enumerate(assignments):
  cp=C/f'worker_{slot}.json';cp.write_text(json.dumps({'configs':assigned},indent=2));log=(O/f'eval_cached_worker_{slot}.log').open('x')
  cmd=[str(R/'.venv/bin/python'),'-B','-m','eval.action4_component_eval_cached','--suite',str(cp.relative_to(R))]
  process=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=log,start_new_session=True);log.close();workers.append(dict(pid=process.pid,slot=slot,configs=assigned,command=cmd))
 receipt=dict(old_eval_pids=old,new_workers=workers,interrupted_uncompleted_claims=interrupted,reason='Repeated PNG decoding and shared queue starvation; raw cache exact read parity passed',training_processes_untouched=True,cpu_total_reserved=68,scoring_implementation_unchanged=True)
 (O/'EVAL_REPLACEMENT.json').write_text(json.dumps(receipt,indent=2));print(receipt)
