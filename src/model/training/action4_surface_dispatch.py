"""Cache-completion handoff to one independent surface-head train/eval pair."""
from pathlib import Path
import json,os,time,subprocess,fcntl
R=Path('/data/zy/vln/a-test')
def run(config):
    assert Path.cwd().resolve()==R
    cfg=json.loads((R/config).read_text());out=R/cfg['output'];cache=R/cfg['cache_output'];out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DISPATCH.json').exists():raise FileExistsError('Never duplicate a dispatched run')
    cache_pid=json.loads((cache/'CACHE_PROCESS.json').read_text())['pid']
    while True:
        complete=(cache/'CACHE_COMPLETE.json').exists();status=subprocess.run(['ps','-p',str(cache_pid),'-o','stat=,args='],capture_output=True,text=True);alive=status.returncode==0 and bool(status.stdout.strip()) and not status.stdout.strip().startswith('Z')
        if (cache/'CACHE_FAILURE.json').exists():raise RuntimeError('Cache failed, no training launched')
        if complete and not alive:break
        if not complete and not alive:raise RuntimeError('Cache process exited without completion')
        if alive and 'model.training.action4_surface_cache' not in status.stdout:raise RuntimeError('Cache PID identity changed')
        (out/'CONTROLLER_STATUS.json').write_text(json.dumps({'status':'WAITING_CACHE','cache_pid':cache_pid}));time.sleep(30)
    for name in ['HEAD_TEST_CUDA.json','PNP_TEST.json','CACHED_UPDATE_TEST.json']:
        assert json.loads((cache/name).read_text())['status']=='PASS'
    manifest=json.loads((cache/'CACHE_COMPLETE.json').read_text());assert manifest['splits']['train']['count']==3024 and manifest['splits']['eval']['count']==378
    free=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<8000:raise RuntimeError('Insufficient headroom; no training started')
    env=dict(os.environ,PYTHONPATH=str(R/'runtime/surface_deps')+':'+str(R/'src'),CUDA_VISIBLE_DEVICES='GPU-b1ae5be6-1dba-7343-e233-59df41402f33',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(R/'runtime'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    workers=[];processes=[]
    for role,module in [('eval','eval.action4_surface_eval'),('train','model.training.action4_surface_train')]:
        with (out/(role+'.log')).open('x') as log:
            command=[str(R/'.venv/bin/python'),'-B','-m',module,'--config',config];p=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
        processes.append(p);workers.append({'pid':p.pid,'role':role,'command':command,'gpu':6,'cpu':1})
    (out/'DISPATCH.json').write_text(json.dumps({'workers':workers,'source_cache':cfg['cache_output'],'cpu_total_upper':71,'epochs':cfg['epochs'],'eval_every':cfg['eval_every']},indent=2))
    while any(p.poll() is None for p in processes):
        failures=[p.returncode for p in processes if p.poll() is not None and p.returncode!=0]
        if failures:
            for p in processes:
                if p.poll() is None:p.terminate()
            (out/'CONTROLLER_FAILURE.json').write_text(json.dumps({'exit_codes':failures,'action':'stopped only this controller-owned counterpart; no automatic retry'}));return
        time.sleep(20)
    (out/'CONTROLLER_COMPLETE.json').write_text(json.dumps({'codes':[p.returncode for p in processes],'accuracy_accepted':False}))
if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());(R/cfg['output']/'CONTROLLER_FAILURE.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
