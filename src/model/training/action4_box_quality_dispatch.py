"""Wait for cache and original B0 completion before admitting new score trial."""
from pathlib import Path
import os,json,time,subprocess,fcntl
R=Path('/data/zy/vln/a-test')

def alive(pid,identity):
    state=subprocess.run(['ps','-p',str(pid),'-o','stat=,args='],capture_output=True,text=True)
    active=state.returncode==0 and bool(state.stdout.strip()) and not state.stdout.strip().startswith('Z')
    if active and identity not in state.stdout:raise RuntimeError('PID identity changed')
    return active

def run(config):
    assert Path.cwd().resolve()==R;cfg=json.loads((R/config).read_text());out=R/cfg['output'];cache=R/cfg['cache_output'];parent=R/'outputs/action4_independent_trials_v001/b0_giou_quality'
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DISPATCH.json').exists():raise FileExistsError('Run already dispatched')
    cache_pid=json.loads((cache/'CACHE_PROCESS.json').read_text())['pid'];parent_pid=json.loads((parent/'PROCESS.json').read_text())['pid']
    while True:
        if (cache/'CACHE_FAILURE.json').exists() or (parent/'FAILURE.json').exists():raise RuntimeError('Prerequisite failed; no automatic retry')
        cache_active=alive(cache_pid,'model.training.action4_box_quality_cache');parent_active=alive(parent_pid,'b0_giou_quality.json')
        ready=(cache/'CACHE_COMPLETE.json').exists() and (parent/'TRAIN_COMPLETE.json').exists() and not cache_active and not parent_active
        if ready:break
        if not cache_active and not (cache/'CACHE_COMPLETE.json').exists():raise RuntimeError('Cache exited without completion')
        (out/'CONTROLLER_STATUS.json').write_text(json.dumps({'status':'WAITING_CACHE_AND_ORIGINAL_B0_80_EPOCHS','cache_pid':cache_pid,'parent_pid':parent_pid}));time.sleep(30)
    assert json.loads((parent/'TRAIN_COMPLETE.json').read_text())['epochs']>=80
    complete=json.loads((cache/'CACHE_COMPLETE.json').read_text());assert complete['splits']['train']['images']==1600 and complete['splits']['eval']['images']==200
    free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>5000
    env=dict(os.environ,PYTHONPATH=str(R/'src'),CUDA_VISIBLE_DEVICES='GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(R/'runtime'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/'preflight.log').open('x') as log:subprocess.run([str(R/'.venv/bin/python'),'-B','tests/check_action4_box_quality_cached_update.py'],env=env,stdout=log,stderr=log,check=True)
    workers=[];processes=[]
    for role,module in [('eval','eval.action4_box_quality_eval'),('train','model.training.action4_box_quality_train')]:
        with (out/(role+'.log')).open('x') as log:
            command=[str(R/'.venv/bin/python'),'-B','-m',module,'--config',config];process=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
        processes.append(process);workers.append({'pid':process.pid,'role':role,'gpu':4,'cpu':1,'command':command})
    (out/'DISPATCH.json').write_text(json.dumps({'workers':workers,'original_B0_complete':True,'epochs':cfg['epochs'],'eval_every':cfg['eval_every'],'cpu_upper_after_parent_release':63},indent=2))
    while any(p.poll() is None for p in processes):
        failures=[p.returncode for p in processes if p.poll() is not None and p.returncode!=0]
        if failures:
            for p in processes:
                if p.poll() is None:p.terminate()
            (out/'CONTROLLER_FAILURE.json').write_text(json.dumps({'exit_codes':failures,'action':'only own counterpart stopped; no retry'}));return
        time.sleep(20)
    (out/'CONTROLLER_COMPLETE.json').write_text(json.dumps({'codes':[p.returncode for p in processes],'accuracy_accepted':False}))
if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());(R/cfg['output']/'CONTROLLER_FAILURE.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
