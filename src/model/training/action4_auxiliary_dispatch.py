"""Admit one additive A0 job by replacing one elastic eval slot, not trainers."""
from pathlib import Path
import json,os,subprocess,signal,time,fcntl


def main():
    R=Path('/data/zy/vln/a-test');suite=R/'outputs/action4_independent_trials_v001';out=R/'outputs/action4_a0_additive_v001/train_run_v001'
    lock=(out/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DISPATCH.json').exists():raise RuntimeError('Auxiliary already dispatched')
    assert json.loads((out.parent/'CUDA_TEST.json').read_text())['status']=='PASS'
    assert (suite/'quality_order_audit_v001/REPORT.json').exists()
    for gpu,needed in ((1,6000),(5,16000)):
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free<needed:raise RuntimeError('Insufficient GPU memory; no process changed')
    replacement=json.loads((suite/'EVAL_REPLACEMENT.json').read_text());old=next(x for x in replacement['new_workers'] if x['slot']==7);pid=old['pid']
    command=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
    if '-m eval.action4_component_eval_cached --suite configs/eval/component_trials_v001/worker_7.json' not in command:raise ValueError('Unexpected elastic worker identity')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        status=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
        if status.returncode!=0 or status.stdout.strip().startswith('Z'):break
        time.sleep(.2)
    else:raise RuntimeError('Old eval process not stopped; no new process started')
    from eval.action4_component_eval import rank
    interrupted=[]
    for cp in json.loads((suite/'SUITE.json').read_text())['configs']:
        cfg=json.loads((R/cp).read_text());directory=R/cfg['output'];q=directory/'eval'
        for claim in q.glob('*.claim'):
            if json.loads(claim.read_text())['pid']==pid and not claim.with_suffix('.done.json').exists():
                interrupted.append(str(claim.relative_to(R)));claim.rename(claim.with_name(claim.name+'.aux_transition'))
        with (directory/'selection.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);done=[json.loads(p.read_text()) for p in q.glob('*.done.json')]
            if done:
                top=max(done,key=lambda d:rank(d['report'],cfg['role']));bp=directory/'BEST.json';best=json.loads(bp.read_text()) if bp.exists() else None
                if best is None or rank(top['report'],cfg['role'])>rank(best['report'],cfg['role']):
                    temporary=directory/'best.aux_reconcile.tmp';os.link(R/top['snapshot'],temporary);os.replace(temporary,directory/'best.pt');bp.write_text(json.dumps(top,indent=2))
            fcntl.flock(f,fcntl.LOCK_UN)
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(R/'runtime'))
    config='configs/model/action4_a0_additive_v001.json';workers=[]
    for role,module,gpu in [('eval','eval.action4_auxiliary_eval','GPU-a77a9d4c-7642-7112-698c-b02673d254cb'),('train','model.training.action4_auxiliary_train','GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8')]:
        log=(out/(role+'.log')).open('x');cmd=[str(R/'.venv/bin/python'),'-B','-m',module,'--config',config]
        process=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1' if role=='eval' else '2',MKL_NUM_THREADS='1' if role=='eval' else '2'),stdout=log,stderr=log,start_new_session=True);log.close();workers.append({'pid':process.pid,'role':role,'command':cmd})
    (out/'DISPATCH.json').write_text(json.dumps({'replaced_elastic_eval':old,'workers':workers,'interrupted_requests':interrupted,'cpu_budget':72,'cpu_accounting':'original60train +7oldEval +4auxTrain +1auxEval=72','original_six_trainers_untouched':True,'gpu1_eval_workers_total':8},indent=2))
    print(workers)


def retry_after_startup_fix():
    R=Path('/data/zy/vln/a-test');parent=R/'outputs/action4_a0_additive_v001';old=parent/'train_run_v001';out=parent/'train_run_v002';out.mkdir(exist_ok=True)
    if any((out/name).exists() for name in ('DISPATCH.json','train.log','eval.log')):raise RuntimeError('Potential prior launch; inspect instead of duplicating')
    assert (old/'FAILURE.json').exists() and not (old/'STATUS.json').exists(), 'Only zero-update failure can restart fresh'
    assert json.loads((parent/'CUDA_DETERMINISTIC_TEST.json').read_text())['status']=='PASS'
    for process in json.loads((old/'DISPATCH.json').read_text())['workers']:
        state=subprocess.run(['ps','-p',str(process['pid']),'-o','stat='],capture_output=True,text=True)
        if state.returncode==0 and not state.stdout.strip().startswith('Z'):raise RuntimeError('Old auxiliary process still active')
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(R/'runtime'));workers=[]
    config='configs/model/action4_a0_additive_v002.json'
    for role,module,gpu in [('eval','eval.action4_auxiliary_eval','GPU-a77a9d4c-7642-7112-698c-b02673d254cb'),('train','model.training.action4_auxiliary_train','GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8')]:
        log=(out/(role+'.log')).open('x');cmd=[str(R/'.venv/bin/python'),'-B','-m',module,'--config',config];threads='1' if role=='eval' else '2'
        process=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads),stdout=log,stderr=log,start_new_session=True);log.close();workers.append({'pid':process.pid,'role':role,'command':cmd})
    receipt={'workers':workers,'replaces_auxiliary_attempt':'train_run_v001 zero updates; failure retained','cpu_total':72,'gpu1_eval_workers':8,'original_six_trainers_untouched':True}
    (out/'DISPATCH.json').write_text(json.dumps(receipt,indent=2));print(receipt)
