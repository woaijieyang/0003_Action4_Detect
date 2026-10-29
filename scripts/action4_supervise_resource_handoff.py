"""Adopt one healthy frozen run after an explicit resource revocation.
No training restart, source mutation, metric change, or automatic retry.
"""
import os,sys,json,time,subprocess,fcntl,math,traceback
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test')
BASE=ROOT/'runtime/action4_runs/p2_b0_formal80_v001'
OUT=ROOT/'outputs/action4_improvement/stage1_baseline/p2_b0_formal80_v001'
AUDIT=ROOT/'outputs/action4_phase2_stage1_v001/gpu0_retirement_v001'

def process(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat=,lstart=,args='],capture_output=True,text=True)
    text=result.stdout.strip()
    if not text or text.split()[0].startswith('Z'):return None
    return text.split(None,1)[1]

def expected_steps(terminal,cfg):
    total=cfg['epochs']*math.ceil(cfg['expected_units']/cfg['batch'])
    if terminal['step']+terminal.get('no_supervision_update_count',0)!=total:raise ValueError('Training traversal mismatch')
    return {0,terminal['step']}|set(range(cfg['eval_every'],terminal['step']+1,cfg['eval_every']))

def main():
    assert Path.cwd().resolve()==ROOT
    spec=json.loads((BASE/'RUN_SPEC.json').read_text());code=ROOT/spec['code_root']
    os.environ['ACTION4_PROJECT_ROOT']=str(ROOT);sys.dont_write_bytecode=True
    sys.path[:]=[str(code/'runtime/surface_deps'),str(code/'src')]+[p for p in sys.path if p and Path(p).resolve()!=ROOT/'src']
    from model.training.action4_run_snapshot import verify_running_snapshot
    from model.training.action4_runtime import atomic_json,canonical_hash
    from eval.action4_stage2_eval import rank_result
    from model.training.action4_best_only_retention import finalize_best_only
    cfg=json.loads((ROOT/spec['config']).read_text());verify_running_snapshot(cfg)
    guard=(OUT/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((OUT/'DISPATCH.json').read_text());attempt=old['attempt_id']
    if (AUDIT/'RESOURCE_DISPATCH.json').exists():raise FileExistsError('Duplicate handoff')
    live=[]
    for row in old['workers']:
        if row['gpus']==[0]:continue
        identity=process(row['pid'])
        if identity is None or spec['config'] not in identity:raise RuntimeError('Existing worker identity mismatch')
        live.append(dict(row,process_identity=identity,adopted=True))
    env=dict(os.environ,ACTION4_ATTEMPT_ID=attempt,ACTION4_PROJECT_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
             TMPDIR=str(ROOT/'runtime'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for worker,gpu in [('migrated0',5),('migrated2',6)]:
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(info[1])<12000:raise RuntimeError('Insufficient replacement memory')
        cmd=[str(ROOT/'.venv/bin/python'),'-I',str(BASE/'launch.py'),'eval.action4_stage2_eval','--worker-id',worker]
        with (OUT/('eval_'+worker+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=info[0].strip()),stdout=log,stderr=log,start_new_session=True)
        identity=process(child.pid)
        if identity is None:raise RuntimeError('Replacement worker failed to start')
        live.append({'role':'eval_'+worker,'pid':child.pid,'gpus':[gpu],'gpu_uuid':info[0].strip(),'process_identity':identity,'command':cmd,'adopted':False})
    receipt={'controller_pid':os.getpid(),'controller_identity':process(os.getpid()),'attempt_id':attempt,'workers':live,'cpu_budget':43,'GPU0_forbidden':True,
             'run_spec':str((BASE/'RUN_SPEC.json').relative_to(ROOT)),'config_id':canonical_hash(cfg),'reason':'user revoked GPU0; orchestration-only resource override',
             'training_restarted':False,'time_unix':time.time()}
    atomic_json(AUDIT/'RESOURCE_DISPATCH.json',receipt)
    while True:
        terminal_path=OUT/'TRAIN_COMPLETE.json';terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        if list(OUT.glob('TRAIN_FAILURE*.json')) or list((OUT/'eval').glob('*.failed.json')):raise RuntimeError('Explicit training/evaluation failure')
        all_stopped=True
        for w in live:
            identity=process(w['pid'])
            if identity is not None:
                if identity!=w['process_identity']:raise RuntimeError('PID identity changed')
                all_stopped=False
            elif w['role']=='train':
                if terminal is None or terminal['attempt_id']!=attempt:raise RuntimeError('Training ended without matching terminal receipt')
            else:
                wid=w['role'][5:];p=OUT/('WORKER_COMPLETE_'+wid+'.json')
                if not p.exists() or json.loads(p.read_text())['attempt_id']!=attempt:raise RuntimeError('Evaluator ended without completion')
        if terminal is not None and all_stopped:
            expected=expected_steps(terminal,cfg)
            if terminal['attempt_id']!=attempt:raise ValueError('Attempt mismatch')
            requests=[p for p in (OUT/'eval').glob('step_*.json') if p.name.count('.')==1]
            reports=[json.loads(p.with_suffix('.done.json').read_text()) for p in requests]
            if {x['step'] for x in reports}!=expected:raise ValueError('Incomplete eval coverage')
            if any(x['config_id']!=canonical_hash(cfg) for x in reports):raise ValueError('Eval config mismatch')
            best=json.loads((OUT/'BEST.json').read_text());winner=max(reports,key=rank_result)
            if best['step']!=winner['step']:raise ValueError('Selection mismatch')
            atomic_json(OUT/'EVAL_COMPLETE.json',{'attempt_id':attempt,'evaluations':len(reports),'all_steps':sorted(expected),'best_step':best['step'],'config_id':canonical_hash(cfg)})
            finalize_best_only(str(OUT),remove_recovery=True)
            atomic_json(OUT/'CONTROLLER_COMPLETE.json',{'attempt_id':attempt,'mode':cfg['mode'],'role':cfg['role'],'best_step':best['step'],'accuracy_accepted':False,'config_id':canonical_hash(cfg),'resource_handoff':str((AUDIT/'RESOURCE_DISPATCH.json').relative_to(ROOT))})
            return
        time.sleep(5)

if __name__=='__main__':
    try:main()
    except BaseException:
        # Preserve running training and surviving evaluators on supervisor errors.
        AUDIT.mkdir(exist_ok=True)
        (AUDIT/'SUPERVISOR_FAILURE.json').write_text(json.dumps({'traceback':traceback.format_exc(),'time_unix':time.time()},indent=2))
        raise
