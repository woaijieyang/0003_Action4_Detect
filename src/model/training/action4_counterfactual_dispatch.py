"""Wait for an owned training slot to finish; run one bounded diagnostic only."""
import json,os,subprocess,time,fcntl,traceback
from pathlib import Path


def main():
    R=Path('/data/zy/vln/a-test');config_path='configs/data/action4_counterfactual_diagnostic_v001.json';cfg=json.loads((R/config_path).read_text());out=R/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (out/'CONTROLLER.json').open('x') as f:json.dump({'pid':os.getpid(),'config':config_path,'wait_for':'b0_baseline complete80 and process exit','automatic_restart':False},f)
    suite=R/'outputs/action4_independent_trials_v001';process=next(p for p in json.loads((suite/'PROCESSES.json').read_text())['processes'] if p.get('config','').endswith('/b0_baseline.json'));job=suite/'b0_baseline'
    (out/'STATUS.json').write_text(json.dumps({'phase':'WAITING_FOR_RELEASED_GPU2_AND_CPU_SLOT','render_started':False}))
    while True:
        if (job/'FAILURE.json').exists():raise RuntimeError('Baseline failed; no automatic render admission')
        if (job/'TRAIN_COMPLETE.json').exists():
            complete=json.loads((job/'TRAIN_COMPLETE.json').read_text())
            if complete['epochs']!=80:raise ValueError('Expected completed80, not a partial run')
            pid_state=subprocess.run(['ps','-p',str(process['pid']),'-o','stat='],capture_output=True,text=True)
            if pid_state.returncode!=0 or pid_state.stdout.strip().startswith('Z'):break
        time.sleep(60)
    free=int(subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<16384:raise RuntimeError('Released GPU headroom unavailable; no competing process stopped')
    frozen=out/'B0_frozen.pt';os.link(suite/'b0_giou_quality/best.pt',frozen)
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(R/'.cache'),BLENDER_USER_CONFIG=str(R/'.cache/blender_counterfactual/config'),BLENDER_USER_SCRIPTS=str(R/'.cache/blender_counterfactual/scripts'),BLENDER_USER_DATAFILES=str(R/'.cache/blender_counterfactual/datafiles'))
    for key in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    (out/'STATUS.json').write_text(json.dumps({'phase':'RENDERING','gpu':2,'cpu':2,'worst_case_total_cpu':64}))
    command=[str(R/'runtime/blender-3.6.23-linux-x64/blender'),'--background','--factory-startup','--disable-autoexec','--threads','2','--python-exit-code','1','--python',str(R/'scripts/action4_counterfactual_render.py'),'--','--config',config_path]
    with (out/'render.log').open('x') as log:subprocess.run(command,cwd=R,env=env,stdout=log,stderr=log,check=True)
    (out/'STATUS.json').write_text(json.dumps({'phase':'EVALUATING_FIXED_MODEL','gpu':2,'cpu':2}))
    with (out/'eval.log').open('x') as log:subprocess.run([str(R/'.venv/bin/python'),'-B','-m','eval.action4_counterfactual_eval','--output',cfg['output']],cwd=R,env=env,stdout=log,stderr=log,check=True)
    (out/'STATUS.json').write_text(json.dumps({'phase':'COMPLETE_VISUAL_REVIEW_REQUIRED','production_accepted':False}))

if __name__=='__main__':
    try:main()
    except Exception:
        out=Path('/data/zy/vln/a-test/outputs/action4_scene_domain_audit_v001/counterfactual_v001');out.mkdir(parents=True,exist_ok=True);(out/'FAILURE.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
