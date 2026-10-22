"""Run the prepared diagnostic in resources released by stopped A0 candidates."""
from pathlib import Path
import json,os,subprocess,traceback


def main():
    R=Path('/data/zy/vln/a-test');cp='configs/data/action4_counterfactual_diagnostic_v002.json';cfg=json.loads((R/cp).read_text());out=R/cfg['output'];out.mkdir(exist_ok=False)
    receipts=json.loads((R/'outputs/action4_independent_trials_v001/PRUNED_A0_TRIALS.json').read_text())['trials']
    for receipt in receipts:
        for pid in [receipt['pid']]+receipt['children']:
            ps=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
            if ps.returncode==0 and not ps.stdout.strip().startswith('Z'):raise RuntimeError('Released owner still active')
    free=int(subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=16384
    os.link(R/'outputs/action4_independent_trials_v001/b0_giou_quality/best.pt',out/'B0_frozen.pt')
    (out/'CONTROLLER.json').write_text(json.dumps({'pid':os.getpid(),'config':cp,'cpu_total_worst_case':54,'gpu':6}))
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(R/'.cache'),BLENDER_USER_CONFIG=str(R/'.cache/blender_counterfactual/config'),BLENDER_USER_SCRIPTS=str(R/'.cache/blender_counterfactual/scripts'),BLENDER_USER_DATAFILES=str(R/'.cache/blender_counterfactual/datafiles'))
    for key in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    (out/'STATUS.json').write_text(json.dumps({'phase':'RENDERING','gpu':6,'cpu':2}))
    cmd=[str(R/'runtime/blender-3.6.23-linux-x64/blender'),'--background','--factory-startup','--disable-autoexec','--threads','2','--python-exit-code','1','--python',str(R/'scripts/action4_counterfactual_render.py'),'--','--config',cp]
    with (out/'render.log').open('x') as f:subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=f,check=True)
    (out/'STATUS.json').write_text(json.dumps({'phase':'EVALUATING_FIXED_MODEL'}))
    with (out/'eval.log').open('x') as f:subprocess.run([str(R/'.venv/bin/python'),'-B','-m','eval.action4_counterfactual_eval','--output',cfg['output']],cwd=R,env=env,stdout=f,stderr=f,check=True)
    (out/'STATUS.json').write_text(json.dumps({'phase':'COMPLETE_VISUAL_REVIEW_REQUIRED','production_accepted':False}))

if __name__=='__main__':
    try:main()
    except Exception:
        out=Path('/data/zy/vln/a-test/outputs/action4_scene_domain_audit_v001/counterfactual_v002');out.mkdir(parents=True,exist_ok=True);(out/'FAILURE.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
