"""Sequential preparation then three CPU-only surface workers, budget<=72."""
from pathlib import Path
import os,json,subprocess
R=Path('/data/zy/vln/a-test')
def run():
    assert Path.cwd().resolve()==R
    out=R/'outputs/action4_surface_labels_v001'
    if (out/'DISPATCH.json').exists():raise FileExistsError('Already dispatched')
    env=dict(os.environ,PYTHONPATH=str(R/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(R/'runtime'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',BLENDER_USER_CONFIG=str(R/'runtime/blender_user/config'),BLENDER_USER_SCRIPTS=str(R/'runtime/blender_user/scripts'),BLENDER_USER_DATAFILES=str(R/'runtime/blender_user/datafiles'))
    with (out/'prepare.log').open('x') as log:subprocess.run([str(R/'.venv/bin/python'),'-B','-m','data.action4_surface_inputs'],env=env,stdout=log,stderr=log,check=True)
    processes=[];workers=[]
    for shard in range(3):
        with (out/f'shard_{shard}.log').open('x') as log:p=subprocess.Popen([str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','1','--python',str(R/'src/data/action4_surface_labels.py'),'--',str(shard)],env=env,stdout=log,stderr=log)
        processes.append(p);workers.append({'pid':p.pid,'shard':shard,'cpu':1})
    (out/'DISPATCH.json').write_text(json.dumps({'workers':workers,'cpu_total_upper':72,'GPU_used':False},indent=2))
    codes=[p.wait() for p in processes]
    (out/'WORKERS_EXIT.json').write_text(json.dumps({'codes':codes,'all_receipts':all((out/f'shard_{i}_COMPLETE.json').exists() for i in range(3)),'training_accepted':False}))
if __name__=='__main__':run()
