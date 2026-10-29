"""Cold recovery of terminal audit CLI failure; never dispatch RGB again."""
from pathlib import Path
import os,json,subprocess,time,fcntl,traceback
from data.action4_paths import PROJECT_ROOT as R
from data import action4_occlusion500_queue as q
from data.action4_occlusion500_similarity import additional_records,accepted,require_proof
OUT=R/'outputs/phrase2_occlusion500_v001'
def write(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def run():
    assert Path.cwd().resolve()==R and q.counts()=={'accepted':500};require_proof(accepted())
    lock=(OUT/'postprocess_recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,{22,23});directory=OUT/'postprocess_v002';directory.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(R/'runtime'))
    def call(stage,module,args=None):
        cmd=[str(R/'.venv/bin/python'),'-m',module]+(args or []);log=(directory/(stage+'.log')).open('x')
        p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(range(64))));log.close()
        write(OUT/'PIPELINE_STATUS.json',{'stage':stage,'pid':os.getpid(),'child_pid':p.pid,'command':cmd,'no_RGB_regeneration':True,'time_unix':time.time()})
        code=p.wait()
        if code:raise RuntimeError((stage,code,str(directory/(stage+'.log'))))
    call('batch_engineering_and_interface_v002','data.action4_occlusion500_audit_v002')
    rows=additional_records();selected=set();covered=set()
    for r in rows:
        tags={(r['scene_id'],s,o['occlusion_bin']) for s,o in zip(['left','right'],r['instances'])}
        if tags-covered:selected.add(r['slot_id']);covered|=tags
    for value in [lambda r:min(o['bbox_center_uv'][0] for o in r['instances']),lambda r:max(o['bbox_center_uv'][1] for o in r['instances']),lambda r:min(o['actual_width_fraction'] for o in r['instances']),lambda r:max(o['actual_width_fraction'] for o in r['instances'])]:
        ordered=sorted(rows,key=value);selected.add(ordered[0]['slot_id']);selected.add(ordered[-1]['slot_id'])
    assert len(covered)==30
    call('final_visual_products','eval.action4_occlusion500_review',['--slots']+list(map(str,sorted(selected)))+['--status','accepted','--output','preview_review_final500'])
    write(OUT/'POSTPROCESS_RECOVERY.json',{'status':'PASS_AUDIT_V002_AND_ACTUAL_INTERFACE_VISUAL_REVIEW_REQUIRED','original_failure_preserved':'outputs/phrase2_occlusion500_v001/postprocess_audit_v001_failure','no_RGB_regenerated':True,'time_unix':time.time()})
    write(OUT/'PIPELINE_STATUS.json',{'stage':'AWAIT_ADDITIONAL500_FINAL_VISUAL_REVIEW','new_images':500,'global_similarity_images':5500,'formal_merge_complete':False,'time_unix':time.time()})
if __name__=='__main__':
    try:run()
    except BaseException:
        write(OUT/'POSTPROCESS_V002_FAILURE.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});raise
