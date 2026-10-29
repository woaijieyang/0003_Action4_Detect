"""Sequential generation -> global similarity -> quota-preserving replacements.

No training and no automatic final visual acceptance. Child work blocks on exit
rather than repeated dataset polling. Final review products are made once.
"""
from pathlib import Path
import fcntl,json,os,subprocess,time,traceback,sys
from data.action4_paths import PROJECT_ROOT as R
OUT=R/'outputs/phrase2_generation_v001'

def write(p,obj):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,p)

def run():
    assert Path.cwd().resolve()==R
    lock=(OUT/'delivery_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,{22,23})
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(R/'runtime'))
    stages=OUT/'pipeline_v002';stages.mkdir(exist_ok=False)
    def call(stage,module,args=None,cpus=None,extra=None):
        cmd=[str(R/'.venv/bin/python'),'-m',module]+(args or [])
        log=(stages/(stage+'.log')).open('x')
        proc=subprocess.Popen(cmd,cwd=R,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                              preexec_fn=lambda:os.sched_setaffinity(0,set(cpus or [22,23])))
        log.close();write(OUT/'PIPELINE_STATUS.json',{'stage':stage,'pid':os.getpid(),'child_pid':proc.pid,'command':cmd,'time_unix':time.time(),'no_training':True})
        rc=proc.wait()
        if rc:raise RuntimeError((stage,rc,'See '+str(stages/(stage+'.log'))))
    for round_id in range(12):
        from data import action4_phrase2_queue as q
        if q.counts().get('accepted',0)<5000:
            call(f'generate_{round_id:02d}','data.action4_phrase2_bulk_controller',extra={'PHRASE2_RUN_TAG':f'bulk_cached_{round_id:02d}'},cpus=sorted(range(96)))
        call(f'similarity_{round_id:02d}','data.action4_phrase2_similarity',['--workers','64'],cpus=list(range(64)))
        report=json.loads((OUT/'SIMILARITY_CURRENT.json').read_text())
        if not report['rejected']:break
        call(f'cull_{round_id:02d}','data.action4_phrase2_similarity',['--cull'])
    else:raise RuntimeError('Similarity replacements did not converge within 12 batches; diagnose distribution, do not relax thresholds')
    call('final_engineering_and_interface','data.action4_phrase2_finalize_v002',cpus=list(range(64)))
    # Deterministic representative review includes extremes and both instances
    # in all scene/occlusion groups; never inspect model scores for admission.
    records=__import__('data.action4_phrase2_similarity',fromlist=['accepted']).accepted()
    selected=set();groups={}
    for r in records:
        key=(r['scene_id'],max(o['occlusion_inframe'] for o in r['instances'])>.35)
        groups.setdefault(key,[]).append(r)
    for values in groups.values():
        values.sort(key=lambda r:r['slot_id'])
        for i in [0,len(values)//2,len(values)-1]:selected.add(values[i]['slot_id'])
    for get in [lambda r:min(o['bbox_center_uv'][0] for o in r['instances']),lambda r:max(o['bbox_center_uv'][0] for o in r['instances']),
                lambda r:min(o['occlusion_inframe'] for o in r['instances']),lambda r:max(o['occlusion_inframe'] for o in r['instances']),
                lambda r:min((o['bbox_full_xyxy'][2]-o['bbox_full_xyxy'][0])/r['width'] for o in r['instances']),
                lambda r:max((o['bbox_full_xyxy'][2]-o['bbox_full_xyxy'][0])/r['width'] for o in r['instances'])]:
        values=sorted(records,key=get);selected.update(r['slot_id'] for r in values[:2]+values[-2:])
    required=OUT/'PREVIEW_REPLACEMENTS_REQUIRED.json'
    if required.exists():selected.update(json.loads(required.read_text())['slots'])
    call('final_visual_products','eval.action4_phrase2_review',['--slots']+list(map(str,sorted(selected)))+['--output','preview_review_final_v002','--status','accepted'],cpus=list(range(64)))
    write(OUT/'PIPELINE_STATUS.json',{'stage':'AWAIT_ROOT_FINAL_VISUAL_REVIEW','time_unix':time.time(),'images':5000,'similarity_pass':True,'no_training':True,'not_formally_released':True})

if __name__=='__main__':
    try:run()
    except BaseException:
        write(OUT/'PIPELINE_FAILURE.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});raise
