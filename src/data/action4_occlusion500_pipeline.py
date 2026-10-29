"""Wait for base release; generate/check additional500 without touching base RGB.

Two real visual gates remain: first128 and final batch. No model is trained.
The final append is a separate command after actual root visual admission.
"""
from pathlib import Path
import fcntl,json,os,subprocess,time,traceback,hashlib
from data.action4_paths import PROJECT_ROOT as R

OUT=R/'outputs/phrase2_occlusion500_v001'

def write(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def enter_execution():
    pointer=R/'configs/execution/action4_active.json';current=json.loads(pointer.read_text());authority=json.loads((R/current['active_ref']['path']).read_text())
    parent=json.loads((R/authority['source_freeze']).read_text());freeze=dict(parent);freeze['parent_execution']=current['active_ref']['path'];freeze['sources']=dict(parent['sources'])
    for p in (R/'src/data').glob('action4_occlusion500_*.py'):freeze['sources'][str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=R/'src/eval/action4_occlusion500_review.py';freeze['sources'][str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=R/'configs/data/action4_phrase2_occlusion500_v001.json';freeze['sources'][str(cfg.relative_to(R))]=hashlib.sha256(cfg.read_bytes()).hexdigest()
    freeze['schema']='action4.occlusion500_source_freeze.v1';freeze['catalog']='outputs/phrase2_occlusion500_v001/OCCLUSION_CATALOG_v001.json'
    freeze['catalog_sha256']=hashlib.sha256((R/freeze['catalog']).read_bytes()).hexdigest()
    ref=OUT/'SOURCE_FREEZE_v001.json';ref.open('x').write(json.dumps(freeze,indent=2)+'\n')
    revision='r'+str(int(authority['revision'][1:])+1).zfill(3)
    authority.update(revision=revision,parent_authority=current['active_ref']['path'],status='PHRASE2_ADDITIONAL500_RUNNING_SEPARATELY',
        source_freeze=str(ref.relative_to(R)),configuration=str(cfg.relative_to(R)),dataset_root='data/phrase2',candidate_root='data/phrase2/.staging_batches/occlusion500_v001',
        target_images=500,total_after_merge=5500,split_counts={'train':400,'eval':50,'test':50},final_split_counts={'train':4400,'eval':550,'test':550},
        slot_plan='outputs/phrase2_occlusion500_v001/SLOTS.json',pipeline='src/data/action4_occlusion500_pipeline.py',generation_policy_revision='occlusion500_split_local_joint_random_v001',
        queue='outputs/phrase2_occlusion500_v001/slots.sqlite3',progress='outputs/phrase2_occlusion500_v001/PIPELINE_STATUS.json',
        generation_progress='outputs/phrase2_occlusion500_v001/STATUS.json',current_dispatch='outputs/phrase2_occlusion500_v001/CURRENT_RUN.json',
        similarity_gate='outputs/phrase2_occlusion500_v001/SIMILARITY_COMPLETE.json',finalizer='src/data/action4_occlusion500_audit.py',
        merge_entrypoint='src/data/action4_occlusion500_merge.py',base5000_formal_release_complete=True,formal_dataset_release_complete=False)
    target=R/f'configs/execution/action4_execution_{revision}.json';target.open('x').write(json.dumps(authority,indent=2)+'\n')
    assert json.loads(pointer.read_text())==current,'Another authority writer changed the pointer'
    write(pointer,{'schema':'action4.active_execution_pointer.v1','active_ref':{'path':str(target.relative_to(R)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}})

def run():
    assert Path.cwd().resolve()==R
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,{22,23})
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(R/'runtime'),XDG_CACHE_HOME=str(R/'runtime/cache'))
    stages=OUT/'pipeline_v001';stages.mkdir(exist_ok=False)
    write(OUT/'PIPELINE_STATUS.json',{'stage':'WAIT_BASE5000_FORMAL_RELEASE','pid':os.getpid(),'actual_RGB_started':False,'time_unix':time.time()})
    while True:
        complete=R/'data/phrase2/DATASET_COMPLETE.json'
        if complete.exists():
            proof=json.loads(complete.read_text())
            delivered=R/'outputs/phrase2_generation_v001/DELIVERY_COMPLETE.json'
            if proof.get('status')=='FORMAL_DATASET_USABLE' and proof.get('images')==5000 and delivered.exists():
                closure=json.loads(delivered.read_text())
                if closure.get('status')=='FORMAL_DATASET_USABLE' and closure['manifest_sha256']==proof['manifest_sha256']:break
        time.sleep(60)
    from data.action4_occlusion500_catalog import build
    catalog=OUT/'OCCLUSION_CATALOG_v001.json'
    if not catalog.exists():build(catalog,require_complete=True)
    else:assert json.loads(catalog.read_text())['base_accepted_snapshot']==5000
    enter_execution()
    def call(stage,module,args=None,cpus=None):
        cmd=[str(R/'.venv/bin/python'),'-m',module]+(args or [])
        log=(stages/(stage+'.log')).open('x');proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            preexec_fn=lambda:os.sched_setaffinity(0,set(cpus or [22,23])));log.close()
        write(OUT/'PIPELINE_STATUS.json',{'stage':stage,'pid':os.getpid(),'child_pid':proc.pid,'command':cmd,'time_unix':time.time(),'no_training':True})
        rc=proc.wait()
        if rc:raise RuntimeError((stage,rc,str(stages/(stage+'.log'))))
    from data import action4_occlusion500_queue as q
    q.initialize(json.loads((OUT/'SLOTS.json').read_text()))
    for iteration in range(12):
        if q.counts().get('accepted',0)<500:
            call(f'generate_{iteration:02d}','data.action4_occlusion500_controller',['--run-tag',f'batch_{iteration:02d}'],list(range(96)))
        call(f'similarity_{iteration:02d}','data.action4_occlusion500_similarity',['--workers','64'],list(range(64)))
        report=json.loads((OUT/'SIMILARITY_CURRENT.json').read_text())
        if not report['rejected']:break
        call(f'cull_{iteration:02d}','data.action4_occlusion500_similarity',['--cull'])
    else:raise RuntimeError('Additional replacements did not converge; do not weaken thresholds or change base5000')
    replacements=OUT/'PREVIEW_REPLACEMENTS_REQUIRED.json'
    if replacements.exists():
        slots=json.loads(replacements.read_text())['slots']
        call('replacement_visual_products','eval.action4_occlusion500_review',['--slots']+list(map(str,slots))+['--status','accepted','--output','preview_review_replacements'],list(range(64)))
        write(OUT/'PIPELINE_STATUS.json',{'stage':'WAIT_PREVIEW_REPLACEMENT_VISUAL_REVIEW','slots':slots,'time_unix':time.time(),'formal_merge_complete':False})
        while not (OUT/'ROOT_PREVIEW_REPLACEMENT_REVIEW.json').exists():time.sleep(60)
    call('batch_engineering_and_interface','data.action4_occlusion500_audit',cpus=list(range(64)))
    from data.action4_occlusion500_similarity import additional_records
    records=additional_records();selected=set();covered=set()
    for r in records:
        tags={(r['scene_id'],s,o['occlusion_bin']) for s,o in zip(['left','right'],r['instances'])}
        if tags-covered:selected.add(r['slot_id']);covered|=tags
    # Include image geometry extremes and any replaced preview identities.
    for value in [lambda r:min(o['bbox_center_uv'][0] for o in r['instances']),lambda r:max(o['bbox_center_uv'][1] for o in r['instances']),
                  lambda r:min(o['actual_width_fraction'] for o in r['instances']),lambda r:max(o['actual_width_fraction'] for o in r['instances'])]:
        ordered=sorted(records,key=value);selected.add(ordered[0]['slot_id']);selected.add(ordered[-1]['slot_id'])
    replacements=OUT/'PREVIEW_REPLACEMENTS_REQUIRED.json'
    if replacements.exists():selected.update(json.loads(replacements.read_text())['slots'])
    call('final_visual_products','eval.action4_occlusion500_review',['--slots']+list(map(str,sorted(selected)))+['--status','accepted','--output','preview_review_final500'],list(range(64)))
    write(OUT/'PIPELINE_STATUS.json',{'stage':'AWAIT_ADDITIONAL500_FINAL_VISUAL_REVIEW','pid':os.getpid(),'new_images':500,'global_similarity_images':5500,
        'formal_merge_complete':False,'next':'Actual root visual review, then data.action4_occlusion500_merge','time_unix':time.time()})

if __name__=='__main__':
    try:run()
    except BaseException:
        write(OUT/'PIPELINE_FAILURE.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});raise
