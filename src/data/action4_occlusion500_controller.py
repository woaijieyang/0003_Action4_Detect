"""Additional500 batch: derived from src/data/action4_phrase2_controller.py; see DERIVATION.json."""
"""Bounded phrase2 preview->bulk orchestration; no training, no duplicate dispatch."""
import json,os,time,subprocess,fcntl,hashlib,traceback
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from data import action4_occlusion500_queue as queue
from data.action4_occlusion500_quality import publish
from data.action4_occlusion500_host_worker import remove_staging

def atomic(path,value):
    temp=Path(str(path)+".tmp");temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)

def promote_preview(out,plan,cfg):
    gate=json.loads((out/"ROOT_PREVIEW_ACCEPTED.json").read_text())
    if gate.get("decision")!="accept_preview128" or set(gate["slot_ids"])!=set(plan["preview_slot_ids"]):raise ValueError("Explicit exact128 root visual review required")
    if gate["configuration_sha256"]!=hashlib.sha256((R/"configs/data/action4_phrase2_occlusion500_v001.json").read_bytes()).hexdigest():raise ValueError("Preview config changed")
    db=queue.connect();rows=db.execute("SELECT * FROM slots WHERE status='preview_checked'").fetchall()
    for row in rows:
        db.execute("BEGIN IMMEDIATE");db.execute("UPDATE slots SET status='checking',owner=? WHERE slot=?",(os.getpid(),row["slot"]));db.execute("COMMIT")
        checked=json.loads(row["record"]);payload=json.loads((R/checked["candidate"]).read_text());record=payload["record"];slot=payload["slot"]
        if gate["rgb_sha256_by_slot"][str(slot["slot_id"])]!=record["rgb_sha256"]:raise ValueError("Reviewed RGB identity mismatch")
        accepted=publish(record,payload["proposal"],slot,checked["stats"]);accepted["surface_validation"]=checked["surface_validation"];accepted["source_candidate_record"]=checked["candidate"]
        (R/payload["runtime_dir"]/"PUBLISHED_RECORD.json").write_text(json.dumps(accepted))
        queue.transition(slot["slot_id"],"accepted",record=accepted)
        remove_staging(payload)
    db.close();queue.publish_manifests();queue.open_bulk();atomic(out/"BULK_OPENED.json",{"preview_promoted":len(rows),"remaining":500-len(rows),"no_RGB_repeated":True})

# The exact, adapted promote_preview helper is prepended during installation.
def run():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-tag',required=True);args=parser.parse_args()
    assert Path.cwd().resolve()==R
    cfg=json.loads((R/'configs/data/action4_phrase2_occlusion500_v001.json').read_text());out=R/cfg['output_root']
    from data.action4_occlusion500_contract import assert_configuration
    assert_configuration(cfg)
    base=json.loads((R/'data/phrase2/DATASET_COMPLETE.json').read_text())
    assert base['status']=='FORMAL_DATASET_USABLE' and base['images']==5000,'Original5000 must finish formal release before additional RGB'
    for split,digest in base['manifest_sha256'].items():assert hashlib.sha256((R/'data/phrase2'/split/'annotations.jsonl').read_bytes()).hexdigest()==digest
    guard=(out/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'STOP.json').exists()
    plan=json.loads((out/'SLOTS.json').read_text());queue.initialize(plan)
    db=queue.connect();assert not db.execute("SELECT count(*) FROM slots WHERE status NOT IN ('pending','accepted','preview_checked')").fetchone()[0];db.close()
    directory=out/'runs'/args.run_tag;directory.mkdir(parents=True,exist_ok=False)
    profile=json.loads((R/'configs/data/action4_phrase2_execution_v002.json').read_text())
    os.sched_setaffinity(0,set(profile['controller_cpu']))
    env=dict(os.environ,PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(R/'runtime'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(R/'runtime/cache'),XDG_CONFIG_HOME=str(R/'runtime/config'),
        CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'),OPTIX_CACHE_PATH=str(R/'runtime/optix_cache'),BLENDER_USER_CONFIG=str(R/'runtime/phrase2_blender/config'),
        BLENDER_USER_DATAFILES=str(R/'runtime/phrase2_blender/datafiles'),BLENDER_USER_SCRIPTS=str(R/'runtime/phrase2_blender/scripts'))
    gpu_uuid={g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in profile['gpu']}
    processes=[];dispatch=[];generation={};start=time.time();last=0
    bulk=(out/'ROOT_PREVIEW_ACCEPTED.json').exists()
    if bulk:queue.open_bulk()
    def launch(role,index,scene=None):
        key=(role,index);n=generation.get(key,0);generation[key]=n+1;worker=f'{role}_{index:02d}_{args.run_tag}_{n:02d}'
        if role=='render':
            p=profile['render_workers'][index];cpus=p['cpu'];gpu=p['gpu']
            cmd=[str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','2','--python-exit-code','1','--python',str(R/'src/data/action4_occlusion500_cached_worker.py'),'--',scene,worker]
        else:
            cpus=profile['quality_cpu'][index];gpu=None;cmd=[str(R/'.venv/bin/python'),'-m','data.action4_occlusion500_host_worker',worker]
        log=(directory/(worker+'.log')).open('x');p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=gpu_uuid[gpu] if gpu is not None else ''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cpus)));log.close()
        processes.append(p);dispatch.append({'role':role,'index':index,'worker':worker,'scene':scene,'pid':p.pid,'cpu_affinity':cpus,'gpu':gpu,'command':cmd,'reaped':False})
        atomic(directory/'DISPATCH.json',{'controller_pid':os.getpid(),'workers':dispatch,'CPU_budget':72,'started_unix':start})
        atomic(out/'CURRENT_RUN.json',{'run':str(directory.relative_to(R)),'controller_pid':os.getpid(),'dispatch':str((directory/'DISPATCH.json').relative_to(R))})
    db=queue.connect();demand={r[0]:r[1] for r in db.execute("SELECT scene,count(*) FROM slots WHERE status='pending' GROUP BY scene")};db.close()
    if not demand:return
    allocation={s:0 for s in demand};scenes=[]
    for _ in range(min(len(profile['render_workers']),sum(demand.values()))):
        scene=max(demand,key=lambda s:demand[s]/(allocation[s]+1));allocation[scene]+=1;scenes.append(scene)
    for i in range(3):launch('quality',i)
    for i,scene in enumerate(scenes):launch('render',i,scene)
    while True:
        if (out/'STOP.json').exists():raise RuntimeError('Additional batch engineering failure; inspect actual evidence')
        counts=queue.counts()
        if not bulk and counts.get('preview_checked',0)==128:
            atomic(out/'PREVIEW_READY.json',{'count':128,'not_formally_merged':True,'requires':'Root RGB/local-input/mask/canonical8 review'})
            if (out/'ROOT_PREVIEW_ACCEPTED.json').exists():promote_preview(out,plan,cfg);bulk=True
        for process,worker in list(zip(processes,dispatch)):
            rc=process.poll()
            if rc is None or worker['reaped']:continue
            worker['reaped']=True;worker['exit_code']=rc
            if rc:raise RuntimeError((worker,rc))
            if worker['role']=='render' and counts.get('accepted',0)<500:
                db=queue.connect();todo={r[0]:r[1] for r in db.execute("SELECT scene,count(*) FROM slots WHERE status='pending' AND enabled=1 GROUP BY scene")};db.close()
                if todo:
                    live={s:sum(w['role']=='render' and w['scene']==s and p.poll() is None for p,w in zip(processes,dispatch)) for s in todo}
                    scene=max(todo,key=lambda s:todo[s]/(live[s]+1));launch('render',worker['index'],scene)
        now=time.time()
        if now-last>30:
            queue.publish_manifests();atomic(out/'STATUS.json',{'counts':counts,'bulk_opened':bulk,'elapsed_seconds':now-start,'candidate_root':cfg['candidate_root'],'formal_merge_complete':False,'time_unix':now});last=now
        if counts.get('accepted',0)==500:
            queue.publish_manifests()
            for p in processes:
                if p.poll() is None:p.wait(timeout=180)
            atomic(out/'ALL_SLOTS_COMPLETE.json',{'count':500,'requires':'Global5500 similarity, batch audit and final visual admission before merge','time_unix':time.time()});return
        time.sleep(5)

if __name__=='__main__':
    try:run()
    except BaseException:
        out=R/'outputs/phrase2_occlusion500_v001';atomic(out/'CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});raise
