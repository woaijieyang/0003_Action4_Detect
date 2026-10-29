"""Resume the existing admitted slot plan with fixed-scene cached workers.

This entry point never resets accepted slots or promotes the preview a second
time. A fresh run directory is mandatory; all dispatches remain attributable.
"""
import fcntl, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from data import action4_phrase2_queue as queue

OUT = R/'outputs/phrase2_generation_v001'

def write(path, obj):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); os.replace(tmp, path)

def run():
    assert Path.cwd().resolve() == R
    guard = (OUT/'controller.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'STOP.json').exists()
    gate = json.loads((OUT/'ROOT_PREVIEW_ACCEPTED.json').read_text())
    cfg_path = R/'configs/data/action4_phrase2_v001.json'
    assert gate['decision'] == 'accept_preview128'
    assert gate['configuration_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    assert (OUT/'BULK_OPENED.json').exists()
    queue.initialize(json.loads((OUT/'SLOTS.json').read_text()))
    db = queue.connect()
    assert not db.execute("SELECT count(*) FROM slots WHERE status NOT IN ('pending','accepted')").fetchone()[0]
    db.close()
    profile = json.loads((R/'configs/data/action4_phrase2_execution_v002.json').read_text())
    tag = os.environ['PHRASE2_RUN_TAG']
    directory = OUT/'runs'/tag; directory.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, set(profile['controller_cpu']))
    queue.open_bulk()
    uuids = {g: subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'], text=True).strip() for g in profile['gpu']}
    env = dict(os.environ, PYTHONPATH=str(R/'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(R/'runtime'),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', XDG_CACHE_HOME=str(R/'runtime/cache'),
               XDG_CONFIG_HOME=str(R/'runtime/config'), CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'), OPTIX_CACHE_PATH=str(R/'runtime/optix_cache'),
               BLENDER_USER_CONFIG=str(R/'runtime/phrase2_blender/config'), BLENDER_USER_DATAFILES=str(R/'runtime/phrase2_blender/datafiles'),
               BLENDER_USER_SCRIPTS=str(R/'runtime/phrase2_blender/scripts'))
    children = []; entries = []; generation = {}; started = time.time(); last = 0
    def launch(role, index, scene=None):
        key = (role, index); n = generation.get(key, 0); generation[key] = n+1
        name = f'{role}_{index:02d}_{tag}_{n:02d}'
        if role == 'render':
            item = profile['render_workers'][index]; cpus=item['cpu']; gpu=item['gpu']
            cmd=[str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','2','--python-exit-code','1','--python',str(R/'src/data/action4_phrase2_cached_worker.py'),'--',scene,name]
        else:
            cpus=profile['quality_cpu'][index]; gpu=None
            cmd=[str(R/'.venv/bin/python'),'-m','data.action4_phrase2_host_worker',name]
        log=(directory/(name+'.log')).open('x')
        p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=uuids[gpu] if gpu is not None else ''),stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cpus)))
        log.close(); item={'role':role,'index':index,'worker':name,'pid':p.pid,'gpu':gpu,'cpu_affinity':cpus,'scene':scene,'command':cmd,'reaped':False}
        entries.append(item);children.append((p,item))
        write(directory/'DISPATCH.json',{'controller_pid':os.getpid(),'workers':entries,'CPU_limit':72,'started_unix':started})
        write(OUT/'BULK_CURRENT.json',{'run':str(directory.relative_to(R)),'controller_pid':os.getpid(),'dispatch':str((directory/'DISPATCH.json').relative_to(R))})
    db=queue.connect(); demand={r[0]:r[1] for r in db.execute("SELECT scene,count(*) FROM slots WHERE status='pending' GROUP BY scene")}; db.close()
    if not demand:
        write(directory/'COMPLETE.json',{'already_complete':True}); return
    scenes=[]
    # Allocate once per batch by remaining scene demand. A worker never reloads
    # environments between images. Its CPU/GPU slot can be recycled only after
    # its entire fixed scene has finished.
    allocation={s:0 for s in demand}
    for _ in profile['render_workers']:
        s=max(demand,key=lambda s:demand[s]/(allocation[s]+1));allocation[s]+=1;scenes.append(s)
    for i in range(len(profile['quality_cpu'])):launch('quality',i)
    for i,s in enumerate(scenes):launch('render',i,s)
    while True:
        if (OUT/'STOP.json').exists():raise RuntimeError('Owned worker reported engineering failure')
        counts=queue.counts()
        for p,w in list(children):
            code=p.poll()
            if code is None or w['reaped']:continue
            w['reaped']=True;w['exit_code']=code
            if code:raise RuntimeError(f'Owned worker failed: {w["worker"]} {code}')
            if w['role']=='render' and counts.get('accepted',0)<5000:
                db=queue.connect();todo={r[0]:r[1] for r in db.execute("SELECT scene,count(*) FROM slots WHERE status='pending' GROUP BY scene")};db.close()
                if todo:
                    live={s:sum(x['role']=='render' and x['scene']==s and q.poll() is None for q,x in children) for s in todo}
                    s=max(todo,key=lambda s:todo[s]/(live[s]+1));launch('render',w['index'],s)
        now=time.time()
        if now-last>=30:
            queue.publish_manifests()
            db=queue.connect();recent=db.execute("SELECT count(*) FROM slots WHERE status='accepted' AND updated>?",(now-300,)).fetchone()[0];db.close()
            status={'counts':counts,'bulk_opened':True,'run':tag,'elapsed_seconds':now-started,'formal_count':counts.get('accepted',0),
                    'fully_released':False,'accepted_images_per_min_last5':recent/5,'cached_render_workers_live':sum(w['role']=='render' and p.poll() is None for p,w in children),'time_unix':now}
            write(OUT/'STATUS.json',status);write(directory/'STATUS.json',status);last=now
        if counts.get('accepted',0)==5000:
            queue.publish_manifests()
            for p,w in children:
                if p.poll() is None:p.wait(timeout=180)
            write(OUT/'ALL_SLOTS_COMPLETE.json',{'count':5000,'run':tag,'time_unix':time.time(),'similarity_and_final_review_required':True})
            write(directory/'COMPLETE.json',{'accepted':5000,'seconds':time.time()-started});return
        time.sleep(5)

if __name__=='__main__':
    try:run()
    except BaseException:
        write(OUT/'BULK_CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});raise
