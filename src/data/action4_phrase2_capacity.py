"""Add owned render capacity without editing the running frozen worker/controller.

CPU affinity, not nominal thread-pool size, bounds simultaneous runnable cores.
New workers use the exact generation implementation and the same transactional
queue. SIGUSR1 drains only an added worker after its current claim completes.
"""
from pathlib import Path
import json,os,sys,time,signal,subprocess,hashlib,traceback,sqlite3
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
OUT=R/'outputs/phrase2_generation_v001'

def write(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

def bounded_worker(scene,name):
    from data import action4_phrase2_queue as queue
    from data.action4_phrase2_worker import run
    state={'drain':False};original_claim=queue.claim
    class Drained(Exception):pass
    def request_drain(signum,frame):state['drain']=True
    signal.signal(signal.SIGUSR1,request_drain)
    def claim(*args,**kwargs):
        if state['drain']:raise Drained()
        return original_claim(*args,**kwargs)
    queue.claim=claim
    try:run(scene,name)
    except Drained:
        # claim() runs before a new slot is owned, outside the frame try/except.
        write(OUT/'workers'/name/'CONTROLLED_DRAIN_COMPLETE.json',{'pid':os.getpid(),'no_slot_owned':True,'no_frame_interrupted':True})

def command(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None

def affinity(pid,cpus,worker):
    value=command(pid)
    if value is None:return False
    if 'action4_phrase2_worker.py' not in value or worker not in value:raise ValueError(('PID identity changed',pid))
    result=subprocess.run(['taskset','-a','-p','-c',','.join(map(str,cpus)),str(pid)],capture_output=True,text=True)
    if result.returncode and command(pid) is not None:raise RuntimeError(result.stderr)
    return result.returncode==0

def supervise():
    import fcntl
    guard=(OUT/'capacity.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert Path.cwd().resolve()==R and (OUT/'BULK_OPENED.json').exists() and not (OUT/'CAPACITY_DISPATCH_v001.json').exists()
    os.sched_setaffinity(0,{70,71})
    dispatch=json.loads((OUT/'DISPATCH_BULK.json').read_text())
    old=[w for w in dispatch['workers'] if w['role']=='render'];assert len(old)==32
    old.sort(key=lambda w:w['worker'])
    expected=set()
    for w in old:
        assert len(w['cpu_affinity'])==2 and 'action4_phrase2_worker.py' in command(w['pid']) and w['worker'] in command(w['pid'])
        assert not expected.intersection(w['cpu_affinity']);expected.update(w['cpu_affinity'])
    assert expected==set(range(64))
    uuids={g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in [4,5,6,7]}
    free={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for g in [4,5,6,7]}
    assert min(free.values())>=20000,('Need enough headroom for up to four extra render contexts per card',free)
    db=sqlite3.connect(OUT/'slots.sqlite3');now=time.time();baseline_n=db.execute("SELECT count(*) FROM slots WHERE status='accepted' AND updated>?",(now-300,)).fetchone()[0];baseline_count=db.execute("SELECT count(*) FROM slots WHERE status='accepted'").fetchone()[0];db.close()
    log=(OUT/'capacity_supervisor_v001.log').open('a')
    env=dict(os.environ,PYTHONPATH=str(R/'src'),TMPDIR=str(R/'runtime'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             XDG_CACHE_HOME=str(R/'runtime/cache'),XDG_CONFIG_HOME=str(R/'runtime/config'),CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'),OPTIX_CACHE_PATH=str(R/'runtime/optix_cache'),
             BLENDER_USER_CONFIG=str(R/'runtime/phrase2_blender/config'),BLENDER_USER_DATAFILES=str(R/'runtime/phrase2_blender/datafiles'),BLENDER_USER_SCRIPTS=str(R/'runtime/phrase2_blender/scripts'))
    scenes=['wall']*2+['table_full_garment']*11+['cloth_lift10_full_garment']*3
    extras=[];processes=[];started=time.time();ramp_done=False;drain_requested=False;restored=set()
    def launch(i):
        parents=old[2*i:2*i+2];cpus=[]
        for w in parents:
            if not affinity(w['pid'],w['cpu_affinity'][:1],w['worker']):raise RuntimeError('Main worker ended before capacity allocation')
            cpus.append(w['cpu_affinity'][1])
        gpu=4+i%4;name=f'render_capacity_{i:02d}_v001'
        cmd=[str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','2','--python-exit-code','1','--python',str(Path(__file__).resolve()),'--','worker',scenes[i],name]
        output=(OUT/(name+'.log')).open('x')
        try:
            proc=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=uuids[gpu]),stdout=output,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cpus)))
        except BaseException:
            for w in parents:affinity(w['pid'],w['cpu_affinity'],w['worker'])
            raise
        finally:output.close()
        processes.append(proc);extras.append({'worker':name,'pid':proc.pid,'gpu':gpu,'scene':scenes[i],'cpu_affinity':cpus,'parents_reduced_to_one_core':parents,'command':cmd})
        write(OUT/'CAPACITY_DISPATCH_v001.json',{'supervisor_pid':os.getpid(),'generation_source_freeze':'SOURCE_FREEZE_v009.json','wrapper_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_workers':32,'added_workers':extras,'current_render_processes':32+len(extras),'target_render_processes':48,'target_per_GPU':12,'CPU_limit':72,'CPU_contract':'32 base workers one core +16 added workers two cores =64 render; host6 + controller/supervisor shared cores70,71 =72','baseline_5min_accepted':baseline_n,'baseline_count':baseline_count,'started_unix':started,'initial_free_MiB':free})
    # First add one per card. Ramp only after actual accepted new-worker data.
    for i in range(4):launch(i)
    while True:
        if (OUT/'STOP.json').exists() or (OUT/'CAPACITY_DRAIN_REQUESTED.json').exists():drain_requested=True
        for i,(proc,w) in enumerate(zip(processes,extras)):
            rc=proc.poll()
            if rc is not None and i not in restored:
                for parent in w['parents_reduced_to_one_core']:affinity(parent['pid'],parent['cpu_affinity'],parent['worker'])
                restored.add(i)
                if rc!=0:
                    write(OUT/'STOP.json',{'source':'owned capacity worker failed','pid':proc.pid,'worker':w['worker'],'exit_code':rc,'no_automatic_restart':True});drain_requested=True
        if not ramp_done and not drain_requested:
            proven=[any((OUT/'workers'/w['worker']).glob('*/PUBLISHED_RECORD.json')) for w in extras[:4]]
            if all(proven):
                # Base memory peak was below26GiB/card; four added contexts at
                # a conservative4GiB each retain a safety reserve on48GiB cards.
                memory={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for g in [4,5,6,7]}
                if min(memory.values())>=16000:
                    for i in range(4,16):launch(i)
                    ramp_done=True
                    write(OUT/'CAPACITY_RAMP_COMPLETE.json',{'renderers':48,'per_GPU':12,'CPU_limit':72,'proof':'each first added worker produced actual host-accepted data before full ramp','free_before_final_ramp_MiB':memory,'time_unix':time.time()})
                else:
                    # Keep the stable36-worker configuration; do not oversubscribe.
                    ramp_done=True;write(OUT/'CAPACITY_RAMP_LIMITED.json',{'renderers':36,'reason':'keep GPU memory reserve','free_MiB':memory})
        elapsed=time.time()-started
        if elapsed>600 and not (OUT/'CAPACITY_10MIN_REVIEW.json').exists():
            db=sqlite3.connect(OUT/'slots.sqlite3');accepted=db.execute("SELECT count(*) FROM slots WHERE status='accepted'").fetchone()[0];db.close()
            rate=(accepted-baseline_count)*60/elapsed;base_rate=baseline_n/5
            regression=baseline_n>=100 and rate<base_rate*.8
            write(OUT/'CAPACITY_10MIN_REVIEW.json',{'accepted':accepted,'actual_images_per_min_since_addition':rate,'prior_5min_images_per_min':base_rate,'not_controlled_benchmark':True,'clear_regression_drain':regression})
            if regression:drain_requested=True
        if drain_requested:
            for proc in processes:
                if proc.poll() is None:os.kill(proc.pid,signal.SIGUSR1)
        live=[w for proc,w in zip(processes,extras) if proc.poll() is None]
        write(OUT/'CAPACITY_STATUS.json',{'live_added_renderers':len(live),'base_renderers':32,'ramp_done':ramp_done,'drain_requested':drain_requested,'CPU_limit':72,'elapsed_seconds':elapsed,'time_unix':time.time()})
        if not live:
            write(OUT/'CAPACITY_COMPLETE.json',{'all_extra_workers_finished':True,'all_parent_affinities_restored_where_live':len(restored)==len(extras),'drained':drain_requested});return
        time.sleep(15)

if __name__=='__main__':
    if '--' in sys.argv:
        args=sys.argv[sys.argv.index('--')+1:];assert args[0]=='worker';bounded_worker(args[1],args[2])
    else:supervise()
