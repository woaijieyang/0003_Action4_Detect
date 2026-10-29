"""Bounded persistent-render dispatch. Never launches training or other jobs."""
from pathlib import Path
import argparse,collections,hashlib,json,os,random,subprocess,time
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import action4_coverage1000_queue as queue

def run(run_id,stage):
    if Path.cwd().resolve()!=R or Path(run_id).name!=run_id:raise ValueError('Project root and safe run ID required')
    out=R/'outputs/coverage1000_v001';cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());plan=json.loads((R/cfg['plan_path']).read_text())
    if cfg['status']!='FROZEN_FOR_PREVIEW' or cfg['protocol_id']!=plan['protocol_id']:raise ValueError('Protocol not frozen')
    if stage=='bulk':
        gate=json.loads((out/'ROOT_PREVIEW_ACCEPTED.json').read_text())
        if gate.get('status')!='PASS_FORMAL128_VISUAL_REVIEW' or gate.get('protocol_id')!=cfg['protocol_id']:raise ValueError('Root visual admission required before bulk')
    elif stage!='preview':raise ValueError('Unknown data stage')
    freeze=json.loads((out/'SOURCE_FREEZE_v001.json').read_text())
    for path,sha in freeze['files'].items():
        if hashlib.sha256(contained_path(R,path,exists=True).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+path)
    if (out/'STOP.json').exists():raise ValueError('Do not restart a stopped batch without diagnosis')
    directory=out/'runs'/run_id;directory.mkdir(parents=True,exist_ok=False)
    queue.initialize(plan)
    db=queue.connect()
    if stage=='bulk':db.execute("UPDATE slots SET enabled=1 WHERE status='pending'")
    selected={row[0]:row[1] for row in db.execute("SELECT scene,count(*) FROM slots WHERE enabled=1 AND status='pending' GROUP BY scene")};db.close()
    if not selected:raise ValueError('No unfinished enabled slots; never rerender completed slots')
    # All 32 render workers have two fixed CPU slots; quality6 + controller2 =72.
    total=sum(selected.values());allocation={s:max(1,int(32*n/total)) for s,n in selected.items()}
    while sum(allocation.values())<32:allocation[max(selected,key=lambda s:selected[s]/allocation[s])]+=1
    while sum(allocation.values())>32:allocation[max(allocation,key=lambda s:allocation[s])]-=1
    scenes=[s for s,n in allocation.items() for _ in range(n)];random.Random(cfg['master_seed']).shuffle(scenes)
    gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    cards={int(a[0]):a for line in gpu_text.splitlines() if(a:=[p.strip() for p in line.split(',')])}
    for card in [4,5,6,7]:
        if int(cards[card][2])>1024 or int(cards[card][3])>10:raise RuntimeError('Assigned render GPU is not idle; preserve other tasks: '+str(card))
    env=os.environ.copy();env.update(PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(R/'runtime'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',XDG_CACHE_HOME=str(R/'runtime/cache'),XDG_CONFIG_HOME=str(R/'runtime/config'),BLENDER_USER_CONFIG=str(R/'runtime/phrase2_blender/config'),BLENDER_USER_DATAFILES=str(R/'runtime/phrase2_blender/datafiles'),BLENDER_USER_SCRIPTS=str(R/'runtime/phrase2_blender/scripts'),CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'),OPTIX_CACHE_PATH=str(R/'runtime/optix_cache'))
    jobs=[];processes=[]
    for i,scene in enumerate(scenes):
        card=4+i//8;cpus=[24+i,72+i] if i<24 else [2*(i-24),2*(i-24)+1];child=env.copy();child['CUDA_VISIBLE_DEVICES']=cards[card][1]
        name=f'{run_id}_r{i:02d}';cmd=['taskset','-c',','.join(map(str,cpus)),str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','2','--python-exit-code','1','--python',str(R/'src/data/action4_coverage1000_worker.py'),'--',scene,name]
        log=directory/(name+'.log')
        with log.open('xb') as f:p=subprocess.Popen(cmd,cwd=R,env=child,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({'kind':'render','pid':p.pid,'worker_id':name,'scene':scene,'GPU':card,'cpu_affinity':cpus,'command':cmd,'log':str(log.relative_to(R))});processes.append(p)
    for i in range(3):
        cpus=[16+2*i,17+2*i];name=f'{run_id}_q{i:02d}';child=env.copy();child['CUDA_VISIBLE_DEVICES']=''
        cmd=['taskset','-c',','.join(map(str,cpus)),str(R/'.venv/bin/python'),'-B','-m','data.action4_coverage1000_host',name];log=directory/(name+'.log')
        with log.open('xb') as f:p=subprocess.Popen(cmd,cwd=R,env=child,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({'kind':'quality','pid':p.pid,'worker_id':name,'cpu_affinity':cpus,'command':cmd,'log':str(log.relative_to(R))});processes.append(p)
    receipt={'controller_pid':os.getpid(),'controller_cpu_affinity':sorted(os.sched_getaffinity(0)),'run_id':run_id,'stage':stage,'jobs':jobs,
        'cpu_budget':72,'render_workers':32,'render_per_GPU':8,'quality_workers':3,'protocol_id':cfg['protocol_id'],'scene_allocation':allocation,
        'started_unix':time.time(),'controller_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/'DISPATCH.json').write_text(json.dumps(receipt,indent=2));(out/'CURRENT_RUN.json').write_text(json.dumps({'run_id':run_id,'stage':stage,'dispatch':str((directory/'DISPATCH.json').relative_to(R))},indent=2))
    start=time.monotonic();first_checked=queue.counts().get('preview_checked',0)+queue.counts().get('accepted',0)
    generations=[0]*32
    while True:
        # Reuse a completed scene's exact CPU/GPU slot for another fixed
        # template. No healthy/failed process is restarted and concurrency
        # never exceeds the original32 render slots (8 per assigned GPU).
        if not(out/'STOP.json').exists():
            db=queue.connect();pending={r[0]:r[1] for r in db.execute("SELECT scene,count(*) FROM slots WHERE enabled=1 AND status='pending' GROUP BY scene")};db.close()
            for index in range(32):
                if processes[index].poll()!=0 or not pending:continue
                active=collections.Counter(jobs[k]['scene'] for k in range(32) if processes[k].poll() is None)
                scene=max(pending,key=lambda s:pending[s]/(active[s]+1));previous=jobs[index];generations[index]+=1
                name=f'{run_id}_r{index:02d}_g{generations[index]:02d}';child=env.copy();child['CUDA_VISIBLE_DEVICES']=cards[previous['GPU']][1]
                cpus=previous['cpu_affinity'];cmd=['taskset','-c',','.join(map(str,cpus)),str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','2','--python-exit-code','1','--python',str(R/'src/data/action4_coverage1000_worker.py'),'--',scene,name]
                log=directory/(name+'.log')
                with log.open('xb') as f:p=subprocess.Popen(cmd,cwd=R,env=child,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                jobs[index]={'kind':'render','pid':p.pid,'worker_id':name,'scene':scene,'GPU':previous['GPU'],'cpu_affinity':cpus,'command':cmd,'log':str(log.relative_to(R))}
                processes[index]=p
                with(directory/'CAPACITY_EVENTS.jsonl').open('a') as f:f.write(json.dumps({'old_pid':previous['pid'],'old_scene':previous['scene'],'new':jobs[index],'reason':'completed_worker_slot_reassigned_to_pending_scene','time_unix':time.time()})+'\n')
            (directory/'CURRENT_CAPACITY.json').write_text(json.dumps({'jobs':jobs,'maximum_render_processes':32,'maximum_per_GPU':8,'CPU_total_budget':72},indent=2))
        counts=queue.counts();live=sum(p.poll() is None for p in processes);failure=[{'pid':p.pid,'exit':p.returncode} for p in processes if p.poll() not in (None,0)]
        elapsed=time.monotonic()-start;checked=counts.get('preview_checked',0)+counts.get('accepted',0)
        status={'status':'RUNNING','counts':counts,'live_processes':live,'failed_processes':failure,'elapsed_seconds':elapsed,
            'engineering_checked_images_per_minute':(checked-first_checked)*60/max(1,elapsed),'formal_6500_released':False,'protocol_id':cfg['protocol_id']}
        if failure or counts.get('failed',0) or (out/'STOP.json').exists():status['status']='FAILED_REQUIRES_DIAGNOSIS'
        elif not live:
            if stage=='preview' and counts.get('preview_checked',0)==128:status['status']='AWAIT_ROOT_FORMAL128_VISUAL_REVIEW'
            elif stage=='bulk' and counts.get('accepted',0)==1000:status['status']='ALL1000_ENGINEERING_ACCEPTED_GLOBAL_AUDITS_REQUIRED';queue.publish_manifests()
            else:status['status']='INCOMPLETE_NO_AUTOMATIC_RESTART'
        temp=directory/'STATUS.json.tmp';temp.write_text(json.dumps(status,indent=2));os.replace(temp,directory/'STATUS.json')
        (out/'PIPELINE_STATUS.json').write_text(json.dumps({**status,'dispatch':str((directory/'DISPATCH.json').relative_to(R))},indent=2))
        if status['status']!='RUNNING':
            (directory/'COMPLETE.json').write_text(json.dumps(status,indent=2));break
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['preview','bulk'],required=True);a=p.parse_args();run(a.run_id,a.stage)
