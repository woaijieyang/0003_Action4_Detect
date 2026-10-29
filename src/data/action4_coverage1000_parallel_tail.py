"""Parallel read-only lookahead of the *same frozen* remaining attempt stream.

GPU workers retain exclusive slot ownership. This helper never renders, claims
a slot, edits a pose, or relaxes a gate. Its cache may advance a pending slot
only across a contiguous prefix proved to fail the production geometry gates.
CPU helpers time-share logical CPUs already inside the72-core affinity set.
"""
from pathlib import Path
import argparse,collections,fcntl,hashlib,json,os,subprocess,sys,time
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
OUT=R/'outputs/coverage1000_v001';FOLDER=OUT/'parallel_tail_v001'

def worker(shard,shards):
    from data import action4_coverage1000_queue as q
    from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
    from data.action4_coverage1000_sampling import propose
    from data.action4_coverage1000_contract import occlusion_ranges,source_eligible
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_scene_visibility import SceneVisibility
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());spec=json.loads((FOLDER/'RANGE.json').read_text());slot=spec['slot']
    runtime=CoverageArmRuntime(slot['scene_id'],True,cfg['arm_asset_version']);trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']};checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees)
    catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text());tested=0;reasons=collections.Counter();passed=[];last=spec['start']+shard-shards;start=time.time()
    for attempt in range(spec['start']+shard,spec['end'],shards):
        if tested%16==0:
            db=q.connect();state=db.execute('SELECT status FROM slots WHERE slot=?',(slot['slot_id'],)).fetchone()[0];db.close()
            if state=='accepted':break
            best=FOLDER/'BEST.json'
            if best.exists() and attempt>json.loads(best.read_text())['attempt']:break
        last=attempt;tested+=1;p,error=propose(runtime,cfg,slot,attempt,catalog,trees,ray_resolution=28)
        if error:reasons[error]+=1;continue
        runtime.apply(p);collision=checker.check()
        if not collision['accepted']:reasons['collision:'+collision['status']]+=1;continue
        sources=visibility.measure(p);reason=None
        for side,m in sources.items():
            lo,hi,inc=occlusion_ranges(slot)[side];o=m['estimated_external_occlusion'];counts=m['occluder_class_counts'];total=sum(counts.values())
            if not max(0,lo-.04)<=o<=min(.79,hi+.04):reason=side+':scene_occlusion';break
            if not source_eligible(o,m['primary_source_class']):reason=side+':scene_source';break
            if o>.35 and m['primary_occluder_object']==side+'_forearm_continuation':reason=side+':upperarm_not_forearm_occlusion';break
            if total>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*total:reason=side+':non_own_primary';break
        if reason:reasons[reason]+=1;continue
        actual=runtime.export_corners()
        if actual['max_proposal_vs_actual_error_px']>.005:raise ValueError('Actual projection mismatch; not an ordinary failure')
        value={'attempt':attempt,'slot_id':slot['slot_id'],'protocol_id':cfg['protocol_id'],'projection_error_px':actual['max_proposal_vs_actual_error_px'],'pair_metrics':p['phrase2']['pair_metrics']}
        passed.append(value)
        with(FOLDER/'best.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);best=FOLDER/'BEST.json'
            if not best.exists() or json.loads(best.read_text())['attempt']>attempt:
                temp=best.with_suffix('.json.tmp');temp.write_text(json.dumps(value));os.replace(temp,best)
    result={'status':'COMPLETE_GEOMETRY_ONLY','shard':shard,'shards':shards,'start':spec['start'],'end':spec['end'],'last_tested':last,
        'tested':tested,'successes':passed,'rejections':dict(reasons),'seconds':time.time()-start,'no_RGB':True}
    (FOLDER/f'worker_{shard:02d}.json').write_text(json.dumps(result));print(json.dumps({k:v for k,v in result.items() if k not in ['successes','rejections']}),flush=True)

def coordinate():
    from data import action4_coverage1000_queue as q
    FOLDER.mkdir(exist_ok=False);db=q.connect();remaining=[dict(r) for r in db.execute("SELECT * FROM slots WHERE status IN ('pending','claimed')")];db.close()
    if len(remaining)!=1:
        (FOLDER/'COMPLETE.json').write_text(json.dumps({'status':'NOT_NEEDED_OR_NOT_SINGLE_SLOT','remaining':len(remaining)}));return
    row=remaining[0];slot=json.loads(row['spec']);start_attempt=row['attempt'];cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text())
    end=min(cfg['max_total_attempts_per_slot'],start_attempt+36000)
    capacity=json.loads((OUT/'runs/bulk_v001/CURRENT_CAPACITY.json').read_text());allcpus=set(c for j in capacity['jobs'] for c in j['cpu_affinity'])|{22,23}
    if len(allcpus)!=72:raise ValueError('Unexpected original CPU budget')
    candidates=[j['cpu_affinity'][0] for j in capacity['jobs'] if j['kind']=='render' and j['pid']!=row['owner']]
    cpus=sorted(set(candidates))[:30];shards=len(cpus)
    if shards<4:raise ValueError('Not enough existing idle-render CPU slots')
    spec={'slot':slot,'start':start_attempt,'end':end,'shards':shards,'protocol_id':cfg['protocol_id'],'role':'read-only production-geometry lookahead','RGB_generated':False}
    (FOLDER/'RANGE.json').write_text(json.dumps(spec,indent=2))
    env=os.environ.copy();env.update(PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(R/'runtime'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',XDG_CACHE_HOME=str(R/'runtime/cache'),XDG_CONFIG_HOME=str(R/'runtime/config'),BLENDER_USER_CONFIG=str(R/'runtime/phrase2_blender/config'),BLENDER_USER_DATAFILES=str(R/'runtime/phrase2_blender/datafiles'),BLENDER_USER_SCRIPTS=str(R/'runtime/phrase2_blender/scripts'))
    children=[];jobs=[]
    for i,cpu in enumerate(cpus):
        cmd=['taskset','-c',str(cpu),str(R/'runtime/blender-3.6.23-linux-x64/blender'),'-b','-t','1','--python-exit-code','1','--python',str(R/'src/data/action4_coverage1000_parallel_tail.py'),'--','worker',str(i),str(shards)]
        with(FOLDER/f'worker_{i:02d}.log').open('xb') as f:p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);jobs.append({'pid':p.pid,'CPU_affinity':[cpu],'command':cmd})
    (FOLDER/'DISPATCH.json').write_text(json.dumps({'jobs':jobs,'existing_CPU_union':sorted(allcpus),'CPU_union_with_helpers':sorted(allcpus|set(cpus)),
        'exclusive_new_CPU_allocated':0,'shared_idle_CPU_tasks':shards,'main_GPU_workers_not_modified':True,'GPU_used_by_helpers':False,'started_unix':time.time()},indent=2))
    while any(p.poll() is None for p in children):time.sleep(1)
    if any(p.returncode for p in children):raise RuntimeError('Lookahead failed; do not alter the original queue')
    reports=[json.loads((FOLDER/f'worker_{i:02d}.json').read_text()) for i in range(shards)];best=FOLDER/'BEST.json'
    if not best.exists():
        (FOLDER/'COMPLETE.json').write_text(json.dumps({'status':'NO_NEW_GEOMETRY_SOLUTION_QUEUE_UNCHANGED','tested':sum(r['tested'] for r in reports)}));return
    candidate=json.loads(best.read_text());target=candidate['attempt']
    # Each strided stream must have covered every predecessor of the minimum.
    for i,r in enumerate(reports):
        first=start_attempt+i;required=first+((target-first)//shards)*shards if first<=target else first-shards
        if r['last_tested']<required:raise ValueError('Incomplete predecessor proof; no queue fast-forward')
    applied=False;already_accepted=False;begin=time.time();moves=[]
    while time.time()-begin<60:
        db=q.connect();db.execute('BEGIN IMMEDIATE');current=db.execute('SELECT * FROM slots WHERE slot=?',(slot['slot_id'],)).fetchone()
        if current['status']=='accepted':already_accepted=True;db.execute('COMMIT');db.close();break
        if current['attempt']>target:db.execute('COMMIT');db.close();break
        if current['status']=='pending' and current['owner'] is None and current['attempt']>=start_attempt:
            previous=current['attempt'];db.execute('UPDATE slots SET attempt=?,error=?,updated=? WHERE slot=?',(target,'proved_geometry_failure_prefix:'+str((FOLDER/'RANGE.json').relative_to(R)),time.time(),slot['slot_id']));db.execute('COMMIT');db.close();applied=True;moves.append({'from_attempt':previous,'to_attempt':target});break
        db.execute('COMMIT');db.close();time.sleep(.02)
    result={'status':'GEOMETRY_PREFIX_APPLIED' if applied else 'ORIGINAL_SLOT_ALREADY_ACCEPTED' if already_accepted else 'CACHE_NOT_APPLIED_OR_ALREADY_PASSED',
        'slot_id':slot['slot_id'],'target_attempt':target,'tested':sum(r['tested'] for r in reports),'moves':moves,'all_predecessor_geometry_failures_proved':True,
        'sampler_config_labels_and_model_unchanged':True,'RGB_generated_by_helper':False,'CPU_union_max':72}
    (FOLDER/'COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    coordinate() if args[0]=='coordinate' else worker(int(args[1]),int(args[2]))
