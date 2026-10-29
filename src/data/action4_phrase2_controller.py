"""Bounded phrase2 preview->bulk orchestration; no training, no duplicate dispatch."""
import json,os,time,subprocess,fcntl,hashlib,traceback
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from data import action4_phrase2_queue as queue
from data.action4_phrase2_quality import publish
from data.action4_phrase2_host_worker import remove_staging

def atomic(path,value):
    temp=Path(str(path)+".tmp");temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)

def promote_preview(out,plan,cfg):
    gate=json.loads((out/"ROOT_PREVIEW_ACCEPTED.json").read_text())
    if gate.get("decision")!="accept_preview128" or set(gate["slot_ids"])!=set(plan["preview_slot_ids"]):raise ValueError("Explicit exact128 root visual review required")
    if gate["configuration_sha256"]!=hashlib.sha256((R/"configs/data/action4_phrase2_v001.json").read_bytes()).hexdigest():raise ValueError("Preview config changed")
    db=queue.connect();rows=db.execute("SELECT * FROM slots WHERE status='preview_checked'").fetchall()
    for row in rows:
        db.execute("BEGIN IMMEDIATE");db.execute("UPDATE slots SET status='checking',owner=? WHERE slot=?",(os.getpid(),row["slot"]));db.execute("COMMIT")
        checked=json.loads(row["record"]);payload=json.loads((R/checked["candidate"]).read_text());record=payload["record"];slot=payload["slot"]
        if gate["rgb_sha256_by_slot"][str(slot["slot_id"])]!=record["rgb_sha256"]:raise ValueError("Reviewed RGB identity mismatch")
        accepted=publish(record,payload["proposal"],slot,checked["stats"]);accepted["surface_validation"]=checked["surface_validation"];accepted["source_candidate_record"]=checked["candidate"]
        (R/payload["runtime_dir"]/"PUBLISHED_RECORD.json").write_text(json.dumps(accepted))
        queue.transition(slot["slot_id"],"accepted",record=accepted)
        remove_staging(payload)
    db.close();queue.publish_manifests();queue.open_bulk();atomic(out/"BULK_OPENED.json",{"preview_promoted":len(rows),"remaining":5000-len(rows),"no_RGB_repeated":True})

def run():
    cfg=json.loads((R/"configs/data/action4_phrase2_v001.json").read_text());out=R/cfg["output_root"]
    guard=(out/"controller.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("Already dispatched; verify existing PIDs before explicit recovery")
    plan=json.loads((out/"SLOTS.json").read_text());queue.initialize(plan)
    # Engineering-only renders under earlier rejected assets consume attempts but never enter the dataset.
    burned={745:14,2283:12,3929:0,1941:22,3428:49}
    db=queue.connect()
    for slot,attempt in burned.items():db.execute("UPDATE slots SET attempt=max(attempt,?) WHERE slot=?",(attempt+1,slot))
    db.close()
    pool=sorted(os.sched_getaffinity(0))[:72]
    if len(pool)<72:raise RuntimeError("72-core budget unavailable")
    os.sched_setaffinity(0,set(pool[70:72]))
    gpu_uuid={g:subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=uuid","--format=csv,noheader"],text=True).strip() for g in [4,5,6,7]}
    for name in ["cache","config","cuda_cache","optix_cache","phrase2_blender/config","phrase2_blender/datafiles","phrase2_blender/scripts"]:(R/"runtime"/name).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",
        XDG_CACHE_HOME=str(R/"runtime/cache"),XDG_CONFIG_HOME=str(R/"runtime/config"),CUDA_CACHE_PATH=str(R/"runtime/cuda_cache"),OPTIX_CACHE_PATH=str(R/"runtime/optix_cache"),
        BLENDER_USER_CONFIG=str(R/"runtime/phrase2_blender/config"),BLENDER_USER_DATAFILES=str(R/"runtime/phrase2_blender/datafiles"),BLENDER_USER_SCRIPTS=str(R/"runtime/phrase2_blender/scripts"))
    processes=[];dispatch=[]
    def launch(role,index,scene=None):
        if role=="render":
            gpu=4+index%4;cpus=pool[index*2:index*2+2];worker=f"render_{index:02d}_{os.environ.get('PHRASE2_RUN_TAG','initial')}"
            cmd=[str(R/"runtime/blender-3.6.23-linux-x64/blender"),"-b","-t","2","--python-exit-code","1","--python",str(R/"src/data/action4_phrase2_worker.py"),"--",scene,worker]
            this_env=dict(env,CUDA_VISIBLE_DEVICES=gpu_uuid[gpu])
        else:
            gpu=None;cpus=pool[64+index*2:66+index*2];worker=f"quality_{index:02d}_{os.environ.get('PHRASE2_RUN_TAG','initial')}";cmd=[str(R/".venv/bin/python"),"-m","data.action4_phrase2_host_worker",worker];this_env=dict(env,CUDA_VISIBLE_DEVICES="")
        log=(out/(worker+".log")).open("x")
        proc=subprocess.Popen(cmd,cwd=R,env=this_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cpus)));log.close()
        processes.append(proc);dispatch.append({"role":role,"worker":worker,"pid":proc.pid,"gpu":gpu,"cpu_affinity":cpus,"scene":scene,"command":cmd})
    for i in range(3):launch("quality",i)
    initial=["wall"]*2+["table_full_garment"]*10+["cloth_lift10_full_garment"]*4
    for i,scene in enumerate(initial):launch("render",i,scene)
    atomic(out/"DISPATCH.json",{"pid":os.getpid(),"workers":dispatch,"CPU_limit":72,"initial_render_processes":16,"bulk_max_render_processes":32,"controller_cpu":pool[70:72],"preview_only_until_explicit_visual_review":True})
    ready=False;bulk=False;last=0;start=time.monotonic()
    while True:
        if (out/"STOP.json").exists():raise RuntimeError("Owned worker reported an engineering failure; no automatic retry")
        for process,worker in zip(processes,dispatch):
            if process.poll() is not None and process.returncode!=0:
                atomic(out/"STOP.json",{"pid":process.pid,"worker":worker,"exit_code":process.returncode});raise RuntimeError("Owned worker failed")
        counts=queue.counts()
        if not ready and counts.get("preview_checked",0)==128:
            atomic(out/"PREVIEW_READY.json",{"count":128,"formal_count":0,"seconds":time.monotonic()-start,"requires":"root full-image/crop/input/label review"});ready=True
        if ready and not bulk and (out/"ROOT_PREVIEW_ACCEPTED.json").exists():
            promote_preview(out,plan,cfg);bulk=True
            extra=["wall"]+["table_full_garment"]*12+["cloth_lift10_full_garment"]*3
            for i,scene in enumerate(extra,16):launch("render",i,scene)
            atomic(out/"DISPATCH_BULK.json",{"workers":dispatch,"render_processes":32,"quality_processes":3,"CPU_compute":70,"controller_CPU":2})
        if time.monotonic()-last>30:
            queue.publish_manifests();atomic(out/"STATUS.json",{"counts":counts,"bulk_opened":bulk,"elapsed_seconds":time.monotonic()-start,"source":"transactional slot queue","formal_count":counts.get("accepted",0)});last=time.monotonic()
        if counts.get("accepted",0)==5000:
            queue.publish_manifests();atomic(out/"ALL_SLOTS_COMPLETE.json",{"count":5000,"seconds":time.monotonic()-start,"final_distribution_and_visual_review_required":True})
            for process in processes:process.wait(timeout=120)
            return
        time.sleep(3)
if __name__=="__main__":
    try:run()
    except BaseException:
        out=R/"outputs/phrase2_generation_v001";atomic(out/"CONTROLLER_FAILURE.json",{"traceback":traceback.format_exc()});raise
