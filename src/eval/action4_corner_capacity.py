"""Bounded full650 service-capacity probe using the actual corner evaluator. No BEST."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_sha,_v2_checked

def save(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temp,path)

def validate(cfg):
    n=cfg.get("capacity_workers")
    if (n not in (16,19,32) or type(n) is not int or cfg.get("mode")!="integration"
        or cfg.get("evaluation_only_engineering") is not True or cfg.get("training_gpus")!=[]
        or cfg.get("a0_variant")!="a0_corner_image" or cfg.get("corner_enabled") is not True
        or cfg.get("sam_enabled") is not True or cfg.get("expected_eval_images")!=650
        or cfg.get("eval_indices") is not None or cfg.get("eval_loader_workers")!=(0 if n==32 else 1)
        or cfg.get("eval_threads")!=1 or len(cfg.get("eval_gpus",[]))!=n):
        raise ValueError("Explicit bounded full650 corner service-only configuration required")
    if type(cfg.get("evaluation_allocator_budget_mib")) is not int or not 0<cfg["evaluation_allocator_budget_mib"]<=(3072 if n==32 else 4096):
        raise ValueError("Bounded capacity allocator required")
    if max(cfg["eval_gpus"].count(g) for g in (4,5,6,7))>(8 if n==32 else 5):raise ValueError("Capacity card worker budget exceeded")
    if cfg["process_cpu_affinity"].get("train")!=([57] if n==32 else [63]):raise ValueError("Unused legacy train affinity slot required")
    used=set()
    for i,gpu in enumerate(cfg["eval_gpus"]):
        cpus=cfg["process_cpu_affinity"]["eval_"+str(i)]
        if type(gpu) is not int or gpu not in (4,5,6,7) or len(cpus)!=(1 if n==32 else 2) or len(set(cpus))!=len(cpus) or set(cpus)&used or not set(cpus)<=set(range(24,56 if n==32 else 62)):
            raise ValueError("Independent authorized two-CPU eval allocations required")
        used.update(cpus)
    if cfg.get("controller_cpu")!=([56] if n==32 else [62]):raise ValueError("Registered capacity controller CPU required")
    return n

def worker(cfg,index):
    import torch
    from model.training.action4_runtime import canonical_hash
    from eval.action4_image_guided_eval import validate_corner_eval_environment,run_corner_evaluation
    from eval.action4_stage2_eval import load_evaluation_data
    n=validate(cfg)
    if type(index) is not int or not 0<=index<n:raise ValueError("Worker index outside registered domain")
    if set(os.sched_getaffinity(0))!=set(cfg["process_cpu_affinity"]["eval_"+str(index)]):
        raise ValueError("Actual CPU allocation differs")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=contained_path(R,cfg["output"],exists=True);start=time.monotonic()
    record={"worker_id":index,"pid":os.getpid(),"formal_training":False,"selection_committed":False}
    try:
        refs=json.loads((out/"CAPACITY_INPUT.json").read_text())
        if refs["config_id"]!=canonical_hash(cfg) or _v2_sha(refs["snapshot"]["path"])!=refs["snapshot"]["sha256"]:
            raise ValueError("Frozen capacity input changed")
        validate_corner_eval_environment(cfg,"cuda:0")
        data=load_evaluation_data(cfg)
        result=run_corner_evaluation(cfg,refs["snapshot"]["path"],out/f"worker_{index:02d}",data)
        if result["evaluation_images"]!=650:raise ValueError("Capacity full-domain evaluation incomplete")
        torch.cuda.synchronize()
        record.update(status="COMPLETE_FULL650_CAPACITY_CALL",images=650,
            wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            report_path=str((out/f"worker_{index:02d}/REPORT.json").relative_to(R)))
    except BaseException:
        record.update(status="FAILED",traceback=traceback.format_exc())
        raise
    finally:
        save(out/f"WORKER_{index:02d}.json",record)

def gpu_state():
    text=subprocess.check_output(["nvidia-smi","-i","4,5,6,7","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    return {int(v[0]):{"uuid":v[1].strip(),"free_mib":int(v[2])}
            for v in (line.split(",") for line in text.strip().splitlines())}

def run(cfg):
    from model.training.action4_runtime import canonical_hash
    from eval.action4_image_guided_eval import export_unchanged_engineering_snapshot
    n=validate(cfg)
    from model.training.action4_resource_policy import process_cpu_affinity
    process_cpu_affinity(cfg,"eval_0",R)  # Same guard as production, before any fan-out.
    if set(os.sched_getaffinity(0))!=set(cfg["controller_cpu"]) or os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU-only capacity controller required")
    out=contained_path(R,cfg["output"],exists=True)
    if (out/"DISPATCH.json").exists():raise FileExistsError("No repeated capacity dispatch")
    if cfg.get("capacity_wait_training_first") is True:
        from model.training.action4_capacity_barrier import settings
        item,barrier=settings(cfg,"eval")
        wait_start=time.monotonic();deadline=wait_start+item["timeout_seconds"]
        names=[f"train_{i}_READY.json" for i in range(4)]
        while not all((barrier/name).exists() for name in names):
            if (barrier/"ABORT.json").exists():raise RuntimeError("Training preparation aborted")
            if time.monotonic()>deadline:raise TimeoutError("Training first-batch preparation timeout")
            time.sleep(.5)
        ready=[json.loads((barrier/name).read_text()) for name in names]
        if any(r["experiment_id"]!=item["experiment_id"] or r["run_spec"]!=cfg["capacity_training_run_spec"] for r in ready):
            raise ValueError("Wrong training readiness")
        save(out/"TRAINING_PREWARM_WAIT.json",{"seconds":time.monotonic()-wait_start,
            "not_eval_service_time":True,"training_ready":ready})
    snapshot=export_unchanged_engineering_snapshot(cfg)
    path=contained_path(R,snapshot,exists=True);path.chmod(0o444)
    refs={"config_id":canonical_hash(cfg),"snapshot":{"path":snapshot,"sha256":_v2_sha(path)},
        "purpose":"service capacity only, integration weights not selected best"}
    save(out/"CAPACITY_INPUT.json",refs);(out/"CAPACITY_INPUT.json").chmod(0o444)
    state=gpu_state()
    if any(g["free_mib"]<32768 for g in state.values()):raise RuntimeError("Insufficient free GPU headroom for probe")
    launcher=contained_path(R,cfg["run_spec"],exists=True).with_name("launch.py")
    jobs=[];samples=[];failure=None;start=time.monotonic()
    try:
        for i,gpu in enumerate(cfg["eval_gpus"]):
            cpus=cfg["process_cpu_affinity"]["eval_"+str(i)]
            env=dict(os.environ,ACTION4_CAPACITY_WORKER_ID=str(i),CUDA_VISIBLE_DEVICES=state[gpu]["uuid"],OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
            for key in ("HOME","TMPDIR","CUDA_CACHE_PATH","TORCH_HOME","HF_HOME","XDG_CACHE_HOME","XDG_CONFIG_HOME","MPLCONFIGDIR"):
                cache=out/"runtime_cache"/str(i)/key;cache.mkdir(parents=True,exist_ok=True)
                env[key]=str(cache)
            cmd=["taskset","-c",",".join(map(str,cpus)),str(R/".venv/bin/python"),"-B",str(launcher),
                 "eval.action4_corner_capacity","--worker-id",str(i)]
            with (out/f"worker_{i:02d}.log").open("x") as log:
                proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((i,proc,gpu,cpus))
        save(out/"DISPATCH.json",{"time":time.time(),"workers":[{"id":i,"pid":p.pid,"gpu":g,"cpus":c} for i,p,g,c in jobs],
             "same_fixed_snapshot":refs["snapshot"],"training_concurrent":False,"no_best_or_queue_commit":True})
        if cfg.get("capacity_barrier") is not None:
            from model.training.action4_capacity_barrier import settings
            item,barrier=settings(cfg,"eval")
            deadline=time.monotonic()+item["timeout_seconds"]
            names=[f"eval_{i}_READY.json" for i in range(n)]+[f"train_{i}_READY.json" for i in range(4)]
            while not all((barrier/name).exists() for name in names):
                if (barrier/"ABORT.json").exists():raise RuntimeError("Training side aborted readiness")
                if any(p.poll() is not None for _,p,_,_ in jobs):raise RuntimeError("Worker ended before joint readiness")
                if time.monotonic()>deadline:raise TimeoutError("Joint preparation timeout")
                time.sleep(.5)
            records=[json.loads((barrier/name).read_text()) for name in names]
            if any(r["experiment_id"]!=item["experiment_id"] for r in records):
                raise ValueError("Mixed readiness identities")
            if (any(r["run_spec"]!=cfg["run_spec"] for r in records if r["role"]=="eval")
                or any(r["run_spec"]!=cfg["capacity_training_run_spec"] for r in records if r["role"]=="train")
                or len({r["pid"] for r in records})!=n+4):
                raise ValueError("Readiness run/PID domains differ")
            live=subprocess.run(["ps","-p",",".join(str(r["pid"]) for r in records),"-o","pid="],capture_output=True,text=True)
            if set(map(int,live.stdout.split()))!={r["pid"] for r in records}:raise RuntimeError("Ready process exited before GO")
            atomic_go={"experiment_id":item["experiment_id"],"time":time.time(),"roles":records}
            save(barrier/"GO.json",atomic_go)
        while True:
            codes=[p.poll() for _,p,_,_ in jobs]
            fresh=gpu_state();samples.append({"elapsed":time.monotonic()-start,"gpu":fresh})
            if any(g["free_mib"]<8192 for g in fresh.values()):raise RuntimeError("Capacity GPU residual margin violated")
            if any(c not in (None,0) for c in codes):raise RuntimeError("Owned capacity worker failed")
            if all(c is not None for c in codes):break
            if time.monotonic()-start>3600:raise TimeoutError("Bounded capacity probe exceeded3600seconds")
            time.sleep(15)
        records=[json.loads((out/f"WORKER_{i:02d}.json").read_text()) for i,_,_,_ in jobs]
        if any(r.get("status")!="COMPLETE_FULL650_CAPACITY_CALL" or r.get("images")!=650 for r in records):
            raise ValueError("Missing complete capacity result")
    except BaseException:
        failure=traceback.format_exc()
        if cfg.get("capacity_barrier") is not None:
            barrier=contained_path(R,cfg["capacity_barrier"]["directory"],exists=True)
            save(barrier/"ABORT.json",{"reason":failure})
        for _,p,_,_ in jobs:
            if p.poll() is None:p.terminate()
        for _,p,_,_ in jobs:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
    elapsed=time.monotonic()-start
    save(out/"RESOURCE_SAMPLES.json",samples)
    result={"status":"CAPACITY_COMPLETE" if failure is None else "CAPACITY_FAILED",
        "wall_seconds":elapsed,"full650_calls":n if failure is None else None,
        "calls_per_second":n/elapsed if failure is None else None,"failure":failure,
        "worker_exit_codes":[p.poll() for _,p,_,_ in jobs],"training_concurrent":None if cfg.get("capacity_barrier") else False,
        "joint_probe_configured":cfg.get("capacity_barrier") is not None,
        "published_best":False,"formal_training_launched":False,"formal_admission":False,
        "scope":"current integration checkpoint, full scored evaluator, cold concurrent service; not accuracy replication"}
    save(out/"CAPACITY_SUMMARY.json",result);print(json.dumps(result),flush=True)
    if failure is not None:raise RuntimeError("Capacity failed; preserved owned-worker evidence")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--worker-id",type=int)
    args=parser.parse_args()
    cfg=json.loads(contained_path(R,args.config,exists=True).read_text())
    from model.training.action4_run_snapshot import verify_running_snapshot
    verify_running_snapshot(cfg)
    if args.worker_id is None:
        try:run(cfg)
        except BaseException:
            if cfg.get("capacity_barrier") is not None:
                gate=contained_path(R,cfg["capacity_barrier"]["directory"],exists=True)
                save(gate/"ABORT.json",{"experiment_id":cfg["capacity_barrier"]["experiment_id"],"reason":traceback.format_exc()})
            raise
    else:worker(cfg,args.worker_id)
if __name__=="__main__":main()
