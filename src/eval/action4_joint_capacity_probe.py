"""Bounded actual-trainer timing under separately running capacity workers. No promotion."""
import argparse,json,os,signal,subprocess,time,traceback,hashlib
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_sha
from model.training.action4_runtime import atomic_json

def process_tmp_path(run_spec):
    # multiprocessing appends pymp-*/listener-*; AF_UNIX is limited to ~108 bytes.
    suffix=hashlib.sha256(str(run_spec).encode()).hexdigest()[:12]
    path=contained_path(R,Path(".cache/ipc")/suffix,exists=False)
    if len(os.fsencode(str(path)))>70:raise ValueError("Project IPC temporary path too long")
    return path

def summarize(samples,expected_steps=20):
    if [s["step"] for s in samples]!=list(range(1,expected_steps+1)):
        raise ValueError("Every actual optimizer update must be observed")
    windows={}
    for left,right in zip(samples,samples[1:]):
        if (left["epoch"]==right["epoch"] and left["global_actual_units"]==right["global_actual_units"]==48
            and left["did_update"] and right["did_update"]):
            dt=(right["status_mtime_ns"]-left["status_mtime_ns"])/1e9
            if dt<=0:raise ValueError("Non-increasing update time")
            eligible=(left["live_eval_workers"]==right["live_eval_workers"]==32
                and min(left["prediction_eval_workers"],right["prediction_eval_workers"])>=16)
            windows.setdefault(right["epoch"],[]).append({"seconds":dt,"joint_overlap":eligible})
    qualified=[v for v in windows.values() if len(v)>=3 and all(x["joint_overlap"] for x in v)]
    interval=min(sum(x["seconds"] for x in v)/len(v) for v in qualified) if qualified else None
    return {"full_batch_intervals":sum(len(v) for v in windows.values()),
        "qualified_joint_windows":len(qualified),"windows":windows,
        "fastest_observed_window_mean_seconds":interval,
        "required_eval_calls_per_second_with20percent_margin":None if interval is None else 1.2/(25*interval),
        "joint_timing_usable":len(qualified)>=2,"formal_admission":False}

def run(cfg):
    if (cfg.get("mode")!="integration" or cfg.get("role")!="A0" or cfg.get("joint_capacity_probe") is not True
        or cfg.get("world_size")!=4 or cfg.get("batch")!=48 or cfg.get("micro")!=2
        or cfg.get("expected_units")!=216 or cfg.get("controller_cpu")!=[57]
        or set(os.sched_getaffinity(0))!={57} or cfg.get("eval_gpus")!=[]):
        raise ValueError("Registered bounded real-trainer capacity probe required")
    out=contained_path(R,cfg["output"],exists=True)
    if (out/"JOINT_DISPATCH.json").exists():raise FileExistsError("No repeated joint trainer")
    cap=contained_path(R,cfg["joint_capacity_eval_output"],exists=True)
    eval_spec=_v2_checked({"path":cfg["joint_capacity_eval_run_spec"],"sha256":cfg["joint_capacity_eval_run_spec_sha256"]})
    eval_cfg=json.loads(contained_path(R,eval_spec["config"],exists=True).read_text())
    if eval_cfg["output"]!=cfg["joint_capacity_eval_output"] or eval_cfg["capacity_workers"]!=32:
        raise ValueError("Bound capacity run differs")
    expected_launcher=str(Path(cfg["joint_capacity_eval_run_spec"]).with_name("launch.py"))
    workers=[];pids=[]
    def refresh_workers():
        nonlocal workers,pids
        if not workers:
            dispatch=json.loads((cap/"DISPATCH.json").read_text());workers=dispatch["workers"]
            if len(workers)!=32:raise ValueError("Full32-worker service must be dispatched")
            pids=[w["pid"] for w in workers]
    def active():
        refresh_workers()
        process=subprocess.run(["ps","-p",",".join(map(str,pids)),"-o","pid=,args="],text=True,capture_output=True)
        if process.returncode not in (0,1):raise RuntimeError("Cannot inspect capacity PIDs")
        raw=process.stdout
        result=set()
        for line in raw.splitlines():
            pid,args=line.strip().split(None,1)
            if expected_launcher not in args or "eval.action4_corner_capacity" not in args:
                raise ValueError("Capacity PID identity changed")
            result.add(int(pid))
        return result
    if cfg.get("joint_warmup_before_eval") is True:
        if cfg.get("capacity_barrier") is None:raise ValueError("Prewarm requires coordinated readiness barrier")
    elif len(active())!=32:raise RuntimeError("Capacity workers already ended before joint launch")
    from model.training.action4_resource_policy import validate_device_policy,process_cpu_affinity
    validate_device_policy(cfg,R)
    cpus=process_cpu_affinity(cfg,"train",R)
    if cpus!=list(range(64,96)):raise ValueError("Training CPU partition differs")
    raw=subprocess.check_output(["nvidia-smi","-i","4,5,6,7","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    g={int(v[0]):(v[1].strip(),int(v[2])) for v in (line.split(",") for line in raw.strip().splitlines())}
    if any(free<20480 for uuid,free in g.values()):raise RuntimeError("Insufficient joint training headroom")
    spec=_v2_checked({"path":cfg["run_spec"],"sha256":_v2_sha(cfg["run_spec"])})
    code=contained_path(R,spec["code_root"],exists=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(g[i][0] for i in [4,5,6,7]),
        PYTHONPATH=str(code/"runtime/surface_deps")+":"+str(code/"src"),
        ACTION4_PROJECT_ROOT=str(R),CUBLAS_WORKSPACE_CONFIG=":4096:8",NCCL_P2P_DISABLE="1",
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",
        ACTION4_ATTEMPT_ID=cfg["joint_probe_attempt"])
    for key in ("HOME","TMPDIR","CUDA_CACHE_PATH","TORCH_HOME","HF_HOME","XDG_CACHE_HOME","XDG_CONFIG_HOME","MPLCONFIGDIR"):
        cache=process_tmp_path(cfg["run_spec"]) if key=="TMPDIR" else out/"runtime_cache"/key
        cache.mkdir(parents=True,exist_ok=True);env[key]=str(cache)
    cmd=["taskset","-c",",".join(map(str,cpus)),str(R/".venv/bin/python"),"-m","torch.distributed.run",
        "--standalone","--nnodes=1","--nproc_per_node=4","--max_restarts=0","--module",
        "model.training.action4_stage2_train","--config",str(contained_path(R,spec["config"],exists=True)),"--stop-after","20"]
    start=time.monotonic();samples=[];seen=set();failure=None
    with (out/"joint_train.log").open("x") as log:
        proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(out/"JOINT_DISPATCH.json",{"pid":proc.pid,"command":cmd,"time":time.time(),"stop_after":20,
        "expected_full_batch":48,"eval_worker_pids_at_launch":pids,"capacity_run_spec":cfg["joint_capacity_eval_run_spec"],
        "warmup_before_eval":cfg.get("joint_warmup_before_eval",False),"selection_committed":False,"formal_training":False})
    try:
        while True:
            path=out/"STATUS.json"
            if path.exists():
                with path.open() as stream:
                    row=json.load(stream);stamp=os.fstat(stream.fileno()).st_mtime_ns
                if row.get("did_update") and row["step"] not in seen:
                    living=active();predicting=0
                    for w in workers:
                        progress=cap/f"worker_{w['id']:02d}/PROGRESS.json"
                        if w["pid"] in living and progress.exists():
                            try:stage=json.loads(progress.read_text())["stage"]
                            except json.JSONDecodeError:stage="unavailable"
                            predicting+=stage=="prediction"
                    record={k:row[k] for k in ("step","epoch","global_actual_units","did_update","seconds")}
                    record.update(status_mtime_ns=stamp,live_eval_workers=len(living),prediction_eval_workers=predicting)
                    samples.append(record);seen.add(row["step"])
                    atomic_json(out/"JOINT_TIMING_PROGRESS.json",{"samples":samples})
            if proc.poll() is not None:break
            if time.monotonic()-start>3600:raise TimeoutError("Bounded joint probe deadline")
            time.sleep(.05 if seen else .5)
        if proc.returncode:raise RuntimeError("Actual training probe failed")
        stop=json.loads((out/"CHECKPOINTED_STOP.json").read_text())
        if stop["step"]!=20 or stop.get("attempt_id")!=cfg["joint_probe_attempt"]:raise ValueError("Unexpected controlled stop identity")
        result=summarize(samples)
        result.update(status="BOUNDED_JOINT_PROBE_COMPLETE",training_exit_code=proc.returncode,
            stop_after=20,wall_seconds=time.monotonic()-start,samples=samples,
            checkpoint_not_selected_best=True,unused_initial_eval_request_not_formal_evaluation=True)
    except BaseException:
        failure=traceback.format_exc()
        if cfg.get("capacity_barrier") is not None:
            gate=contained_path(R,cfg["capacity_barrier"]["directory"],exists=True)
            atomic_json(gate/"ABORT.json",{"experiment_id":cfg["capacity_barrier"]["experiment_id"],"reason":failure})
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        result={"status":"JOINT_PROBE_FAILED","failure":failure,"samples":samples,"formal_admission":False}
    atomic_json(out/"JOINT_CAPACITY_RESULT.json",result)
    if failure:raise RuntimeError("Joint probe failed; separate capacity jobs untouched")

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    cfg=json.loads(contained_path(R,a.config,exists=True).read_text())
    from model.training.action4_run_snapshot import verify_running_snapshot
    try:
        verify_running_snapshot(cfg);run(cfg)
    except BaseException:
        if cfg.get("capacity_barrier") is not None:
            gate=contained_path(R,cfg["capacity_barrier"]["directory"],exists=True)
            atomic_json(gate/"ABORT.json",{"experiment_id":cfg["capacity_barrier"]["experiment_id"],"reason":traceback.format_exc()})
        raise
if __name__=="__main__":main()
