"""Scoped independent-role controller; dynamic RGB eval workers and explicit terminal checks."""
import json,os,uuid,time,fcntl,subprocess,signal,traceback,math
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json,canonical_hash
from eval.action4_stage2_eval import rank_result
from model.training.action4_stage2_core import validate_run_gate
from model.training.action4_best_only_retention import finalize_best_only

def run(config):
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    out=contained_path(R,cfg["output"],exists=True)
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("No duplicate stage launch")
    validate_run_gate(cfg,json.loads(contained_path(R,cfg["integration_or_admission_receipt"],exists=True).read_text()))
    train_gpus=cfg["training_gpus"];eval_gpus=cfg["eval_gpus"]
    if len(train_gpus)!=cfg["world_size"] or len(set(train_gpus))!=len(train_gpus):raise ValueError("Train device topology mismatch")
    if any(g not in range(1,8) for g in train_gpus+eval_gpus) or 1 in train_gpus or eval_gpus.count(1)>8:
        raise ValueError("Authorized GPU scope violated")
    cpu=cfg["world_size"]*(cfg.get("threads",2)+cfg.get("loader_workers",0))+2*len(eval_gpus)+1
    if cpu>cfg["reserved_cpu_budget"] or cfg["project_total_reserved_cpu"]>72:raise ValueError("Project CPU budget exceeded")
    uuid_by_gpu={}
    for g in set(train_gpus+eval_gpus):
        fields=subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if int(fields[1])<cfg.get("min_free_mib",12000):raise RuntimeError("Insufficient available GPU memory")
        uuid_by_gpu[g]=fields[0].strip()
    attempt=uuid.uuid4().hex
    env=dict(os.environ,PYTHONPATH=str(R/"runtime/surface_deps")+":"+str(R/"src"),TMPDIR=str(R/"runtime"),
        PYTHONDONTWRITEBYTECODE="1",ACTION4_ATTEMPT_ID=attempt,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",NCCL_P2P_DISABLE="1")
    specs=[("train",[str(R/".venv/bin/python"),"-m","torch.distributed.run","--standalone","--nnodes=1",
        "--nproc_per_node="+str(cfg["world_size"]),"--max_restarts=0","--module","model.training.action4_stage2_train","--config",config],train_gpus)]
    if cfg.get("integration_resume_step") is not None:
        if cfg["mode"]!="integration":raise ValueError("Automatic resume probe is engineering-only")
        specs[0][1].extend(["--stop-after",str(cfg["integration_resume_step"])])
    for i,g in enumerate(eval_gpus):
        specs.append((f"eval_{i}",[str(R/".venv/bin/python"),"-m","eval.action4_stage2_eval","--config",config,"--worker-id",str(i)],[g]))
    processes=[];workers=[]
    try:
        for name,cmd,gpus in specs:
            with (out/(name+".log")).open("x") as log:
                p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(uuid_by_gpu[g] for g in gpus)),stdout=log,stderr=log,start_new_session=True)
            processes.append(p);workers.append({"role":name,"pid":p.pid,"gpus":gpus,"command":cmd})
        atomic_json(out/"DISPATCH.json",{"attempt_id":attempt,"workers":workers,"cpu_budget":cpu,"mode":cfg["mode"]})
        resumed=False
        while any(p.poll() is None for p in processes):
            if any(p.poll() is not None and p.returncode!=0 for p in processes):raise RuntimeError("Owned stage worker failed")
            if cfg.get("integration_resume_step") is not None and not resumed and processes[0].poll()==0:
                stop=json.loads((out/"CHECKPOINTED_STOP.json").read_text())
                if stop["step"]!=cfg["integration_resume_step"] or stop["attempt_id"]!=attempt:raise ValueError("Controlled stop identity mismatch")
                cmd=specs[0][1][:-2]+["--resume",str((out/"last.pt").relative_to(R))]
                with (out/"train_resume.log").open("x") as log:
                    p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(uuid_by_gpu[g] for g in train_gpus)),stdout=log,stderr=log,start_new_session=True)
                processes.append(p);resumed=True
                atomic_json(out/"RESUME_DISPATCH.json",{"pid":p.pid,"command":cmd,"attempt_id":attempt,"from_step":stop["step"],"scope":"explicit controlled integration recovery"})
            time.sleep(5)
        terminal=json.loads((out/"TRAIN_COMPLETE.json").read_text())
        total=cfg["epochs"]*math.ceil(cfg["expected_units"]/cfg["batch"])
        if terminal["step"]!=total or terminal["attempt_id"]!=attempt:raise ValueError("Training completion mismatch")
        requests=[p for p in (out/"eval").glob("step_*.json") if p.name.count(".")==1]
        reports=[json.loads(p.with_suffix(".done.json").read_text()) for p in requests]
        expected={0,total}|set(range(cfg["eval_every"],total+1,cfg["eval_every"]))
        if {r["step"] for r in reports}!=expected or list((out/"eval").glob("*.failed.json")):raise ValueError("Incomplete optimizer-step evaluations")
        best=json.loads((out/"BEST.json").read_text());winner=max(reports,key=rank_result)
        if best["step"]!=winner["step"]:raise ValueError("Best checkpoint does not match all completed evaluations")
        atomic_json(out/"EVAL_COMPLETE.json",{"attempt_id":attempt,"evaluations":len(reports),"all_steps":sorted(expected),"best_step":best["step"],"config_id":canonical_hash(cfg)})
        # Integration recovery probes retain last only until their paired comparison has finished.
        if cfg["mode"]=="full_training":finalize_best_only(str(out),remove_recovery=True)
        atomic_json(out/"CONTROLLER_COMPLETE.json",{"attempt_id":attempt,"mode":cfg["mode"],"role":cfg["role"],"best_step":best["step"],"accuracy_accepted":False,"config_id":canonical_hash(cfg)})
    except BaseException:
        atomic_json(out/"CONTROLLER_FAILURE.json",{"attempt_id":attempt,"traceback":traceback.format_exc()})
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);run(p.parse_args().config)
