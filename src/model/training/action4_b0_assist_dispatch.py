"""Reconcile complete B0 eval coverage after all original/assist writers exit."""
import json,os,subprocess,time,torch,fcntl,traceback
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json,canonical_hash,link_checkpoint_atomic

def primary_alive(pid,config):
    p=subprocess.run(["ps","-p",str(pid),"-o","stat=,args="],capture_output=True,text=True)
    if p.returncode or not p.stdout.strip() or p.stdout.strip().startswith("Z"):return False
    if "eval.action4_online_b0_eval" not in p.stdout or config not in p.stdout:raise RuntimeError("Primary PID identity changed")
    return True

def run(config):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    lock=(out/"assist_controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"ASSIST_DISPATCH.json").exists():raise FileExistsError("Assist already dispatched")
    primary=json.loads((out/"EVAL_PROCESS.json").read_text())["pid"]
    if not primary_alive(primary,config):raise RuntimeError("Original evaluator not healthy")
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),PYTHONDONTWRITEBYTECODE="1",
             OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    workers=[];processes=[]
    for gpu in [6,7]:
        values=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if int(values[1])<5000:raise RuntimeError("No helper memory headroom")
        command=[str(R/".venv/bin/python"),"-B","-m","eval.action4_b0_eval_assist","--config",config]
        with (out/f"assist_gpu{gpu}.log").open("x") as log:
            p=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=values[0].strip()),stdout=log,stderr=log,start_new_session=True)
        workers.append({"pid":p.pid,"gpu":gpu,"cpu":1,"command":command});processes.append(p)
    atomic_json(out/"ASSIST_DISPATCH.json",{"workers":workers,"primary_pid":primary,"extra_compute_cpu":2,
        "combined_compute_cpu":69,"controllers_cpu_budget":2,"training_changed":False,"final_best_requires":"ALL_EVAL_SELECTION.json"})
    while any(p.poll() is None for p in processes):time.sleep(15)
    if any(p.returncode for p in processes):raise RuntimeError("Assist failed; do not reconcile partial evaluation")
    while primary_alive(primary,config):time.sleep(15)
    queue=out/"eval";requests=[p for p in queue.glob("step_*.json") if p.name.count(".")==1]
    if list(queue.glob("*.failed.json")) or any(not p.with_suffix(".done.json").exists() for p in requests):
        raise RuntimeError("Missing/failed eval after workers exited")
    complete=json.loads((out/"TRAIN_COMPLETE.json").read_text())
    expected=cfg["epochs"]*1600//cfg["batch"]
    if complete["epochs"]!=80 or complete["step"]!=expected:raise ValueError("Training not complete")
    done=[json.loads(p.with_suffix(".done.json").read_text()) for p in sorted(requests)]
    if {d["step"] for d in done}!=set(range(0,expected+1,cfg["eval_every"]))|{expected}:raise ValueError("Incomplete step coverage")
    best=max(done,key=lambda d:(d["report"]["mAP50_95"],d["report"]["overall"]["f1"],-d["step"]))
    source=contained_path(R,best.get("checkpoint",best.get("snapshot","")),exists=True)
    state=torch.load(source,map_location="cpu",weights_only=True)
    if state["step"]!=best["step"] or state["config_id"]!=canonical_hash(cfg):raise ValueError("Best identity mismatch")
    selected=out/f"selected_step_{state['step']:08d}.pt"
    if not selected.exists():os.link(source,selected)
    elif not os.path.samefile(source,selected):raise ValueError("Selected path collision")
    link_checkpoint_atomic(selected,out/"best.pt")
    atomic_json(out/"BEST.json",dict(best,checkpoint=str(selected)))
    atomic_json(out/"ALL_EVAL_SELECTION.json",{"status":"ALL_REQUESTS_SCORED_AND_BEST_RECONCILED","eval_count":len(done),
        "best_step":state["step"],"mAP50_95":best["report"]["mAP50_95"],"dual":best["report"]["overall"]["both_detected_images"],"workers_stopped":True})

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg["output"]/"ASSIST_CONTROLLER_FAILURE.json",{"traceback":traceback.format_exc()});raise
