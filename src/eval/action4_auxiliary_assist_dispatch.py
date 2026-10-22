"""Assist only eval; reconcile once all original writers and requests finish."""
import json,os,time,subprocess,fcntl,traceback,torch,math
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_runtime import atomic_json
from model.training.action4_component_train import config_id

CONFIGS=[("configs/model/action4_a0_additive_v002.json",1),
         ("configs/model/action4_direction_pooled_control_v001.json",2),
         ("configs/model/action4_direction_per_roi_v001.json",1)]

def alive(pid,config):
    p=subprocess.run(["ps","-p",str(pid),"-o","stat=,args="],capture_output=True,text=True)
    if p.returncode or not p.stdout.strip() or p.stdout.strip().startswith("Z"):return False
    if "eval.action4_auxiliary_eval" not in p.stdout or config not in p.stdout:raise RuntimeError("Primary PID identity changed")
    return True

def run():
    root=R/"outputs/action4_aux_eval_acceleration_v001"
    guard=(root/"controller.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/"DISPATCH.json").exists():raise FileExistsError("Assist already launched")
    assert json.loads((root/"PARITY_PASS.json").read_text())["exact_full_report_equal"]
    free=int(subprocess.check_output(["nvidia-smi","-i","4","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True))
    if free<20000:raise RuntimeError("GPU4 headroom insufficient")
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),PYTHONDONTWRITEBYTECODE="1",
             CUDA_VISIBLE_DEVICES="GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f",OMP_NUM_THREADS="1",
             MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    processes=[];workers=[];primaries=[]
    for config,count in CONFIGS:
        cfg=json.loads((R/config).read_text());out=R/cfg["output"];pid=json.loads((out/"EVAL_PROCESS.json").read_text())["pid"]
        if not alive(pid,config):raise RuntimeError("Expected healthy primary; inspect before assistance")
        primaries.append({"config":config,"output":cfg["output"],"pid":pid})
    for config,count in CONFIGS:
        for slot in range(count):
            command=[str(R/".venv/bin/python"),"-B","-m","eval.action4_auxiliary_eval_assist","--config",config,
                     "--cache-root",str(root.relative_to(R))]
            logpath=root/f"assist_{len(workers):02d}.log"
            with logpath.open("x") as log:p=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
            processes.append(p);workers.append({"pid":p.pid,"config":config,"gpu":4,"cpu":1,"log":str(logpath.relative_to(R))})
    atomic_json(root/"DISPATCH.json",{"workers":workers,"original_evaluators":primaries,"added_cpu":4,"training_changed":False})
    while any(p.poll() is None for p in processes):time.sleep(15)
    if any(p.returncode for p in processes):
        atomic_json(root/"ASSIST_FAILURE.json",{"codes":[p.returncode for p in processes],"original_train_eval_untouched":True});return
    for primary in primaries:
        out=R/primary["output"];cfg=json.loads((R/primary["config"]).read_text())
        while True:
            if (out/"FAILURE.json").exists() or list((out/"eval").glob("*.failed.json")):raise RuntimeError("Original run failed; no promotion")
            requests=[p for p in (out/"eval").glob("step_*.json") if p.name.count(".")==1]
            pending=[p for p in requests if not p.with_suffix(".done.json").exists()]
            running=alive(primary["pid"],primary["config"])
            if (out/"TRAIN_COMPLETE.json").exists() and not pending and not running:break
            time.sleep(15)
        completion=json.loads((out/"TRAIN_COMPLETE.json").read_text())
        expected=cfg["epochs"]*math.ceil(3024/cfg["batch"])
        if completion.get("epochs")!=cfg["epochs"] or completion.get("step",completion.get("steps"))!=expected:
            raise ValueError("Training completion count mismatch")
        done=[json.loads(p.with_suffix(".done.json").read_text()) for p in requests]
        if {d["step"] for d in done}!=set(range(0,expected+1,25))|{expected}:
            raise ValueError("Incomplete evaluation step coverage")
        winner=max(done,key=lambda d:(d["report"]["auxiliary"]["all8_at05"],-d["report"]["auxiliary"]["hidden_mean"],-d["step"]))
        source=R/winner["snapshot"]
        if not source.exists():source=out/"best.pt"
        state=torch.load(source,map_location="cpu",weights_only=True)
        if state["step"]!=winner["step"] or state["config_id"]!=config_id(cfg):raise ValueError("Winning snapshot unavailable or mismatched")
        best=out/"best.pt"
        if best.stat().st_ino!=source.stat().st_ino:
            temp=out/"best_assist_final.tmp"
            if temp.exists():temp.unlink()
            os.link(source,temp);os.replace(temp,best)
        atomic_json(out/"BEST.json",winner)
        atomic_json(out/"ALL_EVAL_SELECTION.json",{"status":"ALL_REQUESTS_SCORED_AND_BEST_RECONCILED","eval_count":len(done),"best_step":winner["step"],"primary_writer_stopped":True,"all8":winner["report"]["auxiliary"]["all8_at05"],"snapshots_preserved_until_release_cleanup":True})
    atomic_json(root/"COMPLETE.json",{"status":"ALL_AUXILIARY_EVALS_RECONCILED","workers":len(workers)})
if __name__=="__main__":
    try:run()
    except BaseException:atomic_json(R/"outputs/action4_aux_eval_acceleration_v001/CONTROLLER_FAILURE.json",{"traceback":traceback.format_exc()});raise
