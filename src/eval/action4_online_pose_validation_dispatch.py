"""Finish the authorized online-pose workflow after its train/eval writers exit."""
import json,os,time,subprocess,fcntl,traceback
from data.action4_paths import PROJECT_ROOT as R
from model.action4_trial_export import export
from model.training.action4_runtime import atomic_json

def alive_owned(pid):
    p=subprocess.run(["ps","-p",str(pid),"-o","stat=,args="],capture_output=True,text=True)
    if p.returncode or not p.stdout.strip() or p.stdout.strip().startswith("Z"):return False
    if str(R) not in p.stdout:raise RuntimeError("Recorded PID identity changed")
    return True

def run():
    out=R/"outputs/action4_a0_online_rgb_validation_v001"
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("Validation already dispatched")
    training=R/"outputs/action4_a0_online_rgb_v001"
    while True:
        if (training/"CONTROLLER_FAILURE.json").exists() or list((training/"eval").glob("*.failed.json")):
            raise RuntimeError("Training/evaluation failed; no automatic retry")
        if (training/"CONTROLLER_COMPLETE.json").exists() and (training/"EVAL_COMPLETE.json").exists():
            dispatch=json.loads((training/"DISPATCH.json").read_text())
            pids=[w["pid"] for w in dispatch["workers"]]+json.loads((training/"TRAIN_PROCESS.json").read_text())["pids"]
            if not any(alive_owned(pid) for pid in pids):break
        atomic_json(out/"STATUS.json",{"state":"WAIT_COMPLETE_TRAIN_AND_ALL_EVAL","training":str(training.relative_to(R))})
        time.sleep(30)
    result=export("configs/model/action4_a0_online_rgb_v001.json","A0","models/candidates/a0_online_rgb_v001/best.pt")
    cfg={"a0_bundle":result["path"],"b0_bundle":"models/candidates/b0_photometric_v001/best.pt","b0_scoring_policy":"center",
        "intrinsics_context":"outputs/action4_execution_source_release_v006/source_context.json",
        "demo_reference":"data/action4_reference/demo_standard.mp4",
        "training_completion_receipts":["outputs/action4_a0_online_rgb_v001/TRAIN_COMPLETE.json","outputs/action4_b0_photometric_v001/TRAIN_COMPLETE.json"],
        "scope":"complete online-crop candidate; frozen best, no automatic promotion"}
    path=R/"configs/eval/action4_a0_online_rgb_best_v001.json"
    if path.exists():raise FileExistsError("Profile already exists")
    atomic_json(path,cfg)
    env=dict(os.environ,PYTHONPATH=str(R/"runtime/surface_deps")+":"+str(R/"src"),TMPDIR=str(R/"runtime"),
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    specs=[
        ("video",[str(R/".venv/bin/python"),"-B","-m","eval.action4_staged_video_parallel","--selection",str(path.relative_to(R)),
            "--output",str((out/"video").relative_to(R)),"--devices","4,4,5,5,6,7","--reserved-cpu","8"],env),
        ("development",[str(R/".venv/bin/python"),"-B","-m","eval.action4_staged_eval","--selection",str(path.relative_to(R)),
            "--output",str((out/"development").relative_to(R))],dict(env,CUDA_VISIBLE_DEVICES="GPU-b1ae5be6-1dba-7343-e233-59df41402f33")),
        ("historical_test",[str(R/".venv/bin/python"),"-B","-m","eval.action4_staged_eval","--selection",str(path.relative_to(R)),
            "--output",str((out/"historical_test").relative_to(R)),"--split","test"],dict(env,CUDA_VISIBLE_DEVICES="GPU-a68477c3-1c2a-d557-58f5-405d80e5383e"))]
    workers=[];processes=[]
    for name,cmd,e in specs:
        with (out/(name+".log")).open("x") as log:p=subprocess.Popen(cmd,env=e,stdout=log,stderr=log,start_new_session=True)
        workers.append({"role":name,"pid":p.pid,"command":cmd});processes.append(p)
    atomic_json(out/"DISPATCH.json",{"workers":workers,"selected_A0_step":result["step"],"CPU_budget":44,"selection_frozen_before_test":True})
    while any(p.poll() is None for p in processes):time.sleep(10)
    codes=[p.returncode for p in processes]
    if any(codes):raise RuntimeError("Validation failure "+str(codes))
    for name in ["development/REPORT.json","historical_test/REPORT.json","video/COMPLETE.json"]:
        if not (out/name).exists():raise RuntimeError("Missing validation artifact")
    if json.loads((out/"video/COMPLETE.json").read_text())["frames"]!=2850:raise ValueError("Incomplete video")
    atomic_json(out/"COMPLETE.json",{"status":"ARTIFACTS_READY_ROOT_VISUAL_REVIEW_REQUIRED","selected_A0_step":result["step"],"real_acceptance":False})

if __name__=="__main__":
    out=R/"outputs/action4_a0_online_rgb_validation_v001"
    try:run()
    except BaseException:
        atomic_json(out/"CONTROLLER_FAILURE.json",{"traceback":traceback.format_exc()});raise
