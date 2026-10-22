"""Supervise only this candidate's train/eval children; no automatic retries."""
import json,os,time,fcntl,subprocess,traceback
from pathlib import Path
R=Path("/data/zy/vln/a-test")
def run(config):
    assert Path.cwd().resolve()==R
    cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("Already dispatched")
    proof=json.loads((out/"preflight.log").read_text().splitlines()[-1]);assert proof["passed"]
    for index in (1,4):
        free=int(subprocess.check_output(["nvidia-smi","-i",str(index),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True));assert free>3000
    base=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),
              PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    processes=[];workers=[]
    for role,module,uuid,cpu,gpu in [
        ("eval","eval.action4_box_refinement_eval","GPU-a77a9d4c-7642-7112-698c-b02673d254cb",1,1),
        ("train","model.training.action4_box_refinement_train","GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f",2,4)]:
        env=dict(base,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS=str(cpu),MKL_NUM_THREADS=str(cpu))
        command=[str(R/".venv/bin/python"),"-B","-m",module,"--config",config]
        with (out/(role+".log")).open("x") as log:
            p=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
        processes.append(p);workers.append({"role":role,"pid":p.pid,"cpu":cpu,"gpu":gpu,"command":command})
    (out/"DISPATCH.json").write_text(json.dumps({"workers":workers,"cpu_added":3,"accuracy_accepted":False},indent=2))
    while any(p.poll() is None for p in processes):
        if any(p.poll() is not None and p.returncode!=0 for p in processes):
            for p in processes:
                if p.poll() is None:p.terminate()
            (out/"CONTROLLER_FAILURE.json").write_text(json.dumps({"codes":[p.poll() for p in processes],"action":"stop own counterpart only"}));return
        time.sleep(15)
    (out/"CONTROLLER_COMPLETE.json").write_text(json.dumps({"codes":[p.returncode for p in processes],"accuracy_accepted":False}))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());(R/cfg["output"]/"CONTROLLER_FAILURE.json").write_text(json.dumps({"error":traceback.format_exc()}));raise
