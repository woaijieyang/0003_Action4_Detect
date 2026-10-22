"""One two-rank B0 run and one evaluator; no automatic job retry."""
import os,json,subprocess,time,fcntl,uuid,traceback
from pathlib import Path
R=Path("/data/zy/vln/a-test")
def run(config):
    assert Path.cwd().resolve()==R
    cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("Run already dispatched")
    for name in ("preflight.log","batch16_preflight.log"):
        assert json.loads((out/name).read_text().splitlines()[-1])["passed"]
    for gpu,minimum in ((1,8000),(5,33000),(6,33000)):
        assert int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True))>minimum
    attempt=uuid.uuid4().hex
    base=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),ACTION4_ATTEMPT_ID=attempt,
              PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",NCCL_P2P_DISABLE="1")
    processes=[];workers=[]
    specs=[
        ("eval",[str(R/".venv/bin/python"),"-B","-m","eval.action4_online_b0_eval","--config",config],
         "GPU-a77a9d4c-7642-7112-698c-b02673d254cb","1",[1],3),
        ("train",[str(R/".venv/bin/python"),"-B","-m","torch.distributed.run","--standalone","--nnodes=1","--nproc_per_node=2","--max_restarts=0","--module","model.training.action4_online_b0_train","--config",config],
         "GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8,GPU-b1ae5be6-1dba-7343-e233-59df41402f33","2",[5,6],16)]
    for role,command,devices,threads,gpus,cpu in specs:
        with (out/(role+".log")).open("x") as log:
            p=subprocess.Popen(command,env=dict(base,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads),
                               stdout=log,stderr=log,start_new_session=True)
        processes.append(p);workers.append({"role":role,"pid":p.pid,"gpus":gpus,"cpu_budget":cpu,"command":command})
    (out/"DISPATCH.json").write_text(json.dumps({"attempt_id":attempt,"workers":workers,"added_cpu_budget":19,"global_batch":32,"local_batch":16,"max_restarts":0},indent=2))
    while any(p.poll() is None for p in processes):
        if any(p.poll() is not None and p.returncode!=0 for p in processes):
            for p in processes:
                if p.poll() is None:p.terminate()
            (out/"CONTROLLER_FAILURE.json").write_text(json.dumps({"attempt_id":attempt,"codes":[p.poll() for p in processes],"action":"only own counterpart stopped; no retry"}));return
        time.sleep(15)
    (out/"CONTROLLER_COMPLETE.json").write_text(json.dumps({"attempt_id":attempt,"codes":[p.returncode for p in processes],"accuracy_accepted":False}))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());(R/cfg["output"]/"CONTROLLER_FAILURE.json").write_text(json.dumps({"traceback":traceback.format_exc()}));raise
