"""Bounded four-GPU photometric continuation and dedicated asynchronous eval."""
import os,json,subprocess,time,fcntl,uuid,traceback,signal
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

def stop_owned(processes):
    for p in processes:
        if p.poll() is None:p.terminate()
    for p in processes:
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait()

def run(config):
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    out=contained_path(R,cfg["output"],exists=True)
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("Existing run is never automatically restarted")
    if not json.loads((out/"PREFLIGHT.json").read_text())["passed"]:raise ValueError("Actual RGB preflight missing")
    if not json.loads((out/"dataflow_test.log").read_text().splitlines()[-1])["passed"]:raise ValueError("Dataflow test missing")
    devices={}
    for gpu,minimum in [(1,4000),(4,22000),(5,22000),(6,22000),(7,22000)]:
        values=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if int(values[1])<minimum:raise RuntimeError("Insufficient headroom for assigned GPU"+str(gpu))
        devices[gpu]=values[0].strip()
    attempt=uuid.uuid4().hex
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),ACTION4_ATTEMPT_ID=attempt,
        PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",NCCL_P2P_DISABLE="1")
    specs=[
        ("eval",[str(R/".venv/bin/python"),"-B","-m","eval.action4_online_b0_eval","--config",config],[1],1,3),
        ("train",[str(R/".venv/bin/python"),"-B","-m","torch.distributed.run","--standalone","--nnodes=1","--nproc_per_node=4","--max_restarts=0","--module","model.training.action4_online_b0_train","--config",config],[4,5,6,7],2,64)]
    processes=[];workers=[]
    try:
        for role,command,gpus,threads,budget in specs:
            with (out/(role+".log")).open("x") as log:
                p=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(devices[g] for g in gpus),
                    OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads)),stdout=log,stderr=log,start_new_session=True)
            processes.append(p);workers.append({"role":role,"pid":p.pid,"physical_gpus":gpus,"cpu_budget":budget,"command":command})
        atomic_json(out/"DISPATCH.json",{"attempt_id":attempt,"workers":workers,"compute_cpu_budget":67,"controller_cpu_budget":1,
            "global_batch":32,"local_batch":8,"micro":8,"epochs":80,"total_optimizer_steps":4000,"eval_every":20,"restart_policy":"none"})
        while any(p.poll() is None for p in processes):
            if any(p.poll() is not None and p.returncode!=0 for p in processes):
                raise RuntimeError("Owned train/eval failed; stopping owned counterpart, no retry")
            time.sleep(10)
        atomic_json(out/"CONTROLLER_COMPLETE.json",{"attempt_id":attempt,"codes":[p.returncode for p in processes],"accuracy_accepted":False})
    except BaseException:
        atomic_json(out/"CONTROLLER_FAILURE.json",{"attempt_id":attempt,"traceback":traceback.format_exc(),"codes":[p.poll() for p in processes]})
        stop_owned(processes)
        raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);run(p.parse_args().config)
