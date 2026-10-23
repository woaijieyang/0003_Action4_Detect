"""One bounded online RGB pose trial; no automatic restart or other-job control."""
import json,os,time,subprocess,fcntl,uuid,traceback,signal
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_runtime import atomic_json

def run(config):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    guard=(out/"controller.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("Trial already dispatched")
    assert json.loads((out/"PREFLIGHT.json").read_text())["passed"]
    assert json.loads((out/"reduction_test_strong_fixture.log").read_text().splitlines()[-1])["passed"]
    devices={}
    for gpu,minimum in [(1,3000),(4,10000),(5,10000),(6,10000),(7,10000)]:
        fields=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if int(fields[1])<minimum:raise RuntimeError("Insufficient GPU headroom")
        devices[gpu]=fields[0].strip()
    attempt=uuid.uuid4().hex
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),PYTHONDONTWRITEBYTECODE="1",
        ACTION4_ATTEMPT_ID=attempt,OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",NCCL_P2P_DISABLE="1")
    specs=[
        ("eval",[str(R/".venv/bin/python"),"-B","-m","eval.action4_online_pose_eval","--config",config],[1],1,1),
        ("train",[str(R/".venv/bin/python"),"-B","-m","torch.distributed.run","--standalone","--nnodes=1","--nproc_per_node=4","--max_restarts=0",
            "--module","model.training.action4_online_pose_train","--config",config],[4,5,6,7],2,64)]
    processes=[];workers=[]
    try:
        for role,command,gpus,threads,cpu in specs:
            with (out/(role+".log")).open("x") as log:
                p=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(devices[g] for g in gpus),
                    OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads)),stdout=log,stderr=log,start_new_session=True)
            processes.append(p);workers.append({"role":role,"pid":p.pid,"physical_gpus":gpus,"cpu_budget":cpu,"command":command})
        atomic_json(out/"DISPATCH.json",{"attempt_id":attempt,"workers":workers,"compute_cpu":65,"control_budget":1,
            "epochs":80,"steps":15120,"eval_every":25,"global_batch":16,"local_batch":4,"no_padding":True})
        while any(p.poll() is None for p in processes):
            if any(p.poll() is not None and p.returncode!=0 for p in processes):raise RuntimeError("Owned train/eval failed")
            time.sleep(10)
        atomic_json(out/"CONTROLLER_COMPLETE.json",{"attempt_id":attempt,"codes":[p.returncode for p in processes],"accuracy_accepted":False})
    except BaseException:
        atomic_json(out/"CONTROLLER_FAILURE.json",{"attempt_id":attempt,"traceback":traceback.format_exc()})
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);run(p.parse_args().config)
