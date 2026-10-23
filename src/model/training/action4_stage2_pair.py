"""Joint resource/admission preflight for independently owned A0/B0 controllers."""
import json,os,time,subprocess
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_stage2_core import validate_run_gate
from model.training.action4_runtime import atomic_json

def validate_pair(configs):
    if len(configs)!=2 or [c["role"] for c in configs]!=["A0","B0"]:raise ValueError("Exactly one A0 and one B0 required")
    if configs[0]["mode"]!=configs[1]["mode"] or configs[0]["validation_receipt"]!=configs[1]["validation_receipt"]:
        raise ValueError("Pair must share mode and admitted dataset")
    if sum(c["eval_gpus"].count(1) for c in configs)>8:raise ValueError("GPU1 total eval workers exceeds8")
    budgets=[]
    for c in configs:
        if any(g not in range(1,8) for g in c["training_gpus"]+c["eval_gpus"]) or 1 in c["training_gpus"]:
            raise ValueError("GPU authorization violated")
        cpu=c["world_size"]*(c.get("threads",2)+c.get("loader_workers",0))+2*len(c["eval_gpus"])+1
        budgets.append(cpu)
        if cpu>c["reserved_cpu_budget"]:raise ValueError("Per-role CPU reservation insufficient")
    if sum(budgets)>72 or any(c["project_total_reserved_cpu"]<sum(budgets) or c["project_total_reserved_cpu"]>72 for c in configs):
        raise ValueError("Combined project CPU budget invalid")
    outputs=[contained_path(R,c["output"],exists=False) for c in configs]
    if any(not p.is_relative_to(R/"outputs") or p==R/"outputs" for p in outputs):
        raise ValueError("Training artifacts must use dedicated outputs subdirectories")
    if outputs[0]==outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        raise ValueError("Stage output namespaces overlap")
    counterpart=contained_path(R,configs[0]["evaluation_counterpart_bundle"],exists=True)
    if any(counterpart.is_relative_to(p) for p in outputs):raise ValueError("A0 selection cannot follow a live-changing B0 best alias")
    return {"cpu_total":sum(budgets),"GPU1_eval_workers":sum(c["eval_gpus"].count(1) for c in configs),"outputs":[str(p.relative_to(R)) for p in outputs]}

def run(a0_config,b0_config,output):
    paths=[a0_config,b0_config];configs=[json.loads(contained_path(R,p,exists=True).read_text()) for p in paths]
    resource=validate_pair(configs)
    for cfg in configs:
        gate=json.loads(contained_path(R,cfg["integration_or_admission_receipt"],exists=True).read_text());validate_run_gate(cfg,gate)
        root=contained_path(R,cfg["output"],exists=False)
        if root.exists() and any(root.iterdir()):raise FileExistsError("No automatic reuse/restart of an existing stage output")
    out=contained_path(R,output,exists=False)
    if not out.is_relative_to(R/"outputs") or any(out==contained_path(R,c["output"],exists=False) for c in configs):
        raise ValueError("Pair metadata needs its own outputs namespace")
    out.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    processes=[];workers=[]
    for path,cfg in zip(paths,configs):
        contained_path(R,cfg["output"],exists=False).mkdir(parents=True,exist_ok=True)
        command=[str(R/".venv/bin/python"),"-m","model.training.action4_stage2_dispatch","--config",path]
        with (out/(cfg["role"]+".log")).open("x") as log:
            process=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=log,start_new_session=True)
        processes.append(process);workers.append({"role":cfg["role"],"pid":process.pid,"command":command})
    atomic_json(out/"DISPATCH.json",{"workers":workers,"resources":resource,"mode":configs[0]["mode"]})
    # One role failing does not stop the other healthy independent role.
    while any(p.poll() is None for p in processes):
        atomic_json(out/"STATUS.json",{w["role"]:p.poll() for w,p in zip(workers,processes)});time.sleep(10)
    codes={w["role"]:p.returncode for w,p in zip(workers,processes)}
    atomic_json(out/("COMPLETE.json" if all(v==0 for v in codes.values()) else "FAILURE.json"),{"codes":codes,"resources":resource,"mode":configs[0]["mode"],"accuracy_accepted":False})
    if any(codes.values()):raise RuntimeError("An independent stage failed; inspect its preserved failure")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--a0-config",required=True);p.add_argument("--b0-config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.a0_config,a.b0_config,a.output)
