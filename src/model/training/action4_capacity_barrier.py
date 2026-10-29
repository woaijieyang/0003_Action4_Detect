"""Explicit performance-only rendezvous; never enabled for formal training."""
import json,os,time
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

def settings(cfg,role):
    item=cfg.get("capacity_barrier")
    if item is None:return None
    if (cfg.get("mode")!="integration" or role not in ("train","eval")
        or (role=="train" and cfg.get("joint_capacity_probe") is not True)
        or (role=="eval" and cfg.get("evaluation_only_engineering") is not True)
        or not isinstance(item,dict) or not isinstance(item.get("experiment_id"),str)
        or len(item["experiment_id"])!=32 or item.get("timeout_seconds")!=1800):
        raise ValueError("Bounded performance-only barrier contract required")
    root=contained_path(R,item["directory"],exists=True)
    return item,root

def wait_ready(cfg,role,index,first_batch_units=None):
    selected=settings(cfg,role)
    if selected is None:return
    item,root=selected
    limit=cfg["world_size"] if role=="train" else cfg["capacity_workers"]
    if type(index) is not int or not 0<=index<limit:raise ValueError("Barrier role index invalid")
    if role=="train" and first_batch_units!=cfg["batch"]//cfg["world_size"]:
        raise ValueError("Real full first batch must arrive before training readiness")
    ready=root/(role+"_"+str(index)+"_READY.json")
    if ready.exists():raise FileExistsError("Preserve prior readiness")
    atomic_json(ready,{"experiment_id":item["experiment_id"],"role":role,"index":index,
        "pid":os.getpid(),"run_spec":cfg["run_spec"],"time":time.time(),"first_batch_units":first_batch_units})
    deadline=time.monotonic()+item["timeout_seconds"]
    while True:
        if (root/"ABORT.json").exists():raise RuntimeError("Joint barrier aborted")
        if (root/"GO.json").exists():
            go=json.loads((root/"GO.json").read_text())
            if go.get("experiment_id")!=item["experiment_id"]:raise ValueError("Cross-experiment GO")
            return
        if time.monotonic()>deadline:raise TimeoutError("Joint readiness timeout")
        time.sleep(.1)
