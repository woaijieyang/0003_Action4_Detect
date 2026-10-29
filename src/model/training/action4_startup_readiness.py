"""Host readiness before NCCL and evaluator startup; no training math changes."""
import json,os,re,time
from pathlib import Path
from data.action4_paths import PROJECT_ROOT,contained_path
from model.training.action4_runtime import atomic_json

def wait_data_ready(out,attempt,rank,world,units,timeout=1800.):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",attempt):raise ValueError("Explicit attempt identity required")
    if type(rank) is not int or type(world) is not int or not 0<=rank<world:
        raise ValueError("Invalid rank/world")
    if units<=0 or not 0<timeout<=1800:raise ValueError("Invalid readiness units/timeout")
    directory=contained_path(PROJECT_ROOT,Path(out)/"startup"/attempt,exists=False)
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/f"rank_{rank}_data_ready.json"
    if receipt.exists():raise FileExistsError("Duplicate readiness for same rank/attempt")
    atomic_json(receipt,dict(attempt_id=attempt,rank=rank,world=world,units=units,pid=os.getpid(),ready_unix=time.time()))
    start=time.monotonic()
    while True:
        rows=[]
        for i in range(world):
            path=directory/f"rank_{i}_data_ready.json"
            if path.exists():
                row=json.loads(path.read_text())
                if (row.get("attempt_id"),row.get("rank"),row.get("world"),row.get("units"))!=(attempt,i,world,units):
                    raise ValueError("Data readiness identity/domain mismatch")
                rows.append(row)
        if len(rows)==world:
            result=dict(attempt_id=attempt,rank=rank,world=world,wait_seconds=time.monotonic()-start,
                        all_data_ready_unix=time.time(),ready_pids=[r["pid"] for r in rows])
            atomic_json(directory/f"rank_{rank}_released.json",result)
            return result
        if time.monotonic()-start>=timeout:raise TimeoutError("Data preflight ranks not ready; NCCL not entered")
        time.sleep(.1)

def wait_training_ready(out,attempt,process,timeout=1800.):
    start=time.monotonic()
    while True:
        if process.poll() is not None:raise RuntimeError("Trainer exited before evaluator startup")
        path=Path(out)/"TRAIN_PROCESS.json"
        if path.exists():
            row=json.loads(path.read_text())
            if row.get("attempt_id")!=attempt:raise ValueError("Stale trainer readiness receipt")
            if not row.get("pids") or len(row["pids"])!=len(set(row["pids"])):
                raise ValueError("Invalid actual rank PID list")
            return dict(attempt_id=attempt,wait_seconds=time.monotonic()-start,pids=row["pids"])
        if time.monotonic()-start>=timeout:raise TimeoutError("Training initialization did not finish; evaluators not started")
        time.sleep(.25)
