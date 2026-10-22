"""Single-evaluator recovery and atomic immutable checkpoint selection."""
import os,json
from pathlib import Path
from model.training.action4_runtime import atomic_json

def current_attempt():
    value=os.environ.get("ACTION4_ATTEMPT_ID")
    if not value:raise RuntimeError("Launcher must provide ACTION4_ATTEMPT_ID")
    return value

def is_current_terminal(out,attempt):
    for name in ("TRAIN_COMPLETE.json","TRAIN_FAILURE.json"):
        p=out/name
        if p.exists() and json.loads(p.read_text()).get("attempt_id")==attempt:return True
    return False

def recover_claims(queue,attempt):
    # Caller MUST hold the one-worker flock for this output directory.
    reclaimed=[]
    for p in queue.glob("step_*.claim"):
        if p.with_suffix(".done.json").exists() or p.with_suffix(".failed.json").exists():continue
        reclaimed.append({"path":p.name,"old":p.read_text()});p.unlink()
    if reclaimed:atomic_json(queue.parent/f"RECLAIMS_{attempt}.json",reclaimed)

def commit_result(out,request,snapshot,result,rank):
    best_path=out/"BEST.json"
    best=json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or rank(result)>rank(best):
        selected=out/f"selected_step_{result['step']:08d}.pt"
        if not selected.exists():os.link(snapshot,selected)
        elif selected.stat().st_ino!=snapshot.stat().st_ino:
            raise ValueError("Selected checkpoint name collision")
        result=dict(result,checkpoint=str(selected))
        atomic_json(best_path,result)  # authoritative pointer to immutable weight
        best=result
    if best is not None:
        selected=Path(best["checkpoint"])
        temp=out/f"best_alias_{os.getpid()}.tmp"
        if temp.exists():temp.unlink()
        os.link(selected,temp);os.replace(temp,out/"best.pt")
    # Done is LAST. A crash before it replays this original snapshot safely.
    atomic_json(request.with_suffix(".done.json"),result)
    snapshot.unlink()
