"""Best-only retention after evaluated snapshots have no remaining consumers."""
import json,os
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

def prune_old_selected(output):
    out=contained_path(R,output,exists=True);best=json.loads((out/"BEST.json").read_text())
    current=contained_path(R,best["checkpoint"],exists=True)
    if current.parent!=out or not os.path.samefile(current,out/"best.pt"):
        raise ValueError("Best pointer/alias identity mismatch")
    removed=[]
    for path in out.glob("selected_step_*.pt"):
        if path!=current:
            removed.append(path.name);path.unlink()
    return removed

def finalize_best_only(output,remove_recovery=False):
    out=contained_path(R,output,exists=True)
    train=json.loads((out/"TRAIN_COMPLETE.json").read_text());evaluation=json.loads((out/"EVAL_COMPLETE.json").read_text())
    if train["attempt_id"]!=evaluation["attempt_id"]:raise ValueError("Training/eval attempt mismatch")
    requests=[p for p in (out/"eval").glob("step_*.json") if p.name.count(".")==1]
    if any(not p.with_suffix(".done.json").exists() for p in requests) or list((out/"eval").glob("*.failed.json")):
        raise ValueError("Evaluation consumers not closed")
    removed=prune_old_selected(str(out));best=json.loads((out/"BEST.json").read_text())
    old=contained_path(R,best["checkpoint"],exists=True)
    if not os.path.samefile(old,out/"best.pt"):raise ValueError("Selected best changed")
    best["original_selected_checkpoint"]=best.get("original_selected_checkpoint",str(old))
    best["checkpoint"]=str(out/"best.pt");atomic_json(out/"BEST.json",best)
    if old!=out/"best.pt":old.unlink();removed.append(old.name)
    if remove_recovery and (out/"last.pt").exists():(out/"last.pt").unlink();removed.append("last.pt")
    for claim in (out/"eval").glob("*.claim"):claim.unlink();removed.append(str(claim.relative_to(out)))
    atomic_json(out/"BEST_ONLY_RETENTION.json",{"kept":"best.pt","removed":removed,"recovery_removed":remove_recovery,"best_step":best["step"],"metrics_preserved":True})
    return removed
