"""Best-only retention after evaluated snapshots have no remaining consumers."""
import json,os
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

def _corner_inventory(out):
    """Point-in-time owned-file accounting; no content reads or symlink traversal."""
    import stat,time
    files=[];seen=set();logical=unique=allocated=0;skipped=[]
    for directory,dirs,names in os.walk(out,followlinks=False):
        kept=[]
        for name in dirs:
            path=Path(directory)/name
            if path.is_symlink():skipped.append(str(path.relative_to(out)))
            else:kept.append(name)
        dirs[:]=kept
        for name in names:
            path=Path(directory)/name
            if path==out/"CORNER_RETENTION.json":continue
            info=path.lstat()
            if not stat.S_ISREG(info.st_mode):
                skipped.append(str(path.relative_to(out)));continue
            logical+=info.st_size;key=(info.st_dev,info.st_ino)
            if key not in seen:
                unique+=info.st_size;allocated+=getattr(info,"st_blocks",0)*512;seen.add(key)
            files.append({"path":str(path.relative_to(out)),"bytes":info.st_size})
    return {"at_unix":time.time(),"scope":"owned output tree; excludes CORNER_RETENTION.json and later controller terminal write",
        "file_paths":len(files),"unique_inodes":len(seen),"logical_bytes_including_hardlink_aliases":logical,
        "unique_inode_logical_bytes":unique,"unique_inode_allocated_bytes":allocated,
        "skipped_symlinks_or_nonregular":skipped}


def prune_old_selected(output):
    out=contained_path(R,output,exists=True)
    if (out/"CORNER_EVAL_IDENTITY.json").exists():
        # Corner prediction/calibration artifacts bind original snapshot paths.
        # No pruning is safe until a separate dependency/archive audit.
        if (out/"BEST.json").exists():
            from model.training.action4_async_recovery import corner_result_signature
            best=json.loads((out/"BEST.json").read_text())
            if corner_result_signature(best)!=json.loads((out/"CORNER_EVAL_IDENTITY.json").read_text()) or best["deployment_eligible"] is not True:
                raise ValueError("Invalid corner retained BEST")
            current=contained_path(R,best["checkpoint"],exists=True)
            if current.parent!=out or not os.path.samefile(current,out/"best.pt"):raise ValueError("Corner best pointer/alias mismatch")
        elif (out/"best.pt").exists():raise ValueError("Corner alias without BEST")
        if (out/"PNP_RESEARCH_BEST.json").exists():
            from .action4_async_recovery import checked_binding
            research=json.loads((out/"PNP_RESEARCH_BEST.json").read_text())
            source=checked_binding(research["evaluated_snapshot"])
            if not os.path.samefile(source,out/"pnp_research_best.pt"):raise ValueError("Research best alias mismatch")
        elif (out/"pnp_research_best.pt").exists():raise ValueError("Research alias without manifest")
        return []
    best=json.loads((out/"BEST.json").read_text())
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
    if (out/"CORNER_EVAL_IDENTITY.json").exists():
        prune_old_selected(str(out))
        reports=[json.loads(p.with_suffix(".done.json").read_text()) for p in requests]
        refs=[r["evaluated_snapshot"] for r in reports]
        for ref in refs:
            if not contained_path(R,ref["path"],exists=True).is_file():raise ValueError("Bound corner snapshot missing")
        research=json.loads((out/"PNP_RESEARCH_BEST.json").read_text()) if (out/"PNP_RESEARCH_BEST.json").exists() else None
        research_step=research["step"] if research is not None else None
        retained={"protected_pnp_research_best":research,"status":"RETAINED_CORNER_BOUND_DEPENDENCIES","best_step":evaluation["best_step"],
            "evaluated_snapshot_bindings":refs,"selected_aliases":[str(p.relative_to(R)) for p in sorted(out.glob("selected_step_*.pt"))],
            "recovery":str((out/"last.pt").relative_to(R)) if (out/"last.pt").exists() else None,
            "cleanup_performed":False,"recovery_removed":False,"requested_remove_recovery":remove_recovery,
            "provisional_dependency_hold":True,
            "final_retention_policy":"Keep stage best and explicitly valuable history only; this hold does not designate every rejection permanent",
            "retirement_review_candidates":[{"step":r["step"],"snapshot":r["evaluated_snapshot"],
                "selection_status":r.get("selection_status"),"is_step0_reference":r["step"]==0,
                "review":"Close/rebind prediction and calibration references; retain only selected parent/negative-control/calibration/valuable-history dependencies"}
                for r in reports if r["step"] not in (evaluation["best_step"],research_step)],
            "artifact_bytes":_corner_inventory(out),
            "reason":"Temporary hold until final reference/retirement audit; no currently bound dependency is deleted"}
        atomic_json(out/"CORNER_RETENTION.json",retained)
        return []
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
