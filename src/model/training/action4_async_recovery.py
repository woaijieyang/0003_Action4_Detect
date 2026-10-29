"""Single-evaluator recovery and atomic immutable checkpoint selection."""
import os,json,hashlib
from pathlib import Path
from model.training.action4_runtime import atomic_json
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path

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

CORNER_VARIANT="a0_corner_image"
CORNER_SELECTION_CONTRACT="quality_gate_and_original_pose_protocol.v1"


def checked_binding(binding):
    if not isinstance(binding,dict) or set(binding)!={"path","sha256"}:
        raise ValueError("Explicit evaluation artifact path/SHA binding required")
    p=contained_path(ROOT,binding["path"],exists=True)
    def stamp():
        z=p.stat();return (z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
    before=stamp()
    with p.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=binding["sha256"]:
            raise ValueError("Corner evaluation artifact changed")
    if stamp()!=before:raise ValueError("Corner evaluation artifact raced verification")
    return p


def corner_result_signature(result):
    report=result.get("report",{})
    if result.get("evaluation_variant")!=CORNER_VARIANT or result.get("role")!="A0":
        raise ValueError("Explicit corner result variant/role required")
    if report.get("score_schema")!="action4.pose_projection_all8_002.v1":
        raise ValueError("Corner selection must use original pose protocol")
    for value in (report.get("corner_quality_ready"),report.get("quality_gate_passed_for_selection"),
                  result.get("pose_protocol_eligible"),result.get("deployment_eligible")):
        if type(value) is not bool:raise ValueError("Explicit corner eligibility flags required")
    if (report["corner_quality_ready"]!=report["quality_gate_passed_for_selection"] or
        result["deployment_eligible"]!=(report["quality_gate_passed_for_selection"] and result["pose_protocol_eligible"]) or
        result.get("selection_contract")!=CORNER_SELECTION_CONTRACT):
        raise ValueError("Corner result eligibility conjunction changed")
    source=result.get("corner_eval_source")
    if source!=report.get("corner_eval_source"):raise ValueError("Corner source/report mismatch")
    checked_binding(source)
    from data.action4_sam_native_export import _v2_read
    artifact=result.get("report_artifact")
    if (not isinstance(artifact,dict) or set(artifact)!={"path","sha256"}
            or contained_path(ROOT,artifact["path"],exists=True)!=contained_path(ROOT,result.get("raw_report",""),exists=True)):
        raise ValueError("Corner immutable report path/binding mismatch")
    if _v2_read(artifact,json.load)!=report:
        raise ValueError("Corner immutable report differs at selection commit/completion")
    parent=result.get("parent_stage_fallback")
    if not isinstance(parent,dict) or set(parent)!={"path","sha256","selected_step"}:
        raise ValueError("Bound parent-stage fallback required")
    return {"schema":"action4.corner_eval_run_identity.v1","config_id":result["config_id"],
        "role":"A0","evaluation_variant":CORNER_VARIANT,"corner_eval_source":source,
        "parent_stage_fallback":parent}


def _commit_corner_result(out,request,snapshot,result,rank):
    signature=corner_result_signature(result)
    req=json.loads(request.read_text())
    if (req.get("step"),req.get("role"),req.get("config_id"))!=(result["step"],"A0",result["config_id"]):
        raise ValueError("Corner completed request/config identity mismatch")
    evaluated=result.get("evaluated_snapshot")
    if not isinstance(evaluated,dict) or checked_binding(evaluated)!=contained_path(ROOT,snapshot,exists=True):
        raise ValueError("Corner evaluated snapshot binding mismatch")
    ledger=out/"CORNER_EVAL_IDENTITY.json"
    if ledger.exists():
        if json.loads(ledger.read_text())!=signature:raise ValueError("Mixed corner config/source/parent in one evaluation run")
    else:
        if (out/"BEST.json").exists():raise ValueError("Cannot mix legacy BEST into corner evaluation")
        atomic_json(ledger,signature)
    best_path=out/"BEST.json";best=json.loads(best_path.read_text()) if best_path.exists() else None
    if best is not None:
        if corner_result_signature(best)!=signature or best["deployment_eligible"] is not True:
            raise ValueError("Existing corner BEST is not a same-run eligible result")
        selected=contained_path(ROOT,best["checkpoint"],exists=True)
        if selected.parent!=out or not os.path.samefile(selected,checked_binding(best["evaluated_snapshot"])):
            raise ValueError("Corner BEST checkpoint differs from evaluated snapshot")

    elif (out/"best.pt").exists():raise ValueError("Corner best alias exists without authoritative BEST")
    result=dict(result,selection_status="ELIGIBLE_EVALUATED" if result["deployment_eligible"] else "REJECTED_NOT_DEPLOYABLE")
    if result["deployment_eligible"] and (best is None or rank(result)>rank(best)):
        selected=out/f"selected_step_{result['step']:08d}.pt"
        if not selected.exists():os.link(snapshot,selected)
        elif not os.path.samefile(selected,snapshot):raise ValueError("Corner selected checkpoint name collision")
        checked_binding({"path":str(selected),"sha256":evaluated["sha256"]})
        result=dict(result,checkpoint=str(selected),selection_status="SELECTED_BEST_AT_COMMIT")
        atomic_json(best_path,result);best=result
    if best is not None:
        selected=contained_path(ROOT,best["checkpoint"],exists=True);target=out/"best.pt"
        if not target.exists() or not os.path.samefile(selected,target):
            temp=out/f"best_alias_{os.getpid()}.tmp"
            try:
                if temp.exists():temp.unlink()
                os.link(selected,temp);os.replace(temp,target)
            finally:
                if temp.exists():temp.unlink()
    if result.get("research_selection_policy") is not None:
        from .action4_corner_research_selection import commit
        commit(out,result)
    # Preserve original snapshot paths: frozen prediction/calibration artifacts bind them.
    atomic_json(request.with_suffix(".done.json"),result)
    return result


def commit_result(out,request,snapshot,result,rank):
    if result.get("evaluation_variant")==CORNER_VARIANT or "corner_quality_ready" in result.get("report",{}):
        return _commit_corner_result(out,request,snapshot,result,rank)
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
        target=out/"best.pt"
        if not target.exists() or not os.path.samefile(selected,target):
            temp=out/f"best_alias_{os.getpid()}.tmp"
            try:
                if temp.exists():temp.unlink()
                os.link(selected,temp);os.replace(temp,target)
            finally:
                if temp.exists():temp.unlink()
    # Done is LAST. A crash before it replays this original snapshot safely.
    atomic_json(request.with_suffix(".done.json"),result)
    snapshot.unlink()
