"""Separate PnP-candidate All8 research selection; never upgrades deployment quality."""
import math,json,os
from model.training.action4_runtime import atomic_json
POLICY="action4.pnp_candidate_all8_research_best.v1"

def rank(row):
    r=row["metrics"];p=r["pose_corner_p95_valid_matched_only"]
    return (r["pose_all8_at_002_rate"],-(r["pose_fp"]+r["pose_fn"]),-p if p is not None else -math.inf,-row["step"])

def candidate(result):
    if result.get("research_selection_policy")!=POLICY:raise ValueError("Explicit research policy required")
    report=result["report"];metric=report["candidate_pose_report"]
    if metric.get("score_schema")!="action4.pose_projection_all8_002.v1":raise ValueError("PnP research requires original fixed-ID pose score")
    n=metric["pose_gt_count"];rows=metric["per_gt_failure_reason"]
    if n!=1300 or report.get("evaluation_images")!=650 or len(rows)!=n:raise ValueError("Full650/1300 research candidate required")
    keys=[(x["sample_id"],str(x["instance_id"])) for x in rows]
    if len(set(keys))!=n or len({k[0] for k in keys})!=650:raise ValueError("Research GT denominator/domain differs")
    success=sum(x["all8_at_002"] for x in rows)
    if metric["pose_all8_at_002_success"]!=success or metric["pose_all8_at_002_rate"]!=success/n:raise ValueError("Research count/rate mismatch")
    return dict(schema=POLICY,config_id=result["config_id"],step=result["step"],route="corner_candidate",
        metrics={k:v for k,v in metric.items() if k not in ("per_gt_failure_reason","unmatched_pose_predictions")},
        evaluated_snapshot=result["evaluated_snapshot"],report_artifact=report["candidate_pose_report_artifact"],
        calibration_receipt=report["calibration_receipt"],quality_ready=report["corner_quality_ready"],
        deployment_eligible=result["deployment_eligible"],deployment_admitted=False,quality_gate_used_for_ranking=False)

def commit(out,result):
    """Caller holds the existing best-commit lock; done is still committed last."""
    from .action4_async_recovery import checked_binding
    row=candidate(result);source=checked_binding(row["evaluated_snapshot"])
    checked_binding(row["report_artifact"]);checked_binding(row["calibration_receipt"])
    path=out/"PNP_RESEARCH_BEST.json";best=json.loads(path.read_text()) if path.exists() else None
    if best is not None:
        if best.get("schema")!=POLICY or best.get("config_id")!=row["config_id"]:raise ValueError("Mixed research best identity")
        checked_binding(best["evaluated_snapshot"])
    if best is None or rank(row)>rank(best):
        atomic_json(path,row);best=row
    selected=checked_binding(best["evaluated_snapshot"]);target=out/"pnp_research_best.pt"
    if not target.exists() or not os.path.samefile(selected,target):
        temp=out/f"pnp_research_alias_{os.getpid()}.tmp"
        try:
            if temp.exists():temp.unlink()
            os.link(selected,temp);os.replace(temp,target)
        finally:
            if temp.exists():temp.unlink()
    return best

def verify_complete(out,reports):
    from .action4_async_recovery import checked_binding
    rows=[candidate(r) for r in reports];winner=max(rows,key=rank)
    saved=json.loads((out/"PNP_RESEARCH_BEST.json").read_text())
    if saved!=winner:raise ValueError("PnP research best differs from all completed candidate reports")
    source=checked_binding(saved["evaluated_snapshot"])
    if not os.path.samefile(source,out/"pnp_research_best.pt"):raise ValueError("PnP research best alias changed")
    return dict(policy=POLICY,step=saved["step"],route=saved["route"],quality_ready=saved["quality_ready"],
        deployment_admitted=False,checkpoint="pnp_research_best.pt",manifest="PNP_RESEARCH_BEST.json")
