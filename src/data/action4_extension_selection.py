"""Cross-run ROI extension selection using the existing completed-role verifier.

Creation performs expensive sealed-role verification once. Training-time
validation checks bound seal/config/BEST metadata and recomputes the exact
cross-run policy; it never reruns all evaluations or loads models.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path("/data/zy/vln/a-test")
SCHEMA="action4.roi15_global_best_selection.v1"
POLICY="All8_rate_then_min_FP_plus_FN_then_min_P95_exact_tie_keeps_incumbent"
PROTOCOL="action4.fixed_roi_prediction_adaptation.v1"
SCORE_SCHEMA="action4.pose_projection_all8_002.v1"
INPUT_KEYS=("training_protocol","dataset_root","seed","network","preprocess","roi_source","roi_expansion","roi_size",
    "prediction_roi_manifest","frozen_eval_b0_predictions","evaluation_counterpart_sha256",
    "validation_receipt","validation_receipt_sha256","surface_source","train_backbone","a0_variant","sam_enabled",
    "roi_jitter","photometric","geometry_batch_policy","loss_weights","independent_head_dependencies",
    "b0_scoring_policy","evaluation_scoring_policy","score_schema","a0_selection_contract",
    "surface_lr","lr","weight_decay","betas","eps","grad_clip_norm","warmup_updates","eta_min")


def _path(value):
    path=Path(value);path=path if path.is_absolute() else ROOT/path
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Extension selection path escapes project")
    return path


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _sha(path):
    with _path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def _read(ref):
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:
        raise ValueError("Explicit path/SHA reference required")
    path=_path(ref["path"])
    if _sha(path)!=ref["sha256"]:raise ValueError("Bound extension selection evidence changed")
    return json.loads(path.read_text())


def _ref(path):
    path=_path(path)
    return {"path":str(path.relative_to(ROOT)),"sha256":_sha(path)}


def comparison_key(report):
    """Same first three terms as pose_protocol.rank; local optimizer step is omitted."""
    if report.get("score_schema")!=SCORE_SCHEMA:raise ValueError("Wrong extension pose score schema")
    rate=report["pose_all8_at_002_rate"];p95=report["pose_corner_p95_valid_matched_only"]
    for value in (rate,p95):
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError("Nonfinite/malformed extension comparison metric")
    for key in ("pose_fn","pose_fp"):
        if type(report[key]) is not int or report[key]<0:raise ValueError("Integer pose FP/FN required")
    if rate is not None and not 0<=rate<=1:raise ValueError("All8 rate out of range")
    if p95 is not None and p95<0:raise ValueError("Negative P95")
    return (rate if rate is not None else -1.,-(report["pose_fn"]+report["pose_fp"]),
            -p95 if p95 is not None else -float("inf"))


def _metadata(ref,epochs):
    seal=_read(ref)
    if (seal.get("schema")!="action4.sealed_independent_role.v2" or seal.get("role")!="A0"
            or seal.get("strict_load") is not True or seal.get("frozen_backbone_verified") is not True
            or seal.get("epochs")!=epochs):
        raise ValueError("A completed strictly reloaded frozen-backbone seal is required")
    spec=_read({"path":seal["run_spec"],"sha256":seal["run_spec_sha256"]})
    cfg=json.loads(_path(spec["config"]).read_text())
    if _digest(cfg)!=seal["config_id"] or cfg.get("mode")!="full_training" or cfg.get("epochs")!=epochs:
        raise ValueError("Sealed source config identity differs")
    expected_batch,interval=(48,25) if epochs==50 else (64,10)
    n=cfg.get("expected_units")
    if (type(n) is not int or n<=0 or cfg.get("batch")!=expected_batch or cfg.get("eval_every")!=interval
            or cfg.get("expected_eval_images")!=650 or cfg.get("train_indices") is not None
            or cfg.get("training_protocol")!=PROTOCOL or cfg.get("roi_expansion")!=1.5
            or cfg.get("roi_source")!="frozen_b0_prediction" or cfg.get("train_backbone") is not False):
        raise ValueError("Completed full-domain ROI recipe required")
    total=epochs*math.ceil(n/expected_batch)
    expected_steps=set([0,total]+list(range(interval,total+1,interval)))
    completion=seal.get("completion",{})
    if (seal.get("completed_optimizer_steps")!=total or completion.get("evaluations")!=len(expected_steps)
            or completion.get("best_step")!=seal.get("selected_step") or seal.get("selected_step") not in expected_steps
            or completion.get("config_id")!=seal["config_id"]):
        raise ValueError("Completed optimizer/evaluation/best coverage differs")
    best_path=_path(cfg["output"])/"BEST.json";key=str(best_path.relative_to(ROOT))
    expected_sha=completion.get("evidence_sha256",{}).get(key)
    if expected_sha is None:raise ValueError("True BEST receipt missing from sealed completion evidence")
    best_ref={"path":key,"sha256":expected_sha};best=_read(best_ref)
    if (best.get("role")!="A0" or best.get("config_id")!=seal["config_id"]
            or best.get("step")!=seal["selected_step"]
            or best.get("a0_selection_contract")!="full_all8_training_candidate_v1"):
        raise ValueError("BEST report is not the sealed selected candidate")
    report=best["report"];comparison_key(report)
    ids=[r["sample_id"] for r in report["bbox_only"]["per_image"]]
    if len(ids)!=650 or len(set(ids))!=650:raise ValueError("BEST must contain all650 evaluation images")
    identity=report.get("identity",{})
    if identity.get("A0_step")!=seal["selected_step"] or identity.get("A0_config_id")!=seal["config_id"]:
        raise ValueError("BEST evaluated A0 identity differs")
    return {"seal":seal,"config":cfg,"best":best,"best_ref":best_ref,"eval_ids":set(ids)}


def _resolve(incumbent_ref,predecessor_ref):
    incumbent=_metadata(incumbent_ref,50);predecessor=_metadata(predecessor_ref,10)
    a,b=incumbent["config"],predecessor["config"]
    segment=b.get("continuation_segment")
    expected={"schema":"action4.roi15_best_extension.v1","epochs":10,"batch":64,"eval_every":10,
              "parent_seal":incumbent_ref}
    if (a.get("continuation_segment") is not None or segment!=expected
            or predecessor["seal"].get("continuation_segment")!=segment
            or b.get("formal_epochs")!=10 or b.get("formal_training_units")!=a["expected_units"]
            or b.get("schedule_total_updates")!=10*math.ceil(a["expected_units"]/64)
            or b.get("world_size")!=4 or b.get("micro")!=2 or b["expected_units"]!=a["expected_units"]
            or b.get("parent_bundle_sha256")!=incumbent["seal"]["sha256"]
            or b.get("parent_best_step")!=incumbent["seal"]["selected_step"]):
        raise ValueError("Predecessor must be the completed registered10epoch child of this incumbent")
    if any(a.get(k)!=b.get(k) for k in INPUT_KEYS) or incumbent["eval_ids"]!=predecessor["eval_ids"]:
        raise ValueError("Cross-run data/B0/ROI/recipe or evaluation domain differs")
    winner="predecessor" if comparison_key(predecessor["best"]["report"])>comparison_key(incumbent["best"]["report"]) else "incumbent"
    return incumbent,predecessor,winner


def validate_global_selection(ref):
    """Bound metadata validation only; safe per rank, no model/eval execution."""
    result=_read(ref)
    a,b,winner=_resolve(result["incumbent_seal"],result["predecessor_seal"])
    if (result.get("schema")!=SCHEMA or result.get("status")!="COMPLETED_PREDECESSOR_GLOBAL_BEST_SELECTED"
            or result.get("policy")!=POLICY or result.get("tie_policy")!="retain_incumbent"
            or result.get("local_optimizer_step_used_for_cross_run_rank") is not False
            or result.get("verification")!="existing_sealed_role_full_immutable_verifier"
            or result.get("incumbent_best")!=a["best_ref"] or result.get("predecessor_best")!=b["best_ref"]
            or result.get("winner")!=winner
            or result.get("winner_seal")!=(result["predecessor_seal"] if winner=="predecessor" else result["incumbent_seal"])
            or result.get("formal_training_units")!=a["config"]["expected_units"]):
        raise ValueError("Global selection receipt/rank/source binding differs")
    selected=b if winner=="predecessor" else a
    return {"receipt":result,"winner_seal":selected["seal"],"winner_config":selected["config"],
            "incumbent_config":a["config"],"predecessor_config":b["config"]}
