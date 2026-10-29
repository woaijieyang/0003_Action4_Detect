"""Seal completed roles with immutable per-role loaders; no training or promotion.

--check-only never seals. Production verification always uses the run's own
rank_result and strict loader in a fresh CPU subprocess. Metadata fixtures may
inject rank_fn into inspect(); that injection is not exposed by the CLI.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

ROOT = Path("/data/zy/vln/a-test")
SCHEMA = "action4.sealed_independent_role.v2"
TRAINING_SCHEMA = "action4.sealed_training_candidate.v1"
TRAINING_POLICY = "action4.training_candidate_all8_separate_deployment.v1"
SAM_NATIVE_PROTOCOL = "action4.sam_prediction_roi_enhancement.native_v1"
CORNER_NATIVE_PROTOCOL = "action4.corner_prediction_roi_enhancement.native_v1"
FROZEN_PROTOCOLS = ("action4.fixed_roi_prediction_adaptation.v1", SAM_NATIVE_PROTOCOL, CORNER_NATIVE_PROTOCOL)
TERMINALS = ("TRAIN_COMPLETE.json", "EVAL_COMPLETE.json", "CONTROLLER_COMPLETE.json", "BEST.json")


def path(value, exists=True):
    p = ROOT / value
    if p.is_symlink() or not p.resolve(strict=exists).is_relative_to(ROOT.resolve()):
        raise ValueError("Path outside project or symlink")
    return p.resolve(strict=exists)


def digest(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(value, evidence=None):
    p = path(value)
    content = p.read_bytes()
    result = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))
    if evidence is not None:
        evidence[str(p.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    return result


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer count: " + name)
    return value



def validate_extension_seal_parent(cfg):
    """Pure metadata check usable before immutable worker imports are installed."""
    segment=cfg.get("continuation_segment")
    twenty=isinstance(segment,dict) and segment.get("schema")=="action4.roi15_best_extension.v2"
    epochs=20 if twenty else 10
    keys={"schema","epochs","batch","eval_every","parent_seal"}|({"global_selection"} if twenty else set())
    if (not isinstance(segment,dict) or set(segment)!=keys
            or segment["schema"] not in ("action4.roi15_best_extension.v1","action4.roi15_best_extension.v2")
            or (segment["epochs"],segment["batch"],segment["eval_every"])!=(epochs,64,10)):
        raise ValueError("Unregistered ROI extension segment")
    selection=None
    if twenty:
        try:
            from data.action4_extension_selection import validate_global_selection
        except ModuleNotFoundError:
            # CLI -I has not installed snapshot imports yet. Load only the
            # stdlib metadata helper from this run's bound source tree.
            import importlib.util
            run=read_json(cfg["run_spec"])
            module_path=path(run["code_root"])/"src/data/action4_extension_selection.py"
            spec_module=importlib.util.spec_from_file_location("extension_metadata",module_path)
            metadata=importlib.util.module_from_spec(spec_module);spec_module.loader.exec_module(metadata)
            validate_global_selection=metadata.validate_global_selection
        selection=validate_global_selection(segment["global_selection"])
        if segment["parent_seal"]!=selection["receipt"]["winner_seal"]:
            raise ValueError("20epoch seal parent differs from global selection")
    ref=segment["parent_seal"];parent=read_json(ref["path"])
    if digest(path(ref["path"]))!=ref["sha256"]:
        raise ValueError("Extension parent seal changed")
    if (parent.get("schema")!=SCHEMA or parent.get("role")!="A0" or parent.get("epochs")!=(selection["winner_config"]["epochs"] if selection else 50)
            or parent.get("strict_load") is not True or parent.get("frozen_backbone_verified") is not True
            or cfg.get("parent_bundle_sha256")!=parent.get("sha256")
            or cfg.get("parent_best_step")!=parent.get("selected_step")):
        raise ValueError("Extension requires completed50epoch strictly loaded frozen-backbone best")
    if digest(path(parent["run_spec"]))!=parent["run_spec_sha256"]:
        raise ValueError("Parent run identity changed")
    spec=read_json(parent["run_spec"]);pcfg=read_json(spec["config"])
    if canonical(pcfg)!=parent["config_id"] or (selection is None and pcfg.get("continuation_segment") is not None):
        raise ValueError("Parent config/lineage differs")
    units=pcfg["expected_units"]
    if ((selection is None and (pcfg.get("epochs"),pcfg.get("batch"),pcfg.get("eval_every"))!=(50,48,25))
            or (selection is not None and pcfg!=selection["winner_config"])
            or cfg.get("formal_training_units")!=units or cfg.get("expected_units")!=units
            or cfg.get("world_size")!=4 or cfg.get("formal_epochs")!=epochs or cfg.get("train_indices") is not None
            or cfg.get("schedule_total_updates")!=epochs*math.ceil(units/64)):
        raise ValueError("Formal extension data/topology/scheduler budget differs")
    total=pcfg["epochs"]*math.ceil(units/pcfg["batch"]);interval=pcfg["eval_every"]
    if (parent.get("completed_optimizer_steps")!=total
            or parent.get("completion",{}).get("evaluations")!=len(set([0,total]+list(range(interval,total+1,interval))))):
        raise ValueError("Parent formal evaluation coverage incomplete")
    for key in ("prediction_roi_manifest","frozen_eval_b0_predictions","evaluation_counterpart_sha256",
                "dataset_root","validation_receipt","validation_receipt_sha256","surface_source","network",
                "preprocess","loss_weights","surface_lr","lr","weight_decay","betas","eps","seed",
                "grad_clip_norm","warmup_updates","eta_min","geometry_batch_policy",
                "independent_head_dependencies","b0_scoring_policy","evaluation_scoring_policy",
                "training_protocol","score_schema","a0_selection_contract"):
        if cfg.get(key)!=pcfg.get(key):raise ValueError("Extension changed parent input/recipe: "+key)
    return parent

def validate_formal_recipe(cfg):
    """Only registered50epoch ROI adaptation and explicit10/20 extensions bypass80+."""
    if cfg["mode"]!="full_training" or cfg["role"] not in ("A0","B0"):
        raise ValueError("Only completed formal A0/B0 runs may be sealed")
    if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
        if (cfg.get("role")!="A0" or cfg.get("epochs")!=80 or cfg.get("train_indices") is not None
            or cfg.get("expected_units")!=cfg.get("formal_training_units") or cfg.get("train_backbone") is not False
            or cfg.get("corner_softmax_bias_frozen") is not True or cfg.get("a0_variant")!="a0_corner_image"):
            raise ValueError("Corner sealing requires its complete formal80 recipe")
        return False
    if cfg.get("training_protocol")==SAM_NATIVE_PROTOCOL:
        if (cfg.get("epochs")!=80 or cfg.get("formal_epochs")!=80 or cfg.get("train_indices") is not None
                or cfg.get("expected_units")!=cfg.get("formal_training_units") or cfg.get("train_backbone") is not False):
            raise ValueError("SAM sealing requires complete independent80epoch formal domain")
        return False
    if cfg.get("training_protocol")=="action4.fixed_roi_prediction_adaptation.v1":
        extension=cfg.get("continuation_segment") is not None
        if extension:validate_extension_seal_parent(cfg)
        expected={"role":"A0","epochs":cfg["continuation_segment"]["epochs"] if extension else 50,"roi_expansion":1.5,"roi_size":448,
            "roi_source":"frozen_b0_prediction","batch":64 if extension else 48,"micro":2,"eval_every":10 if extension else 25,
            "expected_eval_images":650,"a0_variant":"a0_baseline","score_schema":"action4.pose_projection_all8_002.v1",
            "a0_selection_contract":"full_all8_training_candidate_v1"}
        if any(cfg.get(k)!=v for k,v in expected.items()) or cfg.get("train_backbone") is not False:
            raise ValueError("Fixed50epoch prediction-ROI sealing contract differs")
        if cfg.get("sam_enabled") is not False or cfg.get("roi_jitter") is not None or cfg.get("photometric") not in (False,None):
            raise ValueError("Unexpected input augmentation or SAM in fixed ROI adaptation")
        if cfg.get("preprocess",{}).get("roi_expansion")!=1.5:
            raise ValueError("Stored ROI preprocessing differs")
        return True
    if cfg["epochs"]<80:
        raise ValueError("Only completed formal80+ A0/B0 runs may be sealed")
    return False


def adaptation_external_inputs(cfg,evidence):
    refs=list(cfg["prediction_roi_manifest"].values())+[cfg["frozen_eval_b0_predictions"]]
    for ref in refs:
        p=path(ref["path"]);actual=digest(p)
        if actual!=ref["sha256"]:raise ValueError("External prediction ROI dependency changed")
        evidence[str(p.relative_to(ROOT))]=actual
    receipt=read_json(cfg["validation_receipt"],evidence)
    if digest(path(cfg["validation_receipt"]))!=cfg["validation_receipt_sha256"]:
        raise ValueError("Validation receipt changed")
    ref=receipt["splits"]["eval"];p=path(ref["manifest"])
    actual=digest(p)
    if actual!=ref["sha256"]:raise ValueError("Evaluation manifest/camera identity changed")
    evidence[str(p.relative_to(ROOT))]=actual
    records=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    ids=[r["sample_id"] for r in records]
    if len(ids)!=650 or len(set(ids))!=650 or any(r["split"]!="eval" for r in records):
        raise ValueError("Expected exact650 published eval domain")
    return set(ids)


def validate_adaptation_eval(done,cfg,expected_ids):
    if done.get("a0_selection_contract")!=cfg["a0_selection_contract"]:
        raise ValueError("A0 selection contract missing or changed")
    report=done["report"];identity=report.get("identity",{})
    if (report.get("roi_expansion")!=1.5 or report.get("frozen_b0_predictions")!=cfg["frozen_eval_b0_predictions"]
        or report.get("B0_forward_calls")!=0 or identity.get("A0_step")!=done["step"]
        or identity.get("A0_config_id")!=canonical(cfg) or identity.get("pose_method")!="network"
        or identity.get("B0_scoring_policy")!=cfg["b0_scoring_policy"] or identity.get("score_threshold")!=.3):
        raise ValueError("Actual adaptation eval ROI/model/cache/decoder contract differs")
    ids=[r["sample_id"] for r in report["bbox_only"]["per_image"]]
    if len(ids)!=650 or len(set(ids))!=650 or set(ids)!=expected_ids:
        raise ValueError("Actual evaluation sample domain differs")


def inspect(run_spec, rank_fn=None, report_sink=None, corner_validator=None):
    """Read completion metadata; rank_fn is supplied by the immutable worker."""
    evidence = {}
    spec = read_json(run_spec, evidence)
    cfg = read_json(spec["config"], evidence)
    if spec.get("schema") != "action4.immutable_run.v1" or spec.get("project_root") != str(ROOT):
        raise ValueError("Invalid immutable run identity")
    if not cfg.get("require_immutable_source") or path(cfg["run_spec"]) != path(run_spec):
        raise ValueError("Configuration RUN_SPEC binding mismatch")
    if not path(spec["config"]).is_relative_to(path(spec["code_root"])):
        raise ValueError("Configuration outside its immutable code root")
    for key in ("epochs", "expected_units", "batch", "eval_every", "expected_eval_images"):
        integer(cfg[key], key, 1)
    adaptation=validate_formal_recipe(cfg)
    if adaptation:
        ref=cfg["adaptation_execution_gate"]
        proof=read_json(ref["path"],evidence)
        if digest(path(ref["path"]))!=ref["sha256"] or proof.get("status")!="ROI_ADAPTATION_EXECUTION_PASS":
            raise ValueError("Adaptation execution admission identity changed")
        if cfg.get("continuation_segment") is not None:
            if proof.get("continuation_segment")!=cfg["continuation_segment"]:
                raise ValueError("Extension execution proof changed")
            parent_ref=cfg["continuation_segment"]["parent_seal"]
            read_json(parent_ref["path"],evidence)
            if cfg["continuation_segment"]["schema"]=="action4.roi15_best_extension.v2":
                global_ref=cfg["continuation_segment"]["global_selection"]
                selection=read_json(global_ref["path"],evidence)
                for field in ("incumbent_seal","predecessor_seal","winner_seal","incumbent_best","predecessor_best"):
                    read_json(selection[field]["path"],evidence)
        for key in ("project_code_sha256","prediction_roi_manifest","parent_A0_sha256"):
            expected=cfg["parent_bundle_sha256"] if key=="parent_A0_sha256" else cfg[key]
            if proof.get(key)!=expected:raise ValueError("Adaptation source/data/parent admission differs")

    out = path(cfg["output"])
    missing = [n for n in TERMINALS if not (out / n).is_file()]
    if missing:
        return spec, cfg, None, {"status": "WAITING_FOR_COMPLETION", "missing": missing}
    sam = cfg.get("training_protocol")==SAM_NATIVE_PROTOCOL
    corner = cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL
    if sam:
        sam_input_contract(cfg,evidence)
        read_json(cfg["integration_or_admission_receipt"],evidence)
    expected_eval_ids=adaptation_external_inputs(cfg,evidence) if adaptation or sam else None
    receipts = {n: read_json(out / n, evidence) for n in TERMINALS}
    train, evaluation, complete, best = (receipts[n] for n in TERMINALS)
    config_id = canonical(cfg)
    if any(x["config_id"] != config_id for x in receipts.values()):
        raise ValueError("Receipt configuration identity mismatch")
    attempt = train["attempt_id"]
    if not isinstance(attempt, str) or not attempt or any(x["attempt_id"] != attempt for x in (evaluation, complete)):
        raise ValueError("Receipt attempt identities differ")
    if any(x["role"] != cfg["role"] for x in (train, complete, best)):
        raise ValueError("Role mismatch")
    if any(x.get("mode") != "full_training" for x in (train, complete)):
        raise ValueError("Nonformal completion")
    step = integer(train["step"], "optimizer step", 1)
    skipped = integer(train.get("no_supervision_update_count", 0), "skipped updates")
    total = cfg["epochs"] * math.ceil(cfg["expected_units"] / cfg["batch"])
    for name, expected in (("epochs", cfg["epochs"]), ("units_per_epoch", cfg["expected_units"])):
        if integer(train[name], name, 1) != expected:
            raise ValueError("Incomplete traversal")
    if step + skipped != total or train["formal_training_complete"] is not True:
        raise ValueError("Incomplete traversal")
    for name in ("data_update_attempt", "planned_update_attempts"):
        if name in train and integer(train[name], name, 1) != total:
            raise ValueError("Traversal attempt mismatch")
    expected = {0, step} | set(range(cfg["eval_every"], step + 1, cfg["eval_every"]))
    if cfg.get("eval_schedule") is not None:
        # Parent inspection also loads the run-bound pure cadence implementation;
        # do not import a possibly newer working-tree module.
        import importlib.util
        relative="src/model/training/action4_eval_schedule.py"
        module_path=path(spec["code_root"])/relative
        if digest(module_path)!=spec["files"].get(relative):
            raise ValueError("Immutable eval schedule source differs")
        module_spec=importlib.util.spec_from_file_location("_sealed_eval_schedule",module_path)
        schedule_module=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(schedule_module)
        expected_eval_steps=schedule_module.expected_eval_steps
        ledger_path=path(cfg["output"])/"EVAL_SCHEDULE_LEDGER.json"
        evidence[str(ledger_path.relative_to(ROOT))]=digest(ledger_path)
        expected=expected_eval_steps(cfg,train,schedule_module.read_schedule_ledger(cfg,path(cfg["output"]),train))
    all_steps = evaluation["all_steps"]
    if not isinstance(all_steps, list) or any(type(s) is not int or s < 0 for s in all_steps):
        raise ValueError("Invalid evaluation steps")
    if len(all_steps) != len(expected) or set(all_steps) != expected or integer(evaluation["evaluations"], "evaluations", 1) != len(expected):
        raise ValueError("Evaluation coverage mismatch")
    best_step = integer(best["step"], "best step")
    if best_step not in expected or any(integer(x["best_step"], "selected step") != best_step for x in (complete, evaluation)):
        raise ValueError("Selection disagreement")
    queue = path(out / "eval")
    if list(queue.glob("*.failed.json")):
        raise ValueError("Failed evaluations remain")
    requests = sorted(p.name for p in queue.glob("step_*.json") if p.name.count(".") == 1)
    dones = sorted(p.name for p in queue.glob("*.done.json"))
    if requests != sorted(f"step_{s:08d}.json" for s in expected) or dones != sorted(f"step_{s:08d}.done.json" for s in expected):
        raise ValueError("Queue not closed")
    reports = []
    for s in sorted(expected):
        request = read_json(queue / f"step_{s:08d}.json", evidence)
        done = read_json(queue / f"step_{s:08d}.done.json", evidence)
        for record in (request, done):
            if type(record["step"]) is not int or record["step"] != s or record["config_id"] != config_id or record["role"] != cfg["role"]:
                raise ValueError("Per-evaluation identity mismatch")
            if "attempt_id" in record and record["attempt_id"] != attempt:
                raise ValueError("Per-evaluation attempt mismatch")
        # Existing requests have no mode/schema fields: exact config_id binds
        # those. Done records carry training_mode and actual report schema.
        if request.get("training_mode", cfg["mode"]) != cfg["mode"] or done["training_mode"] != cfg["mode"]:
            raise ValueError("Per-evaluation mode mismatch")
        expected_snapshot = out / f"eval_step_{s:08d}.pt"
        if path(request["snapshot"], False) != path(expected_snapshot, False):
            raise ValueError("Request snapshot identity mismatch")
        report = done["report"]
        if integer(report["evaluation_images"], "evaluated images", 1) != cfg["expected_eval_images"]:
            raise ValueError("Evaluation image coverage mismatch")
        if cfg["role"] == "B0":
            if report.get("schema") != "action4.bbox_metrics_candidate.v1" or report.get("scoring_policy") != cfg["evaluation_scoring_policy"]:
                raise ValueError("B0 score schema/policy mismatch")
            if integer(report["overall"]["images"], "B0 report images", 1) != cfg["expected_eval_images"]:
                raise ValueError("B0 image coverage mismatch")
        else:
            if report.get("score_schema") != cfg["score_schema"]:
                raise ValueError("A0 score schema mismatch")
            if type(done.get("deployment_eligible")) is not bool:
                raise ValueError("Missing checkpoint eligibility decision")
        if corner:
            binding=report["calibration_receipt"]
            calibration=read_json(binding["path"],evidence)
            if evidence[str(path(binding["path"]).relative_to(ROOT))]!=binding["sha256"]:raise ValueError("Corner calibration receipt changed")
            image_ids=calibration["raw_report"]["image_ids"]
            if calibration["candidate_report"]["image_ids"]!=image_ids:raise ValueError("Corner raw/candidate image domains differ")
        else:
            coverage = report if cfg["role"] == "B0" else report["bbox_only"]
            image_rows = coverage["per_image"]
            image_ids = [row["sample_id"] for row in image_rows]
        if len(image_ids) != cfg["expected_eval_images"] or len(set(image_ids)) != len(image_ids):
            raise ValueError("Per-image report coverage is incomplete or duplicated")
        raw_path = path(done["raw_report"])
        if not raw_path.is_relative_to(out / "eval_runs") or not raw_path.parent.name.startswith(f"step_{s:08d}_{attempt}_"):
            raise ValueError("Raw eval report attempt/path mismatch")
        raw = read_json(raw_path, evidence)
        # Worker appends timing and image count after writing REPORT.json.
        expected_raw=report if corner else {k:v for k,v in report.items() if k not in ("evaluation_wall_seconds","evaluation_images")}
        if raw!=expected_raw:raise ValueError("Raw/report content mismatch")
        if adaptation or sam:validate_adaptation_eval(done,cfg,expected_eval_ids)
        if sam:validate_sam_eval(done,cfg)
        reports.append(done)
    if adaptation or sam:validate_adaptation_eval(best,cfg,expected_eval_ids)
    if sam:validate_sam_eval(best,cfg)
    selected = next(r for r in reports if r["step"] == best_step)
    for key in ("step", "config_id", "role", "training_mode", "report", "raw_report", "deployment_eligible"):
        if selected.get(key) != best.get(key):
            raise ValueError("Selected eval receipt differs")
    if path(best["checkpoint"]) != path(out / "best.pt"):
        raise ValueError("BEST pointer must be the finalized run-local best.pt")
    if rank_fn is not None:
        ranked_reports=[r for r in reports if r.get("deployment_eligible") is True] if corner else reports
        if not ranked_reports:raise ValueError("No admitted corner checkpoint to seal")
        ranks = [rank_fn(r) for r in ranked_reports]
        for rank in ranks:
            if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in rank):
                raise ValueError("Nonfinite selection rank")
        winner = ranked_reports[max(range(len(ranked_reports)), key=lambda i: ranks[i])]
        if winner["step"] != best_step:
            raise ValueError("Selected best is not the immutable rank winner")
    status = {"status": "COMPLETION_METADATA_READY" if rank_fn else "PENDING_IMMUTABLE_RANK",
              "role": cfg["role"], "best_step": best_step, "evaluations": len(expected),
              "config_id": config_id, "attempt_id": attempt, "evidence_sha256": evidence}
    if adaptation:
        status["inference_contract"]={"training_protocol":cfg["training_protocol"],
            "roi_source":cfg["roi_source"],"roi_expansion":cfg["roi_expansion"],"roi_size":cfg["roi_size"],
            "B0_sha256":cfg["evaluation_counterpart_sha256"],"score_schema":cfg["score_schema"],
            "prediction_roi_manifest":cfg["prediction_roi_manifest"],
            "frozen_eval_b0_predictions":cfg["frozen_eval_b0_predictions"],
            "track_input_distribution_accepted":False}
        if cfg.get("continuation_segment") is not None:
            status["inference_contract"]["continuation_segment"]=cfg["continuation_segment"]
    if sam:
        status["inference_contract"]=sam_input_contract(cfg,evidence)
    if corner and corner_validator is not None:
        status["corner_inference_contract"]=corner_validator(cfg,reports,best_step)
        for key in ("stage_reference","calibration_receipt","report_artifact","evaluated_snapshot","corner_eval_source"):
            binding=status["corner_inference_contract"].get(key)
            if binding is not None:evidence[str(path(binding["path"]).relative_to(ROOT))]=binding["sha256"]
    if report_sink is not None:
        report_sink.extend(reports)
    return spec, cfg, receipts, status


def inspect_training_selection(spec, cfg, receipts, status, selection, reports, rank_fn=None):
    """Bind a completed training candidate without lowering deployment gates."""
    if cfg["role"] != "A0" or receipts is None:
        raise ValueError("Completed A0 run required for training-candidate seal")
    if cfg.get("a0_variant") == "a0_corner_image":
        raise ValueError("Corner candidate sealing additionally needs its quality/decoder bundle contract")
    root = path(selection)
    if not root.is_relative_to(ROOT / "outputs"):
        raise ValueError("Training selection must be an owned outputs directory")
    evidence = dict(status["evidence_sha256"])
    identity = read_json(root / "OBSERVER_IDENTITY.json", evidence)
    pointer = read_json(root / "TRAINING_BEST.json", evidence)
    completion = read_json(root / "COMPLETE.json", evidence)
    launch = read_json(root / "LAUNCH.json", evidence)
    if (identity.get("policy") != TRAINING_POLICY or identity.get("config_id") != canonical(cfg)
            or path(identity["run"]) != path(cfg["output"])
            or pointer.get("policy") != TRAINING_POLICY or pointer.get("config_id") != canonical(cfg)
            or pointer.get("sealed") is not True or pointer.get("purpose") != "training_candidate_not_deployment_release"
            or pointer.get("publication_default_changed") is not False):
        raise ValueError("Training-candidate identity or completion mismatch")
    if (completion.get("status") != "CANDIDATE_SELECTION_COMPLETE_PENDING_FINAL_SEAL_AND_RETENTION_AUDIT"
            or completion.get("training_best") != pointer or completion.get("evaluated_reports") != len(reports)
            or completion.get("deployment_release") is not False):
        raise ValueError("Training candidate is not completely selected")
    if (path(launch["run_spec"]) != path(cfg["run_spec"])
            or launch["run_spec_sha256"] != digest(path(cfg["run_spec"]))):
        raise ValueError("Training observer RunSpec differs")
    observer = path(launch["source"])
    if not observer.is_relative_to(root / "source") or observer.stat().st_mode & 0o222 or digest(observer) != launch["source_sha256"]:
        raise ValueError("Frozen candidate observer source changed")
    evidence[str(observer.relative_to(ROOT))] = launch["source_sha256"]
    step = integer(pointer["step"], "training candidate step")
    matches = [v for v in reports if v["step"] == step]
    if len(matches) != 1:
        raise ValueError("Training candidate missing from complete eval domain")
    chosen = matches[0]
    fields = {"score_schema","pose_gt_count","pose_output_count","pose_all8_at_002_success",
              "pose_all8_at_002_rate","pose_fn","pose_fp","pose_corner_p95_valid_matched_only",
              "pose_all8_at_002_precision","pose_instance_precision","pose_groups"}
    if set(pointer.get("metrics",{})) != fields or any(pointer["metrics"][k] != chosen["report"][k] for k in fields):
        raise ValueError("Training candidate metrics differ from original evaluated report")
    done = path(cfg["output"]) / "eval" / f"step_{step:08d}.done.json"
    if pointer.get("source_done") != {"path":str(done.relative_to(ROOT)), "sha256":evidence[str(done.relative_to(ROOT))]}:
        raise ValueError("Training candidate evaluated snapshot/report source differs")
    if pointer.get("deployment_guard_passed") != chosen["deployment_eligible"]:
        raise ValueError("Deployment gate must remain the evaluated separate result")
    checkpoint = path(pointer["checkpoint"])
    if checkpoint != path(root / "selected" / f"step_{step:08d}.pt") or digest(checkpoint) != pointer["checkpoint_sha256"]:
        raise ValueError("Actual training candidate checkpoint differs")
    if rank_fn is not None:
        winner = max(reports, key=lambda value:rank_fn(value["report"],value["step"]))
        if winner["step"] != step:
            raise ValueError("Training candidate is not the full-All8 global rank winner")
    return dict(status, status="TRAINING_CANDIDATE_METADATA_READY" if rank_fn else "PENDING_IMMUTABLE_TRAINING_RANK",
                best_step=step, legacy_deployment_best_step=receipts["BEST.json"]["step"],
                selection_purpose="training_candidate", deployment_guard_passed=chosen["deployment_eligible"],
                deployment_release=False, source_checkpoint=str(checkpoint.relative_to(ROOT)),
                candidate_checkpoint_sha256=pointer["checkpoint_sha256"],
                training_selection=str(root.relative_to(ROOT)), evidence_sha256=evidence)


@contextmanager
def training_selection_lock(selection):
    if selection is None:
        yield
        return
    import fcntl
    with path(path(selection) / "observer.lock").open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Training candidate observer is still active") from error
        yield


def verify_dependencies(spec, cfg, verified_files=None):
    """Verify current data receipt and frozen parent/counterpart byte identities."""
    code = path(spec["code_root"])
    validation = path(cfg["validation_receipt"])
    validation_sha = digest(validation)
    if validation_sha != cfg["validation_receipt_sha256"]:
        raise ValueError("Validation receipt changed after run freeze")
    dependencies = {"validation_receipt": str(validation.relative_to(ROOT)),
                    "validation_receipt_sha256": validation_sha}
    for key in ("parent_bundle", "evaluation_counterpart_bundle"):
        if key not in cfg:
            if key == "evaluation_counterpart_bundle" and cfg["role"] == "B0":
                continue
            raise ValueError("Missing dependency: " + key)
        p = path(cfg[key])
        if not p.is_relative_to(code / "inputs"):
            raise ValueError("Dependency is not a frozen run input: " + key)
        sha = verified_digest(p, verified_files)
        sha_key = "parent_bundle_sha256" if key == "parent_bundle" else "evaluation_counterpart_sha256"
        if sha != cfg[sha_key] or spec["files"].get(str(p.relative_to(code))) != sha:
            raise ValueError("Frozen dependency SHA mismatch: " + key)
        dependencies[key] = str(p.relative_to(ROOT))
        dependencies[sha_key] = sha
    return dependencies


def strict_load_selected(loader,candidate,cfg):
    """Old immutable loaders may not expose the new corner-only keyword."""
    if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
        return loader(str(candidate),cfg["role"],"cpu",allow_corner_candidate=True)
    return loader(str(candidate),cfg["role"],"cpu")


def verify_state(state, cfg, selected_step):
    if state.get("role") != cfg["role"] or state.get("config_id") != canonical(cfg) or state.get("config") != cfg:
        raise ValueError("Actual bundle configuration/role mismatch")
    if state.get("selected_step") != selected_step or type(state.get("selected_step")) is not int:
        raise ValueError("Actual bundle step mismatch")
    if state.get("parent_selected_step") != cfg["parent_best_step"] or path(state["parent_bundle"]) != path(cfg["parent_bundle"]):
        raise ValueError("Actual bundle parent identity mismatch")


def verify_frozen_backbone(state,parent):
    import torch
    if (state.get("backbone_frozen") is not True or state.get("backbone_config")!=parent.get("backbone_config")
        or state.get("backbone_step")!=parent.get("backbone_step")
        or state.get("backbone_parent_step")!=parent.get("backbone_step")):
        raise ValueError("Frozen backbone metadata differs from parent")
    current=state.get("backbone");previous=parent.get("backbone")
    if not isinstance(current,dict) or not isinstance(previous,dict) or set(current)!=set(previous):
        raise ValueError("Frozen backbone parameter/buffer domain differs")
    for name,value in current.items():
        old=previous[name]
        if (not torch.is_tensor(value) or not torch.is_tensor(old) or value.dtype!=old.dtype
            or value.shape!=old.shape or not torch.equal(value,old)):
            raise ValueError("Frozen backbone parameter/buffer changed: "+name)
    return True


def sam_input_contract(cfg,evidence):
    """Preserve and hash SAM inputs separately from baseline evidence."""
    fields=("training_protocol","parent_bundle_sha256","parent_best_step","evaluation_counterpart_sha256",
            "roi_source","roi_expansion","roi_size","prediction_roi_manifest","frozen_eval_b0_predictions",
            "sam_protocol_id","sam_cache","sam_eval_cache","sam_training_associations","sam_mask_augmentation","stage3_parent_seal")
    result={k:cfg[k] for k in fields}
    if (cfg["training_protocol"]!=SAM_NATIVE_PROTOCOL or cfg["roi_source"]!="frozen_b0_prediction"
            or cfg["roi_expansion"]!=1.5 or cfg["roi_size"]!=448 or cfg.get("sam_enabled") is not True):
        raise ValueError("SAM native ROI identity differs")
    def visit(value):
        if isinstance(value,dict):
            if "path" in value and "sha256" in value:
                p=path(value["path"]);actual=digest(p)
                if actual!=value["sha256"]:raise ValueError("SAM external input changed: "+str(p))
                evidence[str(p.relative_to(ROOT))]=actual
            else:
                for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    if cfg.get("stage3_execution_gate") is not None:
        result["stage3_execution_gate"]=cfg["stage3_execution_gate"]
    visit(result)
    # This protocol's eval profile explicitly selects CPU sampling; the train
    # dataset also produces CPU tensors. Record the protocol default explicitly.
    backend=cfg.get("roi_sampling_device","cpu")
    if backend!="cpu":raise ValueError("Unregistered SAM ROI sampling backend")
    result["roi_sampling_device"]=backend
    result["sampling_identity_source"]="native_v1 CPU dataset and frozen eval profile"
    return result


def validate_sam_eval(done,cfg):
    observed=done["report"].get("sam_cache_inputs",{})
    if (done["report"].get("identity",{}).get("ROI_sampling_device")!="cpu"
            or observed.get("manifest")!=cfg["sam_eval_cache"]["manifest"]
            or observed.get("engineering_fixture") is not False
            or observed.get("performance_admissible") is not True
            or observed.get("GT_used_for_prediction") is not False):
        raise ValueError("SAM evaluated cache/native identity differs")


def verify_sam_schema(state):
    from model.action4_release_bundle import SAM_SCHEMA
    if state.get("schema")!=SAM_SCHEMA or not isinstance(state.get("sam_adapter"),dict) or not state["sam_adapter"]:
        raise ValueError("Complete SAM_SCHEMA adapter weights required")


def verify_sam_engineering_bundle(bundle):
    """Read-only CPU proof; deliberately never emits a formal completion/seal."""
    from model.action4_release_bundle import load_bundle
    import torch
    torch.set_num_threads(2)
    candidate=path(bundle);sha=digest(candidate)
    model,base,state=load_bundle(str(candidate),"A0","cpu",expected_sha256=sha)
    verify_sam_schema(state)
    cfg=state["config"];evidence={};contract=sam_input_contract(cfg,evidence)
    verify_state(state,cfg,state["selected_step"])
    parent_model,parent_base,parent=load_bundle(str(path(cfg["parent_bundle"])),"A0","cpu",expected_sha256=cfg["parent_bundle_sha256"])
    if parent["selected_step"]!=cfg["parent_best_step"]:raise ValueError("Parent step changed")
    verify_frozen_backbone(state,parent)
    bmodel,_,bstate=load_bundle(str(path(cfg["evaluation_counterpart_bundle"])),"B0","cpu",expected_sha256=cfg["evaluation_counterpart_sha256"])
    unchanged(evidence)
    if digest(candidate)!=sha:raise ValueError("Engineering bundle changed")
    return {"status":"SAM_ENGINEERING_BUNDLE_CPU_VERIFIED_NOT_FORMAL_SEAL","bundle":str(candidate.relative_to(ROOT)),
            "sha256":sha,"selected_step":state["selected_step"],"strict_load":True,
            "frozen_backbone_verified":True,"backbone_tensor_buffer_count":len(state["backbone"]),
            "adapter_tensor_count":len(state["sam_adapter"]),"B0_selected_step":bstate["selected_step"],
            "inference_contract":contract,"evidence_sha256":evidence,
            "formal_training_complete":False,"formal_best_sealed":False,"default_models_modified":False}


def unchanged(evidence):
    for name, sha in evidence.items():
        if digest(path(name)) != sha:
            raise ValueError("Evidence changed during sealing: " + name)


@contextmanager
def completion_locks(out):
    import fcntl
    with ExitStack() as stack:
        for name in ("controller.lock", "selection.lock"):
            # Existing lock only: never create files in a completed/running run.
            stream = stack.enter_context(path(out / name).open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Run still has an active controller/selector") from error
        yield


def run_worker(run_spec, bundle=None, sha=None, training_selection=None):
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--verify-worker", "--run-spec", str(run_spec)]
    if bundle is not None:
        command += ["--bundle", str(bundle), "--sha256", sha]
    if training_selection is not None:
        command += ["--training-candidate-selection", str(training_selection)]
    runtime = path("outputs/action4_role_sealing_review_v001", False)
    for folder in ("tmp", "cache"):
        path(runtime / folder, False).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
               TMPDIR=str(runtime / "tmp"), XDG_CACHE_HOME=str(runtime / "cache"),
               TORCH_HOME=str(runtime / "cache"))
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def snapshot_pythonpath(code, current_paths=None):
    """Retain interpreter dependencies, remove other project paths/code roots."""
    controlled_sites = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    retained = []
    for value in sys.path if current_paths is None else current_paths:
        if not value:
            continue
        resolved = Path(value).resolve()
        if resolved in controlled_sites or not resolved.is_relative_to(ROOT.resolve()):
            retained.append(value)
    return [str(code / "runtime/surface_deps"), str(code / "src")] + retained


def stat_fingerprint(p):
    st = p.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode)


def verified_digest(p, verified_files=None):
    if verified_files is not None and str(p) in verified_files:
        expected, before = verified_files[str(p)]
        if stat_fingerprint(p) != before:
            raise ValueError("Previously verified immutable file changed: " + str(p))
        # Only readonly files can reuse the successful snapshot content scan.
        if not (before[-1] & 0o222):
            return expected
    return digest(p)


def verify_import_inventory(spec):
    code = path(spec["code_root"])
    for folder in ("src", "runtime/surface_deps"):
        base = code / folder
        if not base.exists():
            continue
        if base.is_symlink():
            raise ValueError("Frozen import root is a symlink")
        for directory, dirs, names in os.walk(base, followlinks=False):
            if any((Path(directory) / n).is_symlink() for n in dirs + names):
                raise ValueError("Frozen import inventory contains symlink")
            for name in names:
                p = Path(directory) / name
                importable = p.suffix in (".py", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib") or ".so." in name
                if importable and str(p.relative_to(code)) not in spec["files"]:
                    raise ValueError("Unlisted frozen importable file: " + str(p.relative_to(code)))


def verify_worker(run_spec, bundle=None, sha=None, training_selection=None):
    """Fresh subprocess only. Imports exactly one immutable project module root."""
    spec = read_json(run_spec)
    cfg = read_json(spec["config"])
    code = path(spec["code_root"])
    verify_import_inventory(spec)
    # Verify executable/configuration files before any snapshot imports.
    # Large frozen inputs are checked by the canonical full-manifest verifier
    # below before any checkpoint is loaded; no content check is skipped.
    before = {}
    for name, expected in spec["files"].items():
        p = path(code / name)
        if not p.is_relative_to(code):
            raise ValueError("Immutable manifest path escapes code root")
        before[str(p)] = (expected, stat_fingerprint(p))
        if not p.is_relative_to(code / "inputs") and digest(p) != expected:
            raise ValueError("Immutable run code identity mismatch: " + name)
        if stat_fingerprint(p) != before[str(p)][1]:
            raise ValueError("Immutable run file changed during pre-import verification")
    os.environ["ACTION4_PROJECT_ROOT"] = str(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path[:] = snapshot_pythonpath(code)
    from model.training.action4_run_snapshot import verify_running_snapshot
    from model.training.action4_runtime import canonical_hash
    from eval.action4_stage2_eval import rank_result
    from model.action4_release_bundle import load_bundle
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    verify_running_snapshot(cfg)
    if cfg.get("training_protocol")==SAM_NATIVE_PROTOCOL:
        from model.training.action4_stage3_prediction_protocol import validate_admission
        validate_admission(cfg,read_json(cfg["integration_or_admission_receipt"]))
    if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
        from model.training.action4_stage4_prediction_protocol import validate_admission
        validate_admission(cfg,read_json(cfg["integration_or_admission_receipt"]))
    # Canonical verifier has now checked ALL content hashes. Detect mutation
    # across that scan, and reuse readonly hashes only while stat identity holds.
    for filename, (_, fingerprint) in before.items():
        if stat_fingerprint(path(filename)) != fingerprint:
            raise ValueError("Immutable snapshot changed during full content verification")
    verified_files = before
    reports = []
    corner_validator=None
    if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
        from eval.action4_corner_sealing import validate_corner_sealing_reports
        corner_validator=validate_corner_sealing_reports
    spec, cfg, receipts, status = inspect(run_spec, rank_fn=None if training_selection is not None else rank_result,
        report_sink=reports,corner_validator=corner_validator)
    if training_selection is not None and receipts is not None:
        from eval.action4_pose_protocol import rank as pose_rank
        status = inspect_training_selection(spec, cfg, receipts, status, training_selection, reports, pose_rank)
    if receipts is None:
        raise ValueError("Cannot verify a run before completion")
    if canonical_hash(cfg) != status["config_id"]:
        raise ValueError("Canonical configuration hash mismatch")
    deps = verify_dependencies(spec, cfg, verified_files)
    from model.action4_best_registry import verify_reference, verify_loaded
    pinned = verify_reference(cfg["role"], cfg["parent_bundle"], cfg["parent_best_step"], cfg["parent_registry"])
    parent_model, parent_base, parent = load_bundle(str(path(cfg["parent_bundle"])), cfg["role"], "cpu")
    verify_loaded(pinned, parent)
    if parent["selected_step"] != cfg["parent_best_step"]:
        raise ValueError("Frozen parent selected step mismatch")
    adaptation_parent = parent if cfg.get("training_protocol") in FROZEN_PROTOCOLS else None
    frozen_backbone_verified=None
    del parent_model, parent_base, parent
    if cfg["role"] == "A0":
        model, base, counterpart = load_bundle(str(path(cfg["evaluation_counterpart_bundle"])), "B0", "cpu")
        deps["counterpart_selected_step"] = counterpart["selected_step"]
        deps["counterpart_config_id"] = counterpart["config_id"]
        if cfg.get("training_protocol") in FROZEN_PROTOCOLS:
            for result in reports:
                if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
                    from data.action4_sam_native_export import _v2_checked
                    cal=_v2_checked(result["report"]["calibration_receipt"])
                    if cal["run_binding"]["b0_model_sha256"]!=cfg["evaluation_counterpart_sha256"]:
                        raise ValueError("Corner evaluated B0 identity differs from loaded counterpart")
                    continue
                identity=result["report"]["identity"]
                if identity.get("B0_step")!=counterpart["selected_step"] or identity.get("B0_config_id")!=counterpart["config_id"]:
                    raise ValueError("Evaluated B0 identity differs from strictly loaded frozen counterpart")
        del model, base, counterpart
    if bundle is not None:
        candidate = path(bundle)
        if digest(candidate) != sha:
            raise ValueError("Copied/sealed bundle SHA mismatch before strict load")
        model, base, state = strict_load_selected(load_bundle,candidate,cfg)
        verify_state(state, cfg, status["best_step"])
        if cfg.get("training_protocol")==SAM_NATIVE_PROTOCOL:
            verify_sam_schema(state)
        if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
            if (state.get("schema")!="action4.stage_bundle.corner_candidate.v1" or not status.get("corner_inference_contract",{}).get("quality_ready")
                or state.get("corner_protocol")!=cfg.get("corner_protocol") or state.get("deployment_admitted") is not False):
                raise ValueError("Corner candidate and external calibrated seal contract differ")
        if adaptation_parent is not None:
            frozen_backbone_verified=verify_frozen_backbone(state,adaptation_parent)
        del model, base, state
        if digest(candidate) != sha:
            raise ValueError("Copied/sealed bundle changed during strict load")
    # Every file already passed real SHA verification above. Readonly files
    # use dev/inode/size/mtime/ctime/mode to detect replacement or writes during
    # this worker; writable files are rehashed. No cross-process hash cache.
    for filename, (expected, _) in verified_files.items():
        if verified_digest(path(filename), verified_files) != expected:
            raise ValueError("Immutable snapshot changed after strict load")
    unchanged(status["evidence_sha256"])
    if verify_dependencies(spec, cfg, verified_files) != deps_without_counterpart_identity(deps):
        raise ValueError("Dependency changed during verification")
    return {"status": status, "dependencies": deps, "config": cfg, "code_root": spec["code_root"],
            "run_spec_sha256": digest(path(run_spec)), "bundle_sha256": sha, "strict_load": bundle is not None,
            "frozen_backbone_verified": frozen_backbone_verified}


def deps_without_counterpart_identity(deps):
    return {k: v for k, v in deps.items() if k not in ("counterpart_selected_step", "counterpart_config_id")}


def seal_role(run_spec, destination, training_selection=None):
    spec, cfg, receipts, status = inspect(run_spec)
    if receipts is None:
        raise ValueError("Run is not complete")
    out, dest = path(cfg["output"]), path(destination, False)
    if not dest.is_relative_to(ROOT / "models/candidates") or dest.exists():
        raise ValueError("New models/candidates destination required")
    with completion_locks(out), training_selection_lock(training_selection):
        reports = []
        spec, cfg, receipts, status = inspect(run_spec, report_sink=reports)
        if training_selection is not None and receipts is not None:
            status = inspect_training_selection(spec, cfg, receipts, status, training_selection, reports)
        if receipts is None:
            raise ValueError("Completion disappeared")
        evidence = status["evidence_sha256"]
        source = path(status["source_checkpoint"]) if training_selection is not None else path(out / "best.pt")
        source_sha = digest(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + ".pending_" + str(os.getpid()))
        temp.mkdir(exist_ok=False)
        copied = temp / (cfg["role"] + ".pt")
        shutil.copyfile(source, copied)
        if digest(source) != source_sha or digest(copied) != source_sha:
            raise ValueError("Source changed while copying; pending evidence retained")
        verified = run_worker(run_spec, copied, source_sha) if training_selection is None else run_worker(run_spec, copied, source_sha, training_selection)
        if verified.get("strict_load") is not True or verified.get("bundle_sha256") != source_sha:
            raise ValueError("Immutable worker did not verify copied bundle identity")
        unchanged(evidence)
        if verified["status"]["evidence_sha256"] != evidence or digest(source) != source_sha:
            raise ValueError("Source or completion evidence changed")
        record = {"schema": TRAINING_SCHEMA if training_selection is not None else SCHEMA, "role": cfg["role"],
                  "bundle": str((dest / copied.name).relative_to(ROOT)), "sha256": source_sha,
                  "selected_step": status["best_step"], "config_id": canonical(cfg),
                  "run_spec": str(path(run_spec).relative_to(ROOT)),
                  "run_spec_sha256": verified["run_spec_sha256"], "actual_code_root": spec["code_root"],
                  "source_checkpoint": str(source.relative_to(ROOT)), "strict_load": True,
                  "completion": verified["status"], **verified["dependencies"],
                  "epochs": cfg["epochs"], "completed_optimizer_steps": receipts["TRAIN_COMPLETE.json"]["step"],
                  "score_schema": cfg.get("score_schema"), "accuracy_accepted": False,
                  "selection_scope": "full development selected; not real/test acceptance",
                  "default_models_modified": False, "time_unix": time.time(),
                  "sealer_source_sha256": digest(Path(__file__).resolve())}
        if cfg.get("training_protocol") in FROZEN_PROTOCOLS:
            if verified.get("frozen_backbone_verified") is not True:raise ValueError("Frozen backbone not verified")
            record["frozen_backbone_verified"]=True
            if cfg.get("continuation_segment") is not None:
                record["continuation_segment"]=cfg["continuation_segment"]
        if training_selection is not None:
            record.update(selection_purpose="training_candidate", deployment_release=False,
                          deployment_guard_passed=verified["status"]["deployment_guard_passed"],
                          training_selection=str(path(training_selection).relative_to(ROOT)))
        (temp / "SEALED_ROLE.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        copied.chmod(0o444)
        (temp / "SEALED_ROLE.json").chmod(0o444)
        if dest.exists():
            raise FileExistsError("Destination appeared concurrently")
        temp.rename(dest)
    return record


def verify_seal(seal_path):
    record = read_json(seal_path)
    if record.get("schema") not in (SCHEMA, TRAINING_SCHEMA) or record.get("strict_load") is not True:
        raise ValueError("Only validated role or explicit training-candidate seals are supported")
    selection = None
    if record["schema"] == TRAINING_SCHEMA:
        if record.get("selection_purpose") != "training_candidate" or record.get("deployment_release") is not False:
            raise ValueError("Training seal cannot become a deployment release")
        selection = record["training_selection"]
    if digest(path(record["run_spec"])) != record["run_spec_sha256"] or digest(path(record["bundle"])) != record["sha256"]:
        raise ValueError("Seal run/bundle identity changed")
    verified = run_worker(record["run_spec"], path(record["bundle"]), record["sha256"]) if selection is None else run_worker(record["run_spec"], path(record["bundle"]), record["sha256"], selection)
    status = verified["status"]
    if verified["config"].get("training_protocol") in FROZEN_PROTOCOLS:
        if record.get("frozen_backbone_verified") is not True or verified.get("frozen_backbone_verified") is not True:
            raise ValueError("Sealed frozen backbone proof differs")
    for key, actual in (("role", status["role"]), ("selected_step", status["best_step"]),
                        ("config_id", status["config_id"]), ("actual_code_root", verified["code_root"])):
        if record[key] != actual:
            raise ValueError("Seal metadata identity mismatch: " + key)
    if record["completion"] != status:
        raise ValueError("Seal completion evidence changed")
    if record["schema"] == TRAINING_SCHEMA and record.get("deployment_guard_passed") != status["deployment_guard_passed"]:
        raise ValueError("Training seal deployment report changed")
    for key, actual in verified["dependencies"].items():
        if record.get(key) != actual:
            raise ValueError("Seal dependency identity mismatch: " + key)
    return {"seal": record, "config": verified["config"], "seal_sha256": digest(path(seal_path))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sam-engineering-bundle",help="CPU proof only; never seals")
    parser.add_argument("--run-spec")
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verify-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bundle", help=argparse.SUPPRESS)
    parser.add_argument("--sha256", help=argparse.SUPPRESS)
    parser.add_argument("--verify-seal")
    parser.add_argument("--training-candidate-selection", help="Completed observer directory; does not promote deployment/default")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Expected verified project working directory")
    if args.verify_sam_engineering_bundle:
        if args.output or args.run_spec or args.verify_seal or args.verify_worker:
            parser.error("Engineering proof cannot be combined with sealing arguments")
        sys.path[:0]=[str(ROOT/"runtime/surface_deps"),str(ROOT/"src")]
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        result=verify_sam_engineering_bundle(args.verify_sam_engineering_bundle)
    elif args.verify_seal:
        result = verify_seal(args.verify_seal)
    elif args.verify_worker:
        result = verify_worker(args.run_spec, args.bundle, args.sha256, args.training_candidate_selection)
    else:
        if not args.run_spec:
            parser.error("--run-spec is required")
        _, _, receipts, status = inspect(args.run_spec)
        if receipts is None:
            result = status
        elif args.check_only:
            result = run_worker(args.run_spec, training_selection=args.training_candidate_selection)
        else:
            if not args.output:
                parser.error("--output is required for sealing")
            result = seal_role(args.run_spec, args.output, args.training_candidate_selection)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
