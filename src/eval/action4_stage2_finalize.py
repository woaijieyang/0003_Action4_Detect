"""Combine validated independent role seals without mixing immutable code roots.

Formal entry: --a0-seal .../SEALED_ROLE.json --b0-seal .../SEALED_ROLE.json.
The legacy config entry remains for non-immutable engineering integrations only.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from data.action4_paths import PROJECT_ROOT as R, contained_path


def _path(value, exists=True):
    p = R / value
    if p.is_symlink() or not p.resolve(strict=exists).is_relative_to(R.resolve()):
        raise ValueError("Pair path outside project or symlink")
    return p.resolve(strict=exists)


def _digest(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_role(seal_path):
    command = [str(R / ".venv/bin/python"), "-I", str(R / "scripts/action4_seal_completed_role.py"),
               "--verify-seal", str(_path(seal_path))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    result = subprocess.run(command, cwd=R, env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def inspect_pair(a0_seal, b0_seal, verifier=None):
    """Metadata fixtures may inject a verifier; production always runs subprocesses."""
    verifier = _verify_role if verifier is None else verifier
    verified = [verifier(_path(value)) for value in (a0_seal, b0_seal)]
    a, b = (item["seal"] for item in verified)
    ac, bc = (item["config"] for item in verified)
    for role, record, cfg, item, original in zip(("A0", "B0"), (a, b), (ac, bc), verified, (a0_seal, b0_seal)):
        if record.get("schema") != "action4.sealed_independent_role.v2" or record.get("strict_load") is not True:
            raise ValueError("Pair requires fully validated v2 role seals")
        if record["role"] != role or cfg["role"] != role or cfg["mode"] != "full_training":
            raise ValueError("Pair role/formal mode mismatch")
        if _digest(_path(original)) != item["seal_sha256"] or _digest(_path(record["bundle"])) != record["sha256"]:
            raise ValueError("Verified seal/bundle changed")
        if record["validation_receipt"] != cfg["validation_receipt"] or record["validation_receipt_sha256"] != cfg["validation_receipt_sha256"]:
            raise ValueError("Seal/config dataset identity mismatch")
        if record.get("score_schema") != cfg.get("score_schema"):
            raise ValueError("Seal/config score identity mismatch")
    if a["validation_receipt"] != b["validation_receipt"] or a["validation_receipt_sha256"] != b["validation_receipt_sha256"]:
        raise ValueError("Pair dataset identity mismatch")
    if a.get("evaluation_counterpart_sha256") != b["sha256"] or ac.get("evaluation_counterpart_sha256") != b["sha256"]:
        raise ValueError("A0 fixed counterpart is not the paired sealed B0")
    if a.get("counterpart_selected_step") != b["selected_step"] or a.get("counterpart_config_id") != b["config_id"]:
        raise ValueError("A0 fixed counterpart step/config is not paired B0")
    if ac["b0_scoring_policy"] != bc["evaluation_scoring_policy"]:
        raise ValueError("Paired B0 scoring policy differs from A0 evaluation")
    return verified


def seal_pair(a0_seal, b0_seal, output):
    verified = inspect_pair(a0_seal, b0_seal)
    a, b = (item["seal"] for item in verified)
    ac, bc = (item["config"] for item in verified)
    out = _path(output, False)
    if not out.is_relative_to(R / "models/candidates") or out.exists():
        raise ValueError("New project-local models/candidates pair destination required")
    # No current best/default lookup and no checkpoint rewriting. Paths point to
    # separately sealed byte identities, each strict-loaded by its own code root.
    profile = {"a0_bundle": a["bundle"], "a0_sha256": a["sha256"],
               "b0_bundle": b["bundle"], "b0_sha256": b["sha256"],
               "b0_scoring_policy": bc["evaluation_scoring_policy"],
               "intrinsics_context": ac["intrinsics_context"],
               "demo_reference": ac.get("demo_reference", "data/action4_reference/demo_standard.mp4"),
               "validation_receipt": a["validation_receipt"],
               "validation_receipt_sha256": a["validation_receipt_sha256"],
               "a0_score_schema": ac["score_schema"],
               "role_seals": {"A0": str(_path(a0_seal).relative_to(R)), "B0": str(_path(b0_seal).relative_to(R))},
               "scope": "formal independent development selection; no real/test acceptance or default promotion"}
    records = {"schema": "action4.sealed_role_pair.v1", "mode": "full_training",
               "roles": {"A0": a, "B0": b},
               "role_seal_sha256": {role: item["seal_sha256"] for role, item in zip(("A0", "B0"), verified)},
               "separate_immutable_loaders": True, "accuracy_accepted": False,
               "default_models_modified": False, "pair_builder_sha256": _digest(Path(__file__).resolve())}
    for original, item in zip((a0_seal, b0_seal), verified):
        if _digest(_path(original)) != item["seal_sha256"] or _digest(_path(item["seal"]["bundle"])) != item["seal"]["sha256"]:
            raise ValueError("Pair source changed before publication")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".pending_" + str(os.getpid()))
    temp.mkdir(exist_ok=False)
    for name, value in (("PROFILE.json", profile), ("SEALED.json", records)):
        p = temp / name
        p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        p.chmod(0o444)
    if out.exists():
        raise FileExistsError("Pair destination appeared concurrently")
    temp.rename(out)
    return profile


def seal(a0_config, b0_config, output, allow_integration=False):
    configs = [json.loads(_path(p).read_text()) for p in (a0_config, b0_config)]
    if any(c.get("require_immutable_source") or c.get("run_spec") or c.get("mode") == "full_training" for c in configs):
        raise ValueError("Formal/immutable config pair entry is unsupported; seal each role with its own RUN_SPEC, then use --a0-seal/--b0-seal")
    return _legacy_seal(a0_config, b0_config, output, allow_integration)


def _legacy_seal(a0_config,b0_config,output,allow_integration=False):
    import torch, math
    from model.training.action4_runtime import canonical_hash, save_checkpoint_atomic, atomic_json
    from model.action4_release_bundle import load_bundle
    configs=[json.loads(contained_path(R,p,exists=True).read_text()) for p in [a0_config,b0_config]]
    if [c["role"] for c in configs]!=["A0","B0"]:raise ValueError("Explicit independent stage order required")
    if configs[0]["mode"]!=configs[1]["mode"] or configs[0]["validation_receipt"]!=configs[1]["validation_receipt"]:
        raise ValueError("Paired stages use different mode/dataset")
    from model.training.action4_run_snapshot import verify_running_snapshot
    for cfg in configs:verify_running_snapshot(cfg)
    mode=configs[0]["mode"]
    if mode=="integration" and not allow_integration:raise ValueError("Engineering checkpoints cannot be sealed as formal models")
    packages=[];receipts=[]
    for cfg in configs:
        root=contained_path(R,cfg["output"],exists=True)
        train=json.loads((root/"TRAIN_COMPLETE.json").read_text())
        complete=json.loads((root/"CONTROLLER_COMPLETE.json").read_text())
        eval_complete=json.loads((root/"EVAL_COMPLETE.json").read_text())
        best=json.loads((root/"BEST.json").read_text())
        if not (train["attempt_id"]==complete["attempt_id"]==eval_complete["attempt_id"]):
            raise ValueError("Completion receipts belong to different attempts")
        if train["role"]!=cfg["role"] or complete["role"]!=cfg["role"] or train["mode"]!=mode or complete["mode"]!=mode:
            raise ValueError("Completion role/mode mismatch")
        expected_steps=cfg["epochs"]*math.ceil(cfg["expected_units"]/cfg["batch"])
        if train["epochs"]!=cfg["epochs"] or train["step"]+train.get("no_supervision_update_count",0)!=expected_steps or train["units_per_epoch"]!=cfg["expected_units"]:
            raise ValueError("Completed training coverage differs from configuration")
        if mode=="full_training" and any(r.get("config_id")!=canonical_hash(cfg) for r in [train,complete,eval_complete]):
            raise ValueError("Formal receipt lacks the exact config identity")
        if mode=="full_training" and (train["epochs"]<80 or not train["formal_training_complete"]):
            raise ValueError("Full training incomplete")
        if complete["best_step"]!=best["step"] or eval_complete["best_step"]!=best["step"]:
            raise ValueError("Selection was not reconciled after all eval workers")
        state=torch.load(root/"best.pt",map_location="cpu",weights_only=True)
        if state["role"]!=cfg["role"] or state["config_id"]!=canonical_hash(cfg) or state["selected_step"]!=best["step"]:
            raise ValueError("Selected model identity mismatch")
        state.update(selection_scope="development-selected independent stage; engineering only" if mode=="integration" else "development-selected; real acceptance remains separate",
                     completed_run_epochs=train["epochs"],completed_run_optimizer_steps=train["step"],
                     selected_report=best["raw_report"],source_checkpoint=str((root/"best.pt").relative_to(R)),
                     accuracy_accepted=False)
        packages.append(state);receipts.append(str((root/"TRAIN_COMPLETE.json").relative_to(R)))
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False);records={}
    for cfg,state in zip(configs,packages):
        folder=out/cfg["role"];folder.mkdir();path=folder/"best.pt";save_checkpoint_atomic(path,state)
        model,base,loaded=load_bundle(str(path),cfg["role"],"cpu")
        records[cfg["role"]]={"path":str(path.relative_to(R)),"step":state["selected_step"],
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"independent_strict_load":True}
        del model,base,loaded
    profile={"a0_bundle":records["A0"]["path"],"b0_bundle":records["B0"]["path"],
        "b0_scoring_policy":configs[1]["evaluation_scoring_policy"],
        "intrinsics_context":configs[0].get("intrinsics_context","outputs/action4_execution_source_release_v006/source_context.json"),
        "demo_reference":"data/action4_reference/demo_standard.mp4","validation_receipt":configs[0]["validation_receipt"],
        "training_completion_receipts":receipts,"scope":mode+" paired selected weights; no automatic production promotion"}
    atomic_json(out/"PROFILE.json",profile)
    atomic_json(out/"SEALED.json",{"mode":mode,"models":records,"stage1_registry_unchanged":True,"accuracy_accepted":False})
    return profile


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-config")
    parser.add_argument("--b0-config")
    parser.add_argument("--a0-seal")
    parser.add_argument("--b0-seal")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-integration", action="store_true")
    args = parser.parse_args()
    if args.a0_seal or args.b0_seal:
        if not (args.a0_seal and args.b0_seal) or args.a0_config or args.b0_config or args.allow_integration:
            parser.error("Use exactly --a0-seal and --b0-seal for formal role pairs")
        result = seal_pair(args.a0_seal, args.b0_seal, args.output)
    else:
        if not (args.a0_config and args.b0_config):
            parser.error("Provide both role seals, or both legacy engineering configs")
        result = seal(args.a0_config, args.b0_config, args.output, args.allow_integration)
    print(json.dumps(result))
