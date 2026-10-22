"""Exact-resume integrity and idempotent checkpoint publication.

Legacy v1 weight files remain readable for explicitly permitted weights-only
uses. Exact resume requires evidence emitted by this implementation; missing
historical evidence is rejected rather than inferred from present files.
"""
from __future__ import annotations
import math
import re
from pathlib import Path
import numpy as np
from .action4_runtime import canonical_hash, sha256_file, load_checkpoint, link_checkpoint_atomic

INTEGRITY_SCHEMA = "action4.resume_integrity.v1"
EPOCH_NAME = re.compile(r"epoch_(\d{3})_step_(\d{8})\.pt")


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Checkpoint " + label + " must be an integer >= " + str(minimum))
    return value


def _finite_tree(value, label):
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise ValueError("Checkpoint nonfinite tensor: " + label)
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, label + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _finite_tree(item, label + "." + str(i))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Checkpoint nonfinite value: " + label)


def validate_resume_checkpoint(state, *, optimizer=None, scheduler=None, device=None):
    """Validate internal consistency before restoring mutable training state."""
    import random
    import torch
    proof = state.get("resume_integrity")
    if not isinstance(proof, dict) or proof.get("schema_version") != INTEGRITY_SCHEMA:
        raise ValueError("Exact resume requires resume_integrity v1; legacy checkpoints lack recovery evidence")
    required = ("model", "optimizer", "scheduler", "scaler", "initialization", "identity", "config",
                "sampler", "rng_by_rank", "roi_rng_by_rank", "top_checkpoints", "mandatory_snapshots")
    if any(key not in state for key in required):
        raise ValueError("Incomplete exact-resume checkpoint state")
    if state.get("epoch_boundary") is not True:
        raise ValueError("Exact resume requires a complete epoch boundary")
    identity, config = state["identity"], state["config"]
    stage = state.get("stage")
    if stage not in ("D0", "A0", "B0", "C1") or stage != identity.get("stage"):
        raise ValueError("Checkpoint stage identity mismatch")
    if canonical_hash(config) != identity.get("config_sha256"):
        raise ValueError("Checkpoint config identity mismatch")
    claim = "component_diagnostic" if identity.get("component_smoke") or identity.get("allow_diagnostic") or stage == "D0" else "accepted_data_model_training"
    if state.get("claim_scope") != claim:
        raise ValueError("Checkpoint claim scope mismatch")
    epoch = _integer(state.get("next_epoch"), "next_epoch", 1)
    step = _integer(state.get("global_step"), "global_step")
    updates = _integer(proof.get("updates_per_epoch"), "updates_per_epoch", 1)
    skipped = _integer(proof.get("nonfinite_updates"), "nonfinite_updates")
    epochs = _integer(config["stages"][stage].get("epochs"), "planned epochs", 1)
    if epoch > epochs or identity.get("planned_updates") != epochs * updates or step != epoch * updates - skipped:
        raise ValueError("Checkpoint epoch/global_step/planned update consistency failure")
    if skipped > config["stages"][stage].get("max_nonfinite_updates", 0):
        raise ValueError("Checkpoint skipped-update budget exceeded")
    world = _integer(identity.get("world_size"), "world_size", 1)
    sampler = {"next_epoch": epoch, "seed": config["runtime"].get("seed", 20261018),
               "global_batch": identity.get("global_batch"), "micro_batch": identity.get("micro_batch"), "world_size": world}
    if state["sampler"] != sampler:
        raise ValueError("Checkpoint sampler identity mismatch")
    saved_scheduler = state["scheduler"]
    if saved_scheduler.get("last_epoch") != step or saved_scheduler.get("_step_count") != step + 1:
        raise ValueError("Checkpoint scheduler progress differs from optimizer update counter")
    opt = state["optimizer"]
    groups = opt.get("param_groups", [])
    ids = [index for group in groups for index in group["params"]]
    if not groups or len(set(ids)) != len(ids) or not set(opt.get("state", {})).issubset(ids):
        raise ValueError("Checkpoint optimizer parameter identities invalid")
    if saved_scheduler.get("_last_lr") != [group.get("lr") for group in groups]:
        raise ValueError("Checkpoint optimizer LR differs from scheduler")
    steps = []
    for value in opt["state"].values():
        counter = value.get("step")
        counter = counter.item() if isinstance(counter, torch.Tensor) and counter.numel() == 1 else counter
        if type(counter) not in (int, float) or not math.isfinite(counter) or counter != int(counter) or not 0 < counter <= step:
            raise ValueError("Checkpoint AdamW step counter invalid")
        steps.append(int(counter))
        if not all(k in value for k in ("exp_avg", "exp_avg_sq")) or (value["exp_avg_sq"] < 0).any().item():
            raise ValueError("Checkpoint AdamW moments invalid")
    if (step > 0 and (not steps or max(steps) != step)) or (step == 0 and steps):
        raise ValueError("Checkpoint optimizer progress differs from global_step")
    if optimizer is not None:
        if len(groups) != len(optimizer.param_groups):
            raise ValueError("Checkpoint optimizer group count changed")
        for saved, current in zip(groups, optimizer.param_groups):
            if saved.get("group_name") != current.get("group_name") or len(saved["params"]) != len(current["params"]):
                raise ValueError("Checkpoint optimizer group identity changed")
            for key in ("weight_decay", "betas", "eps", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
                if saved.get(key) != current.get(key):
                    raise ValueError("Checkpoint optimizer hyperparameter changed: " + key)
            for index, parameter in zip(saved["params"], current["params"]):
                moments = opt["state"].get(index)
                if moments and any(moments[k].shape != parameter.shape for k in ("exp_avg", "exp_avg_sq")):
                    raise ValueError("Checkpoint optimizer moment shape changed")
    if scheduler is not None:
        if saved_scheduler.get("base_lrs") != scheduler.base_lrs:
            raise ValueError("Checkpoint scheduler base learning rates changed")
        expected_lrs = [base * function(step) for base, function in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
        if saved_scheduler["_last_lr"] != expected_lrs:
            raise ValueError("Checkpoint learning rate differs from declared schedule at saved step")
    _finite_tree(state["model"], "model")
    _finite_tree(opt, "optimizer")
    _finite_tree(state["scaler"], "scaler")
    if proof.get("device_type") not in ("cpu", "cuda") or type(proof.get("amp_enabled")) is not bool:
        raise ValueError("Checkpoint lacks device/AMP restoration identity")
    if device is not None and proof["device_type"] != device.type:
        raise ValueError("Exact resume device type changed")
    if bool(state["scaler"]) != proof["amp_enabled"]:
        raise ValueError("Checkpoint AMP enabled state differs from saved scaler")
    scale = state["scaler"].get("scale")
    if state["scaler"]:
        scaler = state["scaler"]
        if type(scale) not in (int, float) or scale <= 0:
            raise ValueError("Checkpoint AMP scale invalid")
        if (type(scaler.get("growth_factor")) not in (int, float) or scaler["growth_factor"] <= 1
                or type(scaler.get("backoff_factor")) not in (int, float) or not 0 < scaler["backoff_factor"] < 1):
            raise ValueError("Checkpoint AMP growth/backoff invalid")
        interval = _integer(scaler.get("growth_interval"), "AMP growth interval", 1)
        tracker = _integer(scaler.get("_growth_tracker"), "AMP growth tracker")
        if tracker >= interval:
            raise ValueError("Checkpoint AMP growth tracker invalid")
    if len(state["rng_by_rank"]) != world or len(state["roi_rng_by_rank"]) != world:
        raise ValueError("Checkpoint per-rank RNG topology mismatch")
    if any(not isinstance(rng.get("cuda"), list) for rng in state["rng_by_rank"]):
        raise ValueError("Checkpoint CUDA RNG state must be a list")
    expected_cuda_shapes = None
    if device is not None and device.type == "cuda":
        count = torch.cuda.device_count()
        if count < 1 or any(len(rng.get("cuda", [])) != count for rng in state["rng_by_rank"]):
            raise ValueError("Checkpoint CUDA RNG visible device count changed")
        expected_cuda_shapes = [torch.cuda.get_rng_state(i).shape for i in range(count)]
    for rng, roi in zip(state["rng_by_rank"], state["roi_rng_by_rank"]):
        random.Random().setstate(rng["python"])
        n = rng["numpy"]
        np.random.RandomState().set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
        torch.Generator().set_state(rng["torch"].cpu())
        cuda = rng.get("cuda")
        if not isinstance(cuda, list) or any(not isinstance(x, torch.Tensor) or x.dtype != torch.uint8
                or x.ndim != 1 or x.numel() == 0 or x.device.type != "cpu" or not x.is_contiguous() for x in cuda):
            raise ValueError("Checkpoint CUDA RNG state invalid")
        if proof["device_type"] == "cuda" and not cuda:
            raise ValueError("Checkpoint CUDA RNG state missing")
        if expected_cuda_shapes is not None and [x.shape for x in cuda] != expected_cuda_shapes:
            raise ValueError("Checkpoint CUDA RNG tensor shape differs from runtime")
        np.random.default_rng().bit_generator.state = roi


def _top_entry(entry):
    name = entry.get("file", "")
    match = EPOCH_NAME.fullmatch(name)
    if not match or type(entry.get("epoch")) is not int or int(match[1]) != entry["epoch"]:
        raise ValueError("Invalid retained checkpoint identity")
    return name, int(match[1]), int(match[2])


def verify_retained(top, latest, state, *, repair=False):
    """Bind historical bytes; only current committed alias is reconstructable."""
    latest_sha = sha256_file(latest)
    latest_name = f"epoch_{state['next_epoch']:03d}_step_{state['global_step']:08d}.pt"
    if not isinstance(top, list) or not 1 <= len(top) <= 3 or len({e.get("file") for e in top}) != len(top):
        raise ValueError("Invalid retained checkpoint index")
    for entry in top:
        name, epoch, step = _top_entry(entry)
        if epoch > state["next_epoch"] or step > state["global_step"]:
            raise ValueError("Retained checkpoint is newer than the committed state")
        current = name == latest_name
        expected = latest_sha if current else entry.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Historical retained checkpoint lacks immutable SHA256 evidence")
        path = latest.parent / name
        if not path.is_file():
            if not current:
                raise ValueError("An older retained checkpoint is missing; latest state cannot reconstruct it")
            if repair:
                link_checkpoint_atomic(latest, path)
            else:
                continue
        if sha256_file(path) != expected:
            raise ValueError("Retained checkpoint content identity changed: " + name)
        if not current:
            saved = load_checkpoint(path)
            if any(saved.get(k) != state.get(k) for k in ("run_id", "stage", "claim_scope", "identity", "config")) or saved.get("next_epoch") != epoch or saved.get("global_step") != step or saved.get("epoch_boundary") is not True:
                raise ValueError("Retained checkpoint run/stage/epoch identity changed")
        entry["sha256"] = expected
    return latest_sha


def bind_previous_top(top, model_dir, current_name):
    for entry in top:
        if entry["file"] == current_name:
            continue
        if not entry.get("sha256") or sha256_file(model_dir / entry["file"]) != entry["sha256"]:
            raise ValueError("Retained checkpoint changed before commit")


def recover_epoch20(state, latest, *, repair=False):
    """Replay mandatory publication only from its exact epoch20 commit."""
    if state["stage"] != "B0" or state["next_epoch"] < 20:
        return
    if state.get("config", {}).get("stages", {}).get("B0", {}).get("eval_every_steps"):
        expected = latest.parent / f"epoch_020_step_{20 * state['resume_integrity']['updates_per_epoch']:08d}.pt"
        if not expected.is_file():raise ValueError("Asynchronous B0 epoch20 recovery snapshot missing")
        retained = load_checkpoint(expected)
        if retained.get("identity") != state["identity"] or retained.get("next_epoch") != 20:
            raise ValueError("Asynchronous epoch20 snapshot identity mismatch")
        return
    snapshots = state.get("mandatory_snapshots", {})
    names = ("epoch_020.pt", "best_through_epoch020.pt")
    if set(snapshots) != set(names):
        raise ValueError("Checkpoint lacks epoch20 snapshot identity evidence")
    at_twenty = state["next_epoch"] == 20
    for name in names:
        entry = snapshots[name]
        source_name = entry["source_file"]
        if name == "epoch_020.pt":
            if source_name != f"epoch_020_step_{entry['global_step']:08d}.pt":
                raise ValueError("Invalid epoch20 snapshot source")
        elif not EPOCH_NAME.fullmatch(source_name):
            raise ValueError("Invalid epoch20 best snapshot source")
        source = latest if name == "epoch_020.pt" else latest.parent / source_name
        expected = entry.get("sha256")
        if at_twenty:
            actual = sha256_file(source)
            if expected is not None and expected != actual:
                raise ValueError("Epoch20 snapshot source identity changed")
            expected = actual
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Epoch20 snapshot lacks immutable SHA256 evidence")
        target = latest.parent / name
        if not target.is_file():
            if not at_twenty:
                raise ValueError("Missing historical epoch20 snapshot cannot be reconstructed from later weights")
            if repair:
                link_checkpoint_atomic(source, target)
            else:
                continue
        if sha256_file(target) != expected:
            raise ValueError("Epoch20 snapshot content identity changed")
        entry["sha256"] = expected


def synchronized_rank0_action(action, *, rank, distributed, label):
    """Publish rank0 success/error before another rank can consume its outputs."""
    import torch
    error = None
    if rank == 0:
        try:
            action()
        except Exception as exc:
            if not distributed:
                raise
            error = {"type": type(exc).__name__, "message": str(exc)}
    if distributed:
        message = [error]
        torch.distributed.broadcast_object_list(message, src=0)
        error = message[0]
    if error is not None:
        raise ValueError(label + " failed on rank0: " + error["type"] + ": " + error["message"])


def synchronize_checkpoint_recovery(state, latest, *, rank, distributed):
    """One repair writer; every rank verifies only after repair is published."""
    import torch
    def repair():
        verify_retained(state["top_checkpoints"], latest, state, repair=True)
        recover_epoch20(state, latest, repair=True)
    synchronized_rank0_action(repair, rank=rank, distributed=distributed, label="Checkpoint recovery")
    error = None
    try:
        verify_retained(state["top_checkpoints"], latest, state)
        recover_epoch20(state, latest)
    except Exception as exc:
        if not distributed:
            raise
        error = {"rank": rank, "type": type(exc).__name__, "message": str(exc)}
    errors = [error]
    if distributed:
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
    failed = [item for item in errors if item is not None]
    if failed:
        raise ValueError("Checkpoint recovery verification failed: " + str(failed))


def cleanup_owned_process_group(function):
    """Release a group created by this train call, including pre-loop failures."""
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        already_initialized = torch.distributed.is_initialized()
        try:
            return function(*args, **kwargs)
        finally:
            if not already_initialized and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    return wrapped
