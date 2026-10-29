"""Pure preflight for loader4->12 continuation; never changes state or authorizes launch.

Inputs are already-read immutable config/source manifests and a checkpoint.
File content, leases and transition receipts must still be verified by the owner.
Ordinary strict resume is deliberately not changed by this module.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class LoaderContinuationPreflight:
    schema: str
    predecessor_config_id: str
    successor_config_id: str
    boundary_step: int
    next_epoch: int
    next_cursor: int
    data_update_attempt: int
    actual_consumed_units: int
    loader_workers_before: int
    loader_workers_after: int
    successor_cpu_ids: tuple
    required_runtime_switch_gates: tuple
    launch_allowed: bool = False


def canonical_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, "Invalid integer: " + name)
    return value


def _relative(value):
    _require(isinstance(value, str) and value and "\\" not in value, "Invalid project-relative path")
    p = PurePosixPath(value)
    _require(not p.is_absolute() and ".." not in p.parts and "." not in value.split("/"),
             "Path escapes project or is not normalized")
    _require(str(p) == value, "Noncanonical relative path")
    return p


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "Invalid SHA256")
    return value


def _code_root(cfg):
    spec = _relative(cfg["run_spec"])
    _require(len(spec.parts) == 4 and spec.parts[:2] == ("runtime", "action4_runs")
             and spec.name == "RUN_SPEC.json", "RunSpec location mismatch")
    return spec.parent / "source"


def _manifest(cfg, files):
    _require(isinstance(files, Mapping) and files, "Nonempty full source manifest required")
    normalized = {}
    for name, sha in files.items():
        normalized[str(_relative(name))] = _sha(sha)
    _require("resolved_config.json" in normalized, "Resolved config absent from source inventory")
    root = _code_root(cfg)
    identities = cfg.get("source_identity")
    _require(isinstance(identities, Mapping) and identities, "Consumed source identity missing")
    actual = {}
    for name, sha in identities.items():
        try:
            rel = _relative(name).relative_to(root)
        except ValueError as error:
            raise ValueError("Consumed source path outside its code root") from error
        _require(str(rel).startswith("src/") and rel.suffix == ".py", "Unexpected consumed source entry")
        actual[str(rel)] = _sha(sha)
    expected = {name: sha for name, sha in normalized.items()
                if name.startswith("src/") and name.endswith(".py")}
    _require(actual == expected and expected, "Consumed source and manifest differ")
    code_id = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    _require(cfg.get("project_code_sha256") == code_id
             and cfg.get("tested_project_code_sha256") == code_id, "Actual consumed source math identity mismatch")
    for key, hash_key in (("parent_bundle", "parent_bundle_sha256"),
                          ("evaluation_counterpart_bundle", "evaluation_counterpart_sha256")):
        try:
            rel = _relative(cfg[key]).relative_to(root)
        except ValueError as error:
            raise ValueError("Frozen dependency outside own snapshot") from error
        _require(str(rel).startswith("inputs/") and normalized.get(str(rel)) == _sha(cfg[hash_key]),
                 "Frozen dependency SHA is not manifest-bound")
    registry = _relative(cfg["parent_registry"])
    try:
        rel = registry.relative_to(root)
    except ValueError as error:
        raise ValueError("Parent registry outside snapshot") from error
    _require(str(rel).startswith("inputs/") and str(rel) in normalized, "Registry missing from manifest")
    return normalized


def _resources(cfg):
    _require(cfg["training_gpus"] == [7], "This continuation preserves physical train GPU7")
    _require(cfg["eval_gpus"] == [4, 6, 7, 7], "Evaluation placement must remain unchanged")
    for values in (cfg["training_gpus"], cfg["eval_gpus"]):
        _require(all(type(g) is int and 4 <= g <= 7 for g in values), "Forbidden GPU")
    affinity = cfg["process_cpu_affinity"]
    _require(isinstance(affinity, Mapping) and set(affinity) == {"train", "eval_0", "eval_1", "eval_2", "eval_3"},
             "Exact worker CPU affinity map required")
    threads = _integer(cfg["threads"], "threads", 1)
    workers = _integer(cfg["loader_workers"], "loader_workers", 1)
    eval_workers = _integer(cfg["eval_loader_workers"], "eval_loader_workers")
    union = set()
    for name, ids in affinity.items():
        _require(isinstance(ids, list) and ids and all(type(v) is int and 24 <= v <= 95 for v in ids),
                 "CPU must be logical IDs24-95")
        _require(len(ids) == len(set(ids)), "Duplicate CPU IDs")
        _require(not union.intersection(ids), "Worker CPU affinities overlap")
        union.update(ids)
        minimum = threads + workers if name == "train" else 2 + eval_workers
        _require(len(ids) >= minimum, "Insufficient worker CPU affinity")
    required = threads + workers + (2 + eval_workers) * 4 + 1
    reserved = _integer(cfg["reserved_cpu_budget"], "reserved_cpu_budget", 1)
    total = _integer(cfg["project_total_reserved_cpu"], "project_total_reserved_cpu", 1)
    _require(len(union) <= 72 and max(required, len(union) + 1) <= reserved <= total <= 72,
             "CPU budget does not cover workers plus controller or exceeds72")
    return tuple(sorted(union))


def validate_loader_continuation(predecessor, successor, checkpoint,
                                 predecessor_manifest, successor_manifest):
    """Return an explicitly non-launchable preflight description or raise ValueError.

    No input is mutated; tensor payloads are retained untouched. Full checkpoint
    SHA/actual tensor-state validation is an UNSATISFIED owner gate, not invented.
    """
    try:
        return _validate(predecessor, successor, checkpoint,
                         predecessor_manifest, successor_manifest)
    except (KeyError, TypeError) as error:
        raise ValueError("Missing or malformed continuation input: " + str(error)) from error


def _validate(old, new, state, old_files, new_files):
    _require(isinstance(old, Mapping) and isinstance(new, Mapping) and isinstance(state, Mapping),
             "Mappings required")
    for cfg in (old, new):
        for key, value in (("role", "A0"), ("mode", "full_training"), ("batch", 48),
                           ("micro", 2), ("world_size", 1), ("epochs", 80),
                           ("eval_every", 10), ("expected_units", 10400),
                           ("expected_eval_images", 650), ("require_immutable_source", True)):
            _require(cfg.get(key) == value and type(cfg.get(key)) is type(value), "Unsupported protocol: " + key)
    _require(old["loader_workers"] == 4 and new["loader_workers"] == 12, "Only explicit loader4->12 allowed")
    _require(set(old) == set(new), "Configuration keys changed")
    old_id, new_id = canonical_config(old), canonical_config(new)
    _require(old_id != new_id, "Independent successor identity required")
    old_root, new_root = _code_root(old), _code_root(new)
    _require(old_root != new_root and old["output"] != new["output"], "Successor must use new run/output")
    for cfg in (old, new):
        out = _relative(cfg["output"])
        _require(out.parts[0] == "outputs" and len(out.parts) > 1, "Owned output path required")
    allowed = {"loader_workers", "process_cpu_affinity", "reserved_cpu_budget",
               "project_total_reserved_cpu", "run_spec", "output", "source_identity",
               "parent_bundle", "evaluation_counterpart_bundle", "parent_registry",
               "integration_or_admission_receipt"}
    for key in old:
        if key not in allowed:
            _require(canonical_config(old[key]) == canonical_config(new[key]), "Semantic field changed: " + key)
    for cfg in (old, new):
        if "integration_or_admission_receipt" in cfg:
            _require(_relative(cfg["integration_or_admission_receipt"]) ==
                     _relative(cfg["output"]) / "ADMISSION.json", "Admission is not bound to output")
    # Relocation preserves relative dependency names, never substitutes another asset.
    for key in ("parent_bundle", "evaluation_counterpart_bundle", "parent_registry"):
        _require(_relative(old[key]).relative_to(old_root) == _relative(new[key]).relative_to(new_root),
                 "Dependency layout changed")
    a, b = _manifest(old, old_files), _manifest(new, new_files)
    # Config bytes change by definition; actual file/config binding remains an
    # explicit owner gate. Every other inventory file must be byte-identical.
    _require({k: v for k, v in a.items() if k != "resolved_config.json"} ==
             {k: v for k, v in b.items() if k != "resolved_config.json"},
             "Source or dependency bytes/inventory changed; separate admission required")
    _resources(old)
    cpus = _resources(new)
    _require(state.get("resumable") is True and state.get("role") == "A0"
             and state.get("config_id") == old_id, "Checkpoint is not exact predecessor state")
    _require(type(state.get("world_size")) is int and state["world_size"] == 1, "Checkpoint topology differs")
    step = _integer(state["step"], "step")
    epoch = _integer(state["epoch"], "epoch")
    cursor = _integer(state["cursor"], "cursor")
    skipped = _integer(state["no_supervision_update_count"], "no_supervision_update_count")
    attempts = _integer(state["data_update_attempt"], "data_update_attempt")
    _require(epoch < 80 and cursor < 10400 and cursor % 48 == 0, "Invalid or terminal next data cursor")
    expected_attempts = epoch * math.ceil(10400 / 48) + cursor // 48
    _require(attempts == expected_attempts and step + skipped == attempts,
             "Checkpoint attempts/optimizer/cursor traversal disagree")
    _require(step > 0, "Continuation requires a completed optimizer update")
    _require(isinstance(state["rng_by_rank"], (list, tuple)) and len(state["rng_by_rank"]) == 1
             and isinstance(state["rng_by_rank"][0], Mapping) and state["rng_by_rank"][0],
             "One rank RNG state required")
    _require(isinstance(state["training_model"], Mapping) and state["training_model"], "Training model state missing")
    optimizer = state["optimizer"]
    _require(isinstance(optimizer, Mapping) and isinstance(optimizer.get("state"), Mapping)
             and optimizer["state"] and isinstance(optimizer.get("param_groups"), list)
             and optimizer["param_groups"], "Optimizer state missing")
    scheduler = state["scheduler"]
    _require(isinstance(scheduler, Mapping) and type(scheduler.get("last_epoch")) is int
             and scheduler["last_epoch"] == step, "Scheduler step differs")
    if "attempt_id" in state:
        _require(isinstance(state["attempt_id"], str) and state["attempt_id"], "Invalid attempt identity")
    gates = tuple((name, "UNSATISFIED") for name in (
        "explicit_user_switch_authorization", "live_single_writer_ownership",
        "immutable_files_and_checkpoint_content_sha", "actual_model_optimizer_rng_state_validation",
        "paired_loader_training_semantics_and_throughput", "safe_controller_transition_and_checkpoint_boundary",
        "original_eval_prefix_drain_and_identity", "lineage_observer_and_sealer_support",
        "project_live_CPU_GPU_lease_and_headroom", "tested_rollback_controller"))
    return LoaderContinuationPreflight("action4.loader_continuation_preflight.v1", old_id, new_id,
                                       step, epoch, cursor, attempts, epoch * 10400 + cursor,
                                       4, 12, cpus, gates)
