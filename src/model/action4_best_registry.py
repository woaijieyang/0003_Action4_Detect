"""Identity-pinned best checkpoint selection for explicit future training/inference."""
import hashlib
import json
from functools import lru_cache

from data.action4_paths import PROJECT_ROOT as R, contained_path
from model.action4_checkpoint_paths import resolve_checkpoint

DEFAULT_REGISTRY = "configs/model/action4_best_registry_v002.json"


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=16)
def _digest(path, identity):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if _identity(path) != identity:
        raise ValueError("Checkpoint changed during identity verification")
    return digest.hexdigest()


def verify_reference(role, checkpoint, expected_step=None, registry=DEFAULT_REGISTRY):
    selection = json.loads(contained_path(R, registry, exists=True).read_text())
    if selection.get("schema") != "action4.selected_parent_registry.v1" or role not in ("A0", "B0"):
        raise ValueError("Unsupported best registry or stage role")
    entry = selection["entries"][role]
    if entry["role"] != role:
        raise ValueError("Best registry role mismatch")
    if expected_step is not None and expected_step != entry["selected_step"]:
        raise ValueError("Requested best step differs from selected parent")
    expected = resolve_checkpoint(entry["path"])
    actual = resolve_checkpoint(checkpoint)
    if actual != expected:
        raise ValueError("Checkpoint path differs from the explicitly selected stage best")
    identity = _identity(actual)
    if identity[2] != entry["bytes"] or _digest(actual, identity) != entry["sha256"]:
        raise ValueError("Selected checkpoint bytes changed")
    return entry


def verify_loaded(entry, state):
    for field in ("role", "selected_step", "architecture", "config_id"):
        if state.get(field) != entry[field]:
            raise ValueError("Loaded best identity differs: " + field)
    if state.get("training_config_id", state["config_id"]) != entry["training_config_id"]:
        raise ValueError("Loaded source training identity differs")
    if state.get("pose_method") != entry.get("pose_method"):
        raise ValueError("Loaded pose decoder differs")
