"""Identity-pinned best checkpoint selection for explicit future training/inference."""
import hashlib
import json
from functools import lru_cache

from data.action4_paths import PROJECT_ROOT as R, contained_path
from model.action4_checkpoint_paths import resolve_checkpoint

DEFAULT_REGISTRY = "configs/model/action4_best_registry_v003.json"


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


def selected_entry_from_bundle(role, checkpoint, expected_sha256, expected_step):
    """Build the complete existing registry contract from the actual selected bundle."""
    import torch
    path=resolve_checkpoint(checkpoint)
    identity=_identity(path)
    if _digest(path,identity)!=expected_sha256:
        raise ValueError("Selected parent SHA mismatch")
    state=torch.load(path,map_location="cpu",weights_only=True)
    if state.get("role")!=role or state.get("selected_step")!=expected_step:
        raise ValueError("Selected parent role/step mismatch")
    for field in ("architecture","config_id"):
        if field not in state:raise ValueError("Selected parent lacks "+field)
    entry=dict(path=str(path.relative_to(R)),sha256=expected_sha256,bytes=identity[2],
        role=role,selected_step=expected_step,architecture=state["architecture"],
        config_id=state["config_id"],training_config_id=state.get("training_config_id",state["config_id"]),
        pose_method=state.get("pose_method"))
    verify_loaded(entry,state)
    if _identity(path)!=identity:raise ValueError("Parent changed during registry construction")
    return entry
