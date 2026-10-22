"""Run identity, configuration, checkpoint and device utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from data.action4_paths import PROJECT_ROOT, checked_root, contained_path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with contained_path(PROJECT_ROOT, path, exists=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def write_json_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def atomic_json(path: Path, value) -> None:
    """Replace only a run-owned evolving manifest/checkpoint index."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                     delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_config(path: str | Path) -> dict:
    path = contained_path(PROJECT_ROOT, path, exists=True)
    content = path.read_text(encoding="utf-8")
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        import yaml
        result = yaml.safe_load(content)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    return result


def merge_config(base: dict, changes: dict) -> dict:
    result = dict(base)
    for key, value in changes.items():
        result[key] = merge_config(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path, local_path: str | Path | None = None) -> dict:
    result = read_config(path)
    if local_path is not None:
        result = merge_config(result, read_config(local_path))
    return result


def input_path(root: Path, path: str | Path) -> Path:
    return contained_path(root, path, exists=True)


def output_path(root: Path, path: str | Path) -> Path:
    return contained_path(root, path)


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id) or run_id in (".", ".."):
        raise ValueError("run_id must be a safe single directory name")
    return run_id


def run_directories(root: Path, run_id: str, *, resume: bool = False) -> tuple[Path, Path]:
    validate_run_id(run_id)
    model_dir = output_path(root, Path("models/action4") / run_id)
    output_dir = output_path(root, Path("outputs/action4") / run_id)
    if resume:
        if not model_dir.is_dir() or not output_dir.is_dir():
            raise ValueError("resume requires the original run directories")
    else:
        if model_dir.exists() or output_dir.exists():
            raise FileExistsError("Use a new run_id; existing runs are not overwritten")
        model_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
    return model_dir, output_dir


def source_identity(root: Path) -> dict:
    sources = {}
    for directory in ("src/data", "src/model", "src/eval"):
        for path in sorted((root / directory).rglob("*.py")):
            sources[str(path.relative_to(root))] = sha256_file(path)
    result = {"files": sources, "sha256": canonical_hash(sources)}
    for name, args in (("commit", ["rev-parse", "HEAD"]), ("working_tree", ["status", "--porcelain"])):
        command = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        result[name] = command.stdout.strip() if command.returncode == 0 else None
    return result


def environment_identity(torch) -> dict:
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda_device_identities": [{"logical_index": i, "uuid": "GPU-" + str(torch.cuda.get_device_properties(i).uuid)}
                                       for i in range(torch.cuda.device_count())],
            "distributed_environment": {key: os.environ.get(key) for key in
                ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_DEBUG",
                 "NCCL_DEBUG_FILE", "RANK", "LOCAL_RANK", "WORLD_SIZE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "command": sys.argv}


def select_device(requested: str = "auto", local_rank: int | None = None):
    import torch
    value = "cuda" if torch.cuda.is_available() else "cpu" if requested == "auto" else requested
    if requested != "auto":
        value = requested
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no usable CUDA device is available")
        device = torch.device("cuda", local_rank if local_rank is not None else device.index or 0)
        torch.cuda.set_device(device)
    return device


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False


def capture_rng() -> dict:
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device topology changed; exact resume is not available")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def load_checkpoint(path: Path) -> dict:
    import torch
    checkpoint = torch.load(contained_path(PROJECT_ROOT, path, exists=True), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "action4.checkpoint.v1":
        raise ValueError("Expected an Action4 checkpoint with explicit architecture/configuration identity")
    return checkpoint


def save_checkpoint_atomic(path: Path, checkpoint: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def link_checkpoint_atomic(source: Path, target: Path) -> None:
    """Run-owned immutable checkpoint aliases share storage on this filesystem."""
    import uuid
    if source.resolve().parent != target.resolve(strict=False).parent:
        raise ValueError("Checkpoint aliases must stay in one run model directory")
    temporary = target.parent / (target.name + "." + uuid.uuid4().hex + ".tmp")
    os.link(source, temporary)
    os.replace(temporary, target)
