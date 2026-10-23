"""Resolve explicit archived checkpoint locations without changing frozen experiment configs."""
import json
from pathlib import Path

from data.action4_paths import PROJECT_ROOT as R, contained_path

LOCATION_MAP = "configs/model/action4_checkpoint_locations_v002.json"


def resolve_checkpoint(path):
    requested = contained_path(R, path, exists=False)
    if requested.is_file():
        return requested
    relative = str(requested.relative_to(R))
    manifest = contained_path(R, LOCATION_MAP, exists=True)
    locations = json.loads(manifest.read_text())
    if locations.get("schema") != "action4.checkpoint_locations.v1":
        raise ValueError("Unsupported checkpoint location map")
    target = locations["locations"].get(relative)
    if target is None:
        if relative in locations.get("deleted", {}):
            raise FileNotFoundError("Checkpoint intentionally retired: " + relative + "; " + locations["deleted"][relative])
        raise FileNotFoundError("No checkpoint or explicit relocation for " + relative)
    actual = contained_path(R, target, exists=True)
    if not actual.is_file() or actual.suffix not in (".pt", ".pth", ".ckpt"):
        raise ValueError("Relocated checkpoint must be a project-local weight file")
    return actual
