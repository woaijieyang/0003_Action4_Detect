"""Verify immutable component identities without granting scene acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/zy/vln/a-test")


def bounded(path: str | Path) -> Path:
    value = Path(path)
    value = (value if value.is_absolute() else ROOT / value).resolve()
    if not value.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen module path escapes project root")
    return value


def verify(registry: str | Path) -> dict:
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError("Run from the authorized project root")
    path = bounded(registry)
    definition = json.loads(path.read_text())
    if definition["schema"] != "action4.frozen_components.v1":
        raise ValueError("Unknown frozen-component schema")
    checked = 0
    components = []
    for entry in definition["components"]:
        if not entry["files"]:
            raise ValueError("Empty component file identity")
        for filename, expected in entry["files"].items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Invalid SHA256 identity")
            actual = hashlib.sha256(bounded(filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Locked component changed: " + filename)
            checked += 1
        components.append({"id": entry["id"], "status": entry["status"]})
    if not checked:
        raise ValueError("No frozen files verified")
    return {
        "status": "FROZEN_COMPONENT_IDENTITIES_MATCH",
        "registry_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "files_verified": checked,
        "components": components,
        "scene_or_training_acceptance_granted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/data/action4_frozen_modules_v002.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.registry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
