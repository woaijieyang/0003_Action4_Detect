"""Thin SAM entrypoint; native execution requires a sealed copied source tree."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data/zy/vln/a-test")
source = Path(__file__).resolve().parents[1]
if not source.is_relative_to(ROOT):
    raise RuntimeError("Entry escaped project")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if os.environ.get("ACTION4_PROJECT_ROOT", str(ROOT)) != str(ROOT):
    raise RuntimeError("Unexpected project-root override")
os.environ["ACTION4_PROJECT_ROOT"] = str(ROOT)
seal = source / "SNAPSHOT_MANIFEST.json"
if seal.exists():
    manifest = json.loads(seal.read_text())
    payload = {k: v for k, v in manifest.items() if k != "identity"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if expected != manifest["identity"]:
        raise RuntimeError("Snapshot manifest changed")
    for relative, sha in manifest["files_sha256"].items():
        p = source / relative
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(source) or p.stat().st_mode & 0o222:
            raise RuntimeError("Snapshot file escaped or is writable: " + relative)
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise RuntimeError("Snapshot file changed: " + relative)
    if manifest["entrypoint"] != "scripts/run_action4_sam_full_dev.py":
        raise RuntimeError("Unexpected sealed entrypoint")
    # Eliminate live-source fallback before importing any project package.
    forbidden = {ROOT, ROOT / "src", ROOT / "upstream/sam3"}
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() not in forbidden]
    sys.path.insert(0, str(source / "upstream/sam3"))
    sys.path.insert(0, str(source / "src"))
else:
    if "--execute-native" in sys.argv:
        raise RuntimeError("Live native SAM entry revoked: use the schema-configured sealed snapshot entrypoint")
    sys.path.insert(0, str(source / "src"))

from eval.action4_sam_image_eval import main
if __name__ == "__main__":
    main()
