"""Check physical default/archival layout and all explicit legacy resolutions."""
import json
import re
from data.action4_paths import PROJECT_ROOT as R
from model.action4_checkpoint_paths import resolve_checkpoint, LOCATION_MAP

def main():
    assert sorted(p.name for p in (R / "models/default").iterdir()) == ["A0.pt", "B0.pt"]
    mapping = json.loads((R / LOCATION_MAP).read_text())["locations"]
    for old, new in mapping.items():
        assert not (R / old).exists(), "Stale physical alias remains"
        assert resolve_checkpoint(old) == R / new
        assert resolve_checkpoint(new) == R / new
        if new.startswith("archive/"):
            assert re.match(r"\d{8}-\d{6}-.+\.pt$", (R / new).name)
            assert "best" not in (R / new).name.lower()
    try:
        resolve_checkpoint("models/nonexistent_unregistered_checkpoint.pt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Unregistered missing checkpoint accepted")
    print({"default_checkpoints": 2, "legacy_paths_resolved": len(mapping), "missing_unregistered_rejected": True})

if __name__ == "__main__":
    main()
