"""Check selected historical/default weights and explicit retirement behavior."""
import json
import re
from data.action4_paths import PROJECT_ROOT as R
from model.action4_checkpoint_paths import resolve_checkpoint, LOCATION_MAP

def main():
    assert sorted(p.name for p in (R / "models/default").iterdir()) == ["A0.pt", "B0.pt"]
    history = json.loads((R / "configs/model/action4_archive_registry_v001.json").read_text())
    expected = {entry["destination"] for entry in history["entries"]}
    assert {str(p.relative_to(R)) for p in (R / "models/archive").iterdir()} == expected
    assert len(expected) == 9
    for name in expected:
        assert re.match(r"\d{8}-\d{6}-.+\.pt$", (R / name).name)
        assert "best" not in (R / name).name.lower()
    catalog = json.loads((R / LOCATION_MAP).read_text())
    for old, new in catalog["locations"].items():
        assert not (R / old).exists(), "Stale physical alias remains"
        assert resolve_checkpoint(old) == R / new
        assert resolve_checkpoint(new) == R / new
    for old in catalog["deleted"]:
        assert not (R / old).exists()
        try:
            resolve_checkpoint(old)
        except FileNotFoundError as exc:
            assert "intentionally retired" in str(exc)
        else:
            raise AssertionError("Retired checkpoint silently replaced")
    print({"default": 2, "historical": 9, "resolved": len(catalog["locations"]),
           "retired_paths_rejected": len(catalog["deleted"])})

if __name__ == "__main__":
    main()
