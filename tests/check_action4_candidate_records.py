"""Actual candidate identity plus adversarial split/status/canonical/projection checks."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from data.action4_paths import PROJECT_ROOT as R
from eval.action4_candidate_record_eval import load_candidate


def main():
    cfg = json.loads((R / "configs/eval/action4_next_data_baseline_v001.json").read_text())
    original = load_candidate(cfg["records"][0])
    cases = []
    changed = copy.deepcopy(original)
    changed["split"] = "test"
    cases.append(changed)
    changed = copy.deepcopy(original)
    changed["status"] = "training_accepted"
    cases.append(changed)
    changed = copy.deepcopy(original)
    changed["corners_3d_mm"][0], changed["corners_3d_mm"][1] = changed["corners_3d_mm"][1], changed["corners_3d_mm"][0]
    cases.append(changed)
    changed = copy.deepcopy(original)
    changed["instances"][0]["canonical8_geometry"]["object_to_capture_cv_m"][0][3] += .01
    cases.append(changed)
    with tempfile.TemporaryDirectory(dir=R / "outputs", prefix="candidate_contract_") as directory:
        path = Path(directory) / "record.json"
        for case in cases:
            path.write_text(json.dumps(case))
            ref = {"path": str(path.relative_to(R)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            try:
                load_candidate(ref)
            except ValueError:
                continue
            raise AssertionError("Unsafe or inconsistent candidate accepted")
    ref = dict(cfg["records"][0], sha256="0" * 64)
    try:
        load_candidate(ref)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed record identity accepted")
    print("PASS: actual candidate accepted; test split, admitted status, corner order, pose inconsistency, bad digest rejected")


if __name__ == "__main__":
    main()
