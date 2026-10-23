"""Real selected parents plus wrong role/path/step/hash/config counterexamples."""
import copy
import json
import tempfile
from pathlib import Path

from data.action4_paths import PROJECT_ROOT as R
from model.action4_best_registry import verify_reference, verify_loaded, DEFAULT_REGISTRY
from model.training.action4_stage2_core import construct
from eval.action4_staged_pipeline import load_pipeline


def must_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Wrong checkpoint identity accepted")


def main():
    selection = json.loads((R / DEFAULT_REGISTRY).read_text())
    for role in ["A0", "B0"]:
        cfg = json.loads((R / ("configs/model/action4_stage2_integration_" + role.lower() + "_v001.json")).read_text())
        cfg["mode"] = "full_training"
        model, state = construct(cfg, "cpu")
        verify_loaded(selection["entries"][role], state)
        bad = dict(state, selected_step=state["selected_step"] + 1)
        must_reject(lambda: verify_loaded(selection["entries"][role], bad))
        del model, state
    a, b = selection["entries"]["A0"], selection["entries"]["B0"]
    must_reject(lambda: verify_reference("A0", b["path"]))
    must_reject(lambda: verify_reference("A0", "models/A0/best.pt"))
    must_reject(lambda: verify_reference("B0", "models/B0/best.pt"))
    must_reject(lambda: verify_reference("A0", a["path"], 13350))
    with tempfile.TemporaryDirectory(dir=R / "outputs", prefix="best_identity_test_") as folder:
        p = Path(folder) / "registry.json"
        altered = copy.deepcopy(selection)
        altered["entries"]["A0"]["sha256"] = "0" * 64
        p.write_text(json.dumps(altered))
        must_reject(lambda: verify_reference("A0", a["path"], registry=str(p)))
    pipeline = load_pipeline(json.loads((R / "configs/eval/action4_next_training_best_v001.json").read_text()), "cpu")
    assert pipeline.identity["A0_step"] == 14000 and pipeline.identity["B0_step"] == 3220
    print("PASS: CPU training construction and inference profile select A0=14000/B0=3220; seven identity counterexamples rejected")


if __name__ == "__main__":
    main()
