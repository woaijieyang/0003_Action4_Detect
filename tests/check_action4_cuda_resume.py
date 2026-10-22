"""Actual single-rank CUDA trainer continuation check; no production acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path("/data/zy/vln/a-test")
CONFIG = "configs/local/action4_cuda_resume_v001.json"
OUTPUT = "outputs/action4_cuda_resume_check_v001/gpu"
RUNS = {k: "cuda_resume_v001_" + k for k in ("a0", "continuous", "resumed")}
GPU = "GPU-b1ae5be6-1dba-7343-e233-59df41402f33"
STATE_FIELDS = ("model", "optimizer", "scheduler", "scaler", "rng_by_rank",
                "roi_rng_by_rank", "sampler", "next_epoch", "global_step",
                "epoch_boundary", "resume_integrity", "mandatory_snapshots",
                "config", "identity", "initialization", "claim_scope", "stage")


def bounded(value, *, exists=False):
    p = (ROOT / value).resolve(strict=exists)
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Outside project: {value}")
    return p


def sha(path):
    h = hashlib.sha256()
    with bounded(path, exists=True).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    p = bounded(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def assert_exact(a, b, path="state"):
    """No numerical tolerance: report first mismatch, never downgrade to PASS."""
    import numpy as np
    import torch
    if type(a) is not type(b):
        raise AssertionError(f"{path}: type mismatch {type(a)} != {type(b)}")
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            delta = float((a.double()-b.double()).abs().max()) if a.shape == b.shape and a.numel() else None
            raise AssertionError(f"{path}: tensor mismatch; max_abs_diff={delta}")
        if a.is_floating_point() and not bool(torch.isfinite(a).all()):
            raise AssertionError(f"{path}: nonfinite tensor")
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise AssertionError(f"{path}: array mismatch")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in a:
            assert_exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length mismatch")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_exact(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and (not math.isfinite(a) or not math.isfinite(b)):
        raise AssertionError(f"{path}: nonfinite scalar")
    elif a != b:
        raise AssertionError(f"{path}: value mismatch {a!r} != {b!r}")


def preflight():
    from data.action4_dataset import Action4Dataset, is_ignored
    from model.training.action4_trainer import _validate_split_isolation
    assert Path.cwd().resolve() == ROOT
    cfg = json.loads(bounded(CONFIG, exists=True).read_text())
    assert cfg["network"] == {"decoder_channels":256,"head_channels":64,"freeze_bn":True,"corner_probability_bias":.1}
    assert cfg["preprocess"] == {"full_size":[1024,768],"roi_size":256,"roi_expansion":1.25}
    assert cfg["runtime"]["workers"] == 2 and cfg["runtime"]["amp_init_scale"] == 8192
    assert cfg["runtime"]["amp"] and cfg["runtime"]["deterministic"]
    assert cfg["stages"]["A0"]["epochs"] == 1 and cfg["stages"]["B0"]["epochs"] == 2
    assert cfg["stages"]["B0"]["predicted_roi_after_epoch"] == 0
    assert cfg["stages"]["B0"]["predicted_roi_probability"] == .5
    assert cfg["video"]["frame_manifest"] is None and cfg["video"]["run_during_component"] is False
    assert cfg["initialization"]["sha256"] == sha(cfg["initialization"]["backbone_weights"])
    sets = {s:Action4Dataset(ROOT, bounded(cfg["data"][s+"_manifest"]), split=s, allow_diagnostic=True) for s in ("train","eval","test")}
    _validate_split_isolation(sets["train"], sets["eval"], sets["test"].records)
    provenance = json.loads(bounded("data/action4_diagnostics/cuda_resume_v001/provenance.json").read_text())
    assert sha(provenance["source_manifest"]) == provenance["source_sha256"]
    original = [json.loads(x) for x in bounded(provenance["source_manifest"]).read_text().splitlines()]
    inputs = {CONFIG:sha(CONFIG), provenance["source_manifest"]:sha(provenance["source_manifest"]),
              cfg["initialization"]["backbone_weights"]:sha(cfg["initialization"]["backbone_weights"])}
    for s, dataset in sets.items():
        inputs[cfg["data"][s+"_manifest"]] = sha(cfg["data"][s+"_manifest"])
        for index, rec in zip(provenance["selection"][s], dataset.records):
            expected = dict(original[index]); expected["split"] = s
            assert rec == expected, "Derived annotation differs beyond split field"
            assert rec["status"] == "diagnostic_pilot_not_training_accepted"
            assert sha(rec["image"]) == rec["image_sha256"]
            inputs[rec["image"]] = sha(rec["image"])
            # Actually decode the existing RGB; no new pixels or labels generated.
            item = dataset[dataset.records.index(rec)]
            assert tuple(item["image"].shape) == (3, rec["height"], rec["width"])
    counts = [sum(not is_ignored(x) for x in r["instances"]) for r in sets["train"].records]
    assert len(counts) == 8 and set(counts) == {0,1,2}
    for s in ("A0","B0"):
        st = cfg["stages"][s]
        assert st["global_batch"] == 8 and st["micro_batch"] == 1
        assert st["warmup_updates"] == 0 and st["freeze_backbone_updates"] == 0
        assert st["max_nonfinite_updates"] == 0
    for folder in ("src/data","src/model","src/eval"):
        for p in sorted(bounded(folder).rglob("*.py")):
            inputs[str(p.relative_to(ROOT))] = sha(p)
    for p in ("tests/check_action4_cuda_resume.py", "scripts/action4_cuda_resume_check.py", "scripts/action4_train.py",
              "data/action4_diagnostics/cuda_resume_v001/provenance.json"):
        inputs[p] = sha(p)
    return {"status":"CPU_PREFLIGHT_PASS", "scope":"existing diagnostic RGB/labels only",
            "train_instance_counts":counts,"a0_roi_count":sum(counts),
            "a0_updates":math.ceil(sum(counts)/8),"b0_updates_per_epoch":1,
            "b0_planned_updates":2,"inputs":inputs}


def log_rows(run):
    rows = []
    for p in sorted(bounded("outputs/action4/"+RUNS[run]).rglob("train_updates.jsonl")):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            row.pop("seconds")  # Wall time is explicitly not part of numerical continuation.
            rows.append(row)
    return sorted(rows, key=lambda r:(r["epoch"],r["global_step"]))


def run_check():
    import torch
    from model.training.action4_runtime import load_checkpoint, environment_identity
    pf = preflight()
    assert os.environ.get("WORLD_SIZE","1") == "1"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert "GPU-"+str(torch.cuda.get_device_properties(0).uuid) == GPU
    out = bounded(OUTPUT)
    if out.exists():
        raise FileExistsError("Diagnostic GPU output already exists; preserve evidence and choose a reviewed new version")
    for run in RUNS.values():
        for prefix in ("models/action4", "outputs/action4"):
            if bounded(prefix+"/"+run).exists():
                raise FileExistsError(f"Existing run must not be reused: {run}")
    out.mkdir(parents=True)
    write_new(out/"preflight.json", pf)
    receipts = []
    try:
        def launch(name, stage, run, *extra):
            for path, digest in pf["inputs"].items():
                assert sha(path) == digest, f"Input/source drift before {name}: {path}"
            argv = [sys.executable, "-B", "scripts/action4_train.py", "--config",CONFIG,
                    "--stage",stage,"--run-id",RUNS[run],"--device","cuda",
                    "--allow-diagnostic","--component-smoke",*extra]
            with (out/(name+".log")).open("xb") as log:
                result = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt = {"name":name,"argv":argv,"returncode":result.returncode}
            receipts.append(receipt)
            write_new(out/(name+"_receipt.json"), receipt)
            if result.returncode:
                raise RuntimeError(f"{name} failed; see preserved log")
        launch("a0_initialization","A0","a0")
        init = "models/action4/"+RUNS["a0"]+"/best.pt"
        init_sha = sha(init)
        launch("b0_uninterrupted","B0","continuous","--init-weights",init)
        launch("b0_stop_epoch1","B0","resumed","--init-weights",init,"--max-updates","1")
        resume = "models/action4/"+RUNS["resumed"]+"/last.pt"
        stopped = load_checkpoint(bounded(resume))
        assert stopped["epoch_boundary"] is True and stopped["next_epoch"] == stopped["global_step"] == 1
        assert stopped["identity"]["planned_updates"] == 2 and stopped["claim_scope"] == "component_diagnostic"
        continuous_first = load_checkpoint(bounded("models/action4/"+RUNS["continuous"]+"/epoch_001_step_00000001.pt"))
        for field in STATE_FIELDS:
            assert_exact(continuous_first[field],stopped[field],"epoch1."+field)
        write_new(out/"epoch1_comparison.json", {"status":"PASS_EXACT","fields":list(STATE_FIELDS),"resume_checkpoint_sha256":sha(resume)})
        del stopped, continuous_first
        launch("b0_exact_resume","B0","resumed","--resume",resume)
        a = load_checkpoint(bounded("models/action4/"+RUNS["continuous"]+"/last.pt"))
        b = load_checkpoint(bounded(resume))
        assert a["epoch_boundary"] is b["epoch_boundary"] is True
        assert a["next_epoch"] == b["next_epoch"] == a["global_step"] == b["global_step"] == 2
        assert a["resume_integrity"]["nonfinite_updates"] == b["resume_integrity"]["nonfinite_updates"] == 0
        assert a["resume_integrity"]["device_type"] == "cuda" and a["resume_integrity"]["amp_enabled"]
        for field in STATE_FIELDS:
            assert_exact(a[field], b[field], "final."+field)
        for state in (a,b):
            assert state["rng_by_rank"][0]["cuda"] and state["scaler"]["scale"] == 8192
        l1,l2 = log_rows("continuous"),log_rows("resumed")
        assert [r["global_step"] for r in l1] == [1,2] and all(r["optimizer_updated"] for r in l1+l2)
        assert_exact(l1,l2,"update_logs_excluding_seconds")
        assert init_sha == sha(init)
        for path,digest in pf["inputs"].items():
            assert sha(path) == digest, f"Input/source drift after execution: {path}"
        report = {"status":"PASS_EXACT_SINGLE_RANK_CUDA_FULL_STATE","claim_scope":"component_diagnostic",
                  "environment":environment_identity(torch),"fields_compared":list(STATE_FIELDS),
                  "comparison":"bit exact tensors and scalars, zero tolerance; only run metadata/top alias paths and wall time excluded",
                  "epoch1_exact":True,"final_exact":True,"next_epoch_loss_and_update_records_exact":True,
                  "updates":l1,"receipts":receipts,"initialization_sha256":init_sha,
                  "final_checkpoints":{run:sha("models/action4/"+RUNS[run]+"/last.pt") for run in ("continuous","resumed")},
                  "limitations":["Single-rank GPU6 only; no 2/4-rank CUDA continuation claim",
                                "Two B0 updates check state continuation, not optimization convergence or real-video quality",
                                "Predicted ROI mixing enabled from epoch0; untrained detections may yield zero accepted predicted ROIs",
                                "Exact determinism verified only on this actual runtime/hardware; failures are not tolerated or hidden",
                                "Diagnostic source RGB is 812x616, resized by actual full1024x768/ROI256 preprocessing; not native camera-resolution training"]}
        write_new(out/"report.json",report)
        print(json.dumps({"status":report["status"],"output":OUTPUT}))
    except Exception as exc:
        write_new(out/"failure.json",{"status":"FAIL","error":str(exc),"traceback":traceback.format_exc(),"receipts":receipts})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(),indent=2))
    else:
        run_check()


if __name__ == "__main__":
    main()
