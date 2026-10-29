"""Bounded real-data ROI parity diagnostic; not whole-cache/training admission.

Verifies selected native payloads against immutable producer receipts. It never
constructs a partially verified SamCacheProvider or changes its admission gates.
"""
import copy
import json
import time
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as ROOT, contained_path
from data.action4_sam_native_export import (
    _v2_checked, _v2_read, _v2_rgb, _v2_payload, _check_payload, canonical)
from data.action4_sam_roi_inputs import (
    ImageFrame, NativeMaskIdentity, native_key, verify_image_frame,
    verify_native_prediction, prepare_sam_roi_inputs, RoiRequest)
from data.action4_sam_cache_provider import resolve_training_roi
from data.action4_pose_rgb_batch import prepare_pose_rgb
from data.action4_prediction_roi_manifest import associate
from model.action4_sam_feature_adapter import FeatureIdentity, MaskAux


def run(config):
    started = time.monotonic()
    torch.set_num_threads(1)
    spec = _v2_checked(config["cache_spec"])
    index = _v2_checked(spec["manifest"])
    receipt = _v2_checked(index["producer_receipt"])
    assert receipt["index_payload_sha256"] == canonical(
        {k: v for k, v in index.items() if k != "producer_receipt"})
    lock = _v2_checked(receipt["native_lock"])
    assert lock["identity"] == canonical({k: v for k, v in lock.items() if k != "identity"})
    assert lock["config_identity"] == canonical(lock["config"])
    assert spec["expected_sources"] == receipt["sources"] == index["sources"]
    assert index["split"] == "train" and index["rgb_variant"] == "native_original"
    cfg = lock["config"]
    prompts = _v2_read({"path": cfg["manifest"], "sha256": cfg["manifest_sha256"]},
                       lambda f: [json.loads(x) for x in f if x.strip()])
    associations = _v2_checked(config["associations"])
    summary = _v2_checked(config["prediction_manifest"]["summary"])
    assert summary["B0_predictions"] == associations["predictions"] == receipt["source_contract"]["predictions"]
    assert summary["annotations"] == associations["annotations"]
    assert associations["cache_manifest"] == spec["manifest"]
    records = _v2_read(associations["annotations"], lambda f: [json.loads(x) for x in f if x.strip()])
    rows = _v2_read(config["prediction_manifest"]["rows"], lambda f: [json.loads(x) for x in f if x.strip()])
    saved = _v2_checked(summary["B0_predictions"])
    ids = config["image_ids"]
    assert len(ids) == len(set(ids)) and 1 <= len(ids) <= 16
    by_record = {r["sample_id"]: (i, r) for i, r in enumerate(records)}
    by_prompt = {r["image_id"]: r for r in prompts}
    by_entry = {r["image_id"]: r for r in index["images"]}
    by_saved = {r["sample_id"]: r["instances"] for r in saved}
    by_mapping = {(r["image_id"], r["prediction_id"]): r for r in receipt["mapping"]}
    by_assoc = {(r["image_id"], str(r["gt_instance_id"])): r for r in associations["rows"]}
    reports = []
    for image_id in ids:
        image_index, record = by_record[image_id]
        entry, prompt = by_entry[image_id], by_prompt[image_id]
        rgb = _v2_rgb(entry["image_path"], entry["source_sha256"])
        assert record["image"] == entry["image_path"] and record["rgb_sha256"] == entry["source_sha256"]
        frame = verify_image_frame(ImageFrame(image_id, rgb, entry["pixel_space_id"],
            entry["source_sha256"], entry["variant_id"], entry["variant_sha256"]))
        predicted = {}
        for cache in entry["predictions"]:
            pid = cache["prediction_id"]
            candidate = next(c for c in prompt["candidates"] if c["candidate_id"] == pid)
            mapping = by_mapping[(image_id, pid)]
            assert mapping["cache_sha256"] == cache["sha256"]
            assert mapping["identity_binding"] == candidate["identity_binding"]
            assert mapping["prompted"] and mapping["status"] == "available"
            key = canonical({"lock": lock["identity"], "image_id": image_id,
                "image_sha256": prompt["image_sha256"], "candidate": candidate, "text": cfg["native"]["text"]})
            expected = contained_path(ROOT, cfg["output"], exists=True) / "jobs" / key / "COMPLETED.json"
            assert contained_path(ROOT, mapping["completed_marker"], exists=True) == expected
            marker = _v2_checked({"path": mapping["completed_marker"], "sha256": mapping["completed_marker_sha256"]})
            assert marker["task_key"] == key and marker["lock_identity"] == lock["identity"]
            assert marker["files_sha256"][marker["result_path"]] == mapping["result_sha256"]
            native = _v2_checked({"path": marker["result_path"], "sha256": mapping["result_sha256"]})
            assert native["close_success"] and native["candidate_id"] == pid
            assert native["source_receipt"] == receipt["source_contract"]
            payload = _v2_payload({"path": cache["path"], "sha256": cache["sha256"]})
            _check_payload(payload, prompt, candidate, entry, spec["expected_sources"],
                           "available", None, mapping["sam_binding"], native, lock)
            ni = NativeMaskIdentity(image_id, pid, frame.pixel_space_id)
            predicted[native_key(ni)] = verify_native_prediction(payload, frame, ni, spec["expected_sources"])
        expected_rows, _ = associate(record, by_saved[image_id])
        for row in expected_rows:
            row.update(image_index=image_index, image_path=record["image"], RGB_sha256=record["rgb_sha256"])
        actual_rows = [r for r in rows if r["sample_id"] == image_id]
        assert actual_rows == expected_rows
        selected = [r for r in actual_rows if r["status"] == "matched_unique"]
        requests, units = [], []
        for row in selected:
            item = {"image": torch.from_numpy(rgb.copy()).permute(2, 0, 1), "record": record,
                    "instance": row["GT_instance_index"], "frozen_prediction_roi": row,
                    "prediction_manifest_identity": canonical(config["prediction_manifest"])}
            rgb_roi, geom, meta = prepare_pose_rgb([item], boxes=[row["bbox_for_a0"]],
                roi_expansion=1.5, roi_source="frozen_b0_prediction")
            meta[0].update(frozen_prediction_roi=copy.deepcopy(row), prediction_manifest_identity=item["prediction_manifest_identity"])
            unit = {"roi": rgb_roi[0], "metadata": meta[0], "geometry": {k: v[0] for k, v in geom.items()}}
            assoc = by_assoc[(image_id, str(row["GT_instance_id"]))]
            roi, scope = resolve_training_roi(item, unit, frame, assoc)
            identity = FeatureIdentity(image_id, row["prediction_id"], canonical(roi.record()), "action4.roi448.v1")
            requests.append(RoiRequest(identity, 0, roi.continuous_bounds, record["camera"]["K"], frame.pixel_space_id))
            units.append((unit, roi))
        prepared = prepare_sam_roi_inputs([frame], requests, predicted,
                    expected_sources=spec["expected_sources"], mask_aux_factory=MaskAux)
        reverse = prepare_sam_roi_inputs([frame], requests[::-1], predicted,
                    expected_sources=spec["expected_sources"], mask_aux_factory=MaskAux)
        missing = prepare_sam_roi_inputs([frame], requests, {},
                    expected_sources=spec["expected_sources"], mask_aux_factory=MaskAux)
        for i, (unit, roi) in enumerate(units):
            assert torch.equal(prepared.rgb[i], unit["roi"])
            assert np.allclose(prepared.A448[i], roi.affine(), rtol=0, atol=1e-7)
            assert np.allclose(prepared.A112[i], unit["metadata"]["affine_hm"], rtol=0, atol=1e-7)
            assert torch.equal(torch.tensor(prepared.K112[i], dtype=unit["geometry"]["K"].dtype), unit["geometry"]["K"])
            assert torch.equal(prepared.rgb[i], reverse.rgb[-1-i])
            assert torch.equal(prepared.mask_aux.visible[i], reverse.mask_aux.visible[-1-i])
            assert torch.equal(prepared.rgb[i], missing.rgb[i])
            assert np.array_equal(prepared.K112[i], missing.K112[i])
        assert not bool(missing.mask_aux.available.any())
        reports.append({"image_id": image_id, "instances": len(units), "RGB_A448_A112_K112_equal": True,
            "instance_reorder_preserved": True, "missing_mask_RGB_K_unchanged": True,
            "native_payloads_verified": len(predicted), "mask_pixels448": prepared.mask_aux.visible.sum(dim=(-2,-1)).tolist(),
            "roi_specs": [r.record() for _, r in units]})
    return {"status": "BOUNDED_REAL_INPUT_PARITY_PASS_NOT_TRAINING_ADMISSION", "config": config,
        "images": reports, "image_count": len(reports), "instance_count": sum(r["instances"] for r in reports),
        "whole_cache_revalidated": False, "provider_admission_bypassed": False, "models_called": False,
        "trained": False, "accuracy_measured": False, "seconds": time.monotonic()-started}
