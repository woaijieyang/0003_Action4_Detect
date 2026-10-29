"""Bound native RGB derivative pilot; no labels, prompts, models or training wiring."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .action4_paths import PROJECT_ROOT as ROOT, contained_path
from .action4_rgb_variants import generate_from_file, load_policy, VARIANTS
from .action4_sam_roi_inputs import array_digest

SCHEMA = "action4.rgb_variant_manifest.v1"

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _bytes(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Explicit path/SHA binding required")
    raw = contained_path(ROOT, binding["path"], exists=True).read_bytes()
    if _sha(raw) != binding["sha256"]:
        raise ValueError("Bound input bytes changed")
    return raw

def _binding(path):
    path = contained_path(ROOT, path, exists=True)
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha(path.read_bytes())}

def _write_json(path, value):
    # Same-directory atomic publication; pre-existing outputs are never reused.
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    tmp.replace(path)

def _selection(annotations, image_ids):
    if (not isinstance(image_ids, list) or not image_ids or
            any(not isinstance(s, str) or not s.strip() for s in image_ids) or
            len(set(image_ids)) != len(image_ids)):
        raise ValueError("Explicit nonempty unique image IDs required")
    records = [json.loads(line) for line in _bytes(annotations).decode("utf-8").splitlines() if line.strip()]
    by_id = {}
    for row in records:
        sid = row["sample_id"]
        if sid in by_id or row.get("split") != "train" or row.get("status") != "training_accepted":
            raise ValueError("Unique accepted train annotation domain required")
        by_id[sid] = row
    if any(s not in by_id for s in image_ids):
        raise ValueError("Requested image missing from bound annotations")
    return [by_id[s] for s in image_ids], len(records)

def _decode(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode not in ("RGB", "RGBA") or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Single native RGB/RGBA image required")
        return np.array(image.convert("RGB"), dtype=np.uint8)

def _recipe(row, variant, seed, policy):
    for key in ("width", "height"):
        if type(row[key]) is not int or row[key] < 1:
            raise ValueError("Positive native dimensions required")
    source_path = contained_path(ROOT, row["image"], exists=True)
    rgb, meta, receipt = generate_from_file(source_path, row["rgb_sha256"],
        image_id=row["sample_id"], variant=variant, seed=seed, policy=policy)
    if list(rgb.shape) != [row["height"], row["width"], 3]:
        raise ValueError("Annotation/source native shape mismatch")
    return rgb, meta, receipt

def materialize(annotations, policy_binding, image_ids, output, *, seed=0):
    """Create a new pilot directory and return bound MANIFEST; READY is required."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Explicit nonnegative integer seed required")
    records, total = _selection(annotations, image_ids)
    _bytes(policy_binding)
    policy = load_policy(policy_binding["path"], policy_binding["sha256"])
    out = contained_path(ROOT, output, exists=False)
    out.mkdir(parents=True, exist_ok=False)
    expected = [{"image_id": sid, "variant": v} for sid in image_ids for v in VARIANTS]
    _write_json(out/"EXPECTED.json", {"schema": SCHEMA, "annotations": annotations,
        "policy": policy_binding, "seed": seed, "expected": expected})
    expected_binding = _binding(out/"EXPECTED.json")
    try:
        rows = []
        for ordinal, record in enumerate(records):
            for variant in VARIANTS:
                rgb, recipe, source_receipt = _recipe(record, variant, seed, policy)
                source = {"path": source_receipt["path"], "sha256": record["rgb_sha256"]}
                if variant == "original":
                    actual = dict(source)
                else:
                    path = out/(str(ordinal).zfill(6)+"_"+variant+".png")
                    # Exclusive filename; PNG encoder is lossless, verified below.
                    with path.open("xb") as stream:
                        Image.fromarray(rgb).save(stream, format="PNG")
                        stream.flush()
                        os.fsync(stream.fileno())
                    actual = _binding(path)
                actual_pixels = _decode(_bytes(actual))
                if not np.array_equal(actual_pixels, rgb):
                    raise ValueError("Written native PNG differs from generated pixels")
                rows.append({"image_id": record["sample_id"], "variant": variant,
                    "original_source": source, "actual_input": actual,
                    "native_shape": list(rgb.shape), "pixel_space": "native_integer_centers",
                    "actual_pixels_sha256": array_digest(actual_pixels), "recipe": recipe,
                    "source_receipt": source_receipt,
                    "legacy_leaf_key": "native_original" if variant == "original" else None,
                    "legacy_alias_admitted": False})
        manifest = {"schema": SCHEMA, "status": "COMPLETE_PILOT_NOT_TRAINING_ADMITTED",
            "annotations": annotations, "policy": policy_binding, "seed": seed,
            "expected_registration": expected_binding,
            "image_ids": list(image_ids), "expected": expected, "rows": rows,
            "scope": {"kind": "explicit_requested_selection", "requested_images": len(records),
                "annotation_images": total, "variant_rows": len(rows),
                "dataset_units_added": 0, "formal_stage3_admitted": False},
            "contains_GT_labels": False, "SAM_prompts": False}
        # Validate all entries against independently bound source/recipe before publication.
        _validate(manifest)
        _write_json(out/"MANIFEST.json", manifest)
        binding = _binding(out/"MANIFEST.json")
        _write_json(out/"READY.json", {"status": "READY_PIXEL_PILOT_ONLY", "manifest": binding,
            "expected_registration": expected_binding})
        return binding
    except BaseException as error:
        _write_json(out/"FAILED.json", {"status": "FAILED_NOT_READY", "error": str(error),
            "error_type": type(error).__name__})
        raise

def _validate(manifest):
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "COMPLETE_PILOT_NOT_TRAINING_ADMITTED"
            or manifest.get("contains_GT_labels") is not False or manifest.get("SAM_prompts") is not False):
        raise ValueError("Unknown pixel pilot manifest")
    registration = manifest.get("expected_registration")
    if not isinstance(registration, dict):
        raise ValueError("Missing predeclared expected registration")
    expected_record = json.loads(_bytes(registration))
    if expected_record != {"schema": SCHEMA, "annotations": manifest["annotations"],
            "policy": manifest["policy"], "seed": manifest["seed"], "expected": manifest["expected"]}:
        raise ValueError("Variant manifest differs from predeclared expected domain or inputs")
    records, total = _selection(manifest["annotations"], manifest["image_ids"])
    _bytes(manifest["policy"])
    policy = load_policy(manifest["policy"]["path"], manifest["policy"]["sha256"])
    expected = [{"image_id": r["sample_id"], "variant": v} for r in records for v in VARIANTS]
    rows = manifest["rows"]
    if (manifest["expected"] != expected or len(rows) != len(expected) or
            manifest["scope"] != {"kind": "explicit_requested_selection", "requested_images": len(records),
                "annotation_images": total, "variant_rows": len(expected),
                "dataset_units_added": 0, "formal_stage3_admitted": False}):
        raise ValueError("Incomplete or inflated variant domain")
    for entry, key in zip(rows, expected):
        if (entry["image_id"], entry["variant"]) != (key["image_id"], key["variant"]):
            raise ValueError("Variant image/order/coverage mismatch")
        record = records[manifest["image_ids"].index(key["image_id"])]
        rgb, recipe, receipt = _recipe(record, key["variant"], manifest["seed"], policy)
        source = {"path": receipt["path"], "sha256": record["rgb_sha256"]}
        if (entry["original_source"] != source or entry["recipe"] != recipe or entry["source_receipt"] != receipt
                or entry["native_shape"] != list(rgb.shape) or entry["pixel_space"] != "native_integer_centers"
                or entry["actual_pixels_sha256"] != array_digest(rgb)
                or entry["legacy_alias_admitted"] is not False
                or entry["legacy_leaf_key"] != ("native_original" if key["variant"] == "original" else None)):
            raise ValueError("Variant source/recipe/geometry identity mismatch")
        if key["variant"] == "original" and entry["actual_input"] != source:
            raise ValueError("Original must reference exact original file")
        raw = _bytes(entry["actual_input"])
        if key["variant"] != "original":
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError("Derived RGB must be lossless RGB PNG")
        if not np.array_equal(_decode(raw), rgb):
            raise ValueError("Actual input pixels differ from bound recipe")
    return True

def _verify_manifest_uncached(binding):
    path = contained_path(ROOT, binding["path"], exists=True)
    ready = json.loads(contained_path(ROOT, path.parent/"READY.json", exists=True).read_bytes())
    manifest = json.loads(_bytes(binding))
    registration = manifest.get("expected_registration")
    if not isinstance(registration, dict):
        raise ValueError("Missing predeclared expected registration")
    expected_path = contained_path(ROOT, registration.get("path",""), exists=True)
    if expected_path != contained_path(ROOT, path.parent/"EXPECTED.json", exists=True):
        raise ValueError("Foreign predeclared expected registration")
    if ready != {"status": "READY_PIXEL_PILOT_ONLY", "manifest": binding,
                 "expected_registration": registration} or (path.parent/"FAILED.json").exists():
        raise ValueError("Partial/failed/unbound pilot is not READY")
    for row in manifest["rows"]:
        p = contained_path(ROOT, row["actual_input"]["path"], exists=True)
        if row["variant"] != "original" and not p.is_relative_to(path.parent):
            raise ValueError("Derived image escaped pilot output")
    _validate(manifest)
    return manifest


# Process-local successful proofs only. No disk receipt is trusted as a substitute
# for the first complete recipe/pixel verification in this process.
from collections import OrderedDict
import copy
_MANIFEST_PROOFS=OrderedDict()

def _manifest_fingerprints(binding,manifest):
    path=contained_path(ROOT,binding["path"],exists=True)
    if (path.parent/"FAILED.json").exists():raise ValueError("Failed pilot is not reusable")
    paths=[path,path.parent/"READY.json"]
    paths += [contained_path(ROOT,manifest[k]["path"],exists=True) for k in ("expected_registration","annotations","policy")]
    for row in manifest["rows"]:
        paths += [contained_path(ROOT,row[k]["path"],exists=True) for k in ("original_source","actual_input")]
    result=[]
    for p in sorted(set(paths)):
        resolved=contained_path(ROOT,p,exists=True).resolve(strict=True)
        st=resolved.stat()
        result.append((str(p),str(resolved),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns))
    return tuple(result)

def verify_manifest(binding):
    manifest=json.loads(_bytes(binding))
    before=_manifest_fingerprints(binding,manifest)
    key=(os.getpid(),binding["path"],binding["sha256"],before)
    if key in _MANIFEST_PROOFS:
        if _manifest_fingerprints(binding,manifest)!=before:raise ValueError("Manifest inputs changed during reuse")
        _MANIFEST_PROOFS.move_to_end(key)
        return copy.deepcopy(_MANIFEST_PROOFS[key])
    result=_verify_manifest_uncached(binding)
    if _manifest_fingerprints(binding,result)!=before:raise ValueError("Manifest inputs changed during verification")
    _MANIFEST_PROOFS[key]=copy.deepcopy(result)
    while len(_MANIFEST_PROOFS)>4:_MANIFEST_PROOFS.popitem(last=False)
    return result
