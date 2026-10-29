"""Fixed prediction-side native RGB variants; no dataset/model/cache generation.

Call generate_variant on original native uint8 RGB only. Parameters are fixed per
image/source/policy, while select_variant changes only the image-level choice.
Source file SHA supplied to the array API is a caller assertion; use
generate_from_file for a verified same-byte decode. All results bind actual pixels.
"""
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .action4_paths import PROJECT_ROOT, contained_path
from .action4_sam_roi_inputs import array_digest

SCHEMA = "action4.native_rgb_variant.v1"
POLICY_SCHEMA = "action4.native_rgb_variant_policy.v1"
VARIANTS = ("original", "light", "blur")
DEFAULT_POLICY = {
    "schema": POLICY_SCHEMA,
    "variants": list(VARIANTS),
    "probability": 0.8,
    "operation_probability": 0.5,
    "add_range": [-0.098, 0.098],
    "multiply_range": [0.6, 1.4],
    "contrast_range": [0.5, 2.2],
    "blur_sigma_range": [0.05, 1.2],
    "light_order": ["add", "multiply", "contrast"],
    "channel_mode": "shared_rgb",
    "contrast_center": 0.5,
    "blur_algorithm": "separable_gaussian_radius_ceil3sigma_reflect101_float64",
    "quantization": "clip_then_numpy_rint_uint8",
    "selection": "sha256_image_epoch_uniform_three_v1",
    "epoch_numbering": "one_based",
    "decode": "Pillow_RGB_RGBA_to_RGB_discard_alpha_no_exif_transform",
    "scope": "native_train_only_preparation_not_integrated",
}

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _sha(value):
    return hashlib.sha256(value).hexdigest()

def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Explicit lowercase SHA256 required")

def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an explicit bounded integer")

def _image_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty image_id required")

def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != set(DEFAULT_POLICY):
        raise ValueError("Unknown or missing RGB policy fields")
    p = copy.deepcopy(policy)
    variable = {"add_range", "multiply_range", "contrast_range", "blur_sigma_range"}
    for key in set(p) - variable:
        if type(p[key]) is not type(DEFAULT_POLICY[key]) or p[key] != DEFAULT_POLICY[key]:
            raise ValueError("Unsupported RGB policy field: " + key)
    for key in variable:
        values = p[key]
        bounds = DEFAULT_POLICY[key]
        if (not isinstance(values, list) or len(values) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or
            not bounds[0] <= values[0] <= values[1] <= bounds[1]):
            raise ValueError("Invalid RGB policy interval: " + key)
    return p

def load_policy(path, expected_sha256):
    _digest(expected_sha256)
    raw = contained_path(PROJECT_ROOT, path, exists=True).read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("RGB policy file SHA mismatch")
    return validate_policy(json.loads(raw))

def _rgb(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError("Native RGB must be an actual uint8 ndarray, not normalized floats")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("Native RGB shape must be positive H x W x 3")
    return np.ascontiguousarray(value)

def _uniform(key, name):
    # Explicit hash draws avoid global RNG and NumPy generator-version dependence.
    return int.from_bytes(hashlib.sha256(key + b"|" + name.encode()).digest()[:8], "big") / 2**64

def select_variant(image_id, epoch, *, seed=0, policy=None):
    """One choice per image/epoch; no instance ID, rank, worker, or call-order input."""
    _image_id(image_id)
    _integer(epoch, "epoch", 1)
    _integer(seed, "seed")
    p = validate_policy(DEFAULT_POLICY if policy is None else policy)
    key = _canonical({"image_id": image_id, "epoch": epoch, "seed": seed, "policy": p})
    index = int.from_bytes(hashlib.sha256(key).digest(), "big") % len(VARIANTS)
    return VARIANTS[index]

def _parameters(image_id, source_sha256, source_pixels_sha256, variant, seed, policy):
    key = _canonical({"image_id": image_id, "source_sha256": source_sha256,
        "source_pixels_sha256": source_pixels_sha256, "variant": variant, "seed": seed, "policy": policy})
    params = {"enabled": False, "operations": {}, "values": {}}
    if variant == "original":
        return params
    params["enabled"] = _uniform(key, "enabled") < policy["probability"]
    names = policy["light_order"] if variant == "light" else ["blur"]
    for name in names:
        params["operations"][name] = _uniform(key, name + "_enabled") < policy["operation_probability"]
        interval = policy["blur_sigma_range" if name == "blur" else name + "_range"]
        params["values"][name] = interval[0] + _uniform(key, name + "_value") * (interval[1] - interval[0])
    return params

def _blur(rgb, sigma):
    radius = int(math.ceil(3 * sigma))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-axis * axis / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = rgb
    for dimension in (0, 1):
        pads = [(0, 0), (0, 0), (0, 0)]
        pads[dimension] = (radius, radius)
        padded = np.pad(out, pads, mode="reflect")
        window = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=dimension)
        out = np.einsum("...k,k->...", window, kernel, optimize=False)
    return out

def generate_variant(rgb, *, image_id, source_sha256, variant, seed=0, policy=None):
    """Return (readonly RGB, metadata); supplied source SHA remains caller-asserted."""
    source = _rgb(rgb).copy()
    _image_id(image_id)
    _digest(source_sha256)
    _integer(seed, "seed")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("Unknown native RGB variant")
    p = validate_policy(DEFAULT_POLICY if policy is None else policy)
    source_pixels = array_digest(source)
    params = _parameters(image_id, source_sha256, source_pixels, variant, seed, p)
    out = source.copy()
    if params["enabled"]:
        values = source.astype(np.float64) / 255
        for operation, enabled in params["operations"].items():
            if not enabled:
                continue
            amount = params["values"][operation]
            if operation == "add":
                values = values + amount
            elif operation == "multiply":
                values = values * amount
            elif operation == "contrast":
                values = (values - p["contrast_center"]) * amount + p["contrast_center"]
            else:
                values = _blur(values, amount)
            values = np.clip(values, 0, 1)
        out = np.rint(values * 255).astype(np.uint8)
    # Bytes-backed arrays cannot have WRITEABLE re-enabled by downstream callers.
    out = np.frombuffer(out.tobytes(), dtype=np.uint8).reshape(out.shape)
    metadata = {"schema": SCHEMA, "image_id": image_id, "source_sha256": source_sha256,
        "source_pixels_sha256": source_pixels, "native_shape": list(source.shape),
        "color_space": "RGB_uint8_sRGB_code_values", "variant": variant, "seed": seed,
        "policy": p, "policy_sha256": _sha(_canonical(p)), "parameters": params,
        "variant_sha256": array_digest(out), "source_file_verified": False,
        "no_post_roi_transform": True}
    metadata["variant_id"] = "native_" + variant + ":" + _sha(_canonical(metadata))
    return out, metadata

def generate_from_file(path, expected_sha256, *, image_id, variant, seed=0, policy=None):
    """Hash and decode the exact same bytes; no EXIF geometry or alpha compositing."""
    _digest(expected_sha256)
    source_path = contained_path(PROJECT_ROOT, path, exists=True)
    raw = source_path.read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("Native source file SHA mismatch")
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode not in ("RGB", "RGBA") or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Only single-frame native RGB/RGBA sources are supported")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    out, meta = generate_variant(rgb, image_id=image_id, source_sha256=expected_sha256,
        variant=variant, seed=seed, policy=policy)
    # Provenance receipt is separate from variant identity, so array/file callers
    # produce the same cache identity for the same verified pixels and parameters.
    return out, meta, {"source_file_verified": True, "source_sha256": expected_sha256,
        "source_pixels_sha256": array_digest(rgb), "path": str(source_path.relative_to(PROJECT_ROOT)),
        "variant_id": meta["variant_id"]}

def verify_variant(rgb, metadata, *, source_rgb, image_id, source_sha256, variant, seed=0, policy=None):
    """Recompute from independently held source/arguments; reject rehashed wrong RGB."""
    expected, expected_meta = generate_variant(source_rgb, image_id=image_id,
        source_sha256=source_sha256, variant=variant, seed=seed, policy=policy)
    actual = _rgb(rgb)
    if metadata != expected_meta or not np.array_equal(actual, expected):
        raise ValueError("RGB variant payload/identity differs from independent source recipe")
    return True

def assert_cache_compatible(metadata, cache_metadata):
    """Necessary variant guard only; existing cache validators still verify all else."""
    for field in ("source_sha256", "variant_id", "variant_sha256"):
        if metadata.get(field) != cache_metadata.get(field) or not metadata.get(field):
            raise ValueError("Cannot reuse native mask/cache across RGB variants")
