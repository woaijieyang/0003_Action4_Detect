"""Versioned D0 method/data identity. This is admission evidence, not accuracy."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from data.action4_dataset import is_ignored
from data.action4_paths import checked_root
from data.action4_split_protocol import binding_identity
from .action4_runtime import canonical_hash, input_path, sha256_file

SCHEMA = "action4.d0_gate.v2"
SOURCE_FILES = (
    "src/model/action4_network.py", "src/model/action4_losses.py",
    "src/model/action4_decode.py", "src/data/action4_dataset.py",
    "src/data/action4_adapter.py", "src/model/training/action4_trainer.py",
    "src/model/training/action4_gate.py", "src/model/training/action4_runtime.py",
    "src/eval/action4_runner.py", "src/eval/action4_metrics.py",
    "src/data/action4_split_protocol.py", "src/model/training/action4_stage_policy.py",
    "src/model/training/action4_initialization.py",
    "src/eval/action4_step_eval.py",
)
PROTOCOL = {
    "schema_version": SCHEMA,
    "fixed_rois": "first non-ignored instances in immutable training manifest order",
    "formal_roi_count": 32,
    "ideal_heatmap_p95_max_px": 1.0,
    "learned_heatmap_p95_max_px": 2.0,
    "zero_baseline_improvement_min": 0.8,
    "maximum_updates": 500,
    "geometry_gate_required": True,
    "backbone_nonzero_gradient_and_parameter_change_required": True,
}


def build_gate_binding(root: str | Path, config: dict, train_manifest: str | Path,
                       records: list[dict], *, component: bool = False) -> dict:
    """Recompute identities from current bytes; never trust a saved source digest."""
    root = checked_root(root)
    extension = config.get("server_extension")
    if not component and (not isinstance(extension, dict)
                          or extension.get("schema_version") != "action4.server_extension.v1"):
        raise ValueError("Formal gate requires the declared action4.server_extension.v1 protocol")
    roi_count = config["stages"]["D0"].get("roi_count", 32)
    if type(roi_count) is not int or roi_count < 1:
        raise ValueError("D0 fixed ROI count must be a positive integer")
    if not component and roi_count != PROTOCOL["formal_roi_count"]:
        raise ValueError("Formal D0 requires exactly 32 fixed ROIs")
    rois, image_hashes = [], {}
    for sample_index, record in enumerate(records):
        for instance_index, instance in enumerate(record["instances"]):
            if is_ignored(instance):
                continue
            image = input_path(root, record["image"])
            if str(image) not in image_hashes:
                image_hashes[str(image)] = sha256_file(image)
            rois.append({
                "sample_index": sample_index, "instance_index": instance_index,
                "sample_id": record["sample_id"], "instance_id": instance.get("instance_id"),
                "record_sha256": canonical_hash(record),
                "image_sha256": image_hashes[str(image)],
            })
            if len(rois) == roi_count:
                break
        if len(rois) == roi_count:
            break
    if len(rois) != roi_count:
        raise ValueError("Not enough distinct non-ignored fixed D0 ROIs")
    initializer = config.get("initialization", {})
    weight_path = initializer.get("backbone_weights")
    if weight_path is None:
        if not component:
            raise ValueError("Formal gate requires verified ImageNet initialization")
        weight_identity = {"mode": "random_component_only"}
    else:
        weight_sha = sha256_file(input_path(root, weight_path))
        if initializer.get("sha256") != weight_sha:
            raise ValueError("Initialization weight SHA differs from the declared identity")
        weight_identity = {"mode": "imagenet_backbone", "sha256": weight_sha}
    # Runtime parallelism/output paths are deliberately not part of method
    # identity. Seed, AMP, deterministic mode and D0 optimization are relevant.
    method = {key: config.get(key) for key in
              ("network", "preprocess", "decode", "heatmap", "heatmaps", "loss", "losses")}
    method["D0"] = config["stages"]["D0"]
    method["runtime"] = {key: config.get("runtime", {}).get(key) for key in
                         ("seed", "amp", "deterministic")}
    def source_sha(name):
        try:
            return sha256_file(input_path(root, name))
        except FileNotFoundError:
            # Isolated CPU fixtures may omit source/config copies. Such a
            # component record cannot pass validate_d0_gate under any flag.
            if component:
                return None
            raise
    method["runtime"]["amp_init_scale"] = config.get("runtime", {}).get("amp_init_scale", 65536.)
    # Standalone legacy gate callers may provide only method settings.
    # Any explicit data block still goes through the full protocol checks.
    protocol_config = config if "data" in config else {**config, "data": {}}
    protocol = {"split_protocol_binding": binding_identity(root, protocol_config),
                "gate": PROTOCOL, "declared": config.get("protocol"), "server_extension": extension,
                "evaluation_sha256": source_sha("configs/eval/action4_protocol.json")}
    return copy.deepcopy({
        "protocol_sha256": canonical_hash(protocol),
        "source_sha256": {name: source_sha(name) for name in SOURCE_FILES},
        "method": method, "initialization": weight_identity,
        "train_manifest_sha256": sha256_file(input_path(root, train_manifest)),
        "fixed_rois": rois, "fixed_rois_sha256": canonical_hash(rois),
    })


def validate_d0_gate(gate: dict, expected_binding: dict) -> None:
    """Reject legacy/changed identities and independently check recorded measures."""
    if gate.get("schema_version") != SCHEMA:
        raise ValueError("D0 requires action4.d0_gate.v2; historical gates cannot be promoted")
    if gate.get("component_diagnostic") is not False or gate.get("gate_pass") is not True:
        raise ValueError("D0 gate is diagnostic or failed")
    if gate.get("binding") != expected_binding or gate.get("binding_sha256") != canonical_hash(expected_binding):
        raise ValueError("D0 gate method/data/initialization/fixed-ROI binding changed")
    report = gate.get("epe_report") or {}
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    values = [report.get("ideal_heatmap_p95_px"), report.get("epe_p95_heatmap_px"),
              report.get("improvement_over_zero_p95")]
    if (not all(finite(value) for value in values)
            or report.get("rois") != PROTOCOL["formal_roi_count"]
            or not 0 <= values[0] <= PROTOCOL["ideal_heatmap_p95_max_px"]
            or not 0 <= values[1] <= PROTOCOL["learned_heatmap_p95_max_px"]
            or not PROTOCOL["zero_baseline_improvement_min"] <= values[2] <= 1
            or gate.get("geometry_checks", {}).get("gate_pass") is not True
            or gate.get("finite_nonzero_backbone_gradient") is not True
            or gate.get("backbone_parameter_changed") is not True
            or type(gate.get("global_step")) is not int
            or not 0 < gate["global_step"] <= PROTOCOL["maximum_updates"]):
        raise ValueError("D0 measured evidence does not meet the unchanged formal thresholds")


def make_d0_gate(binding: dict, evidence: dict, *, component: bool, global_step: int) -> dict:
    result = {**evidence, "schema_version": SCHEMA, "component_diagnostic": bool(component),
              "global_step": global_step, "binding": binding,
              "binding_sha256": canonical_hash(binding)}
    if component:
        result["gate_pass"] = False
    elif result.get("gate_pass") is True:
        validate_d0_gate(result, binding)
    return result


def validate_reused_d0_gate(gate, expected_binding, receipt):
    """Explicit reuse of measured D0 for reviewed scheduling-only changes."""
    validate_d0_gate(gate, gate['binding'])
    if receipt.get('schema') != 'action4.d0_schedule_reuse.v1':
        raise ValueError('Explicit scheduling-only reuse receipt required')
    if receipt.get('original_binding_sha256') != canonical_hash(gate['binding']) or receipt.get('current_binding_sha256') != canonical_hash(expected_binding):
        raise ValueError('D0 reuse receipt identity changed')
    before=copy.deepcopy(gate['binding']);after=copy.deepcopy(expected_binding)
    old=before.pop('source_sha256');new=after.pop('source_sha256')
    if before != after:raise ValueError('D0 method, data, geometry, or initialization changed; cannot reuse')
    changes={p:{'before':old.get(p),'after':new.get(p)} for p in set(old)|set(new) if old.get(p)!=new.get(p)}
    allowed={'src/model/training/action4_trainer.py','src/model/training/action4_gate.py','src/eval/action4_step_eval.py','src/model/training/action4_initialization.py','src/model/training/action4_checkpoint.py'}
    if not set(changes)<=allowed or changes!=receipt.get('source_changes'):
        raise ValueError('Unreviewed D0 source changes')
    if receipt.get('authorization')!='user_requested_reuse_for_async_eval_scheduling_only':
        raise ValueError('Missing explicit reuse authorization')
