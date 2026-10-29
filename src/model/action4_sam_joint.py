"""Shared SAM geometry forward and explicit training-item/serialization contracts."""
from dataclasses import dataclass
import torch
from torch import nn
from model.action4_geometry_forward import instancewise_geometry
from model.action4_sam_feature_adapter import (
    FeatureIdentity, MaskAux, SamFeatureAdapter, PROTOCOL_ID, _identities,
)

SAM_VARIANT = "a0_sam_image"
SAM_ARCHITECTURE = "joint_pose_sam"
QUALITY_KEY = "sam_mask_quality"
QUALITY_PREFIX = "sam_adapter.quality_head"
BUNDLE_PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "format": "geometry_plus_sam_adapter.v1",
    "geometry_batch_policy": "instancewise_head_v1",
    "rgb_size": 448, "c5_channels": 2048, "c5_size": 14,
    "gdr_input_channels": 71,
}


CORNER_VARIANT = "a0_corner_image"
CORNER_ARCHITECTURE = "joint_pose_sam_corner_candidate"
CORNER_PROTOCOL = {
    "protocol_id":"action4.corner_candidate_model.v1",
    "grid":224,"fixed_ids":8,"feature_path":"shared_enhancedC5_FPN128",
    "quality_epoch_basis":"one_based","quality_warmup_epochs":5,
    "decoder_status":"pending","async_eval_status":"pending",
    "quality_ready":False,"deployment_admitted":False,
    "output":"raw_pose_and_typed_corner_observations_only",
}


@dataclass(frozen=True)
class CornerCandidateOutput:
    raw_pose: dict
    sam_aux: object
    corners: object


def corner_variant(cfg):
    flag=cfg.get("corner_enabled",False)
    if type(flag) is not bool or flag!=(cfg.get("a0_variant")==CORNER_VARIANT):
        raise ValueError("Explicit corner variant/flag mismatch")
    if not flag and any(k in cfg for k in ("corner_protocol","corner_counterpart")):
        raise ValueError("Undeclared corner configuration")
    return flag


def validate_corner_config(cfg):
    if not corner_variant(cfg):return
    from model.action4_corner_supervision import LOSS_WEIGHTS,parameter_loss_dependencies
    if cfg.get("mode")!="integration":
        from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL
        proof=cfg.get("stage4_execution_gate")
        if (cfg.get("mode")!="full_training" or cfg.get("training_protocol")!=CORNER_NATIVE_PROTOCOL
                or not isinstance(proof,dict) or set(proof)!={"path","sha256"}):
            raise ValueError("Corner candidate research only: trainer/decoder/async admission pending")
    if type(cfg.get("corner_softmax_bias_frozen",False)) is not bool:raise ValueError("Explicit corner softmax bias policy required")
    if cfg.get("corner_protocol")!=CORNER_PROTOCOL:
        raise ValueError("Corner protocol/readiness/one-based epoch contract mismatch")
    from data.action4_prediction_roi_data import roi_finetune
    peak=2e-5 if roi_finetune(cfg) else 1e-4
    if cfg.get("corner_lr")!=peak or cfg.get("corner_quality_lr")!=peak:
        raise ValueError("Corner candidate learning rates must be explicit")
    if any(cfg.get("loss_weights",{}).get(k)!=v for k,v in LOSS_WEIGHTS.items()):
        raise ValueError("Corner loss configuration incomplete or changed")
    counterpart=cfg.get("corner_counterpart")
    if not isinstance(counterpart,dict) or set(counterpart)!={"B0","train_sam_cache","eval_sam_cache"}:
        raise ValueError("Explicit frozen B0/train/eval SAM counterpart bindings required")
    from data.action4_paths import PROJECT_ROOT,contained_path
    for ref in counterpart.values():
        if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:
            raise ValueError("Counterpart path/SHA binding required")
        contained_path(PROJECT_ROOT,ref["path"],exists=False)
        h=ref["sha256"]
        if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("Counterpart SHA invalid")
    from model.training.action4_supervision import A0_KEYS
    if set(cfg.get("loss_weights",{}))!=set(A0_KEYS)|{QUALITY_KEY}|set(LOSS_WEIGHTS):
        raise ValueError("Corner training requires exactly all A0/SAM/five-corner loss keys")
    dependencies=cfg.get("independent_head_dependencies",[])
    required=parameter_loss_dependencies("corner_head")
    corner_entries=[entry for entry in dependencies if any(p=="corner_head" or p.startswith("corner_head.") for p in entry["parameter_prefixes"])]
    if corner_entries!=required:
        raise ValueError("Exact corner independent parameter/loss ownership required")
    shared=[entry for entry in dependencies if set(entry["parameter_prefixes"])=={"sam_adapter.encoder","sam_adapter.projection"}]
    expected_shared=set(A0_KEYS)|set(LOSS_WEIGHTS).difference({"corner_quality_bce"})
    if len(shared)!=1 or set(shared[0]["losses"])!=expected_shared:
        raise ValueError("Shared SAM adapter requires exactly its A0/localization loss dependencies")
    surface={"parameter_prefixes":["head.surface"],"losses":list(A0_KEYS)}
    pose={"parameter_prefixes":["head.pose"],"losses":[k for k in A0_KEYS if k not in ("xyz","region","visible_mask","amodal_mask")]}
    if surface not in dependencies or pose not in dependencies:
        raise ValueError("Surface/pose owners must retain only actual A0 gradient paths")


def sam_variant(cfg):
    variant = cfg.get("a0_variant", "a0_baseline")
    flag = cfg.get("sam_enabled", False)
    if type(flag) is not bool or variant not in ("a0_baseline", SAM_VARIANT, CORNER_VARIANT):
        raise ValueError("Unknown A0 variant or nonboolean sam_enabled")
    corner_variant(cfg)
    if (variant in (SAM_VARIANT,CORNER_VARIANT)) != flag:
        raise ValueError("a0_variant and sam_enabled must agree explicitly")
    if flag:
        if cfg.get("role") != "A0" or cfg.get("train_backbone", False):
            raise ValueError("SAM A0 contract requires role A0 and frozen RGB backbone")
        if cfg.get("sam_protocol_id") != PROTOCOL_ID:
            raise ValueError("SAM protocol identity missing or mismatched")
    return flag


def sam_geometry_forward(head, adapter, features, rgb, K, canonical, *,
                         mask_aux=None, identities=None, return_aux=False):
    identities = _identities(() if identities is None else identities, len(rgb))
    if K.shape != (len(rgb), 3, 3) or canonical.shape != (len(rgb), 8, 3):
        raise ValueError("SAM joint forward requires matching batched K and fixed8 CAD")
    if not torch.isfinite(K).all() or not torch.isfinite(canonical).all():
        raise ValueError("Non-finite SAM joint geometry input")
    auxiliary = adapter(features, rgb=rgb, identities=identities, mask_aux=mask_aux)
    output = instancewise_geometry(head, auxiliary.c5, K, canonical)
    return (output, auxiliary) if return_aux else output


class SamGeometryHead(nn.Module):
    """Bundle-load interface on existing C5. It owns NO RGB backbone."""
    def __init__(self, geometry, sam_adapter):
        super().__init__()
        self.geometry = geometry
        self.sam_adapter = sam_adapter

    def forward(self, features, K, canonical, *, rgb, mask_aux=None, identities=None, return_aux=False):
        return sam_geometry_forward(self.geometry, self.sam_adapter, features, rgb, K, canonical,
                                    mask_aux=mask_aux, identities=identities, return_aux=return_aux)


def sam_items_enabled(items):
    flags = [item.get("sam_enabled", False) for item in items]
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("sam_enabled item marker must be boolean")
    enabled = any(flags)
    if enabled and not all(flags):
        raise ValueError("Mixed baseline/SAM items: all SAM ranks/dummies need the fixed variant marker")
    return enabled


def validate_sam_item(item):
    if item.get("sam_enabled") is not True:
        raise ValueError("Enhanced batches require sam_enabled=True on every item, including dummy/missing")
    identity = item.get("identity")
    _identities((identity,), 1)
    aux = item.get("mask_aux")
    if not isinstance(aux, MaskAux) or aux.identities != (identity,) or aux.protocol_id != PROTOCOL_ID:
        raise ValueError("Per-item MaskAux/identity/protocol mismatch")
    if tuple(aux.visible.shape) != (1, 1, 448, 448) or tuple(aux.distance.shape) != (1, 1, 448, 448):
        raise ValueError("Per-item mask/distance must retain singleton batch")
    if tuple(aux.available.shape) != (1, 1, 1, 1) or tuple(aux.dropout.shape) != (1,):
        raise ValueError("Per-item mask availability/dropout shape mismatch")
    if tuple(aux.candidate_overlap.shape) != (1, 1):
        raise ValueError("Per-item candidate overlap shape mismatch")
    if item.get("gt_identity") != identity:
        raise ValueError("Separate GT identity must match the current ROI instance")
    if type(item.get("sam_quality_supervised")) is not bool:
        raise ValueError("Explicit boolean sam_quality_supervised required")
    gt = item.get("gt_visible448")
    if not isinstance(gt, torch.Tensor) or tuple(gt.shape) != (1, 448, 448):
        raise ValueError("Separate gt_visible448 [1,448,448] required (unknown may contain NaN)")
    return aux


def sam_quality_eligible(item):
    aux = validate_sam_item(item)
    return (bool(item["active"]) and item["sam_quality_supervised"]
            and bool(aux.available.flatten()[0]) and bool(aux.visible.bool().any())
            and not bool(aux.dropout[0]))


def stack_sam_items(items, device, dtype):
    auxiliaries = [validate_sam_item(item) for item in items]
    ids = tuple(item["identity"] for item in items)
    _identities(ids, len(items))
    floating = lambda name: torch.cat([getattr(aux, name) for aux in auxiliaries]).to(device=device, dtype=dtype)
    aux = MaskAux(floating("visible"), floating("distance"), floating("available"),
                  floating("candidate_overlap"), torch.cat([x.dropout for x in auxiliaries]).to(device),
                  ids)
    gt = torch.stack([item["gt_visible448"] for item in items]).to(device=device, dtype=dtype)
    supervised = torch.tensor([bool(item["active"]) and item["sam_quality_supervised"] for item in items],
                              dtype=torch.bool, device=device)
    return aux, ids, gt, supervised


def validate_mask_augmentation_config(cfg):
    if cfg.get("sam_mask_augmentation") is None:return
    if (cfg.get("role")!="A0" or cfg.get("sam_enabled") is not True
            or cfg.get("a0_variant") not in (SAM_VARIANT,CORNER_VARIANT)):
        raise ValueError("sam_mask_augmentation requires explicit SAM-enabled A0 variant; baseline/B0 configuration refused")
    corner_variant(cfg)


def validate_sam_training_config(cfg):
    validate_mask_augmentation_config(cfg)
    if not sam_variant(cfg):
        return
    validate_corner_config(cfg)
    expected = {"surface_lr": 1e-5, "lr": 5e-5, "sam_adapter_lr": 1e-5 if corner_variant(cfg) else 1e-4,
                "sam_quality_lr": 1e-4, "weight_decay": 1e-4}
    from data.action4_prediction_roi_data import roi_finetune
    if roi_finetune(cfg):expected.update(surface_lr=2e-6,lr=1e-5,sam_adapter_lr=2e-6,sam_quality_lr=2e-5)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError("Stage3 SAM learning-rate/decay contract mismatch: " + key)
    if cfg.get("loss_weights", {}).get(QUALITY_KEY) != 0.1:
        raise ValueError("Explicit SAM quality weight 0.1 required")
    dependencies = cfg.get("independent_head_dependencies", [])
    owners = []
    for entry in dependencies:
        prefixes = entry["parameter_prefixes"]
        if any(QUALITY_PREFIX == p or QUALITY_PREFIX.startswith(p + ".") or p.startswith(QUALITY_PREFIX + ".")
               for p in prefixes):
            owners.append(entry)
    if len(owners) != 1 or owners[0]["parameter_prefixes"] != [QUALITY_PREFIX] or owners[0]["losses"] != [QUALITY_KEY]:
        raise ValueError("Quality head requires exact independent ownership; shared adapter must not be cleared with quality")
