"""Self-contained stage weights; explicit best-weight warm starts, no fake resume."""
import copy
import torch
from model.action4_online_b0 import OnlineB0
from model.action4_component_experiments import Candidate
from model.action4_surface_head import SurfaceHead
from model.action4_gdr_pose import GDRPose
from data.action4_paths import PROJECT_ROOT as R, contained_path
from model.training.action4_runtime import canonical_hash, save_checkpoint_atomic

SCHEMA = "action4.stage_bundle.v1"
SAM_SCHEMA = "action4.stage_bundle.sam.v1"
CORNER_SCHEMA = "action4.stage_bundle.corner_candidate.v1"
B0_GUIDANCE_SCHEMA = "action4.stage_bundle.b0_guidance.v1"

def _load_verified_checkpoint(path,expected_sha256):
    """Hash and deserialize one resolved file handle; reject replacement races."""
    import hashlib,os
    from pathlib import Path
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("Explicit parent SHA256 required")
    resolved=Path(path).resolve(strict=True)
    if not resolved.is_relative_to(R.resolve()):raise ValueError("Parent checkpoint escaped project")
    def fingerprint(stat):return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    before=fingerprint(resolved.stat())
    with resolved.open("rb") as stream:
        if fingerprint(os.fstat(stream.fileno()))!=before:raise ValueError("Parent replaced before verified read")
        if hashlib.file_digest(stream,"sha256").hexdigest()!=expected_sha256:
            raise ValueError("Consumed parent checkpoint SHA mismatch")
        if fingerprint(os.fstat(stream.fileno()))!=before or fingerprint(resolved.stat())!=before:
            raise ValueError("Parent replaced/modified after hash")
        stream.seek(0)
        state=torch.load(stream,map_location="cpu",weights_only=True)
        if fingerprint(os.fstat(stream.fileno()))!=before or fingerprint(resolved.stat())!=before:
            raise ValueError("Parent replaced/modified during weight load")
    return state


def load_bundle(path, role, device="cpu", *, allow_corner_candidate=False, expected_sha256=None):
    from model.action4_checkpoint_paths import resolve_checkpoint
    resolved=resolve_checkpoint(path)
    state = (torch.load(resolved,map_location="cpu",weights_only=True) if expected_sha256 is None
             else _load_verified_checkpoint(resolved,expected_sha256))
    if type(allow_corner_candidate) is not bool:raise ValueError("Explicit research-loader permission required")
    corner=state.get("schema")==CORNER_SCHEMA
    if corner and not allow_corner_candidate:raise ValueError("Unadmitted corner candidate: explicit research loader required; decoder/quality/eval pending")
    if not corner and any(k in state for k in ("corner_head","corner_protocol","corner_counterpart","corner_parent")):
        raise ValueError("Undeclared corner state/schema")
    if state.get("schema") not in (SCHEMA, SAM_SCHEMA, B0_GUIDANCE_SCHEMA, CORNER_SCHEMA) or state.get("role") != role:
        raise ValueError("Stage bundle role/schema mismatch")
    if (role != "A0" and state.get("schema") in (SAM_SCHEMA,CORNER_SCHEMA)) or (role != "B0" and state.get("schema") == B0_GUIDANCE_SCHEMA):
        raise ValueError("SAM bundle schema is A0-only")
    cfg = state["config"]
    if canonical_hash(cfg) != state["config_id"]:
        raise ValueError("Embedded configuration identity mismatch")
    base = None
    if role == "B0":
        from model.action4_b0_fusion import GuidedB0,enabled,ARCHITECTURE,PROTOCOL
        enhanced=state.get("schema")==B0_GUIDANCE_SCHEMA
        if enhanced!=enabled(cfg) or enhanced!=(state.get("architecture")==ARCHITECTURE):raise ValueError("B0 guidance schema/architecture/config mismatch")
        if enhanced and state.get("guidance_protocol")!=PROTOCOL:raise ValueError("B0 guidance protocol mismatch")
        if not enhanced and "guidance_protocol" in state:raise ValueError("Undeclared guidance state")
        model = GuidedB0(cfg["network"]) if enhanced else OnlineB0(cfg["network"])
    elif role == "A0":
        from model.action4_sam_joint import sam_variant, SAM_ARCHITECTURE, BUNDLE_PROTOCOL, SamGeometryHead, CORNER_ARCHITECTURE, CORNER_PROTOCOL, corner_variant, validate_corner_config
        from model.action4_sam_feature_adapter import SamFeatureAdapter
        enhanced = state["architecture"] in (SAM_ARCHITECTURE,CORNER_ARCHITECTURE)
        if corner!=(state["architecture"]==CORNER_ARCHITECTURE) or corner!=corner_variant(cfg):raise ValueError("Corner schema/architecture/config mismatch")
        if corner:
            validate_corner_config(cfg)
            if (state.get("corner_protocol")!=CORNER_PROTOCOL or state.get("corner_counterpart")!=cfg["corner_counterpart"]
                    or state.get("quality_ready") is not False or state.get("deployment_admitted") is not False
                    or state.get("pose_method")!="network_raw_corner_research_only" or "corner_head" not in state):
                raise ValueError("Incomplete/unadmitted corner bundle contract")
            expected_parent={"path":cfg["parent_bundle"],"sha256":cfg["parent_bundle_sha256"],"selected_step":cfg["parent_best_step"]}
            if (state.get("corner_parent")!=expected_parent or state.get("parent_bundle")!=expected_parent["path"]
                    or state.get("parent_selected_step")!=expected_parent["selected_step"]):
                raise ValueError("Corner parent initialization identity mismatch")
        if enhanced != (state["schema"] in (SAM_SCHEMA,CORNER_SCHEMA)):
            raise ValueError("A0 bundle SAM schema/architecture mismatch")
        if state["architecture"] not in ("surface", "joint_pose", SAM_ARCHITECTURE, CORNER_ARCHITECTURE):
            raise ValueError("Unsupported A0 bundle architecture")
        if enhanced != sam_variant(cfg):
            raise ValueError("A0 bundle architecture/config SAM variant mismatch")
        if not enhanced and any(k in state for k in ("sam_adapter", "sam_protocol")):
            raise ValueError("Baseline bundle carries undeclared SAM parameters")
        if enhanced and state.get("sam_protocol") != BUNDLE_PROTOCOL:
            raise ValueError("SAM bundle protocol identity mismatch")
        bc = state["backbone_config"]
        base = Candidate("A0", bc["network"], bc["variant"])
        base.load_state_dict(state["backbone"], strict=True)
        base.to(device).eval().requires_grad_(False)
        model = SurfaceHead() if state["architecture"] == "surface" else GDRPose()
    else:
        raise ValueError("Unsupported stage")
    model.load_state_dict(state["model"], strict=True)
    if role == "A0" and enhanced:
        adapter = SamFeatureAdapter()
        adapter.load_state_dict(state["sam_adapter"], strict=True)
        if corner:
            from model.action4_corner_supervision import CornerObservationHead
            from model.action4_joint_surface_rgb import JointSurfaceRGB
            corner_head=CornerObservationHead()
            corner_head.load_state_dict(state["corner_head"],strict=True)
            model=JointSurfaceRGB.from_corner_components(model,base,state,adapter,corner_head)
        else:
            model = SamGeometryHead(model, adapter)
    return model.to(device).eval(), base, state

def warm_start(path, role, device="cuda:0", learning_rate=None):
    """Return selected weights + FRESH AdamW; starts a new experiment at step zero.

    Caller must supply validated training data and loss. A0 frozen backbone is
    returned separately; no validation/test target is an inference input.
    """
    model, base, state = load_bundle(path, role, device)
    if state.get("architecture") in ("joint_pose_sam","online_b0_guidance"):
        raise ValueError("Enhanced warm start requires explicit stage2_core.construct/optimizer_for SAM contract")
    model.train()
    cfg = state["config"]
    if role == "B0":
        groups = [
            {"params": model.detector.backbone.parameters(), "lr": cfg["lr_backbone"]},
            {"params": [p for n,p in model.detector.named_parameters() if not n.startswith("backbone.")], "lr": cfg["lr_detector"]},
            {"params": model.refiner.parameters(), "lr": cfg["lr_refiner"]},
        ]
        if learning_rate is not None:
            for group in groups: group["lr"] = learning_rate
        optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    elif state["architecture"] == "joint_pose":
        groups = [
            {"params":model.surface.parameters(), "lr":learning_rate if learning_rate is not None else cfg["surface_lr"]},
            {"params":model.pose.parameters(), "lr":learning_rate if learning_rate is not None else cfg["lr"]},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate if learning_rate is not None else cfg["lr"], weight_decay=.0001)
    return model, base, optimizer, state

def export(selection, destination):
    import json
    cfg = json.loads(contained_path(R, selection, exists=True).read_text())
    if cfg.get("a0_kind") == "joint_pose_sam":
        raise ValueError("Legacy export cannot serialize enhanced SAM weights; use complete bundle_snapshot")
    target = contained_path(R, destination, exists=False)
    states = {}
    for role, key in (("A0","a0"),("B0","b0")):
        training_cfg = json.loads(contained_path(R, cfg[key+"_config"], exists=True).read_text())
        source = torch.load(contained_path(R,cfg[key+"_checkpoint"],exists=True), map_location="cpu", weights_only=True)
        if source["config_id"] != canonical_hash(training_cfg):
            raise ValueError("Source configuration mismatch")
        if role == "B0" and (training_cfg.get("b0_variant")=="b0_fusion_image" or source.get("architecture")=="online_b0_guidance"):
            raise ValueError("Legacy export cannot serialize guidedB0; use complete bundle_snapshot")
        if role == "A0":
            from model.action4_sam_joint import sam_variant, SAM_ARCHITECTURE
            training_keys=source.get("training_model", {})
            if (sam_variant(training_cfg) or source.get("architecture")==SAM_ARCHITECTURE
                    or "sam_adapter" in source or "sam_protocol" in source
                    or any(k.startswith("sam_adapter.") for k in training_keys)):
                raise ValueError("Legacy export cannot serialize enhanced SAM weights; use complete bundle_snapshot")
        state = {
            "schema":SCHEMA, "role":role, "config":training_cfg,
            "config_id":source["config_id"], "selected_step":source["step"],
            "model":source["model"] if role=="B0" else source["head"],
            "architecture":"online_b0" if role=="B0" else cfg.get("a0_kind","joint_pose"),
            "selection_scope":"development-selected; real acceptance remains separate",
            "exact_optimizer_resume":False,
            "continuation":"warm_start selected weights with fresh optimizer; no last/best optimizer mixing",
            "source_checkpoint":cfg[key+"_checkpoint"],
        }
        if role=="A0":
            parent=torch.load(contained_path(R,cfg["a0_backbone"],exists=True),map_location="cpu",weights_only=True)
            if parent.get("role")!="A0": raise ValueError("Wrong backbone role")
            state.update(backbone=parent["model"],backbone_config=parent["config"],
                         backbone_step=parent["step"],pose_method=cfg["pose_method"])
        states[role]=state
    target.mkdir(exist_ok=False)
    for role,state in states.items():
        (target/role).mkdir()
        save_checkpoint_atomic(target/role/"best.pt", state)
    return {role:str((target/role/"best.pt").relative_to(R)) for role in states}

if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--destination",required=True)
    a=p.parse_args();print(json.dumps(export(a.selection,a.destination)))
