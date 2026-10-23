"""Shared independent-role loss, best initialization and self-contained snapshot contract."""
import copy
import torch
from model.action4_joint_surface_rgb import JointSurfaceRGB
from model.action4_release_bundle import load_bundle,SCHEMA
from model.action4_sparse_surface_losses import sparse_surface_losses
from model.action4_pose_maps import geometry_losses
from model.training.action4_online_b0_batch import batch_losses
from model.training.action4_runtime import canonical_hash

def construct(cfg,device):
    role=cfg["role"]
    pinned=None
    if cfg["mode"]=="full_training":
        from model.action4_best_registry import verify_reference,DEFAULT_REGISTRY
        pinned=verify_reference(role,cfg["parent_bundle"],cfg["parent_best_step"],cfg.get("parent_registry",DEFAULT_REGISTRY))
    if role=="A0":
        model=JointSurfaceRGB(cfg["parent_bundle"],cfg["train_backbone"],device)
        parent=model.parent
    elif role=="B0":
        model,_,parent=load_bundle(cfg["parent_bundle"],"B0",device)
    else:raise ValueError("Independent A0/B0 role required")
    if parent["selected_step"]!=cfg["parent_best_step"]:raise ValueError("Selected best parent mismatch")
    if role=="B0" and parent["config"]["network"]!=cfg["network"]:raise ValueError("B0 architecture changed")
    if pinned is not None:
        from model.action4_best_registry import verify_loaded
        verify_loaded(pinned,parent)
    return model.to(device),parent

def optimizer_for(model,cfg):
    if cfg["role"]=="A0":
        groups=[{"params":model.head.surface.parameters(),"lr":cfg["surface_lr"]},
                {"params":model.head.pose.parameters(),"lr":cfg["lr"]}]
        if model.train_backbone:groups.append({"params":model.base.backbone.parameters(),"lr":cfg["lr_backbone"]})
    else:
        groups=[{"params":model.detector.backbone.parameters(),"lr":cfg["lr_backbone"]},
                {"params":[p for n,p in model.detector.named_parameters() if not n.startswith("backbone.")],"lr":cfg["lr_detector"]},
                {"params":model.refiner.parameters(),"lr":cfg["lr_refiner"]}]
    return torch.optim.AdamW(groups,weight_decay=cfg.get("weight_decay",.0001))

def losses_on_batch(model,items,cfg,device,corner_normalizers=None):
    if cfg["role"]=="B0":
        return batch_losses(model,[x["item"] for x in items],cfg,device,
                            augmentation_seeds=[x["augmentation_seed"] for x in items] if cfg.get("photometric") else None)
    rois=torch.stack([x["roi"] for x in items]).to(device)
    geometry={k:torch.stack([x["geometry"][k] for x in items]).to(device) for k in items[0]["geometry"]}
    targets={"masks":torch.stack([x["surface_masks"] for x in items]).to(device),
             "sparse":[{k:v.to(device) for k,v in x["surface_sparse"].items()} for x in items]}
    output=model(rois,geometry["K"],geometry["canonical"])
    loss=sparse_surface_losses(output["surface"],targets)
    loss.update(geometry_losses(output,geometry,corner_normalizers))
    return loss

def objective(losses,cfg,active_count,global_count,world_size):
    # DDP averages gradients. Surface/box terms are per-object means; corner terms
    # already divide by global visible/hidden counts. Zero-weight padding is explicit.
    if global_count<=0 or active_count<0:raise ValueError("Invalid optimizer batch mass")
    return sum(cfg.get("loss_weights",{}).get(k,1.)*v*(
        (world_size if active_count else 0.) if cfg["role"]=="A0" and k in ("visible_corners","hidden_corners")
        else world_size*active_count/global_count) for k,v in losses.items())

def bundle_snapshot(model,parent,cfg,step):
    # CPU values are independent of subsequent optimizer updates.
    cpu=lambda state:{k:v.detach().cpu().clone() for k,v in state.items()}
    common={"schema":SCHEMA,"role":cfg["role"],"config":copy.deepcopy(cfg),"config_id":canonical_hash(cfg),
            "selected_step":int(step),"parent_selected_step":parent["selected_step"],"parent_bundle":cfg["parent_bundle"],
            "training_scope":cfg["mode"],"exact_optimizer_resume":False,
            "continuation":"best weights with fresh optimizer; exact recovery stored separately",
            "selection_scope":"snapshot awaiting development evaluation","resumable":False}
    if cfg["role"]=="A0":
        common.update(architecture="joint_pose",model=cpu(model.head.state_dict()),backbone=cpu(model.base.state_dict()),
            backbone_config=copy.deepcopy(parent["backbone_config"]),
            backbone_step=step if model.train_backbone else parent["backbone_step"],
            backbone_parent_step=parent["backbone_step"],backbone_frozen=not model.train_backbone,
            surface_frozen=False,pose_method="network")
    else:common.update(architecture="online_b0",model=cpu(model.state_dict()),inference_scoring_policy=cfg["evaluation_scoring_policy"])
    return common


def validate_run_gate(cfg,gate):
    """An integration receipt cannot silently authorize a full-data training run."""
    if not gate.get("launch_allowed") or gate.get("mode")!=cfg["mode"]:
        raise ValueError("Run mode differs from its explicit launch/admission gate")
    if cfg["mode"]=="full_training":
        if gate.get("formal_stage2_allowed") is not True:raise ValueError("Formal second stage not admitted")
        if gate.get("validation_receipt")!=cfg["validation_receipt"] or gate.get("role_units",{}).get(cfg["role"])!=cfg["expected_units"]:
            raise ValueError("Formal dataset/role coverage differs from admission")
        if gate.get("eval_images")!=cfg["expected_eval_images"]:raise ValueError("Evaluation dataset differs from admission")
        if cfg["role"]=="A0" and gate.get("surface_source")!=cfg["surface_source"]:
            raise ValueError("Surface sidecar version differs from admission")
        if cfg["role"]=="A0":
            import json
            from data.action4_paths import PROJECT_ROOT as R,contained_path
            receipt=json.loads(contained_path(R,cfg["validation_receipt"],exists=True).read_text())
            bindings=cfg["surface_source"].get("manifest_sha256",{})
            if any(bindings.get(split)!=receipt["splits"][split]["sha256"] for split in ("train","eval","test")):
                raise ValueError("Formal surface sidecars are not bound to the same three dataset manifests")
    elif cfg["mode"]!="integration":raise ValueError("Unknown run mode")
