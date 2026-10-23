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

def load_bundle(path, role, device="cpu"):
    from model.action4_checkpoint_paths import resolve_checkpoint
    state = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=True)
    if state.get("schema") != SCHEMA or state.get("role") != role:
        raise ValueError("Stage bundle role/schema mismatch")
    cfg = state["config"]
    if canonical_hash(cfg) != state["config_id"]:
        raise ValueError("Embedded configuration identity mismatch")
    base = None
    if role == "B0":
        model = OnlineB0(cfg["network"])
    elif role == "A0":
        bc = state["backbone_config"]
        base = Candidate("A0", bc["network"], bc["variant"])
        base.load_state_dict(state["backbone"], strict=True)
        base.to(device).eval().requires_grad_(False)
        model = SurfaceHead() if state["architecture"] == "surface" else GDRPose()
    else:
        raise ValueError("Unsupported stage")
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), base, state

def warm_start(path, role, device="cuda:0", learning_rate=None):
    """Return selected weights + FRESH AdamW; starts a new experiment at step zero.

    Caller must supply validated training data and loss. A0 frozen backbone is
    returned separately; no validation/test target is an inference input.
    """
    model, base, state = load_bundle(path, role, device)
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
    target = contained_path(R, destination, exists=False)
    target.mkdir(exist_ok=False)
    states = {}
    for role, key in (("A0","a0"),("B0","b0")):
        training_cfg = json.loads(contained_path(R, cfg[key+"_config"], exists=True).read_text())
        source = torch.load(contained_path(R,cfg[key+"_checkpoint"],exists=True), map_location="cpu", weights_only=True)
        if source["config_id"] != canonical_hash(training_cfg):
            raise ValueError("Source configuration mismatch")
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
    for role,state in states.items():
        (target/role).mkdir()
        save_checkpoint_atomic(target/role/"best.pt", state)
    return {role:str((target/role/"best.pt").relative_to(R)) for role in states}

if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--destination",required=True)
    a=p.parse_args();print(json.dumps(export(a.selection,a.destination)))
