"""Export completed independent trials; never publish a partial or bare pose head."""
import copy,json,math,torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import canonical_hash,save_checkpoint_atomic
from model.action4_release_bundle import SCHEMA

def completed_best(cfg,role):
    out=contained_path(R,cfg["output"],exists=True)
    complete=json.loads((out/"TRAIN_COMPLETE.json").read_text())
    size=1600 if role=="B0" else 3024
    total=cfg["epochs"]*math.ceil(size/cfg["batch"])
    if complete.get("epochs")!=80 or complete.get("step",complete.get("steps"))!=total:
        raise ValueError("Full eighty-epoch training required")
    queue=out/"eval";requests=[p for p in queue.glob("step_*.json") if p.name.count(".")==1]
    expected=set(range(0,total+1,cfg["eval_every"]))|{total}
    if list(queue.glob("*.failed.json")) or any(not p.with_suffix(".done.json").exists() for p in requests):
        raise ValueError("Unfinished/failed evaluation")
    reports=[json.loads(p.with_suffix(".done.json").read_text()) for p in requests]
    if {x["step"] for x in reports}!=expected:raise ValueError("Evaluation coverage mismatch")
    if role=="B0":
        if not (out/"ALL_EVAL_SELECTION.json").exists():raise ValueError("All evaluator writers must reconcile")
        rank=lambda x:(x["report"]["mAP50_95"],x["report"]["overall"]["f1"],-x["step"])
    else:
        if not (out/"EVAL_COMPLETE.json").exists():raise ValueError("A0 evaluator still active")
        rank=lambda x:(x["report"]["all8"],x["report"]["valid"],-(x["report"]["hidden_mean_valid_only"] or 0),-x["step"])
    winner=max(reports,key=rank);best=json.loads((out/"BEST.json").read_text())
    if winner["step"]!=best["step"]:raise ValueError("Published best is not global completed best")
    state=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    if state["step"]!=best["step"] or state["config_id"]!=canonical_hash(cfg):
        raise ValueError("Best checkpoint configuration mismatch")
    return state,best

def compose_pose_bundle(parent,surface,pose,cfg):
    if parent["role"]!="A0" or parent["backbone_step"]!=6300 or surface["step"]!=cfg["frozen_surface_step"]:
        raise ValueError("Pose/surface/backbone lineage mismatch")
    model={"surface."+k:v for k,v in surface["head"].items()}
    model.update(pose["model"])
    if set(model)!=set(parent["model"]) or any(model[k].shape!=parent["model"][k].shape for k in model):
        raise ValueError("Combined RGB model topology mismatch")
    runtime_cfg=dict(cfg,surface_lr=0.)
    bundle=dict(parent,model=model,config=runtime_cfg,config_id=canonical_hash(runtime_cfg),
        training_config=cfg,training_config_id=canonical_hash(cfg),selected_step=pose["step"],
        pose_method="network",source_checkpoint=cfg["output"]+"/best.pt",
        surface_frozen=True,frozen_surface_step=surface["step"],
        selection_scope="GTROI selected pose candidate; full predicted ROI and real video still required")
    return bundle

def export(config,role,output):
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    state,best=completed_best(cfg,role)
    if role=="A0":
        parent=torch.load(contained_path(R,cfg["parent_bundle"],exists=True),map_location="cpu",weights_only=True)
        surface=torch.load(contained_path(R,cfg["frozen_surface_checkpoint"],exists=True),map_location="cpu",weights_only=True)
        bundle=compose_pose_bundle(parent,surface,state,cfg)
    else:
        bundle={"schema":SCHEMA,"role":"B0","architecture":"online_b0","config":cfg,"config_id":canonical_hash(cfg),
            "selected_step":state["step"],"model":state["model"],"source_checkpoint":cfg["output"]+"/best.pt",
            "inference_scoring_policy":cfg["evaluation_scoring_policy"],"exact_optimizer_resume":False,
            "continuation":"selected best weights with fresh optimizer",
            "selection_scope":"development-selected; real video validation required"}
    destination=contained_path(R,output,exists=False)
    if destination.exists():raise FileExistsError("Never replace a sealed model automatically")
    save_checkpoint_atomic(destination,bundle)
    return {"path":str(destination.relative_to(R)),"step":state["step"],"role":role,"real_acceptance":False}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--role",choices=["A0","B0"],required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(export(a.config,a.role,a.output)))
