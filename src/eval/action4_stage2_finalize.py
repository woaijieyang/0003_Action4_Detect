"""Seal independent development-selected roles without promoting engineering runs."""
import json,hashlib,torch,math
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import canonical_hash,save_checkpoint_atomic,atomic_json
from model.action4_release_bundle import load_bundle

def seal(a0_config,b0_config,output,allow_integration=False):
    configs=[json.loads(contained_path(R,p,exists=True).read_text()) for p in [a0_config,b0_config]]
    if [c["role"] for c in configs]!=["A0","B0"]:raise ValueError("Explicit independent stage order required")
    if configs[0]["mode"]!=configs[1]["mode"] or configs[0]["validation_receipt"]!=configs[1]["validation_receipt"]:
        raise ValueError("Paired stages use different mode/dataset")
    mode=configs[0]["mode"]
    if mode=="integration" and not allow_integration:raise ValueError("Engineering checkpoints cannot be sealed as formal models")
    packages=[];receipts=[]
    for cfg in configs:
        root=contained_path(R,cfg["output"],exists=True)
        train=json.loads((root/"TRAIN_COMPLETE.json").read_text())
        complete=json.loads((root/"CONTROLLER_COMPLETE.json").read_text())
        eval_complete=json.loads((root/"EVAL_COMPLETE.json").read_text())
        best=json.loads((root/"BEST.json").read_text())
        if not (train["attempt_id"]==complete["attempt_id"]==eval_complete["attempt_id"]):
            raise ValueError("Completion receipts belong to different attempts")
        if train["role"]!=cfg["role"] or complete["role"]!=cfg["role"] or train["mode"]!=mode or complete["mode"]!=mode:
            raise ValueError("Completion role/mode mismatch")
        expected_steps=cfg["epochs"]*math.ceil(cfg["expected_units"]/cfg["batch"])
        if train["epochs"]!=cfg["epochs"] or train["step"]!=expected_steps or train["units_per_epoch"]!=cfg["expected_units"]:
            raise ValueError("Completed training coverage differs from configuration")
        if mode=="full_training" and any(r.get("config_id")!=canonical_hash(cfg) for r in [train,complete,eval_complete]):
            raise ValueError("Formal receipt lacks the exact config identity")
        if mode=="full_training" and (train["epochs"]<80 or not train["formal_training_complete"]):
            raise ValueError("Full training incomplete")
        if complete["best_step"]!=best["step"] or eval_complete["best_step"]!=best["step"]:
            raise ValueError("Selection was not reconciled after all eval workers")
        state=torch.load(root/"best.pt",map_location="cpu",weights_only=True)
        if state["role"]!=cfg["role"] or state["config_id"]!=canonical_hash(cfg) or state["selected_step"]!=best["step"]:
            raise ValueError("Selected model identity mismatch")
        state.update(selection_scope="development-selected independent stage; engineering only" if mode=="integration" else "development-selected; real acceptance remains separate",
                     completed_run_epochs=train["epochs"],completed_run_optimizer_steps=train["step"],
                     selected_report=best["raw_report"],source_checkpoint=str((root/"best.pt").relative_to(R)),
                     accuracy_accepted=False)
        packages.append(state);receipts.append(str((root/"TRAIN_COMPLETE.json").relative_to(R)))
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False);records={}
    for cfg,state in zip(configs,packages):
        folder=out/cfg["role"];folder.mkdir();path=folder/"best.pt";save_checkpoint_atomic(path,state)
        model,base,loaded=load_bundle(str(path),cfg["role"],"cpu")
        records[cfg["role"]]={"path":str(path.relative_to(R)),"step":state["selected_step"],
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"independent_strict_load":True}
        del model,base,loaded
    profile={"a0_bundle":records["A0"]["path"],"b0_bundle":records["B0"]["path"],
        "b0_scoring_policy":configs[1]["evaluation_scoring_policy"],
        "intrinsics_context":"outputs/action4_execution_source_release_v006/source_context.json",
        "demo_reference":"data/action4_reference/demo_standard.mp4","validation_receipt":configs[0]["validation_receipt"],
        "training_completion_receipts":receipts,"scope":mode+" paired selected weights; no automatic production promotion"}
    atomic_json(out/"PROFILE.json",profile)
    atomic_json(out/"SEALED.json",{"mode":mode,"models":records,"stage1_registry_unchanged":True,"accuracy_accepted":False})
    return profile

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--a0-config",required=True);p.add_argument("--b0-config",required=True);p.add_argument("--output",required=True);p.add_argument("--allow-integration",action="store_true")
    a=p.parse_args();seal(a.a0_config,a.b0_config,a.output,a.allow_integration)
