"""CPU-only pose-head adaptation to frozen best surface predictions."""
import os,json,math,time,traceback,fcntl,numpy as np,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_pose_map_dataset import load_maps
from model.action4_pose_maps import PoseFromMaps,geometry_losses
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash
from model.training.action4_async_recovery import current_attempt

def publish(model,cfg,step,out):
    queue=out/"eval";queue.mkdir(exist_ok=True);request=queue/f"step_{step:08d}.json"
    if request.exists():return
    while True:
        if list(queue.glob("*.failed.json")):raise RuntimeError("Pose eval failure")
        pending=[p for p in queue.glob("step_*.json") if p.name.count(".")==1 and not p.with_suffix(".done.json").exists()]
        if len(pending)<8:break
        time.sleep(3)
    path=out/f"eval_step_{step:08d}.pt"
    save_checkpoint_atomic(path,{"schema":"action4.pose_maps_eval.v1","model":model.state_dict(),"step":step,"config_id":canonical_hash(cfg),"role":"A0","resumable":False})
    atomic_json(request,{"step":step,"snapshot":str(path.relative_to(R))})

def train(config,resume=None):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"];attempt=current_attempt()
    lock=(out/"train.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"TRAIN_PROCESS.json").exists() and not resume:raise FileExistsError("No automatic repeat")
    torch.set_num_threads(cfg["train_threads"]);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg["seed"])
    maps,geometry,meta=load_maps(cfg["cache_output"],"train")
    if len(maps)!=3024:raise ValueError("All3024 training targets required")
    parent=torch.load(R/cfg["parent_bundle"],map_location="cpu",weights_only=True)
    if parent["role"]!="A0" or parent["selected_step"]!=cfg["parent_expected_step"]:raise ValueError("Selected A0 parent required")
    model=PoseFromMaps()
    model.load_state_dict({k:v for k,v in parent["model"].items() if k.startswith("pose.")},strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=.0001)
    total=cfg["epochs"]*math.ceil(len(maps)/cfg["batch"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=cfg["lr"]*.01)
    start_epoch=cursor=step=0
    if resume:
        state=torch.load(R/resume,map_location="cpu",weights_only=True)
        if state.get("schema")!="action4.pose_maps_train.v1" or state["config_id"]!=canonical_hash(cfg):
            raise ValueError("Only explicit same-run recovery allowed")
        if state["step"]>=total or (out/"TRAIN_COMPLETE.json").exists():raise ValueError("Completed training is not restarted")
        model.load_state_dict(state["model"],strict=True);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"]);start_epoch,cursor,step=state["epoch"],state["cursor"],state["step"]
    atomic_json(out/"TRAIN_PROCESS.json",{"pid":os.getpid(),"attempt_id":attempt,"CPU_only":True,"threads":cfg["train_threads"]})
    if step%cfg["eval_every"]==0:publish(model,cfg,step,out)
    for epoch in range(start_epoch,cfg["epochs"]):
        order=np.random.default_rng(cfg["seed"]+epoch*1000003).permutation(len(maps))
        first_index=cursor if epoch==start_epoch else 0
        for first in range(first_index,len(order),cfg["batch"]):
            start=time.monotonic();indices=order[first:first+cfg["batch"]]
            g={k:v[indices] for k,v in geometry.items()}
            model.train();optimizer.zero_grad(set_to_none=True)
            output=model(maps[indices],g["K"],g["canonical"])
            losses=geometry_losses(output,g);loss=sum(cfg["loss_weights"][k]*v for k,v in losses.items())
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite geometry loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            status={"epoch":epoch+1,"step":step,"cursor":first+len(indices),"losses":{k:float(v.detach()) for k,v in losses.items()},"gradient_norm":float(norm),"seconds":time.monotonic()-start,"surface_frozen":True}
            atomic_json(out/"STATUS.json",status)
            with (out/"train.jsonl").open("a") as f:f.write(json.dumps(status)+"\n")
            if step%cfg["eval_every"]==0 or first+len(indices)==len(maps):
                save_checkpoint_atomic(out/"last.pt",{"schema":"action4.pose_maps_train.v1","model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"step":step, "epoch":epoch+1 if first+len(indices)==len(maps) else epoch,"cursor":0 if first+len(indices)==len(maps) else first+len(indices),
                    "rng_cpu":torch.get_rng_state(),"config_id":canonical_hash(cfg),"role":"A0","resumable":True,"purpose":"explicit same-run failure recovery; changed runs start selected best"})
            if step%cfg["eval_every"]==0:publish(model,cfg,step,out)
    publish(model,cfg,step,out);atomic_json(out/"TRAIN_COMPLETE.json",{"epochs":cfg["epochs"],"step":step,"attempt_id":attempt})

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");a=p.parse_args()
    try:train(a.config,a.resume)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg["output"]/"TRAIN_FAILURE.json",{"attempt_id":current_attempt(),"traceback":traceback.format_exc()});raise
