"""Actual RGB jitter -> frozen surface -> trainable pose; exact distributed reductions."""
import os,json,time,math,fcntl,traceback,datetime,hashlib,numpy as np,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from data.action4_paths import PROJECT_ROOT as R
from data.action4_online_pose_data import PoseRGBTrainingData,PoseEpochBatches
from model.action4_online_pose_rgb import OnlinePoseRGB
from model.action4_pose_maps import geometry_losses
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash
from model.training.action4_component_train import seed,rng_state,restore_rng
from model.training.action4_async_recovery import current_attempt

POINT_LOSSES=("visible_corners","hidden_corners")
def distributed_objective(losses,weights,world):
    # DDP averages rank gradients: compensate only globally normalized point sums.
    return sum(weights[k]*v*(world if k in POINT_LOSSES else 1.) for k,v in losses.items())

def publish(model,cfg,step,out):
    queue=out/"eval";queue.mkdir(exist_ok=True);request=queue/f"step_{step:08d}.json"
    if request.exists():return
    while True:
        if list(queue.glob("*.failed.json")):raise RuntimeError("Async eval failure")
        pending=[p for p in queue.glob("step_*.json") if p.name.count(".")==1 and not p.with_suffix(".done.json").exists()]
        if len(pending)<8:break
        time.sleep(3)
    path=out/f"eval_step_{step:08d}.pt"
    save_checkpoint_atomic(path,{"schema":"action4.pose_maps_eval.v1","role":"A0","model":model.pose.state_dict(),
        "step":step,"config_id":canonical_hash(cfg),"resumable":False})
    atomic_json(request,{"step":step,"snapshot":str(path.relative_to(R))})

def train(config,resume=None):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"];attempt=current_attempt()
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
    if world!=cfg["world_size"]:raise ValueError("World size differs from frozen config")
    torch.cuda.set_device(local);device=f"cuda:{local}"
    dist.init_process_group("nccl",timeout=datetime.timedelta(seconds=180),device_id=torch.device(device))
    guard=(out/"train.lock").open("a")
    if rank==0:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in cfg["source_identity"].items():
            if hashlib.sha256((R/name).read_bytes()).hexdigest()!=digest:raise ValueError("Training source changed: "+name)
    dist.barrier();torch.set_num_threads(2);seed(cfg["seed"]);rng=np.random.default_rng(cfg["seed"]+91)
    data=PoseRGBTrainingData(cfg["validation_receipt"],cfg["seed"],cfg["roi_jitter"])
    if len(data)!=3024:raise ValueError("Expected complete3024 target training set")
    model=OnlinePoseRGB(cfg["parent_bundle"]).to(device)
    if model.parent_step!=cfg["parent_expected_step"]:raise ValueError("Selected-best parent mismatch")
    optimizer=torch.optim.AdamW(model.pose.parameters(),lr=cfg["lr"],weight_decay=.0001)
    total=cfg["epochs"]*(len(data)//cfg["batch"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=cfg["lr"]*.01)
    epoch=cursor=step=0
    if resume:
        state=torch.load(R/resume,map_location="cpu",weights_only=True)
        if state["config_id"]!=canonical_hash(cfg) or state["world_size"]!=world or not state["resumable"]:
            raise ValueError("Only explicit same-run recovery allowed")
        if state["step"]>=total or (out/"TRAIN_COMPLETE.json").exists():raise ValueError("Completed run cannot restart")
        model.pose.load_state_dict(state["model"],strict=True);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        epoch,cursor,step=state["epoch"],state["cursor"],state["step"];restore_rng(state["rng_by_rank"][rank],rng)
    elif (out/"last.pt").exists():raise FileExistsError("Explicit recovery required")
    model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    pids=[None]*world;dist.all_gather_object(pids,os.getpid())
    if rank==0:
        atomic_json(out/"TRAIN_PROCESS.json",{"pids":pids,"attempt_id":attempt,"world_size":world,"frozen_backbone_surface":True})
        if step%cfg["eval_every"]==0:publish(model.module,cfg,step,out)
    sampler=PoseEpochBatches(len(data),cfg["batch"],world,rank,cfg["seed"])
    loader=DataLoader(data,batch_sampler=sampler,collate_fn=list,num_workers=cfg["loader_workers"],persistent_workers=True,
        prefetch_factor=2,generator=torch.Generator().manual_seed(cfg["seed"]))
    for ep in range(epoch,cfg["epochs"]):
        start=cursor if ep==epoch else 0;sampler.set_epoch(ep,start);seen=start
        for items in loader:
            begin=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            rois=torch.stack([x["roi"] for x in items]).to(device)
            g={k:torch.stack([x["geometry"][k] for x in items]).to(device) for k in items[0]["geometry"]}
            counts=torch.stack([g["visible"].sum(),(~g["visible"]).sum()]).to(dtype=torch.float32)
            dist.all_reduce(counts,op=dist.ReduceOp.SUM)
            output=model(rois,g["K"],g["canonical"])
            losses=geometry_losses(output,g,{"visible":counts[0],"hidden":counts[1]})
            loss=distributed_objective(losses,cfg["loss_weights"],world)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite pose objective")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.module.pose.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1;seen+=len(items)*world
            ne,nc=(ep+1,0) if seen==len(data) else (ep,seen)
            values=torch.stack([losses[k].detach()*(world if k in POINT_LOSSES else 1.) for k in sorted(losses)])
            dist.all_reduce(values);values/=world
            if rank==0:
                status={"step":step,"epoch":ep+1,"cursor":seen,"losses":dict(zip(sorted(losses),values.cpu().tolist())),
                    "gradient_norm":float(norm),"seconds":time.monotonic()-begin,"world_size":world,
                    "global_visible_hidden_counts":counts.cpu().tolist(),"peak_memory_mib":torch.cuda.max_memory_allocated()/1048576,
                    "first_target":{"sample_id":items[0]["metadata"]["sample_id"],"instance_id":items[0]["metadata"]["instance_id"],"crop_box":items[0]["metadata"]["crop_box"]}}
                atomic_json(out/"STATUS.json",status)
                with (out/"train.jsonl").open("a") as f:f.write(json.dumps(status)+"\n")
            if step%cfg["eval_every"]==0 or nc==0:
                states=[None]*world if rank==0 else None;dist.gather_object(rng_state(rng),states,dst=0)
                if rank==0:
                    save_checkpoint_atomic(out/"last.pt",{"schema":"action4.online_pose_train.v1","role":"A0","model":model.module.pose.state_dict(),
                        "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng_by_rank":states,"world_size":world,
                        "config_id":canonical_hash(cfg),"epoch":ne,"cursor":nc,"step":step,"resumable":True})
            if step%cfg["eval_every"]==0:
                if rank==0:publish(model.module,cfg,step,out)
                dist.barrier()
        cursor=0
    if rank==0:
        publish(model.module,cfg,step,out);atomic_json(out/"TRAIN_COMPLETE.json",{"epochs":cfg["epochs"],"step":step,"attempt_id":attempt})
    dist.barrier();dist.destroy_process_group()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");a=p.parse_args()
    try:train(a.config,a.resume)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg["output"]/f"TRAIN_FAILURE_rank{os.environ.get('RANK','unknown')}.json",
            {"attempt_id":current_attempt(),"traceback":traceback.format_exc()});raise
