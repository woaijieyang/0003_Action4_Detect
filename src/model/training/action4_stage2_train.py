"""Independent-role DDP training with explicit tail mass and full RGB model snapshots."""
import json,os,math,time,fcntl,hashlib,datetime,traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_stage2_data import Stage2Units,Stage2Batches
from model.training.action4_stage2_core import construct,optimizer_for,losses_on_batch,objective,bundle_snapshot,validate_run_gate
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash
from model.training.action4_component_train import seed,rng_state,restore_rng
from model.training.action4_async_recovery import current_attempt

def publish(model,parent,cfg,step,out):
    queue=out/"eval";queue.mkdir(exist_ok=True);request=queue/f"step_{step:08d}.json"
    if request.exists():return
    while True:
        if list(queue.glob("*.failed.json")):raise RuntimeError("Actual async eval failure")
        pending=[p for p in queue.glob("step_*.json") if p.name.count(".")==1 and not p.with_suffix(".done.json").exists()]
        if len(pending)<cfg.get("max_pending",8):break
        time.sleep(2)
    snapshot=out/f"eval_step_{step:08d}.pt"
    save_checkpoint_atomic(snapshot,bundle_snapshot(model,parent,cfg,step))
    atomic_json(request,{"snapshot":str(snapshot.relative_to(R)),"step":step,"role":cfg["role"],"config_id":canonical_hash(cfg)})

def train(config,resume=None,stop_after=None):
    cfg=json.loads(contained_path(R,config,exists=True).read_text());out=contained_path(R,cfg["output"],exists=True)
    if (out/"TRAIN_COMPLETE.json").exists():raise FileExistsError("Terminal training must not restart")
    validate_run_gate(cfg,json.loads(contained_path(R,cfg["integration_or_admission_receipt"],exists=True).read_text()))
    if cfg["role"] not in ("A0","B0") or cfg["mode"] not in ("integration","full_training"):raise ValueError("Explicit role/mode required")
    if cfg["eval_every"]!={"A0":25,"B0":20}[cfg["role"]]:raise ValueError("Optimizer-step eval protocol changed")
    if stop_after is not None and cfg["mode"]!="integration":raise ValueError("Controlled stop probe only allowed in explicit integration mode")
    if cfg["mode"]=="full_training" and cfg["epochs"]<80:raise ValueError("Full stage requires at least80epochs")
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
    if world!=cfg["world_size"]:raise ValueError("DDP topology mismatch")
    torch.cuda.set_device(local);device=f"cuda:{local}";attempt=current_attempt()
    dist.init_process_group("nccl",device_id=torch.device(device),timeout=datetime.timedelta(seconds=180))
    guard=(out/"train.lock").open("a")
    if rank==0:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in cfg["source_identity"].items():
            if hashlib.sha256(contained_path(R,name,exists=True).read_bytes()).hexdigest()!=digest:raise ValueError("Training source changed: "+name)
    dist.barrier();torch.set_num_threads(cfg.get("threads",2));seed(cfg["seed"]);rng=np.random.default_rng(cfg["seed"]+rank)
    data=Stage2Units(cfg)
    if len(data)!=cfg["expected_units"]:raise ValueError("Admitted role unit count changed")
    model,parent=construct(cfg,device);optimizer=optimizer_for(model,cfg)
    total_steps=cfg["epochs"]*math.ceil(len(data)/cfg["batch"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_steps,eta_min=min(g["lr"] for g in optimizer.param_groups)*.01)
    epoch=cursor=step=0
    if resume:
        state=torch.load(contained_path(R,resume,exists=True),map_location="cpu",weights_only=True)
        if not state.get("resumable") or state["config_id"]!=canonical_hash(cfg) or state["world_size"]!=world:
            raise ValueError("Recovery must match exact role/config/topology")
        model.load_state_dict(state["training_model"],strict=True);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        epoch,cursor,step=state["epoch"],state["cursor"],state["step"];restore_rng(state["rng_by_rank"][rank],rng)
    elif (out/"last.pt").exists():raise FileExistsError("Explicit same-run recovery required; new experiments start from best")
    model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    pids=[None]*world;dist.all_gather_object(pids,os.getpid())
    if rank==0:
        atomic_json(out/"TRAIN_PROCESS.json",{"pids":pids,"attempt_id":attempt,"role":cfg["role"],"units":len(data),"resumed_from_step":step if resume else None})
        if step%cfg["eval_every"]==0:publish(model.module,parent,cfg,step,out)
    sampler=Stage2Batches(len(data),cfg["batch"],world,rank,cfg["seed"])
    workers=cfg.get("loader_workers",0)
    loader=DataLoader(data,batch_sampler=sampler,collate_fn=list,num_workers=workers,
        persistent_workers=workers>0,**({"prefetch_factor":2} if workers else {}),
        generator=torch.Generator().manual_seed(cfg["seed"]))
    for ep in range(epoch,cfg["epochs"]):
        start=cursor if ep==epoch else 0;sampler.set_epoch(ep,start);seen=start
        for items in loader:
            begin=time.monotonic();global_n=items[0]["global_count"];local_n=sum(x["active"] for x in items)
            check=torch.tensor(local_n,device=device);dist.all_reduce(check)
            if int(check)!=global_n:raise ValueError("Distributed sample mass mismatch")
            counts=None
            if cfg["role"]=="A0":
                counts=torch.tensor([sum(int(x["geometry"]["visible"].sum()) for x in items if x["active"]),
                                     sum(int((~x["geometry"]["visible"]).sum()) for x in items if x["active"])],device=device,dtype=torch.float32)
                dist.all_reduce(counts);counts={"visible":counts[0],"hidden":counts[1]}
            model.train();optimizer.zero_grad(set_to_none=True);logged={}
            micro_count=math.ceil(math.ceil(global_n/world)/cfg["micro"])
            for m in range(micro_count):
                mini=items[m*cfg["micro"]:(m+1)*cfg["micro"]]
                if not mini:mini=[dict(items[0],active=False)]
                active=sum(x["active"] for x in mini)
                context=model.no_sync() if m+1<micro_count else __import__("contextlib").nullcontext()
                with context:
                    losses=losses_on_batch(model,mini,cfg,device,counts)
                    loss=objective(losses,cfg,active,global_n,world)
                    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite role loss")
                    loss.backward()
                for key,value in losses.items():
                    logged[key]=logged.get(key,0.)+float(objective({key:value.detach()},cfg,active,global_n,world))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1;seen+=global_n
            ne,nc=(ep+1,0) if seen==len(data) else (ep,seen)
            values=torch.tensor([logged[k] for k in sorted(logged)],device=device);dist.all_reduce(values);values/=world
            status={"role":cfg["role"],"mode":cfg["mode"],"step":step,"epoch":ep+1,"cursor":seen,"global_actual_units":global_n,
                    "losses":dict(zip(sorted(logged),values.cpu().tolist())),"gradient_norm":float(norm),"seconds":time.monotonic()-begin}
            if rank==0:
                atomic_json(out/"STATUS.json",status)
                with (out/"train.jsonl").open("a") as f:f.write(json.dumps(status)+"\n")
            if step%cfg["eval_every"]==0 or nc==0 or (stop_after is not None and step>=stop_after):
                states=[None]*world if rank==0 else None;dist.gather_object(rng_state(rng),states,dst=0)
                if rank==0:
                    save_checkpoint_atomic(out/"last.pt",{"training_model":model.module.state_dict(),"optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict(),"rng_by_rank":states,"world_size":world,"epoch":ne,"cursor":nc,"step":step,
                        "role":cfg["role"],"config_id":canonical_hash(cfg),"resumable":True})
            if step%cfg["eval_every"]==0:
                if rank==0:publish(model.module,parent,cfg,step,out)
                dist.barrier()
            if stop_after is not None and step>=stop_after:
                if rank==0:atomic_json(out/"CHECKPOINTED_STOP.json",{"attempt_id":attempt,"step":step,"mode":"integration","completion":False})
                dist.barrier();dist.destroy_process_group();return
        if seen!=len(data):raise ValueError("Epoch coverage incomplete")
        cursor=0
    if rank==0:
        publish(model.module,parent,cfg,step,out)
        atomic_json(out/"TRAIN_COMPLETE.json",{"epochs":cfg["epochs"],"step":step,"attempt_id":attempt,"mode":cfg["mode"],"units_per_epoch":len(data),"role":cfg["role"],"formal_training_complete":cfg["mode"]=="full_training","config_id":canonical_hash(cfg)})
    dist.barrier();dist.destroy_process_group()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");p.add_argument("--stop-after",type=int);a=p.parse_args()
    try:train(a.config,a.resume,a.stop_after)
    except BaseException:
        cfg=json.loads(contained_path(R,a.config,exists=True).read_text())
        atomic_json(contained_path(R,cfg["output"],exists=True)/f"TRAIN_FAILURE_rank{os.environ.get('RANK','unknown')}.json",{"attempt_id":current_attempt(),"traceback":traceback.format_exc()});raise
