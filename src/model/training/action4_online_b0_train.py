"""End-to-end B0 continuation; best parents,80epochs,20step async eval."""
import json,os,math,time,fcntl,hashlib,traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from model.action4_online_b0 import OnlineB0
from model.training.action4_online_b0_batch import batch_losses
from model.training.action4_component_train import seed,rng_state,restore_rng
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash
from model.training.action4_async_recovery import current_attempt

def publish(model,cfg,step,out):
    queue=out/"eval";queue.mkdir(exist_ok=True);request=queue/f"step_{step:08d}.json"
    if request.exists():return
    while len([p for p in queue.glob("step_*.json") if p.name.count(".")==1 and not p.with_suffix(".done.json").exists()])>=8:
        if list(queue.glob("*.failed.json")):raise RuntimeError("Async eval failure")
        time.sleep(3)
    path=out/f"eval_step_{step:08d}.pt"
    save_checkpoint_atomic(path,{"model":model.state_dict(),"config_id":canonical_hash(cfg),"step":step,"role":"B0","resumable":False})
    atomic_json(request,{"snapshot":str(path.relative_to(R)),"step":step})

def train(config,resume=None,stop_after=None):
    attempt=current_attempt();cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
    if world!=2:raise ValueError("This run requires exactly2ranks")
    torch.cuda.set_device(local);device=f"cuda:{local}";dist.init_process_group("nccl")
    guard=(out/"train.lock").open("a")
    if rank==0:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if rank==0:
        for path,digest in {**cfg["source_identity"],**cfg["data_identity"]}.items():
            if hashlib.sha256((R/path).read_bytes()).hexdigest()!=digest:raise ValueError("Identity changed: "+path)
    dist.barrier()
    torch.set_num_threads(2);seed(cfg["seed"]);rng=np.random.default_rng(cfg["seed"]+91)
    data=ComponentData(cfg["validation_receipt"],"train","B0");assert len(data)==1600
    model=OnlineB0(cfg["network"]).to(device)
    if cfg.get("best_bundle"):
        from model.action4_release_bundle import load_bundle
        model,_,parent=load_bundle(cfg["best_bundle"],"B0",device)
        if parent["selected_step"]!=cfg["parent_best_step"]:raise ValueError("Selected best step mismatch")
    else:
        base=torch.load(R/cfg["base_best"],map_location="cpu",weights_only=True)
        head=torch.load(R/cfg["adapter_best"],map_location="cpu",weights_only=True)
        if base["step"]!=5220 or head["step"]!=6300:raise ValueError("Unexpected selected parent step")
        model.detector.load_state_dict(base["model"],strict=True);model.refiner.load_state_dict(head["head"],strict=True);del base,head
    groups=[{"params":model.detector.backbone.parameters(),"lr":cfg["lr_backbone"]},
            {"params":[p for n,p in model.detector.named_parameters() if not n.startswith("backbone.")],"lr":cfg["lr_detector"]},
            {"params":model.refiner.parameters(),"lr":cfg["lr_refiner"]}]
    opt=torch.optim.AdamW(groups,weight_decay=.0001)
    total_steps=cfg["epochs"]*math.ceil(len(data)/cfg["batch"])
    def factor(step):
        if step<20:return max(.05,(step+1)/20)
        return .01+.99*(1+math.cos(math.pi*min(1,(step-20)/max(1,total_steps-20))))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    epoch=cursor=step=0
    if resume:
        state=torch.load(R/resume,map_location="cpu",weights_only=True)
        if not state.get("resumable") or state["config_id"]!=canonical_hash(cfg):raise ValueError("Exact same-run recovery required")
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        if state["world_size"]!=world:raise ValueError("Recovery world size mismatch")
        restore_rng(state["rng_by_rank"][rank],rng);epoch,cursor,step=state["epoch"],state["cursor"],state["step"]
    elif (out/"last.pt").exists():raise FileExistsError("Explicit recovery required")
    model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    pids=[None]*world;dist.all_gather_object(pids,os.getpid())
    if rank==0:
        atomic_json(out/"TRAIN_PROCESS.json",{"pids":pids,"attempt_id":attempt,"world_size":world})
        publish(model.module,cfg,step,out)
    dist.barrier()
    for ep in range(epoch,cfg["epochs"]):
        order=np.random.default_rng(cfg["seed"]+ep*1000003).permutation(len(data)).tolist()
        start=cursor if ep==epoch else 0
        if len(order)%cfg["batch"] or cfg["batch"]%world:raise ValueError("No padded/duplicated training samples allowed")
        local_batch=cfg["batch"]//world
        batches=[order[i+rank*local_batch:i+(rank+1)*local_batch] for i in range(start,len(order),cfg["batch"])]
        loader=DataLoader(data,batch_sampler=batches,collate_fn=list,num_workers=cfg["loader_workers"],
                          prefetch_factor=2,generator=torch.Generator().manual_seed(cfg["seed"]+ep))
        seen=start
        for items in loader:
            begin=time.monotonic();model.train();opt.zero_grad(set_to_none=True);logged={}
            for first in range(0,len(items),cfg["micro"]):
                mini=items[first:first+cfg["micro"]];losses=batch_losses(model,mini,cfg,device)
                fraction=len(mini)/len(items);loss=sum(losses.values())*fraction
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite B0 loss")
                loss.backward()
                for k,v in losses.items():logged[k]=logged.get(k,0)+float(v.detach())*fraction
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step();step+=1;seen+=len(items)*world
            ne,nc=(ep+1,0) if seen==len(data) else (ep,seen)
            values=torch.tensor([logged[k] for k in sorted(logged)],device=device);dist.all_reduce(values);values/=world
            logged=dict(zip(sorted(logged),values.cpu().tolist()))
            status={"step":step,"epoch":ep+1,"cursor":seen,"losses":logged,"gradient_norm":float(norm),"seconds":time.monotonic()-begin,"peak_memory_mib":torch.cuda.max_memory_allocated()/1048576}
            if rank==0:
                atomic_json(out/"STATUS.json",status)
                with (out/"train.jsonl").open("a") as f:f.write(json.dumps(status)+"\n")
            if step%cfg["eval_every"]==0 or nc==0 or (stop_after and step>=stop_after):
                states=[None]*world if rank==0 else None;dist.gather_object(rng_state(rng),states,dst=0)
                if rank==0:save_checkpoint_atomic(out/"last.pt",{"model":model.module.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"rng_by_rank":states,"world_size":world,"config_id":canonical_hash(cfg),"epoch":ne,"cursor":nc,"step":step,"role":"B0","resumable":True})
            if step%cfg["eval_every"]==0:
                if rank==0:publish(model.module,cfg,step,out)
                dist.barrier()
            if stop_after and step>=stop_after:return
        cursor=0
    if rank==0:
        publish(model.module,cfg,step,out);atomic_json(out/"TRAIN_COMPLETE.json",{"epochs":cfg["epochs"],"step":step,"attempt_id":attempt})
    dist.barrier();dist.destroy_process_group()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");p.add_argument("--stop-after",type=int);a=p.parse_args()
    try:train(a.config,a.resume,a.stop_after)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg["output"]/f"TRAIN_FAILURE_rank{os.environ.get('RANK','unknown')}.json",{"attempt_id":current_attempt(),"traceback":traceback.format_exc()});raise
