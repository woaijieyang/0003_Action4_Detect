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
from model.training.action4_supervision import local_counts,weighted_objective,has_supervision,step_if_supervised
from model.training.action4_stage2_core import loss_sums_on_batch,update_loss_weights
from model.training.action4_run_snapshot import verify_running_snapshot
from model.training.action4_eval_schedule import validate_schedule,eval_due,resume_ledger
from model.training.action4_eval_publication import coordinated_publish

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
    verify_running_snapshot(cfg)
    if (out/"TRAIN_COMPLETE.json").exists():raise FileExistsError("Terminal training must not restart")
    validate_run_gate(cfg,json.loads(contained_path(R,cfg["integration_or_admission_receipt"],exists=True).read_text()))
    if cfg["role"] not in ("A0","B0") or cfg["mode"] not in ("integration","full_training"):raise ValueError("Explicit role/mode required")
    validate_schedule(cfg)
    if cfg["eval_every"] not in ({10,25} if cfg["role"]=="A0" else {20}):raise ValueError("Optimizer-step eval protocol changed")
    if stop_after is not None and cfg["mode"]!="integration":raise ValueError("Controlled stop probe only allowed in explicit integration mode")
    from data.action4_prediction_roi_data import roi_finetune
    if cfg["mode"]=="full_training" and cfg["epochs"]<80 and cfg.get("training_protocol")!="action4.fixed_roi_prediction_adaptation.v1" and not roi_finetune(cfg):raise ValueError("Full stage requires at least80epochs")
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    requested=cfg.get('worker_fd_soft_limit',16384)
    if soft<requested:resource.setrlimit(resource.RLIMIT_NOFILE,(min(requested,hard),hard))
    if resource.getrlimit(resource.RLIMIT_NOFILE)[0]<requested:
        raise RuntimeError('Per-process descriptor budget cannot support tensor-prefetch transport')
    # Enhanced missing/incompatible cache fails before CUDA/DDP/model initialization.
    from model.action4_sam_joint import sam_variant
    data=None
    if sam_variant(cfg):
        from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
        data=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
    if world!=cfg["world_size"]:raise ValueError("DDP topology mismatch")
    if cfg.get("synchronize_data_preflight",False):
        if data is None:raise ValueError("Data readiness contract requires initialized prediction inputs")
        from model.training.action4_startup_readiness import wait_data_ready
        wait_data_ready(out,current_attempt(),rank,world,len(data),
            timeout=cfg.get("training_startup_timeout_s",1800.))
    torch.cuda.set_device(local)
    from model.training.action4_resource_policy import apply_allocator_budget
    apply_allocator_budget(cfg,"training",local)
    device=f"cuda:{local}";attempt=current_attempt()
    dist.init_process_group("nccl",device_id=torch.device(device),timeout=datetime.timedelta(seconds=180))
    guard=(out/"train.lock").open("a")
    if rank==0:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in cfg["source_identity"].items():
            if hashlib.sha256(contained_path(R,name,exists=True).read_bytes()).hexdigest()!=digest:raise ValueError("Training source changed: "+name)
    dist.barrier();torch.set_num_threads(cfg.get("threads",2));seed(cfg["seed"]);rng=np.random.default_rng(cfg["seed"]+rank)
    if data is None:data=Stage2Units(cfg)
    if len(data)!=cfg["expected_units"]:raise ValueError("Admitted role unit count changed")
    model,parent=construct(cfg,device);optimizer=optimizer_for(model,cfg)
    total_steps=cfg["epochs"]*math.ceil(len(data)/cfg["batch"])
    from model.training.action4_roi_adaptation import make_scheduler
    scheduler=make_scheduler(optimizer,cfg,total_steps)
    epoch=cursor=step=0
    data_attempt=zero_updates=0
    schedule_ledger=[];scheduled=cfg.get("eval_schedule") is not None
    route_count_enabled=cfg.get("sam_roi_routing") is not None
    if route_count_enabled:
        from model.training.action4_route_counts import (empty_counts,reduce_route_update,advance_counts,
            checkpoint_route_counts,restore_route_counts,route_log_fields)
        route_cumulative=empty_counts()
    repair_lineage=None
    if resume:
        if cfg.get("scoring_repair") is not None:
            from model.training.action4_scoring_repair_continuation import load_scoring_repair_state
            state,repair_lineage=load_scoring_repair_state(cfg,resume)
        else:
            state=torch.load(contained_path(R,resume,exists=True),map_location="cpu",weights_only=True)
            if not state.get("resumable") or state["config_id"]!=canonical_hash(cfg) or state["world_size"]!=world:
                raise ValueError("Recovery must match exact role/config/topology")
        model.load_state_dict(state["training_model"],strict=True);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        epoch,cursor,step=state["epoch"],state["cursor"],state["step"];restore_rng(state["rng_by_rank"][rank],rng)
        data_attempt=state.get("data_update_attempt",step);zero_updates=state.get("no_supervision_update_count",0)
        schedule_ledger=resume_ledger(cfg,state)
        if route_count_enabled:
            route_cumulative=restore_route_counts(state.get("roi_route_counts"),
                epoch=epoch,cursor=cursor,units_per_epoch=len(data))
    elif (out/"last.pt").exists():raise FileExistsError("Explicit same-run recovery required; new experiments start from best")
    model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    pids=[None]*world;dist.all_gather_object(pids,os.getpid())
    if rank==0:
        atomic_json(out/"TRAIN_PROCESS.json",{"pids":pids,"attempt_id":attempt,"role":cfg["role"],"units":len(data),"resumed_from_step":step if resume else None})
        if repair_lineage is not None:atomic_json(out/"SCORING_REPAIR_RESUMED.json",repair_lineage)
    # At an epoch boundary, a recovery checkpoint belongs to the preceding epoch.
    last_epoch=max(1,epoch if cursor==0 else epoch+1)
    if eval_due(cfg,step,last_epoch):
        coordinated_publish(out,rank=rank,attempt=attempt,step=step,config_id=canonical_hash(cfg),
            publish_fn=lambda:publish(model.module,parent,cfg,step,out))
    sampler=Stage2Batches(len(data),cfg["batch"],world,rank,cfg["seed"])
    workers=cfg.get("loader_workers",0)
    loader=DataLoader(data,batch_sampler=sampler,collate_fn=list,num_workers=workers,
        persistent_workers=workers>0,**({"prefetch_factor":2} if workers else {}),
        generator=torch.Generator().manual_seed(cfg["seed"]))
    capacity_ready_waited=False
    for ep in range(epoch,cfg["epochs"]):
        epoch_one_based=ep+1  # Stored data epoch is zero-based; corner warmup converts exactly here.
        start=cursor if ep==epoch else 0;sampler.set_epoch(ep,start);seen=start
        for items in loader:
            if cfg.get("capacity_barrier") is not None and not capacity_ready_waited:
                if stop_after!=20:raise ValueError("Performance barrier requires explicit bounded stop20")
                from model.training.action4_capacity_barrier import wait_ready
                wait_ready(cfg,"train",rank,first_batch_units=len(items))
                capacity_ready_waited=True

            begin=time.monotonic();global_n=items[0]["global_count"];local_n=sum(x["active"] for x in items)
            check=torch.tensor(local_n,device=device);dist.all_reduce(check)
            if int(check)!=global_n:raise ValueError("Distributed sample mass mismatch")
            data_attempt+=1
            if route_count_enabled:
                # Entire rank update before microbatching, including no-supervision data.
                route_update=reduce_route_update(items,enabled=True,device=device,all_reduce=dist.all_reduce)
                if route_update["active_units"]!=global_n:raise ValueError("Global route count differs from consumed unit mass")
                route_cumulative=advance_counts(route_cumulative,route_update,epoch=ep,cursor=seen,units_per_epoch=len(data))
            if cfg["role"]=="A0":
                from model.action4_sam_joint import sam_variant,corner_variant
                local=local_counts(items,sam_enabled=sam_variant(cfg),corner_enabled=corner_variant(cfg));keys=sorted(local)
                mass=torch.tensor([local[k] for k in keys],device=device,dtype=torch.float64)
                dist.all_reduce(mass);counts=dict(zip(keys,mass.cpu().tolist()))
            else:
                # Baseline B0 retains its established image-mean objective.
                enabled=any(v>0 for v in cfg.get('loss_weights',{}).values()) if cfg.get('loss_weights') else True
                counts={'b0_image':global_n if enabled else 0}
            weights=update_loss_weights(cfg,epoch_one_based)
            supervised=has_supervision(counts,weights)
            model.train();optimizer.zero_grad(set_to_none=True);logged={}
            norm=0.;did_update=False
            micro_count=math.ceil(math.ceil(global_n/world)/cfg["micro"])
            for m in range(micro_count if supervised else 0):
                mini=items[m*cfg["micro"]:(m+1)*cfg["micro"]]
                if not mini:mini=[dict(items[0],active=False)]
                active=sum(x["active"] for x in mini)
                context=model.no_sync() if m+1<micro_count else __import__("contextlib").nullcontext()
                with context:
                    if cfg['role']=='A0':
                        losses=loss_sums_on_batch(model,mini,cfg,device,epoch_one_based=epoch_one_based)
                        loss=weighted_objective(losses,counts,weights,world)
                    else:
                        actual=[x for x in mini if x['active']]
                        losses=losses_on_batch(model,actual or mini[:1],cfg,device)
                        loss=objective(losses,cfg,active,global_n,world)
                    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite role loss")
                    loss.backward()
                for key,value in losses.items():
                    contribution=(weights.get(key,1.)*world*value.detach()/counts[key] if counts[key]>0 else value.detach()*0) if cfg['role']=='A0' else objective({key:value.detach()},cfg,active,global_n,world)
                    logged[key]=logged.get(key,0.)+float(contribution)
            did_update,norm=step_if_supervised(model.module,optimizer,scheduler,counts,weights,
                cfg.get('independent_head_dependencies',[]))
            if did_update:step+=1
            else:zero_updates+=1
            seen+=global_n
            ne,nc=(ep+1,0) if seen==len(data) else (ep,seen)
            if not logged:logged={'no_supervision_update':0.}
            values=torch.tensor([logged[k] for k in sorted(logged)],device=device);dist.all_reduce(values);values/=world
            status={"role":cfg["role"],"mode":cfg["mode"],"step":step,"epoch":ep+1,"cursor":seen,"global_actual_units":global_n,
                    "losses":dict(zip(sorted(logged),values.cpu().tolist())),"gradient_norm":float(norm),"seconds":time.monotonic()-begin,
                    "data_update_attempt":data_attempt,"no_supervision_update_count":zero_updates,"did_update":did_update,"eligible_counts":counts}
            if route_count_enabled:
                status.update(route_log_fields(route_update,route_cumulative,
                    epoch=ep,cursor_after=seen,units_per_epoch=len(data)))
                status["roi_route_progress"]["attempt_id"]=attempt
            if rank==0:
                atomic_json(out/"STATUS.json",status)
                with (out/"train.jsonl").open("a") as f:f.write(json.dumps(status)+"\n")
                if scheduled:
                    schedule_ledger.append({k:status[k] for k in ("epoch","cursor","step","did_update",
                        "data_update_attempt","no_supervision_update_count")})
            evaluation_due=did_update and eval_due(cfg,step,epoch_one_based)
            if evaluation_due or nc==0 or not did_update or (stop_after is not None and step>=stop_after):
                states=[None]*world if rank==0 else None;dist.gather_object(rng_state(rng),states,dst=0)
                if rank==0:
                    save_checkpoint_atomic(out/"last.pt",{"training_model":model.module.state_dict(),"optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict(),"rng_by_rank":states,"world_size":world,"epoch":ne,"cursor":nc,"step":step,
                        "role":cfg["role"],"config_id":canonical_hash(cfg),"resumable":True,
                        "data_update_attempt":data_attempt,"no_supervision_update_count":zero_updates,
                        **({"eval_schedule_ledger":schedule_ledger} if scheduled else {}),
                        **({"roi_route_counts":checkpoint_route_counts(route_cumulative,
                            epoch=ne,cursor=nc,units_per_epoch=len(data))} if route_count_enabled else {})})
            if evaluation_due:
                coordinated_publish(out,rank=rank,attempt=attempt,step=step,config_id=canonical_hash(cfg),
                    publish_fn=lambda:publish(model.module,parent,cfg,step,out))
                dist.barrier()
            if stop_after is not None and step>=stop_after:
                if rank==0:atomic_json(out/"CHECKPOINTED_STOP.json",{"attempt_id":attempt,"step":step,"mode":"integration","completion":False})
                dist.barrier();dist.destroy_process_group();return
        if seen!=len(data):raise ValueError("Epoch coverage incomplete")
        cursor=0
    if step==0:raise RuntimeError("No supervised optimizer update in completed data epochs")
    coordinated_publish(out,rank=rank,attempt=attempt,step=step,config_id=canonical_hash(cfg),
        publish_fn=lambda:publish(model.module,parent,cfg,step,out))
    if rank==0:
        if scheduled:atomic_json(out/"EVAL_SCHEDULE_LEDGER.json",schedule_ledger)
        atomic_json(out/"TRAIN_COMPLETE.json",{"epochs":cfg["epochs"],"step":step,"attempt_id":attempt,"mode":cfg["mode"],"units_per_epoch":len(data),"role":cfg["role"],"formal_training_complete":cfg["mode"]=="full_training","config_id":canonical_hash(cfg),"data_update_attempt":data_attempt,"no_supervision_update_count":zero_updates,"planned_update_attempts":total_steps,
            **({"eval_schedule_ledger_sha256":hashlib.sha256((out/"EVAL_SCHEDULE_LEDGER.json").read_bytes()).hexdigest()} if scheduled else {}),
            **({"roi_route_counts":checkpoint_route_counts(route_cumulative,
                epoch=cfg["epochs"],cursor=0,units_per_epoch=len(data))} if route_count_enabled else {})})
    dist.barrier();dist.destroy_process_group()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");p.add_argument("--stop-after",type=int);a=p.parse_args()
    try:train(a.config,a.resume,a.stop_after)
    except BaseException:
        cfg=json.loads(contained_path(R,a.config,exists=True).read_text())
        atomic_json(contained_path(R,cfg["output"],exists=True)/f"TRAIN_FAILURE_rank{os.environ.get('RANK','unknown')}.json",{"attempt_id":current_attempt(),"traceback":traceback.format_exc()});raise
