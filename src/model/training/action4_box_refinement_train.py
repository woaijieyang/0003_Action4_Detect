"""Full1600 train images, independent best initialization,20step async eval."""
import os,json,math,time,hashlib,fcntl,traceback
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_box_refinement_data import load_inputs,attach_train_targets
from model.action4_box_refinement import BoxRefinement,loss
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash

def publish(model,cfg,step,out):
    queue=out/"eval";queue.mkdir(exist_ok=True);request=queue/f"step_{step:08d}.json"
    if request.exists():return
    while len([p for p in queue.glob("step_*.json") if p.name.count(".")==1 and not p.with_suffix(".done.json").exists()])>=8:
        if list(queue.glob("*.failed.json")):raise RuntimeError("Asynchronous eval failed")
        time.sleep(3)
    snapshot=out/f"eval_step_{step:08d}.pt"
    save_checkpoint_atomic(snapshot,{"head":model.state_dict(),"step":step,"config_id":canonical_hash(cfg),"resumable":False})
    atomic_json(request,{"step":step,"snapshot":str(snapshot.relative_to(R))})

def train(config,resume=None):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    lock=(out/"train.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in cfg["source_identity"].items():
        if hashlib.sha256((R/path).read_bytes()).hexdigest()!=digest:raise ValueError("Source changed: "+path)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    rows=attach_train_targets(load_inputs(cfg["cache_output"],"train"))
    model=BoxRefinement().cuda()
    initial=torch.load(R/cfg["appearance_best"],map_location="cpu",weights_only=True)
    model.appearance.load_state_dict(initial["head"],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg["epochs"]*math.ceil(len(rows)/cfg["batch_images"]),eta_min=cfg["lr"]*.01)
    epoch=cursor=step=0
    if resume:
        state=torch.load(R/resume,map_location="cpu",weights_only=True)
        if state["config_id"]!=canonical_hash(cfg) or not state["resumable"]:raise ValueError("Exact recovery requires same-run state")
        model.load_state_dict(state["head"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        epoch,cursor,step=state["epoch"],state["cursor"],state["step"];torch.set_rng_state(state["rng_cpu"]);torch.cuda.set_rng_state_all(state["rng_cuda"])
    elif (out/"last.pt").exists():raise FileExistsError("Existing run requires explicit recovery")
    atomic_json(out/"TRAIN_PROCESS.json",{"pid":os.getpid()})
    publish(model,cfg,step,out)
    for ep in range(epoch,cfg["epochs"]):
        order=np.random.default_rng(cfg["seed"]+ep*1000003).permutation(len(rows))
        for start in range(cursor if ep==epoch else 0,len(rows),cfg["batch_images"]):
            batch=[rows[i] for i in order[start:start+cfg["batch_images"]]]
            tensors={k:torch.cat([x[k] for x in batch]).cuda() for k in ("features","geometry","priors","boxes","targets","matched","supervision_valid")}
            model.train();optimizer.zero_grad(set_to_none=True)
            outputs=model(*[tensors[k] for k in ("features","geometry","priors","boxes")])
            losses=loss(outputs,tensors["boxes"],tensors["targets"],tensors["matched"],tensors["supervision_valid"])
            total=sum(losses.values())
            if not torch.isfinite(total):raise FloatingPointError("Nonfinite training loss")
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            seen=start+len(batch);ne,nc=(ep+1,0) if seen==len(rows) else (ep,seen)
            status={"step":step,"epoch":ep+1,"cursor_images":seen,"losses":{k:float(v.detach()) for k,v in losses.items()},"gradient_norm":float(norm)}
            atomic_json(out/"STATUS.json",status)
            with (out/"train.jsonl").open("a") as f:f.write(json.dumps(status)+"\n")
            if step%cfg["eval_every"]==0 or nc==0:
                save_checkpoint_atomic(out/"last.pt",{"head":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"config_id":canonical_hash(cfg),"epoch":ne,"cursor":nc,"step":step,"rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all(),"resumable":True})
            if step%cfg["eval_every"]==0:publish(model,cfg,step,out)
        cursor=0
    publish(model,cfg,step,out);atomic_json(out/"TRAIN_COMPLETE.json",{"epochs":cfg["epochs"],"steps":step})

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");a=p.parse_args()
    try:train(a.config,a.resume)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg["output"]/"TRAIN_FAILURE.json",{"traceback":traceback.format_exc()});raise
