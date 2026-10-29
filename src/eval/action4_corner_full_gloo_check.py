"""Full corner model CPU/Gloo decomposition and same-state recovery check, not formal launch."""
import os,json,copy,hashlib,datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_stage2_core import construct,optimizer_for,loss_sums_on_batch,update_loss_weights
from model.training.action4_supervision import local_counts,weighted_objective,has_supervision,step_if_supervised
from model.training.action4_runtime import canonical_hash,save_checkpoint_atomic
from model.training.action4_component_train import seed,rng_state,restore_rng


def run(output_directory,rank=None):
    out=contained_path(R,output_directory,exists=True);role="reference" if rank is None else "rank_"+str(rank)
    spec=json.loads((out/"INPUTS.json").read_text());device=spec.get("device","cpu");backend=spec.get("backend","gloo")
    if device=="cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=spec["gpu_map"][role]:raise ValueError("Probe GPU mapping differs")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(spec["allocator_gib"]*1024**3/torch.cuda.get_device_properties(0).total_memory)
    elif device!="cpu":raise ValueError("Unregistered probe device")
    torch.set_num_threads(3);torch.set_num_interop_threads(1);seed(19);rng=np.random.default_rng(19)
    cfg=json.loads((R/spec["config"]["path"]).read_text())
    for name in ["config","items"]:
        ref=spec[name];p=contained_path(R,ref["path"],exists=True)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=ref["sha256"]:raise ValueError("Owned integration input changed")
    if spec.get("corner_softmax_bias_frozen",False):cfg["corner_softmax_bias_frozen"]=True
    (out/(role+"_CONFIG.json")).write_text(json.dumps(cfg,indent=2))
    # This is a local, hash-bound cache created by the prior real dataset loader.
    items=torch.load(R/spec["items"]["path"],map_location="cpu",weights_only=False)
    assert len(items)==2 and [i['data_index'] for i in items]==[0,1]
    world=1 if rank is None else 2
    if rank is not None:
        dist.init_process_group(backend,init_method='file://'+str(out/'rendezvous'),rank=rank,world_size=world,timeout=datetime.timedelta(minutes=10))
    model,parent=construct(cfg,device);opt=optimizer_for(model,cfg);scheduler=torch.optim.lr_scheduler.StepLR(opt,100)
    if any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):raise ValueError('Stochastic layers need a separately registered reference')
    frozen={k:v.detach().clone() for k,v in model.base.state_dict().items()}
    gauge=model.corner_head.localization.heatmap.bias.detach().clone()
    wrap=model if rank is None else torch.nn.parallel.DistributedDataParallel(model,device_ids=[0] if device=="cuda:0" else None,broadcast_buffers=False)
    steps=0;records=[]
    for attempt,epoch in enumerate([1,6,6,6],1):
        current=copy.deepcopy(items)
        if attempt>=3:current[1]['active']=False
        if attempt==4:current[0]['active']=False
        mine=current if rank is None else [current[rank]]
        counts=local_counts(mine,sam_enabled=True,corner_enabled=True)
        if rank is not None:
            keys=sorted(counts);v=torch.tensor([counts[k] for k in keys],dtype=torch.float64,device=device);dist.all_reduce(v);counts=dict(zip(keys,v.tolist()))
        weights=update_loss_weights(cfg,epoch);model.train();assert not model.base.training
        opt.zero_grad(set_to_none=True);objective_value=0.
        if has_supervision(counts,weights):
            for item in mine:
                sums=loss_sums_on_batch(wrap,[item],cfg,device,epoch_one_based=epoch)
                objective=weighted_objective(sums,counts,weights,world)
                objective.backward();objective_value+=float(objective.detach())/world
            did,norm=step_if_supervised(model,opt,scheduler,counts,weights,cfg['independent_head_dependencies']);assert did;steps+=1
        else:
            before={k:v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()}
            before_opt=copy.deepcopy(opt.state_dict());before_sched=copy.deepcopy(scheduler.state_dict())
            did,norm=step_if_supervised(model,opt,scheduler,counts,weights,cfg['independent_head_dependencies']);assert not did
            for k,v in model.state_dict().items():assert torch.equal(before[k],v) if torch.is_tensor(v) else before[k]==v
            assert scheduler.state_dict()==before_sched
            after=opt.state_dict();assert after['param_groups']==before_opt['param_groups']
            for k,vals in before_opt['state'].items():
                for name,v in vals.items():assert torch.equal(v,after['state'][k][name]) if torch.is_tensor(v) else v==after['state'][k][name]
            del before,before_opt,after
        if rank is not None:
            v=torch.tensor(objective_value,dtype=torch.float64,device=device);dist.all_reduce(v);objective_value=float(v)
        records.append(dict(attempt=attempt,optimizer_step=steps,epoch_branch=epoch,counts=counts,objective=objective_value,gradient_norm=norm,did_update=did))
        (out/(role+'_STATUS.json')).write_text(json.dumps(records[-1]))
        if attempt==1 and rank is not None:
            path=out/(role+'_resume.pt')
            save_checkpoint_atomic(path,dict(training_model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                rng_by_rank={rank:rng_state(rng)},world_size=world,config_id=canonical_hash(cfg),resumable=True,epoch=0,cursor=2,step=steps,
                data_update_attempt=attempt,no_supervision_update_count=0))
            del wrap,model,opt,scheduler
            model,parent=construct(cfg,device);opt=optimizer_for(model,cfg);scheduler=torch.optim.lr_scheduler.StepLR(opt,100)
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert state['resumable'] and state['world_size']==world and state['config_id']==canonical_hash(cfg)
            model.load_state_dict(state['training_model'],strict=True);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            restore_rng(state['rng_by_rank'][rank],rng);steps=state['step'];del state
            wrap=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0] if device=="cuda:0" else None,broadcast_buffers=False)
    assert steps==3 and all(torch.equal(frozen[k],v) for k,v in model.base.state_dict().items())
    if cfg.get("corner_softmax_bias_frozen",False):assert torch.equal(gauge,model.corner_head.localization.heatmap.bias)
    save_checkpoint_atomic(out/(role+'_FINAL.pt'),dict(frozen_heatmap_bias=model.corner_head.localization.heatmap.bias.detach().clone(),parameters={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad},optimizer=opt.state_dict(),scheduler=scheduler.state_dict()))
    result=dict(status='FULL_MODEL_ROLE_COMPLETE_NOT_EFFECT',rank=rank,world=world,records=records,optimizer_steps=steps,
        actual_resume_after_step1=rank is not None,frozen_backbone_unchanged=True,global_zero_skipped=True,CPU=sorted(os.sched_getaffinity(0)),device=device,backend=backend,
        peak_cuda_allocated=torch.cuda.max_memory_allocated() if device=="cuda:0" else 0,
        precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32))
    (out/(role+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if rank is not None:dist.destroy_process_group()
