"""Two real-input corner updates using existing core; not formal stage4 training."""
from pathlib import Path
import copy,hashlib,json,time,os
import torch
from model.training.action4_stage2_core import construct,optimizer_for,loss_sums_on_batch,update_loss_weights,bundle_snapshot
from model.training.action4_supervision import local_counts,weighted_objective,step_if_supervised,A0_KEYS
from model.action4_sam_joint import CORNER_PROTOCOL,stack_sam_items
from model.action4_corner_supervision import LOSS_WEIGHTS,parameter_loss_dependencies
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from data.action4_stage2_data import Stage2Units
from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL
from data.action4_paths import PROJECT_ROOT as R
from model.action4_release_bundle import load_bundle


def run(output_directory="outputs/action4_stage4_real_input_updates_v001"):
    from data.action4_paths import contained_path
    out=contained_path(R,output_directory,exists=False);out.mkdir(exist_ok=False)
    torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.manual_seed(19)
    path=R/"outputs/action4_stage3_native_prediction_v001/real_gpu_updates_v001/ENGINEERING_REAL_STEP2.pt"
    parent=torch.load(path,map_location="cpu",weights_only=True);cfg=copy.deepcopy(parent["config"])
    cfg.update(mode="integration",job_id="stage4_real_input_two_updates_not_formal",a0_variant="a0_corner_image",corner_enabled=True,
        training_protocol=CORNER_NATIVE_PROTOCOL,parent_bundle=str(path.relative_to(R)),parent_best_step=parent["selected_step"],
        parent_bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),corner_protocol=CORNER_PROTOCOL,
        corner_lr=1e-4,corner_quality_lr=1e-4,sam_adapter_lr=1e-5,train_indices=list(range(8)),sam_cache_validation_workers=3)
    cfg.pop("a0_selection_contract",None)
    cfg["corner_counterpart"]={"B0":{"path":cfg["evaluation_counterpart_bundle"],"sha256":cfg["evaluation_counterpart_sha256"]},
        "train_sam_cache":cfg["sam_cache"]["manifest"],"eval_sam_cache":cfg["sam_eval_cache"]["manifest"]}
    cfg["loss_weights"].update(LOSS_WEIGHTS)
    cfg["independent_head_dependencies"]=[
        {"parameter_prefixes":["head.surface"],"losses":list(A0_KEYS)},
        {"parameter_prefixes":["head.pose"],"losses":[k for k in A0_KEYS if k not in ("xyz","region","visible_mask","amodal_mask")]},
        {"parameter_prefixes":["sam_adapter.encoder","sam_adapter.projection"],"losses":list(A0_KEYS)+[k for k in LOSS_WEIGHTS if k!="corner_quality_bce"]},
        {"parameter_prefixes":["sam_adapter.quality_head"],"losses":["sam_mask_quality"]},
        *parameter_loss_dependencies("corner_head")]
    (out/"CONFIG.json").write_text(json.dumps(cfg,indent=2));(out/"STATUS.json").write_text(json.dumps(dict(stage="loading_native_inputs",pid=os.getpid())))
    units=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux));items=[]
    for i in range(len(units)):
        item=units[(i,0,True,2)]
        if item['mask_aux'].available.item()>0:items.append(item)
        if len(items)==2:break
    if len(items)!=2:raise RuntimeError("Bounded input selection did not provide two current masks; retain failure")
    torch.save(items,out/"PREPARED_REAL_ITEMS.pt")
    model,parent=construct(cfg,"cpu");optimizer=optimizer_for(model,cfg)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100)
    frozen={k:v.detach().clone() for k,v in model.base.state_dict().items()}
    q0={k:v.detach().clone() for k,v in model.corner_head.quality_head.state_dict().items()}
    count=local_counts(items,sam_enabled=True,corner_enabled=True);events=[]
    for update,epoch in enumerate([1,6],1):
        model.train();assert not model.base.training
        optimizer.zero_grad(set_to_none=True)
        sums=loss_sums_on_batch(model,items,cfg,"cpu",epoch_one_based=epoch)
        if update==1:
            grads=torch.autograd.grad(sums['corner_heatmap_kl'],tuple(model.sam_adapter.projection.parameters()),retain_graph=True,allow_unused=True)
            corner_to_adapter=sum(float(g.abs().sum()) for g in grads if g is not None)
            assert corner_to_adapter>0
        loss=weighted_objective(sums,count,update_loss_weights(cfg,epoch),1)
        assert torch.isfinite(loss);loss.backward()
        did,norm=step_if_supervised(model,optimizer,scheduler,count,update_loss_weights(cfg,epoch),cfg['independent_head_dependencies'])
        assert did
        if update==1:
            assert all(torch.equal(q0[k],v) for k,v in model.corner_head.quality_head.state_dict().items())
            assert all(p not in optimizer.state for p in model.corner_head.quality_head.parameters())
        else:assert any(not torch.equal(q0[k],v) for k,v in model.corner_head.quality_head.state_dict().items())
        events.append(dict(update=update,epoch_branch_only_not_completed_epoch=epoch,objective=float(loss.detach()),gradient_norm=norm,eligible=count))
        (out/"UPDATES.json").write_text(json.dumps(events,indent=2))
        (out/"STATUS.json").write_text(json.dumps(dict(stage="real_update_complete",update=update,pid=os.getpid())))
        del sums,loss
    assert all(torch.equal(frozen[k],v) for k,v in model.base.state_dict().items())
    # A zero-supervision attempt after real AdamW momentum must not update any state.
    before={k:v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()};opt_before=copy.deepcopy(optimizer.state_dict());sched_before=copy.deepcopy(scheduler.state_dict())
    did,_=step_if_supervised(model,optimizer,scheduler,{k:0 for k in count},update_loss_weights(cfg,6),cfg['independent_head_dependencies'])
    assert not did and scheduler.state_dict()==sched_before
    assert all(torch.equal(before[k],v) if torch.is_tensor(v) else before[k]==v for k,v in model.state_dict().items())
    after=optimizer.state_dict();assert after['param_groups']==opt_before['param_groups']
    for key,values in opt_before['state'].items():
        for k,v in values.items():assert torch.equal(v,after['state'][key][k]) if torch.is_tensor(v) else v==after['state'][key][k]
    del before,opt_before,after
    bundle=bundle_snapshot(model,parent,cfg,2);file=out/"ENGINEERING_CORNER_STEP2.pt";torch.save(bundle,file)
    try:load_bundle(file,"A0","cpu")
    except ValueError:pass
    else:raise AssertionError("Unadmitted corner candidate loaded through default loader")
    loaded,_,state=load_bundle(file,"A0","cpu",allow_corner_candidate=True)
    rgb=torch.stack([x['roi'] for x in items]);K=torch.stack([x['geometry']['K'] for x in items]);cad=torch.stack([x['geometry']['canonical'] for x in items])
    aux,ids,_,_=stack_sam_items(items,"cpu",rgb.dtype)
    model.eval();loaded.eval()
    with torch.inference_mode():
        a=model(rgb,K,cad,mask_aux=aux,identities=ids,return_corner=True)
        b=loaded(rgb,K,cad,mask_aux=aux,identities=ids,return_corner=True)
    for x,y in [(a.raw_pose['translation'],b.raw_pose['translation']),(a.raw_pose['corners'],b.raw_pose['corners']),
        (a.corners.localization.logits,b.corners.localization.logits),(a.corners.quality_logits,b.corners.quality_logits)]:torch.testing.assert_close(x,y,rtol=0,atol=0)
    report=dict(status="REAL_NATIVE_ROI_FULL_CORNER_TWO_UPDATE_STRICT_RELOAD_PASS_NOT_FORMAL_OR_ACCURACY",updates=2,
        parent_engineering_step=2,parent_is_not_formal_best=True,units=[x['data_index'] for x in items],roi_expansion=1.5,
        corner_loss_to_SAM_projection_absolute_gradient=corner_to_adapter,backbone_parameters_and_buffers_unchanged=True,
        quality_head_warmup_no_optimizer_state=True,quality_head_active_branch_updated=True,zero_supervision_no_state_update=True,
        strict_reload_exact_forward=True,default_loader_rejected=True,events=events,
        formal_stage4_gate_open=False,full_eval_executed=False,formal_training_modified=False)
    (out/"COMPLETE.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
