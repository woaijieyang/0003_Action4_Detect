"""Actual bundled A0: live geometry losses, gradients, frozen buffers and strict reload."""
import copy,hashlib,json,time
from pathlib import Path
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_stage2_data import Stage2Units
from model.action4_joint_surface_rgb import JointSurfaceRGB
from model.training.action4_stage2_core import loss_sums_on_batch
from model.training.action4_supervision import local_counts,weighted_objective,step_if_supervised

def main():
    torch.set_num_threads(2);torch.manual_seed(123);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg={'role':'A0','mode':'integration','dataset_root':'data/phrase2','seed':1729,
         'train_indices':[0,7999,8799,8800],'roi_jitter':{'center_fraction':.15,'scale_range':[.65,1.35]},
         'surface_source':json.loads((R/'configs/data/action4_phrase2_consumer_v001.json').read_text())['surface_source']}
    ds=Stage2Units(cfg);items=[ds[(i,0,True,4)] for i in range(4)]
    device='cuda:0';model=JointSurfaceRGB('models/default/A0.pt',False,device).train()
    def tensor_hash(state):
        h=hashlib.sha256()
        for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=tensor_hash(model.base.state_dict());opt=torch.optim.AdamW(model.head.parameters(),lr=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
    reference=copy.deepcopy(model);refopt=torch.optim.AdamW(reference.head.parameters(),lr=1e-5)
    refsch=torch.optim.lr_scheduler.CosineAnnealingLR(refopt,10)
    weights={'xyz':1.,'region':.1,'visible_mask':.5,'amodal_mask':.5,'rotation_point_matching':1.,'centroid':1.,'relative_depth':1.,'visible_corners':1.,'hidden_corners':1.,'positive_depth':1.}
    started=time.monotonic();gradient={};loss_values=[];gradient_equivalence=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);model.train();counts=local_counts(items);logged=0.
        for item in items:
            sums=loss_sums_on_batch(model,[item],cfg,device);loss=weighted_objective(sums,counts,weights,1)
            if not torch.isfinite(loss):raise AssertionError('Nonfinite actual A0 loss')
            loss.backward();logged+=float(loss.detach())
        gradient={group:sum(float(p.grad.abs().sum()) for p in getattr(model.head,group).parameters() if p.grad is not None) for group in ['surface','pose']}
        if not all(v>0 for v in gradient.values()):raise AssertionError('Disconnected actual A0 head')
        refopt.zero_grad(set_to_none=True);reference.train()
        ref_sums=loss_sums_on_batch(reference,items,cfg,device)
        ref_loss=weighted_objective(ref_sums,counts,weights,1);ref_loss.backward()
        ga=torch.cat([p.grad.flatten() for p in model.head.parameters() if p.grad is not None])
        gb=torch.cat([p.grad.flatten() for p in reference.head.parameters() if p.grad is not None])
        relative=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(gb).clamp_min(1e-12))
        gradient_equivalence.append(relative)
        if relative>1e-5:raise AssertionError('Actual A0 micro-batch gradient differs: '+str(relative))
        step_if_supervised(model,opt,sch,counts,weights);step_if_supervised(reference,refopt,refsch,counts,weights)
        for a,b in zip(model.head.parameters(),reference.head.parameters()):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4)
        loss_values.append(logged)
    if tensor_hash(model.base.state_dict())!=before:raise AssertionError('Frozen parameters/buffers changed')
    if any(module.training for module in model.base.modules()):raise AssertionError('Frozen base in train mode')
    state={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}
    model.eval();x=torch.stack([items[0]['roi']]).to(device);K=items[0]['geometry']['K'][None].to(device);cad=items[0]['geometry']['canonical'][None].to(device)
    with torch.no_grad():a=model(x,K,cad)['corners'].clone()
    model.head.load_state_dict(state,strict=True)
    with torch.no_grad():b=model(x,K,cad)['corners']
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    result={'status':'PASS_ACTUAL_A0_TWO_UPDATES_ENGINEERING_ONLY','units':4,'losses':loss_values,'head_gradient_l1':gradient,
            'frozen_parameters_and_buffers_unchanged':True,'strict_reload_equal':True,'seconds':time.monotonic()-started,
            'accuracy_improvement':None,'default_weights_modified':False,'actual_model_microbatch_relative_gradient_errors':gradient_equivalence}
    (R/'outputs/action4_phase2_stage1_v001/A0_ACTUAL_UPDATE_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
