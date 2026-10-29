"""Actual two-rank NCCL AdamW comparison with unequal masks and dummy micro-batches."""
import copy,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.training.action4_supervision import weighted_objective,step_if_supervised
from data.action4_paths import PROJECT_ROOT as R

def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    if world not in (2,3,4):raise ValueError('This regression requires two, three or four ranks')
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(123)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=device)
    model=torch.nn.Linear(2,2).to(device);ref=copy.deepcopy(model)
    ddp=DDP(model,device_ids=[device.index]);opt=torch.optim.AdamW(ddp.parameters(),lr=.01,weight_decay=.1)
    ropt=torch.optim.AdamW(ref.parameters(),lr=.01,weight_decay=.1)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10);rsch=torch.optim.lr_scheduler.CosineAnnealingLR(ropt,10)
    x=torch.arange(10,device=device,dtype=torch.float32).reshape(5,2)/10;y=torch.tensor([1.,.2,.3,.4,1.],device=device)
    masks={'visible':torch.tensor([1,1,0,0,0],device=device,dtype=torch.bool),
           'hidden':torch.tensor([0,0,1,1,1],device=device,dtype=torch.bool)}
    chunks=({2:[[[0],[1,2],[]],[[3],[],[4]]],3:[[[0],[1],[]],[[2],[3],[]],[[4],[],[]]],4:[[[0],[1],[]],[[2],[],[]],[[3],[],[]],[[4],[],[]]]}[world][rank])
    counts={k:sum(int(v[ids].sum()) for ids in chunks) for k,v in masks.items()}
    c=torch.tensor(list(counts.values()),device=device,dtype=torch.float64);dist.all_reduce(c);counts=dict(zip(counts,c.tolist()))
    for update in range(2):
        opt.zero_grad(set_to_none=True);ropt.zero_grad(set_to_none=True)
        for mi,ids in enumerate(chunks):
            ix=torch.tensor(ids or [0],device=device);actual=bool(ids)
            ctx=ddp.no_sync() if mi<2 else __import__('contextlib').nullcontext()
            with ctx:
                pred=ddp(x[ix]);sums={k:((pred[:,j]-y[ix]).square()*m[ix]*actual).sum() for j,(k,m) in enumerate(masks.items())}
                weighted_objective(sums,counts,{},world).backward()
        pred=ref(x);loss=sum(((pred[:,j]-y).square()*m).sum()/counts[k] for j,(k,m) in enumerate(masks.items()));loss.backward()
        max_grad=max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(),ref.parameters()))
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-5)
        step_if_supervised(model,opt,sch,counts,{})
        step_if_supervised(ref,ropt,rsch,counts,{})
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4)
    saved=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict());sched=copy.deepcopy(sch.state_dict())
    opt.zero_grad(set_to_none=True)
    if step_if_supervised(model,opt,sch,{k:0 for k in counts},{})[0]:raise AssertionError('Zero update applied')
    for k,v in saved.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    assert sch.state_dict()==sched
    for key,v in state['state'].items():
        for name,value in v.items():torch.testing.assert_close(value,opt.state_dict()['state'][key][name],rtol=0,atol=0)
    if rank==0:
        out=R/f'outputs/action4_phase2_stage1_v001/DDP_UPDATE_CHECK_WORLD{world}.json'
        out.write_text(json.dumps({'status':'PASS_DISTRIBUTED_NCCL_ENGINEERING_ONLY','world_size':world,'updates':2,'max_gradient_abs_error':max_grad,
            'zero_supervision_state_unchanged':True,'actual_model_accuracy_test':False},indent=2));print(out.read_text())
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
