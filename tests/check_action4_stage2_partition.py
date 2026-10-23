"""Exact data coverage and gradient mass across uneven DDP tail batches."""
import torch
from data.action4_stage2_data import Stage2Batches
from model.training.action4_stage2_core import objective

def main():
    cases=0
    for size in [1,3,5,17]:
        for world in [2,3,4]:
            batch=world*2
            samplers=[Stage2Batches(size,batch,world,r,42) for r in range(world)]
            batches=[list(s) for s in samplers]
            actual=[unit[0] for rank in batches for chunk in rank for unit in chunk if unit[2]]
            assert sorted(actual)==list(range(size))
            assert len({len(rank) for rank in batches})==1
            for step in range(len(batches[0])):
                chunk=[rank[step] for rank in batches]
                assert sum(unit[2] for rank in chunk for unit in rank)==chunk[0][0][3]
            cases+=1
    parameter=torch.tensor(.7,requires_grad=True);values=torch.tensor([1.,2.,5.])
    expected=(parameter*values).square().mean()
    ranks=[]
    for rank_values in [values[:2],values[2:]]:
        loss=(parameter*rank_values).square().mean()
        ranks.append(objective({"box":loss},{"role":"B0"},len(rank_values),3,2))
    result=sum(ranks)/2
    assert torch.allclose(expected,result)
    a=torch.autograd.grad(expected,parameter,retain_graph=True)[0];b=torch.autograd.grad(result,parameter)[0];assert torch.allclose(a,b)
    parameter=torch.tensor(.7,requires_grad=True)
    actual=objective({"box":parameter.square()},{"role":"B0"},1,1,2)
    dummy=objective({"box":(parameter*999).square()},{"role":"B0"},0,1,2)
    assert torch.allclose((actual+dummy)/2,parameter.square())
    parameter=torch.tensor(.7,requires_grad=True)
    visibility=torch.tensor([[True]*2+[False]*6,[True]*5+[False]*3,[True]*7+[False]])
    errors=(parameter*values[:,None]-torch.arange(8).float()[None]/9).square()
    expected=(parameter*values).square().mean()+errors[visibility].mean()+errors[~visibility].mean()
    parts=[]
    for ids in [slice(0,2),slice(2,3)]:
        local=errors[ids];mask=visibility[ids];n=len(local)
        ls={"xyz":(parameter*values[ids]).square().mean(),"visible_corners":(local*mask).sum()/visibility.sum(),"hidden_corners":(local*(~mask)).sum()/(~visibility).sum()}
        parts.append(objective(ls,{"role":"A0"},n,3,2))
    got=sum(parts)/2
    assert torch.allclose(expected,got,atol=1e-6)
    a=torch.autograd.grad(expected,parameter,retain_graph=True)[0];b=torch.autograd.grad(got,parameter)[0];assert torch.allclose(a,b,atol=1e-6)
    print({"partition_cases":cases,"no_dropped_or_duplicate_actual_units":True,"uneven_gradient_mass":True,"dummy_contribution_zero":True})
if __name__=="__main__":main()
