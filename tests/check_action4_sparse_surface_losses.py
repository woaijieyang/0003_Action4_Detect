"""Sparse surface loss matches dense loss at integer pixels; gradients stay connected."""
import torch
from torch.nn import functional as F
from model.action4_surface_head import surface_losses
from model.action4_sparse_surface_losses import sparse_surface_losses,sample_points

def main():
    torch.manual_seed(3);output={"xyz":torch.randn(2,3,12,12,requires_grad=True),"regions":torch.randn(2,65,12,12,requires_grad=True),"masks":torch.randn(2,2,12,12,requires_grad=True)}
    target={"xyz":torch.zeros(2,3,12,12),"regions":torch.full((2,12,12),-100,dtype=torch.long),"valid":torch.zeros(2,12,12,dtype=torch.bool),"masks":torch.randint(0,2,(2,2,12,12)).float()};sparse=[]
    for b,indices in enumerate([[0,17,83],[14,25,36,47,58,69,80,91,102]]):
        idx=torch.tensor(indices);x=idx%12;y=idx//12;xyz=torch.rand(len(idx),3);region=torch.randint(1,65,(len(idx),))
        target["xyz"][b,:,y,x]=xyz.T;target["regions"][b,y,x]=region;target["valid"][b,y,x]=True
        sparse.append({"xy_hm":torch.stack([x,y],-1).float(),"xyz":xyz,"regions":region})
    dense=surface_losses(output,target);other=sparse_surface_losses(output,{"sparse":sparse,"masks":target["masks"]})
    assert all(torch.allclose(dense[k],other[k],atol=1e-6) for k in dense)
    ga=torch.autograd.grad(sum(dense.values()),tuple(output.values()),retain_graph=True);gb=torch.autograd.grad(sum(other.values()),tuple(output.values()))
    assert all(torch.allclose(a,b,atol=1e-7) for a,b in zip(ga,gb))
    feature=torch.randn(3,12,12);xy=torch.rand(19,2)*11
    reference=F.grid_sample(feature[None],((xy+.5)/12*2-1)[None,None],align_corners=False)[0,:,0].T
    assert torch.allclose(sample_points(feature,xy),reference,atol=2e-6)
    try:sample_points(feature,torch.tensor([[-1.,2.]]))
    except ValueError:pass
    else:raise AssertionError("Out-of-grid supervision accepted")
    print("PASS: dense/sparse loss and gradient equality, bilinear oracle, invalid coordinate rejection")
if __name__=="__main__":main()
