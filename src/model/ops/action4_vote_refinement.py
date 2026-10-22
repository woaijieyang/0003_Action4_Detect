"""Robust refinement of PVNet vote evidence, without GT inputs or auto adoption."""
import torch
from torch.nn import functional as F


def refine_votes(mask, directions, initial, iterations=10):
    """Tukey IRLS on ray-normal residuals. Validity means numerical support only.

    Inputs: BHW object mask, BHWK2 direction field, BK2 initial intersections.
    A valid result is not a calibrated confidence or proof of point accuracy.
    """
    if directions.shape[:3] != mask.shape or directions.shape[-2:] != (8,2):
        raise ValueError("Expected BHW mask and BHW82 directions")
    if initial.shape != (mask.shape[0],8,2):
        raise ValueError("Expected eight initial points per object")
    results=[];validities=[];fractions=[]
    for batch in range(len(mask)):
        yx=torch.nonzero(mask[batch].bool())
        if len(yx)<5:
            results.append(initial[batch]);validities.append(torch.zeros(8,dtype=torch.bool,device=initial.device));fractions.append(initial.new_zeros(8));continue
        pixels=yx[:,[1,0]].double()
        raw=directions[batch,yx[:,0],yx[:,1]].double()
        norm=torch.linalg.vector_norm(raw,dim=-1)
        rays=F.normalize(raw,dim=-1)
        normals=torch.stack((rays[...,1],-rays[...,0]),-1)
        rhs_pixel=(normals*pixels[:,None]).sum(-1)
        point=initial[batch].double()
        usable=norm>1e-8
        weights=usable.double()
        for _ in range(iterations):
            residual=(normals*point[None]).sum(-1)-rhs_pixel
            scale=(residual.abs().sort(dim=0, stable=True).values[(len(residual)-1)//2]*1.4826).clamp_min(1e-4)
            u=residual/(4.685*scale)
            forward=((point[None]-pixels[:,None])*rays).sum(-1)>0
            weights=(1-u.square()).clamp_min(0).square()*(u.abs()<1)*forward*usable
            matrix=torch.einsum('nk,nki,nkj->kij',weights,normals,normals)
            rhs=torch.einsum('nk,nki,nk->ki',weights,normals,rhs_pixel)
            solution,info=torch.linalg.solve_ex(matrix,rhs)
            point=torch.where((info==0)[:,None],solution,point)
        eigen=torch.linalg.eigvalsh(matrix)
        valid=(info==0)&torch.isfinite(point).all(-1)&((weights>.05).sum(0)>=5)&(eigen[:,0]>1e-8)&(eigen[:,0]/eigen[:,1].clamp_min(1e-12)>1e-6)
        results.append(point.to(initial.dtype));validities.append(valid);fractions.append((weights>.05).float().mean(0))
    return {"points":torch.stack(results),"valid":torch.stack(validities),"support_fraction":torch.stack(fractions)}
