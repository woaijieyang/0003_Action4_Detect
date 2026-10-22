"""Versioned GDR loss repair; original running v001 remains immutable."""
import torch
from torch.nn import functional as F
from model.action4_surface_head import surface_losses

def pose_losses(output,targets,geometry,corner_normalizers=None):
    losses=surface_losses(output["surface"],targets)
    canonical=geometry["canonical"]
    extent=(canonical.amax(1)-canonical.amin(1)).amax(-1)
    predicted_rotated=torch.bmm(canonical,output["rotation"].transpose(1,2))
    reference_rotated=torch.bmm(canonical,geometry["rotation"].transpose(1,2))
    # Upstream rotation-only PM: coordinate L1 mean *3, normalized by extent.
    losses["rotation_point_matching"]=(torch.abs(predicted_rotated-reference_rotated).mean((1,2))*3/extent).mean()
    K=geometry["K"]
    def center(t):
        hom=torch.bmm(K,t[:,:,None]).squeeze(-1)
        return hom[:,:2]/t[:,2:].clamp_min(1e-4)
    size=output["surface"]["xyz"].shape[-1]
    losses["centroid"]=F.l1_loss(center(output["translation"])/size,center(geometry["translation"])/size)
    scale=(K[:,0,0]+K[:,1,1])*.5*extent/size
    losses["relative_depth"]=F.l1_loss(output["depth"]/scale,geometry["translation"][:,2]/scale)
    errors=F.smooth_l1_loss(output["corners"]/geometry["diagonal"][:,None,None],
                          geometry["corners"]/geometry["diagonal"][:,None,None],
                          beta=.05,reduction="none").mean(-1)
    visible=geometry["visible"].bool();hidden=~visible
    if corner_normalizers is None:
        corner_normalizers={"visible":visible.sum(),"hidden":hidden.sum()}
    for name,mask in (("visible",visible),("hidden",hidden)):
        denominator=torch.as_tensor(corner_normalizers[name],device=errors.device,dtype=errors.dtype).clamp_min(1)
        losses[name+"_corners"]=(errors*mask).sum()/denominator
    losses["positive_depth"]=(F.relu(.005-output["camera_corners"][:,:,2]).mean(-1)/extent).mean()
    return losses

def microbatch_objective(losses,weights,micro_fraction):
    # Corner terms already use optimizer-batch denominators; don't weight twice.
    return sum(weights[k]*v*(1. if k in ("visible_corners","hidden_corners") else micro_fraction)
               for k,v in losses.items())
