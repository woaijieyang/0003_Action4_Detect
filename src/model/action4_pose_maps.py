"""GDR ConvPnP on frozen predicted surface maps; no target map input."""
import torch
from torch import nn
from torch.nn import functional as F
from model.heads.gdr_conv_pnp import ConvPnPNet
from model.heads.gdr_rotation import ortho6d_to_mat_batch

def surface_to_maps(surface,K):
    xyz=surface["xyz"];size=xyz.shape[-1];device=xyz.device;dtype=xyz.dtype
    y,x=torch.meshgrid(torch.arange(size,device=device,dtype=dtype),torch.arange(size,device=device,dtype=dtype),indexing="ij")
    rays=torch.stack(((x[None]-K[:,0,2,None,None])/K[:,0,0,None,None],
                      (y[None]-K[:,1,2,None,None])/K[:,1,1,None,None]),1)
    resize=lambda v:F.interpolate(v,size=(64,64),mode="bilinear",align_corners=False)
    return torch.cat((resize(xyz),resize(rays),resize(surface["regions"][:,1:].softmax(1)),resize(surface["masks"].sigmoid())),1)

class PoseFromMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose=ConvPnPNet(nIn=71,featdim=128,rot_dim=6,num_regions=64,drop_prob=0,mask_attention_type="concat")
    def forward(self,maps,K,canonical):
        if maps.ndim!=4 or maps.shape[1:]!=(71,64,64):raise ValueError("Predicted71channel maps required")
        extent=canonical.amax(1)-canonical.amin(1)
        r,t=self.pose(maps[:,:5],region=maps[:,5:69],extents=extent,mask_attention=maps[:,69:])
        rotation=ortho6d_to_mat_batch(r);size=112
        centroid=(t[:,:2]+.5)*size
        depth=F.softplus(t[:,2])*((K[:,0,0]+K[:,1,1])*.5*extent.amax(-1)/size)
        translation=torch.stack(((centroid[:,0]-K[:,0,2])/K[:,0,0]*depth,
                                 (centroid[:,1]-K[:,1,2])/K[:,1,1]*depth,depth),-1)
        camera=torch.bmm(canonical,rotation.transpose(1,2))+translation[:,None]
        hom=torch.bmm(camera,K.transpose(1,2))
        return {"rotation":rotation,"translation":translation,"depth":depth,"camera_corners":camera,
                "corners":hom[:,:,:2]/camera[:,:,2:].clamp_min(1e-4)}

def geometry_losses(output,geometry,normalizers=None):
    canonical=geometry["canonical"];extent=(canonical.amax(1)-canonical.amin(1)).amax(-1)
    pr=torch.bmm(canonical,output["rotation"].transpose(1,2))
    gt=torch.bmm(canonical,geometry["rotation"].transpose(1,2))
    losses={"rotation_point_matching":(torch.abs(pr-gt).mean((1,2))*3/extent).mean()}
    K=geometry["K"]
    def center(t):
        h=torch.bmm(K,t[:,:,None]).squeeze(-1)
        return h[:,:2]/t[:,2:].clamp_min(1e-4)
    losses["centroid"]=F.l1_loss(center(output["translation"])/112,center(geometry["translation"])/112)
    scale=(K[:,0,0]+K[:,1,1])*.5*extent/112
    losses["relative_depth"]=F.l1_loss(output["depth"]/scale,geometry["translation"][:,2]/scale)
    errors=F.smooth_l1_loss(output["corners"]/geometry["diagonal"][:,None,None],geometry["corners"]/geometry["diagonal"][:,None,None],beta=.05,reduction="none").mean(-1)
    visible=geometry["visible"].bool()
    if normalizers is None:normalizers={"visible":visible.sum(),"hidden":(~visible).sum()}
    for name,mask in [("visible",visible),("hidden",~visible)]:
        losses[name+"_corners"]=(errors*mask).sum()/torch.as_tensor(normalizers[name],device=errors.device).clamp_min(1)
    losses["positive_depth"]=(F.relu(.005-output["camera_corners"][:,:,2]).mean(-1)/extent).mean()
    return losses
