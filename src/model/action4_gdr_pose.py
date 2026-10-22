"""GDR-derived joint geometry -> ConvPnP -> fixed canonical-eight projection.

Network inputs: frozen A0 RGB features, crop intrinsics, fixed CAD dimensions.
Target pose, corners and visibility are accepted by losses only.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model.action4_surface_head import SurfaceHead,surface_losses
from model.heads.gdr_conv_pnp import ConvPnPNet
from model.heads.gdr_rotation import ortho6d_to_mat_batch

class GDRPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.surface=SurfaceHead()
        # XYZ3 + normalized camera rays2 + region64 + visible/amodal masks2.
        self.pose=ConvPnPNet(nIn=71,featdim=128,rot_dim=6,num_regions=64,
                             drop_prob=0,mask_attention_type="concat")
        with torch.no_grad():
            self.pose.fc_r.bias.copy_(torch.tensor([1.,0,0,0,1.,0]))
            self.pose.fc_t.bias.copy_(torch.tensor([0.,0.,1.85458654]))

    def forward(self,features,K,canonical):
        surface=self.surface(features)
        if K.shape!=(len(features),3,3) or canonical.shape!=(len(features),8,3):
            raise ValueError("Batched crop K and fixed8 CAD required")
        size=surface["xyz"].shape[-1]
        dtype=features.dtype;device=features.device
        y,x=torch.meshgrid(torch.arange(size,device=device,dtype=dtype),
                           torch.arange(size,device=device,dtype=dtype),indexing="ij")
        rays=torch.stack(((x[None]-K[:,0,2,None,None])/K[:,0,0,None,None],
                          (y[None]-K[:,1,2,None,None])/K[:,1,1,None,None]),1)
        xyz=F.interpolate(surface["xyz"],size=(64,64),mode="bilinear",align_corners=False)
        rays=F.interpolate(rays,size=(64,64),mode="bilinear",align_corners=False)
        region=F.interpolate(surface["regions"][:,1:].softmax(1),size=(64,64),mode="bilinear",align_corners=False)
        masks=F.interpolate(surface["masks"].sigmoid(),size=(64,64),mode="bilinear",align_corners=False)
        extent=canonical.amax(1)-canonical.amin(1)
        rot6d,t=self.pose(torch.cat((xyz,rays),1),region=region,extents=extent,mask_attention=masks)
        rotation=ortho6d_to_mat_batch(rot6d)
        # Ego rotation + crop centroid + positive relative depth. Explicitly adapted,
        # not the upstream allocentric-centroid conversion.
        centroid=(t[:,:2]+.5)*size
        depth=F.softplus(t[:,2]) * ((K[:,0,0]+K[:,1,1])*.5*extent.amax(-1)/size)
        translation=torch.stack(((centroid[:,0]-K[:,0,2])/K[:,0,0]*depth,
                                 (centroid[:,1]-K[:,1,2])/K[:,1,1]*depth,depth),-1)
        camera=torch.bmm(canonical,rotation.transpose(1,2))+translation[:,None]
        hom=torch.bmm(camera,K.transpose(1,2))
        corners=hom[:,:,:2]/camera[:,:,2:].clamp_min(1e-4)
        return {"surface":surface,"rotation":rotation,"translation":translation,
                "camera_corners":camera,"corners":corners,"depth":depth}

def pose_losses(output,targets,geometry):
    losses=surface_losses(output["surface"],targets)
    reference=torch.bmm(geometry["canonical"],geometry["rotation"].transpose(1,2))+geometry["translation"][:,None]
    extent=(geometry["canonical"].amax(1)-geometry["canonical"].amin(1)).amax(-1)
    # Fixed ID correspondence; never minimize over permutations.
    losses["point_matching"]=(torch.abs(output["camera_corners"]-reference).mean((1,2))/extent).mean()
    error=F.smooth_l1_loss(output["corners"]/geometry["diagonal"][:,None,None],
                         geometry["corners"]/geometry["diagonal"][:,None,None],
                         beta=.05,reduction="none").mean(-1)
    visible=geometry["visible"].bool()
    losses["visible_corners"]=(error*visible).sum()/visible.sum().clamp_min(1)
    hidden=~visible
    losses["hidden_corners"]=(error*hidden).sum()/hidden.sum().clamp_min(1)
    losses["positive_depth"]=F.relu(.005-output["camera_corners"][:,:,2]).mean()/extent.mean()
    return losses
