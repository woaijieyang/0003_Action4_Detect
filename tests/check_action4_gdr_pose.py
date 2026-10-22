"""Upstream port equivalence and actual-data end-to-end geometry gradients."""
import json,torch
from torch import nn
from pathlib import Path
from model.heads.gdr_conv_pnp import ConvPnPNet
from model.heads.gdr_region_head import normal_init,constant_init,get_norm
from model.action4_gdr_pose import GDRPose,pose_losses
from data.action4_gdr_pose_data import geometry
from data.action4_paths import PROJECT_ROOT as R
torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
# Run upstream source with equivalent init/norm and disabled DropBlock dependencies.
source=(R/"upstream/gdr_net/core/gdrn_modeling/models/conv_pnp_net.py").read_text()
source="\n".join(l for l in source.splitlines() if not l.startswith(("from mmcv","from core.utils")))
scope={"normal_init":normal_init,"constant_init":constant_init,"get_norm":get_norm,
       "DropBlock2D":lambda **k:nn.Identity(),"LinearScheduler":lambda *a,**k:nn.Identity()}
exec(compile(source,"upstream_conv_pnp","exec"),scope)
original=scope["ConvPnPNet"](71,rot_dim=6,num_regions=64,mask_attention_type="concat").eval()
ported=ConvPnPNet(71,rot_dim=6,num_regions=64,mask_attention_type="concat").eval()
ported.load_state_dict(original.state_dict(),strict=True)
coor=torch.rand(2,5,64,64);regions=torch.rand(2,64,64,64).softmax(1);masks=torch.rand(2,2,64,64);extent=torch.tensor([[.07,.03,.04]]).repeat(2,1)
a=original(coor.clone(),regions,extent,masks);b=ported(coor.clone(),regions,extent,masks)
assert all(torch.equal(x,y) for x,y in zip(a,b))
manifest=json.loads((R/"outputs/action4_gdr_surface_candidate_v001/CACHE_COMPLETE.json").read_text())
part=manifest["splits"]["train"]["parts"][0]
data=torch.load(R/part["path"],map_location="cpu",weights_only=True)
meta=data["metadata"][:2];geom=geometry(meta,"train");target={k:v[:2].cuda() for k,v in data["targets"].items()}
geom={k:v.cuda() for k,v in geom.items()}
camera=torch.bmm(geom["canonical"],geom["rotation"].transpose(1,2))+geom["translation"][:,None]
hom=torch.bmm(camera,geom["K"].transpose(1,2));oracle=hom[:,:,:2]/hom[:,:,2:]
err=float((oracle-geom["corners"]).abs().max());assert err<.001
model=GDRPose().cuda()
model.surface.load_state_dict(torch.load(R/"outputs/action4_gdr_surface_candidate_v001/train_run_v001/best.pt",map_location="cpu",weights_only=True)["head"],strict=True)
x=data["features"][:2].cuda();out=model(x,geom["K"],geom["canonical"])
losses=pose_losses(out,target,geom)
assert all(torch.isfinite(v) for v in losses.values())
# Geometry-only losses must reach every predicted geometric representation.
total=losses["point_matching"]+losses["visible_corners"]+losses["hidden_corners"]
grads=torch.autograd.grad(total,[out["surface"][k] for k in ("xyz","regions","masks")],retain_graph=True)
routes={k:float(g.norm()) for k,g in zip(("xyz","regions","masks"),grads)}
assert all(v>0 for v in routes.values())
assert grads[2][:,0].norm()>0 and grads[2][:,1].norm()>0
total.backward()
assert model.pose.fc_r.weight.grad.norm()>0 and model.pose.fc_t.weight.grad.norm()>0
assert any(p.grad is not None and p.grad.norm()>0 for p in model.surface.parameters())
print(json.dumps({"passed":True,"upstream_forward_exact":True,"actual_ROIs":2,"GT_pose_projection_max_px":err,"geometry_gradient_routes":routes,"losses":{k:float(v.detach()) for k,v in losses.items()},"scope":"engineering signal test, not accuracy"}))
