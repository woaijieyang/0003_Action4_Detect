"""Actual predicted-surface inputs: forward/loss parity and supervised gradients."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_gdr_pose_data import geometry
from model.action4_gdr_pose import GDRPose
from model.action4_gdr_pose_losses_v002 import pose_losses
from model.action4_pose_maps import PoseFromMaps,surface_to_maps,geometry_losses

torch.set_num_threads(1);torch.set_num_interop_threads(1)
base=R/"outputs/action4_gdr_surface_candidate_v001"
manifest=json.loads((base/"CACHE_COMPLETE.json").read_text())
part=torch.load(R/manifest["splits"]["train"]["parts"][0]["path"],map_location="cpu",weights_only=True)
features=part["features"][:2];meta=part["metadata"][:2]
g=geometry(meta,"train");targets={k:v[:2] for k,v in part["targets"].items()}
parent=torch.load(R/"models/A0/best.pt",map_location="cpu",weights_only=True)
surface=torch.load(base/"train_run_v001/best.pt",map_location="cpu",weights_only=True)
original=GDRPose().eval();original.load_state_dict(parent["model"],strict=True)
original.surface.load_state_dict(surface["head"],strict=True);original.surface.requires_grad_(False)
candidate=PoseFromMaps().eval();candidate.pose.load_state_dict(original.pose.state_dict(),strict=True)
with torch.no_grad():
    frozen_surface=original.surface(features)
    maps=surface_to_maps(frozen_surface,g["K"])
    expected=original(features,g["K"],g["canonical"])
actual=candidate(maps,g["K"],g["canonical"])
delta=max(float((actual[k]-expected[k]).detach().abs().max()) for k in actual)
assert delta<1e-5
old_losses=pose_losses(dict(expected,surface=frozen_surface),targets,g)
new_losses=geometry_losses(actual,g)
assert all(torch.allclose(v,old_losses[k],atol=1e-7,rtol=1e-6) for k,v in new_losses.items())
maps=maps.detach().requires_grad_(True)
actual=candidate(maps,g["K"],g["canonical"]);losses=geometry_losses(actual,g);sum(losses.values()).backward()
groups={"xyz":maps.grad[:,:3],"rays":maps.grad[:,3:5],"regions":maps.grad[:,5:69],"masks":maps.grad[:,69:]}
norms={k:float(v.norm()) for k,v in groups.items()}
assert all(value>0 for value in norms.values())
assert candidate.pose.fc_r.weight.grad.norm()>0 and candidate.pose.fc_t.weight.grad.norm()>0
report={"passed":True,"frozen_surface_step":surface["step"],"pose_parent_step":parent["selected_step"],"actual_ROIs":2,
    "forward_max_delta":delta,"pose_loss_parity":True,"map_input_gradients":norms,
    "rotation_gradient":float(candidate.pose.fc_r.weight.grad.norm()),"translation_gradient":float(candidate.pose.fc_t.weight.grad.norm()),
    "scope":"actual input/target dataflow parity and gradients, not accuracy"}
(R/"outputs/action4_a0_frozen_surface_pose_v001/PARITY_AND_GRADIENTS.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
