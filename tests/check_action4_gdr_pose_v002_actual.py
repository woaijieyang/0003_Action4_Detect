"""Actual cached training objects exercise v002 full loss/backward."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_gdr_pose_data import geometry
from model.action4_gdr_pose import GDRPose
from model.action4_gdr_pose_losses_v002 import pose_losses,microbatch_objective
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
cfg=json.loads((R/"configs/model/action4_gdr_pose_v002.json").read_text())
manifest=json.loads((R/"outputs/action4_gdr_surface_candidate_v001/CACHE_COMPLETE.json").read_text())
part=torch.load(R/manifest["splits"]["train"]["parts"][0]["path"],map_location="cpu",weights_only=True)
g={k:v.cuda() for k,v in geometry(part["metadata"][:2],"train").items()}
y={k:v[:2].cuda() for k,v in part["targets"].items()}
model=GDRPose().cuda()
state=torch.load(R/"outputs/action4_gdr_pose_candidate_v001/pipeline_check.pt",map_location="cpu",weights_only=True)
model.load_state_dict(state["head"],strict=True)
out=model(part["features"][:2].cuda(),g["K"],g["canonical"])
losses=pose_losses(out,y,g);total=microbatch_objective(losses,cfg["loss_weights"],1.)
assert torch.isfinite(total)
geo=sum(losses[k] for k in ("rotation_point_matching","centroid","relative_depth","visible_corners","hidden_corners"))
grads=torch.autograd.grad(geo,[out["surface"][k] for k in ("xyz","regions","masks")],retain_graph=True)
assert all(torch.isfinite(x).all() and x.norm()>0 for x in grads)
total.backward()
assert model.pose.fc_r.weight.grad.norm()>0 and model.pose.fc_t.weight.grad.norm()>0
print(json.dumps({"passed":True,"source_snapshot_step":state["step"],"actual_train_ROIs":2,"losses":{k:float(v.detach()) for k,v in losses.items()},"geometry_gradient_norms":[float(x.norm()) for x in grads],"scope":"actual-data loss/backward contract, not candidate accuracy"}))
