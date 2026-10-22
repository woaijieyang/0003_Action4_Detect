"""Real predicted-map optimizer update and exact save/reload continuation."""
import io,json,time,torch
from data.action4_paths import PROJECT_ROOT as R
from model.action4_pose_maps import PoseFromMaps,geometry_losses
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
out=R/"outputs/action4_a0_frozen_surface_pose_v001"
manifest=json.loads((out/"MAPS_COMPLETE.json").read_text())
part=torch.load(R/manifest["splits"]["train"]["parts"][0]["path"],map_location="cpu",weights_only=True)
x=part["predicted_maps"];g=part["geometry_targets"]
parent=torch.load(R/"models/A0/best.pt",map_location="cpu",weights_only=True)
model=PoseFromMaps();model.load_state_dict({k:v for k,v in parent["model"].items() if k.startswith("pose.")},strict=True)
opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
def step(m,o):
    o.zero_grad(set_to_none=True);losses=geometry_losses(m(x,g["K"],g["canonical"]),g)
    loss=sum(losses.values());assert torch.isfinite(loss);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
    return float(loss.detach()),float(norm)
before=next(model.parameters()).detach().clone();begin=time.monotonic();loss,norm=step(model,opt);elapsed=time.monotonic()-begin
assert not torch.equal(before,next(model.parameters()))
buffer=io.BytesIO();torch.save({"model":model.state_dict(),"optimizer":opt.state_dict()},buffer);buffer.seek(0)
saved=torch.load(buffer,map_location="cpu",weights_only=True)
recovered=PoseFromMaps();recovered.load_state_dict(saved["model"],strict=True)
opt2=torch.optim.AdamW(recovered.parameters(),lr=.0002,weight_decay=.0001);opt2.load_state_dict(saved["optimizer"])
step(model,opt);step(recovered,opt2)
assert all(torch.equal(v,recovered.state_dict()[k]) for k,v in model.state_dict().items())
report={"passed":True,"real_batch":len(x),"step_seconds_one_CPU":elapsed,"loss":loss,"gradient_norm":norm,
        "weights_update":True,"save_load_next_update_exact":True,"scope":"optimizer engineering with real predicted maps, not accuracy"}
(out/"UPDATE_RECOVERY_CHECK.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
