"""Real CUDA crop/update preflight without changing sealed weights."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_online_pose_data import PoseRGBTrainingData,PoseEpochBatches
from data.action4_pose_rgb_batch import prepare_pose_rgb
from model.action4_online_pose_rgb import OnlinePoseRGB
from model.action4_pose_maps import geometry_losses

torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
cfg=json.loads((R/"configs/model/action4_a0_online_rgb_v001.json").read_text());out=R/cfg["output"]
data=PoseRGBTrainingData(cfg["validation_receipt"],cfg["seed"],cfg["roi_jitter"])
units=next(iter(PoseEpochBatches(len(data),16,4,0,cfg["seed"])))
items=[data[u] for u in units]
rois=torch.stack([x["roi"] for x in items]).cuda()
g={k:torch.stack([x["geometry"][k] for x in items]).cuda() for k in items[0]["geometry"]}
raw=[data.data[u[0]] for u in units]
direct,_,_=prepare_pose_rgb(raw,device="cuda:0",boxes=[x["metadata"]["crop_box"] for x in items])
pixel_delta=float((rois-direct).abs().max());assert pixel_delta<.01
model=OnlinePoseRGB(cfg["parent_bundle"]).cuda().train()
buffers={k:v.detach().cpu().clone() for k,v in model.backbone.named_buffers()}
frozen={p.data_ptr() for p in model.backbone.parameters()}|{p.data_ptr() for p in model.surface.parameters()}
trainable={p.data_ptr() for p in model.pose.parameters()};assert not frozen&trainable
optimizer=torch.optim.AdamW(model.pose.parameters(),lr=1e-6)
before=next(model.pose.parameters()).detach().clone()
output=model(rois,g["K"],g["canonical"]);losses=geometry_losses(output,g);loss=sum(losses.values())
assert torch.isfinite(loss);loss.backward()
norm=torch.nn.utils.clip_grad_norm_(model.pose.parameters(),1.,error_if_nonfinite=True);optimizer.step()
assert not torch.equal(before,next(model.pose.parameters()))
assert all(p.grad is None for p in model.backbone.parameters()) and all(p.grad is None for p in model.surface.parameters())
assert all(torch.equal(v.cpu(),buffers[k]) for k,v in model.backbone.named_buffers())
report={"passed":True,"actual_local_batch":len(items),"CPU_vs_GPU_crop_max_normalized_RGB_error":pixel_delta,
    "loss":float(loss.detach()),"gradient_norm":float(norm),"pose_update":True,"frozen_parameters_have_no_grad":True,
    "backbone_buffers_unchanged":True,"peak_memory_mib":torch.cuda.max_memory_allocated()/1048576,
    "scope":"actual RGB CUDA training interface, not accuracy"}
(out/"PREFLIGHT.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
