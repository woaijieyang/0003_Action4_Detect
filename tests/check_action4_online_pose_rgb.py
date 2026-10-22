"""Actual RGB-jitter forward parity and isolated trainable-pose update."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_online_pose_data import PoseRGBTrainingData,PoseEpochBatches
from model.action4_online_pose_rgb import OnlinePoseRGB
from model.action4_release_bundle import load_bundle
from model.action4_pose_maps import geometry_losses
torch.set_num_threads(1);torch.set_num_interop_threads(1)
data=PoseRGBTrainingData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json",20261023,{"center_fraction":.15,"scale_range":[.65,1.35]})
first=[data[(i,0)] for i in [0,1]];changed=data[(0,1)]
assert not torch.equal(first[0]["roi"],changed["roi"])
assert torch.equal(first[0]["geometry"]["rotation"],changed["geometry"]["rotation"])
rois=torch.stack([x["roi"] for x in first]);g={k:torch.stack([x["geometry"][k] for x in first]) for k in first[0]["geometry"]}
path="models/candidates/a0_frozen_surface_pose_v001/best.pt"
model=OnlinePoseRGB(path).eval();head,base,state=load_bundle(path,"A0","cpu")
with torch.no_grad():
 expected=head(base.backbone(rois),g["K"],g["canonical"])["corners"]
 actual=model(rois,g["K"],g["canonical"])["corners"]
delta=float((actual-expected).abs().max());assert delta<.001
model.train()
assert all(not p.requires_grad for p in model.backbone.parameters()) and all(not p.requires_grad for p in model.surface.parameters())
before=next(model.pose.parameters()).detach().clone()
opt=torch.optim.AdamW(model.pose.parameters(),lr=1e-6)
losses=geometry_losses(model(rois,g["K"],g["canonical"]),g);loss=sum(losses.values())
opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.pose.parameters(),1.,error_if_nonfinite=True);opt.step()
assert not torch.equal(before,next(model.pose.parameters()))
assert all(p.grad is None for p in model.backbone.parameters()) and all(p.grad is None for p in model.surface.parameters())
identities=[]
for rank in range(4):
 sampler=PoseEpochBatches(3024,16,4,rank,7);sampler.set_epoch(3)
 for batch in sampler:
  assert len(batch)==4 and all(ep==3 for i,ep in batch);identities.extend(i for i,ep in batch)
assert sorted(identities)==list(range(3024))
report={"passed":True,"actual_RGB_targets":2,"next_epoch_changes_crop":True,"global_pose_targets_unchanged":True,
"forward_parity_px":delta,"only_pose_parameters_updated":True,"rank_target_coverage_exact":True,"loss":float(loss.detach()),
"scope":"RGB crop/update engineering, not new accuracy"}
out=R/"outputs/action4_a0_roi_contract_v001";(out/"ONLINE_RGB_CHECK.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
