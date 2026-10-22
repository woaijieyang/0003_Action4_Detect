"""Real RGB update verifies refined losses reach B0 image features."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images
from eval.action4_component_pipeline import load_trained_component
from eval.action4_box_quality_pool import candidate_union as old_union
from model.action4_proposal_pool import candidate_union
from model.action4_box_quality import box_features
from model.action4_online_b0 import OnlineB0,proposal_features
from model.training.action4_online_b0_batch import image_loss
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(5)
teacher,cfg=load_trained_component(R/"outputs/action4_box_quality_candidate_v001/base_B0_best.pt","B0","cuda:0")
model=OnlineB0(cfg["network"]).cuda();model.detector.load_state_dict(teacher.state_dict(),strict=True);del teacher
state=torch.load(R/"outputs/action4_box_refinement_v001/final_best_frozen.pt",map_location="cpu",weights_only=True)
model.refiner.load_state_dict(state["head"],strict=True)
data=ComponentData("outputs/action4_independent_trials_v001/DATA_VALIDATED.json","train","B0")
item=data[0];model.eval()
with torch.no_grad():
    x,aff=prepare_full_images([item["image"]],size=tuple(cfg["preprocess"]["full_size"]),device="cuda:0")
    dense,f=model(x)
    old,oldboxes=old_union(dense,aff[0],cfg);new,boxes=candidate_union(dense,aff[0],cfg)
    assert old==new and torch.equal(oldboxes,boxes)
    oldf,oldg=box_features(f,boxes,tuple(cfg["preprocess"]["full_size"]))
    newf,newg=proposal_features(f,boxes,tuple(cfg["preprocess"]["full_size"]))
    error=float((oldf-newf).abs().max())
    assert torch.allclose(oldf,newf,atol=.001,rtol=.0001) and torch.equal(oldg,newg)
model.train();losses=image_loss(model,item,cfg,"cuda:0")
weight=next(model.detector.backbone.parameters())
routes={}
for name in ("refined_quality","refined_giou","refined_regression"):
    gradient=torch.autograd.grad(losses[name],weight,retain_graph=True,allow_unused=True)[0]
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm()>0
    routes[name]=float(gradient.norm())
optimizer=torch.optim.AdamW(model.parameters(),lr=.000001)
optimizer.zero_grad();sum(losses.values()).backward()
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
assert all(torch.isfinite(p).all() for p in model.parameters())
print(json.dumps({"passed":True,"image":item["record"]["sample_id"],"candidate_union_exact":True,"pooled_feature_max_error":error,"refined_loss_to_backbone_gradient":routes,"update_gradient_norm":float(norm),"scope":"actualRGB gradient/update engineering test, not accuracy"}))
