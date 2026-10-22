"""Actual-cache update and signal-routing checks, not accuracy acceptance."""
import json,copy,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_box_refinement_data import load_inputs,attach_train_targets
from model.action4_box_refinement import BoxRefinement,loss
torch.set_num_threads(2);torch.manual_seed(7)
rows=attach_train_targets(load_inputs("outputs/action4_box_quality_candidate_v001","train"))
batch=rows[:16]
t={k:torch.cat([x[k] for x in batch]).cuda() for k in ("features","geometry","priors","boxes","targets","matched","supervision_valid")}
m=BoxRefinement().cuda()
m.appearance.load_state_dict(torch.load(R/"outputs/action4_box_quality_candidate_v001/train_run_v001/best.pt",map_location="cpu",weights_only=True)["head"])
args=[t[k] for k in ("features","geometry","priors","boxes")]
out=m(*args)
assert torch.allclose(out[0],t["boxes"],atol=.001,rtol=1e-5)
assert torch.allclose(out[1].sigmoid(),t["priors"][:,0]*t["priors"][:,1],atol=.00011)
ls=loss(out,t["boxes"],t["targets"],t["matched"],t["supervision_valid"])
# Each box loss really reaches the box head; quality target is detached from box.
checks={}
for name in ("giou","regression","quality"):
 g=torch.autograd.grad(ls[name],(m.delta.weight,m.correction.weight),retain_graph=True,allow_unused=True)
 checks[name]=[None if x is None else float(x.norm()) for x in g]
assert checks["giou"][0]>0 and checks["regression"][0]>0 and checks["quality"][1]>0
assert checks["quality"][0] is None or checks["quality"][0]==0
opt=torch.optim.AdamW(m.parameters(),lr=.0002)
initial=float(sum(ls.values()).detach())
initial_parts={k:float(v.detach()) for k,v in ls.items()}
for _ in range(12):
 opt.zero_grad();ls=loss(m(*args),t["boxes"],t["targets"],t["matched"],t["supervision_valid"])
 total=sum(ls.values());assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
print(json.dumps({"initial_parts":initial_parts,"final_parts":{k:float(v.detach()) for k,v in ls.items()}}),flush=True)
assert float(ls["giou"].detach()) < initial_parts["giou"]
assert float(ls["regression"].detach()) < initial_parts["regression"]
# Dynamic IoU quality targets change after box updates; total loss is not a fixed objective comparison.
assert m.appearance.appearance[1].weight.grad.norm()>0
assert m.appearance.score.weight.grad.norm()>0
restored=BoxRefinement().cuda();restored.load_state_dict(m.state_dict())
assert torch.equal(restored(*args)[0],m(*args)[0])
report={"scope":"actual16train-image engineering check, not generalization","passed":True,"loss_before":initial,"loss_after":float(total.detach()),"gradient_routes":checks,"positive_proposals":int(t["matched"].sum()),"all_train_images_loaded":len(rows)}
print(json.dumps(report))
