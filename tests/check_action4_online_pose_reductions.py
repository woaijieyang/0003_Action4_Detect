"""Unequal visible counts must reduce as one optimizer batch, not mean rank ratios."""
import torch,json
from model.action4_pose_maps import geometry_losses
from model.training.action4_online_pose_train import distributed_objective
from eval.action4_online_pose_eval import joint_summary
torch.set_num_threads(1)
n=16;cad=torch.tensor([[x,y,z] for x in [-.035,.035] for y in [-.016,.016] for z in [-.022,.022]])[None].repeat(n,1,1)
K=torch.tensor([[100.,0,56],[0,100,56],[0,0,1]])[None].repeat(n,1,1)
R=torch.eye(3)[None].repeat(n,1,1);t=torch.tensor([0.,0.,1.])[None].repeat(n,1)
camera=cad+t[:,None];h=torch.bmm(camera,K.transpose(1,2));corners=h[:,:,:2]/h[:,:,2:]
visible=torch.zeros(n,8,dtype=torch.bool);visible[:4]=True;visible[8:12,::2]=True;visible[12:,0]=True
g={"K":K,"canonical":cad,"rotation":R,"translation":t,"corners":corners,"diagonal":torch.ones(n)*50,"visible":visible}
def predict(p):
    shift=torch.arange(1,n+1,dtype=torch.float32)[:,None]/n*p*torch.tensor([[.1,-.2,.05]])
    translation=t+shift;cam=cad+translation[:,None];hom=torch.bmm(cam,K.transpose(1,2))
    return {"rotation":R,"translation":translation,"depth":translation[:,2],"camera_corners":cam,"corners":hom[:,:,:2]/hom[:,:,2:]}
p=torch.tensor(.5,requires_grad=True);full=geometry_losses(predict(p),g);weights={k:1. for k in full};objective=sum(full.values());objective.backward();expected_grad=p.grad.clone()
q=torch.tensor(.5,requires_grad=True);output=predict(q);combined=0;naive=0
normalizers={"visible":visible.sum(),"hidden":(~visible).sum()}
for rank in range(4):
    sl=slice(rank*4,(rank+1)*4);local_out={k:v[sl] for k,v in output.items()};local_g={k:v[sl] for k,v in g.items()}
    combined=combined+distributed_objective(geometry_losses(local_out,local_g,normalizers),weights,4)/4
    naive=naive+sum(geometry_losses(local_out,local_g).values())/4
combined.backward()
assert torch.allclose(objective,combined,atol=1e-7,rtol=1e-6)
assert torch.allclose(expected_grad,q.grad,atol=1e-7,rtol=1e-6)
assert abs(float(naive.detach()-objective.detach()))>1e-5
summary=joint_summary([{"all8":True},{"all8":False},{"all8":True},{"all8":False}],
 {"total_GT":6,"fixed_B0_prediction_count":5,"fixed_B0_missed_GT":2,"fixed_B0_false_positives":1})
assert summary["joint_precision"]==2/5 and summary["joint_recall"]==2/6 and summary["joint_f1"]==4/11
print(json.dumps({"passed":True,"global_loss":float(objective.detach()),"rank_reduction_loss":float(combined.detach()),
 "global_gradient":float(expected_grad),"rank_gradient":float(q.grad),"incorrect_rank_mean_loss":float(naive.detach()),
 "misses_and_FP_kept":True,"scope":"reduction/metric regression, not accuracy"}))
