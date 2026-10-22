"""Microbatch invariance, zero-target and decoupled-rotation regressions."""
import json,torch
from model.action4_gdr_pose_losses_v002 import pose_losses,microbatch_objective
from model.action4_gdr_pose import pose_losses as old_losses
torch.set_num_threads(1)
B=16;dtype=torch.float64
canonical=torch.tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=dtype)[None].repeat(B,1,1)*torch.tensor([.035,.016,.022],dtype=dtype)
rotation=torch.eye(3,dtype=dtype)[None].repeat(B,1,1);translation=torch.tensor([[0.,0.,.7]],dtype=dtype).repeat(B,1)
K=torch.tensor([[[100.,0,56],[0,100,56],[0,0,1]]],dtype=dtype).repeat(B,1,1)
camera=canonical+translation[:,None];hom=camera@K.transpose(1,2);corners=hom[:,:,:2]/hom[:,:,2:]
visible=torch.zeros(B,8,dtype=torch.bool)
for i,n in enumerate([0,1,2,3,8,8,8,8,1,1,1,1,7,6,5,4]):visible[i,:n]=True
g={"canonical":canonical,"rotation":rotation,"translation":translation,"K":K,"corners":corners,"diagonal":torch.ones(B,dtype=dtype)*50,"visible":visible}
surface={"xyz":torch.zeros(B,3,2,2,dtype=dtype),"regions":torch.zeros(B,65,2,2,dtype=dtype),"masks":torch.zeros(B,2,2,2,dtype=dtype)}
target={"xyz":surface["xyz"].clone(),"regions":torch.ones(B,2,2,dtype=torch.long),"valid":torch.ones(B,2,2,dtype=torch.bool),"masks":torch.zeros_like(surface["masks"])}
def sliced(data,a,z):return {k:v[a:z] for k,v in data.items()}
def calculate(micro,old=False):
    shift=torch.tensor(.7,dtype=dtype,requires_grad=True);total=0.
    perturb=torch.arange(1,B+1,dtype=dtype)[:,None,None]*shift
    for a in range(0,B,micro):
        z=min(a+micro,B);o={"surface":sliced(surface,a,z),"rotation":rotation[a:z],"translation":translation[a:z],"depth":translation[a:z,2],"camera_corners":camera[a:z],"corners":corners[a:z]+perturb[a:z]}
        losses=(old_losses(o,sliced(target,a,z),sliced(g,a,z)) if old else pose_losses(o,sliced(target,a,z),sliced(g,a,z),{"visible":visible.sum(),"hidden":(~visible).sum()}))
        if old:total+=(losses["visible_corners"]+losses["hidden_corners"])*(z-a)/B
        else:total+=microbatch_objective(losses,{k:float(k in ("visible_corners","hidden_corners")) for k in losses},(z-a)/B)
    grad=torch.autograd.grad(total,shift)[0]
    return [float(total),float(grad)]
old16=calculate(16,True);old4=calculate(4,True)
assert max(abs(a-b) for a,b in zip(old16,old4))>1e-4
new={m:calculate(m) for m in (16,8,4,1)}
assert max(abs(x-y) for row in new.values() for x,y in zip(row,new[16]))<1e-12
# Rotation-only PM remains informative under large translation errors.
theta=torch.tensor(.3,dtype=dtype,requires_grad=True)
zero=theta*0;one=zero+1;c=theta.cos();s=theta.sin()
rot=torch.stack([c,-s,zero,s,c,zero,zero,zero,one]).reshape(1,3,3).repeat(B,1,1)
t=translation+.1
o={"surface":surface,"rotation":rot,"translation":t,"depth":t[:,2],"camera_corners":canonical@rot.transpose(1,2)+t[:,None],"corners":corners}
v=pose_losses(o,target,g)
grad=float(torch.autograd.grad(v["rotation_point_matching"],theta)[0]);assert abs(grad)>.01
old=old_losses(o,target,g)
# Ground truth pose gives zero pose losses, independent of surface auxiliaries.
o.update(rotation=rotation,translation=translation,depth=translation[:,2],camera_corners=camera,corners=corners)
zero_losses=pose_losses(o,target,g)
for name in ("rotation_point_matching","centroid","relative_depth","visible_corners","hidden_corners","positive_depth"):assert abs(float(zero_losses[name]))<1e-12
for value in (True,False):
    all_g=dict(g,visible=torch.full_like(visible,value))
    assert all(torch.isfinite(x) for x in pose_losses(o,target,all_g).values())
print(json.dumps({"passed":True,"old_micro16":old16,"old_micro4":old4,"new_micro_results":new,"rotation_gradient_with_translation_error":grad,"GT_pose_losses_zero":True,"all_visible_and_all_hidden_finite":True,"scope":"loss regression proof, not accuracy"}))
