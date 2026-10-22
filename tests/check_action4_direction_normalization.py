"""Direction gradient must not depend on microbatch partition or visible area."""
import json
from pathlib import Path
import torch
from model.action4_direction_losses import direction_trial_loss

def main():
    torch.set_num_threads(2);torch.manual_seed(7)
    vectors=torch.randn(4,8,2,12,12,requires_grad=True)
    logits=torch.randn(4,2,12,12,requires_grad=True)
    points=torch.randn(4,8,2)*20
    mask=torch.zeros(4,12,12,dtype=torch.bool)
    for i,n in enumerate([1,3,6,12]):mask[i,:n,:n]=True
    mask[3]=False  # Include an empty ROI in only one microbatch.
    output={"direction_vectors":vectors,"foreground_logits":logits}
    full=direction_trial_loss(output,points,mask,"per_roi")["direction"]
    g=torch.autograd.grad(full,vectors,retain_graph=True)[0]
    pieces=[]
    for start in [0,2]:
        o={k:v[start:start+2] for k,v in output.items()}
        pieces.append(direction_trial_loss(o,points[start:start+2],mask[start:start+2],"per_roi")["direction"]*.5)
    split=sum(pieces);sg=torch.autograd.grad(split,vectors,retain_graph=True)[0]
    assert torch.allclose(full,split,atol=1e-7) and torch.allclose(g,sg,atol=1e-7)
    changed=points.clone();changed[:,7,0]+=10
    changed_loss=direction_trial_loss(output,changed,mask,"per_roi")["direction"]
    cg=torch.autograd.grad(changed_loss,vectors,retain_graph=True)[0]
    assert not torch.allclose(g[:,7],cg[:,7])
    assert torch.equal(g[:,:7],cg[:,:7])
    empty=direction_trial_loss(output,points,torch.zeros_like(mask),"per_roi")["direction"]
    eg=torch.autograd.grad(empty,vectors)[0];assert empty.item()==0 and torch.equal(eg,torch.zeros_like(eg))
    result={"status":"PASS","microbatch_loss_and_gradient_invariant":True,"fixed_ID7_label_changes_only_ID7_direction_gradient":True,"empty_mask_zero_direction_gradient":True,"scope":"constructed gradient contract, not model accuracy"}
    out=Path("outputs/action4_direction_normalization_v001");out.mkdir(exist_ok=True)
    (out/"LOSS_TEST.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
