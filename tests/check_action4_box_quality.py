"""Box-aware score must depend on box content and learn continuous IoU targets."""
from pathlib import Path
import json,torch

def main():
    torch.set_num_threads(1);torch.manual_seed(23)
    assert Path('src/model/action4_box_quality.py').exists(),'Box-aware quality module missing'
    from model.action4_box_quality import box_features,BoxAwareQuality,varifocal_loss
    feature=torch.arange(2*8*8,dtype=torch.float32).reshape(1,2,8,8)
    boxes=torch.tensor([[0.,0.,12.,12.],[16.,16.,28.,28.]])
    pooled,geometry=box_features(feature,boxes,(32,32));assert pooled.shape==(2,18) and not torch.equal(pooled[0],pooled[1])
    model=BoxAwareQuality(channels=2);logits=model(pooled,geometry);target=torch.tensor([.8,0.]);loss=varifocal_loss(logits,target);loss.backward();assert model.score.weight.grad.norm()>0
    perfect=torch.tensor([10.,-10.]);wrong=-perfect
    assert varifocal_loss(perfect,torch.tensor([1.,0.]))<varifocal_loss(wrong,torch.tensor([1.,0.]))
    out=Path('outputs/action4_box_quality_candidate_v001');out.mkdir(exist_ok=True);result={'status':'PASS','box_change_changes_input_features':True,'continuous_quality_target_has_gradient':True,'scope':'constructed feature/loss contract, not accuracy'};(out/'MODULE_TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
