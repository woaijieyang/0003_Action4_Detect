"""Surface head contract: XYZ/region supervision must produce real gradients."""
from pathlib import Path
import json
import torch

def main(device="cpu"):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1);torch.manual_seed(23)
    assert Path('src/model/action4_surface_head.py').exists(), 'Dense surface head not implemented'
    from model.action4_surface_head import SurfaceHead,surface_losses
    model=SurfaceHead().to(device);features=torch.randn(1,2048,14,14,device=device);out=model(features)
    assert out['xyz'].shape==(1,3,112,112) and out['masks'].shape==(1,2,112,112) and out['regions'].shape==(1,65,112,112)
    valid=torch.zeros(1,112,112,dtype=torch.bool,device=device);valid[:,40:70,40:70]=True
    target={'xyz':torch.rand_like(out['xyz']),'valid':valid,'regions':torch.full((1,112,112),-100,dtype=torch.long,device=device),'masks':torch.zeros_like(out['masks'])};target['regions'][valid]=7;target['masks'][:,:,40:70,40:70]=1
    loss=surface_losses(out,target);g=torch.autograd.grad(loss['xyz'],out['xyz'],retain_graph=True)[0];assert g[:,:,valid[0]].abs().sum()>0 and g[:,:,~valid[0]].abs().sum()==0
    old=model.head.features[-1].weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=.001);opt.zero_grad();sum(loss.values()).backward();opt.step();assert not torch.equal(old,model.head.features[-1].weight)
    result={'status':'PASS','output_grid':[112,112],'XYZ_changes_parameters':True,'invalid_XYZ_pixels_zero_gradient':True,'scope':'constructed feature/head gradient test, not accuracy'}
    outpath=Path('outputs/action4_gdr_surface_candidate_v001');outpath.mkdir(exist_ok=True);(outpath/('HEAD_TEST_CUDA.json' if device!='cpu' else 'HEAD_TEST.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args();main(a.device)
