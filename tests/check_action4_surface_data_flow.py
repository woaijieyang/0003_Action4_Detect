"""One real synthetic ROI through the new GDR-derived head and losses."""
from pathlib import Path
import json,torch

def main():
    torch.set_num_threads(2);torch.manual_seed(23)
    assert Path('src/data/action4_surface_data.py').exists(),'Surface supervision adapter not implemented'
    from data.action4_surface_data import prepare_surface_batch
    from data.action4_component_data import ComponentData
    from model.action4_components import load_component
    from model.action4_surface_head import SurfaceHead,surface_losses
    data=ComponentData('outputs/action4_independent_trials_v001/DATA_VALIDATED.json','train','A0');base,state=load_component('models/action4/independent_components_v001/A0.pt',role='A0');base.eval();base.requires_grad_(False)
    x,targets=prepare_surface_batch([data[0]],state['preprocess'],'train')
    with torch.no_grad():features=base.backbone(x)
    model=SurfaceHead();output=model(features);losses=surface_losses(output,targets);sum(losses.values()).backward()
    assert all(p.grad is None for p in base.parameters())
    assert model.head.features[-1].weight.grad.norm()>0
    assert targets['valid'].sum()>0 and (targets['regions'][targets['valid']]>=1).all()
    result={'status':'PASS','sample_id':data[0]['record']['sample_id'],'valid_surface_points':int(targets['valid'].sum()),'losses':{k:float(v.detach()) for k,v in losses.items()},'head_gradient_norm':float(model.head.features[-1].weight.grad.norm()),'frozen_backbone_no_grad':True,'scope':'one real training ROI data/gradient check, not accuracy'}
    Path('outputs/action4_gdr_surface_candidate_v001/DATA_FLOW_TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
