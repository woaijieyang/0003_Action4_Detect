"""Real cached-feature GPU update and predicted-PnP evaluation integration."""
from pathlib import Path
import json,torch
from model.action4_surface_head import SurfaceHead,surface_losses
from eval.action4_surface_eval import evaluate

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(23)
    root=Path.cwd();out=root/'outputs/action4_gdr_surface_candidate_v001';data=torch.load(out/'cache/train_00000.pt',map_location='cpu',weights_only=True);model=SurfaceHead().to('cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    x=data['features'][:2].to('cuda:0');target={k:v[:2].to('cuda:0') for k,v in data['targets'].items()};losses=surface_losses(model(x),target);loss=losses['xyz']+losses['mask']+.1*losses['region'];assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);before=model.head.features[-1].weight.detach().clone();opt.step();assert not torch.equal(before,model.head.features[-1].weight)
    report=evaluate(model,data['features'][:2],data['metadata'][:2]);assert report['rois']==2
    result={'status':'PASS','actual_cached_train_ROIs':2,'finite_loss':float(loss.detach()),'gradient_norm':float(norm),'head_parameters_changed':True,'predicted_pose_eval_completed':True,'scope':'one-update engineering integration; no accuracy or convergence claim'}
    (out/'CACHED_UPDATE_TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
