"""CPU checks of upstream GIoU agreement and actual hidden-target gradient flow."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,torch,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from torchvision.ops import generalized_box_iou_loss
from model.action4_component_experiments import aligned_iou_giou,corner_loss
from model.training.action4_component_train import construct
from data.action4_component_data import ComponentData,roi_batch

def main():
 torch.set_num_threads(2);torch.manual_seed(87)
 xy=torch.rand(128,2)*1000;wh=torch.rand(128,2)*200+2;a=torch.cat([xy,xy+wh],-1).requires_grad_();xy2=xy+torch.randn(128,2)*100;wh2=torch.rand(128,2)*200+2;b=torch.cat([xy2,xy2+wh2],-1)
 # Translate pairs together to satisfy upstream nonnegative-coordinate contract.
 shift=torch.minimum(a.detach().amin(),b.amin()).clamp_max(0);a2=a-shift;b2=b-shift
 local=1-aligned_iou_giou(a2,b2)[1];official=generalized_box_iou_loss(a2,b2,eps=1e-9);delta=float((local-official).abs().max())
 gl=torch.autograd.grad(local.sum(),a,retain_graph=True)[0];go=torch.autograd.grad(official.sum(),a)[0];gd=float((gl-go).abs().max());assert delta<1e-5 and gd<1e-5
 cfg=json.loads((R/'configs/model/component_trials_v001/a0_joint.json').read_text());cfg['preprocess']=dict(cfg['preprocess'],roi_size=64);data=ComponentData(cfg['validation_receipt'],'train','A0')
 chosen=next(i for i,(ri,gi) in enumerate(data.units) if not all(data.records[ri]['instances'][gi]['corner_visible']))
 model=construct(cfg,'cpu');x,meta=roi_batch([data[chosen]],cfg,'cpu',np.random.default_rng(1),False);out=model(x);loss=corner_loss(out,meta,'joint')['hidden'];g=torch.autograd.grad(loss,model.correction_head.weight,retain_graph=True)[0]
 altered=dict(meta);altered['points_hm']=meta['points_hm'].clone();altered['points_hm'][~meta['visible']]+=3
 other=corner_loss(out,altered,'joint')['hidden'];g2=torch.autograd.grad(other,model.correction_head.weight)[0];assert g.abs().sum()>0 and not torch.equal(g,g2)
 report=dict(status='PASS',torchvision_giou_loss_max_difference=delta,torchvision_giou_gradient_max_difference=gd,hidden_correction_gradient_norm=float(g.norm()),hidden_target_perturb_changes_gradient=True,scope='mechanism and formula checks, not accuracy')
 (R/'outputs/action4_independent_trials_v001/LOSS_FLOW_TEST.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
