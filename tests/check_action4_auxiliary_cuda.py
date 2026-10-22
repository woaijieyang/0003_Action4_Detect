"""Actual 448ROI CUDA training and voting check before auxiliary launch."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from pathlib import Path
import sys,json,torch,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from model.training.action4_auxiliary_train import build
from model.action4_a0_enhancement import evidence_loss,direction_supervision,auxiliary_vote_evidence
from data.action4_component_data import ComponentData
from data.action4_direction_targets import prepare_evidence_batch

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cfg=json.loads((R/'configs/model/action4_a0_additive_v001.json').read_text());model,pp=build(cfg,'cuda:0');data=ComponentData(cfg['validation_receipt'],'train','A0');x,points,mask,objects=prepare_evidence_batch([data[i] for i in range(4)],pp,'cuda:0')
    before={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()};opt=torch.optim.AdamW(model.trainable_parameters(),lr=.001);opt.zero_grad();loss=evidence_loss(model(x),points,mask);sum(loss.values()).backward();assert all(p.grad is None for p in model.base.parameters());opt.step()
    assert all(torch.equal(v,model.base.state_dict()[k].cpu()) for k,v in before.items())
    directions,_=direction_supervision(points,mask);fg=torch.stack((~mask,mask),1).float()*20-10
    simulated={'corner_logits':model(x)['corner_logits'],'direction_vectors':directions,'foreground_logits':fg};votes=auxiliary_vote_evidence(simulated);error=float(torch.linalg.vector_norm(votes['auxiliary_corners_hm']-points,dim=-1).max());assert votes['numerically_valid'].all() and error<.01
    report={'status':'PASS','actual_roi_size':pp['roi_size'],'real_data_training_loss':{k:float(v.detach()) for k,v in loss.items()},'frozen_base_exact':True,'constructed_cuda_voting_error_px':error,'scope':'real sample update and constructed CUDA votes, not learned accuracy'}
    (R/'outputs/action4_a0_additive_v001/CUDA_DETERMINISTIC_TEST.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
