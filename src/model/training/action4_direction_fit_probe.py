"""Train-only fixed16 fitting probe, never a deployable checkpoint."""
import json
import torch
import numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_direction_targets import prepare_evidence_batch
from model.training.action4_auxiliary_train import build
from model.action4_direction_losses import direction_trial_loss
from model.action4_a0_enhancement import auxiliary_vote_evidence
from model.training.action4_runtime import atomic_json


def run(output):
    torch.set_num_threads(2);torch.manual_seed(20261024);out=R/output
    cfg=json.loads((R/'configs/model/action4_a0_additive_v002.json').read_text());model,preprocess=build(cfg,'cuda:0')
    state=torch.load(R/'outputs/action4_direction_normalization_v001/selected_best.pt',map_location='cpu',weights_only=True);model.evidence.load_state_dict(state['head'],strict=True)
    selections=json.loads((R/'outputs/action4_direction_fit_audit_v001/REPORT.json').read_text())['selection']['train'];chosen=selections[:8]+selections[64:72]
    data=ComponentData(cfg['validation_receipt'],'train','A0');lookup={(data.records[ri]['sample_id'],data.records[ri]['instances'][gi]['instance_id']):i for i,(ri,gi) in enumerate(data.units)}
    items=[data[lookup[(s['sample_id'],s['instance_id'])]] for s in chosen];cached=[]
    for start in range(0,16,4):
        x,points,mask,objects=prepare_evidence_batch(items[start:start+4],preprocess,'cuda:0')
        with torch.no_grad():features=model.base.decoder(model.base.backbone(x));logits=model.base.corner_head(features)
        diagonal=[]
        for obj in objects:
            box=np.array(obj['bbox_full_xyxy']);extent=box[2:]-box[:2];diagonal.append(112/2.25*np.linalg.norm(extent)/max(extent))
        cached.append((x,points,mask,features,logits,torch.tensor(diagonal,device='cuda:0')))
    optimizer=torch.optim.AdamW(model.evidence.parameters(),lr=.001,weight_decay=.0001);history=[]
    for step in range(501):
        if step%100==0:
            errors=[];total_loss=0
            with torch.no_grad():
                for x,points,mask,features,logits,diag in cached:
                    pred=model.evidence(features,x);pred['corner_logits']=logits
                    total_loss+=float(sum(direction_trial_loss(pred,points,mask,'per_roi').values()))/4
                    torch.manual_seed(22);votes=auxiliary_vote_evidence(pred)['auxiliary_corners_hm'];errors.extend((torch.linalg.vector_norm(votes-points,dim=-1)/diag[:,None]).cpu().tolist())
            result={'step':step,'all8_at05':sum(max(e)<=.05 for e in errors),'mean_corner_error':float(np.mean(errors)),'loss':total_loss};history.append(result);atomic_json(out/'STATUS.json',result)
        if step==500:break
        optimizer.zero_grad(set_to_none=True)
        for x,points,mask,features,logits,diag in cached:
            pred=model.evidence(features,x);loss=sum(direction_trial_loss(pred,points,mask,'per_roi').values())/4;loss.backward()
        torch.nn.utils.clip_grad_norm_(model.evidence.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    atomic_json(out/'REPORT.json',{'scope':'train-only fixed16 repeated500update fit probe, no generalization claim','initial_step':state['step'],'examples':chosen,'history':history,'production_checkpoint_written':False,'frozen_base_features_cached':True})

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    try:run(a.output)
    except BaseException:
        atomic_json(R/a.output/'FAILURE.json',{'traceback':traceback.format_exc()});raise
