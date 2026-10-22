"""Isolate mask and direction errors; oracle labels are diagnostic only."""
import json
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_direction_targets import prepare_evidence_batch
from eval.action4_component_eval_cached import CachedComponentData
from model.action4_a0_enhancement import auxiliary_vote_evidence,direction_supervision
from model.training.action4_auxiliary_train import build
from model.training.action4_runtime import atomic_json


def run(output):
    torch.set_num_threads(2);out=R/output
    cfg=json.loads((R/'configs/model/action4_a0_additive_v002.json').read_text())
    model,preprocess=build(cfg,'cuda:0')
    state=torch.load(out/'auxiliary.pt',map_location='cpu',weights_only=True)
    model.evidence.load_state_dict(state['head'],strict=True);model.eval()
    data=CachedComponentData(cfg['validation_receipt'],'eval','A0');rows=[]
    with torch.inference_mode():
        for start in range(0,len(data),4):
            items=[data[i] for i in range(start,min(start+4,len(data)))];x,points,mask,objects=prepare_evidence_batch(items,preprocess,'cuda:0');pred=model(x)
            target,support=direction_supervision(points,mask)
            unit=torch.nn.functional.normalize(pred['direction_vectors'],dim=2)
            angle=(unit*target).sum(2).clamp(-1,1).acos()*180/np.pi
            angle=(angle*support).sum((-1,-2))/support.sum((-1,-2)).clamp_min(1)
            variants={}
            for name in ['predicted','oracle_mask']+(['oracle_mask_direction'] if start<16 else []):
                values=dict(pred)
                if name!='predicted':values['foreground_logits']=torch.stack((~mask,mask),1).float()*20
                if name=='oracle_mask_direction':values['direction_vectors']=target
                torch.manual_seed(20261022+start);torch.cuda.manual_seed_all(20261022+start)
                variants[name]=auxiliary_vote_evidence(values)
            for i,obj in enumerate(objects):
                box=np.array(obj['bbox_full_xyxy']);extent=box[2:]-box[:2]
                diagonal=(preprocess['roi_size']/4)/preprocess['roi_expansion']*np.linalg.norm(extent)/max(extent)
                rows.append({'sample_id':items[i]['record']['sample_id'],'instance_id':obj['instance_id'],'visible':obj['corner_visible'],'heavy':obj['occlusion_inframe']>=.4,'angle_degrees':angle[i].cpu().tolist(),'errors':{name:(torch.linalg.vector_norm(v['auxiliary_corners_hm'][i]-points[i],dim=-1)/diagonal).cpu().tolist() for name,v in variants.items()}})
            atomic_json(out/'STATUS.json',{'completed_rois':len(rows),'total':len(data)})
    scores={}
    for name in ['predicted','oracle_mask','oracle_mask_direction']:
        rr=[r for r in rows if name in r['errors']]
        scores[name]={'rois':len(rr),'all8_at05':sum(max(r['errors'][name])<=.05 for r in rr),'heavy_all8_at05':sum(r['heavy'] and max(r['errors'][name])<=.05 for r in rr),'hidden_mean':float(np.mean([e for r in rr for e,v in zip(r['errors'][name],r['visible']) if not v]))}
    atomic_json(out/'REPORT.json',{'scope':'oracle GT-mask/GT-direction diagnostic only, forbidden production inputs','checkpoint_step':state['step'],'scores':scores,'rows':rows,'production_accepted':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
