"""Does the learned foreground signal distinguish camera bodies from accessories?"""
import json,torch,numpy as np
from PIL import Image
from model.training.action4_auxiliary_train import build
from model.training.action4_runtime import atomic_json
from data.action4_dataset import prepare_rois
from model.training.action4_component_train import bounded
from eval.action4_metrics import bbox_iou
from data.action4_paths import PROJECT_ROOT as R


def run(output):
    out=bounded(output);torch.set_num_threads(2);state=torch.load(out/'auxiliary.pt',map_location='cpu',weights_only=True);model,preprocess=build(state['config'],'cuda:0');model.evidence.load_state_dict(state['head'],strict=True);model.eval();step=state['step'];del state
    cases=json.loads((R/'outputs/action4_scene_domain_audit_v001/counterfactual_v002/EVALUATION.json').read_text())['rows'];rows=[]
    with torch.inference_mode():
        for case in cases:
            predictions=case['predictions']
            if not predictions:continue
            with Image.open(bounded(case['image'],True)) as image:rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1)
            crops,affines=prepare_rois([rgb],[p['bbox_full_xyxy'] for p in predictions],[0]*len(predictions),roi_size=448,expansion=2.25,device='cuda:0')
            probability=model(crops)['foreground_logits'].softmax(1)[:,1]
            for i,prediction in enumerate(predictions):
                score=probability[i].flatten();matched=max((bbox_iou(prediction['bbox_full_xyxy'],g) for g in case['reference_boxes']),default=0)>=.5
                rows.append({'sample_id':case['sample_id'],'template':case['template'],'cameras_present':case['cameras_present'],'patches_present':case['patches_present'],'bbox':prediction['bbox_full_xyxy'],'B0_score':prediction['score'],'matches_camera_body':matched,'mask_positive_fraction':float((score>.5).float().mean()),'top64_mean_probability':float(score.topk(64).values.mean()),'mean_probability':float(score.mean())})
    groups={}
    for label in (False,True):
        selected=[r for r in rows if r['matches_camera_body']==label];groups[str(label)]={key:[min(r[key] for r in selected),float(np.median([r[key] for r in selected])),max(r[key] for r in selected)] for key in ('mask_positive_fraction','top64_mean_probability','mean_probability')};groups[str(label)]['count']=len(selected)
    atomic_json(out/'REPORT.json',{'scope':'fixed-head camera-body vs accessory counterfactual ROI signal audit; no learned threshold or deployment','auxiliary_step':step,'groups_min_median_max':groups,'rows':rows,'B0_score_changed':False})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
