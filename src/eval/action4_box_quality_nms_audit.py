"""Fixed2x2 score/NMS comparison, no training or threshold sweep."""
import json
import torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_box_quality import load_box_split
from model.action4_box_quality import BoxAwareQuality
from eval.action4_box_quality_decode import select_boxes
from eval.action4_bbox_metrics import score_boxes
from eval.action4_metrics import bbox_iou,is_ignored
from model.training.action4_runtime import atomic_json

def standard_nms(candidates,scores):
    result=[]
    for i in np.argsort(-np.asarray(scores),kind='stable'):
        if scores[i]<.001:continue
        row=dict(candidates[i],score=float(scores[i]))
        if any(bbox_iou(row['bbox_full_xyxy'],other['bbox_full_xyxy'])>=.5 for other in result):continue
        result.append(row)
        if len(result)==2:break
    return result

def run():
    torch.set_num_threads(1);out=R/'outputs/action4_box_quality_nms_audit_v001';data=load_box_split(R/'outputs/action4_box_quality_candidate_v001','eval');records=[json.loads(l) for l in (R/'data/eval/annotations.jsonl').read_text().splitlines()];state=torch.load(out/'head.pt',map_location='cpu',weights_only=True);model=BoxAwareQuality().to('cuda:0');model.load_state_dict(state['head'],strict=True);model.eval();predictions={k:[] for k in ['old_legacy','new_legacy','old_nms05','new_nms05']}
    with torch.inference_mode():
        for row in data:
            new=model(row['features'].to('cuda:0'),row['geometry'].to('cuda:0')).sigmoid().cpu().numpy();old=np.array([c['old_product_score'] for c in row['candidates']])
            for name,scores,selector in [('old_legacy',old,select_boxes),('new_legacy',new,select_boxes),('old_nms05',old,standard_nms),('new_nms05',new,standard_nms)]:predictions[name].append({'sample_id':row['sample_id'],'instances':selector(row['candidates'],scores)})
    overlap=[]
    for record in records:
        targets=[x for x in record['instances'] if not is_ignored(x)]
        if len(targets)==2:overlap.append({'sample_id':record['sample_id'],'iou':bbox_iou(targets[0]['bbox_full_xyxy'],targets[1]['bbox_full_xyxy'])})
    report={'scope':'fixed200 development2x2 score/NMS comparison; no independent acceptance claim','head_step':state['step'],'metrics':{k:score_boxes(v,records,score_threshold=.3) for k,v in predictions.items()},'GT_pair_max_iou':max(x['iou'] for x in overlap),'GT_pairs_over05':[x for x in overlap if x['iou']>=.5],'production_changed':False}
    atomic_json(out/'REPORT.json',report);atomic_json(out/'PREDICTIONS.json',predictions)
if __name__=='__main__':run()
