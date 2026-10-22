"""Frozen score heads, cached candidates and established Soft-NMS defaults."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_box_quality import load_box_split
from model.action4_box_quality import BoxAwareQuality
from eval.action4_soft_nms import select
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json

def run():
    torch.set_num_threads(1);out=R/'outputs/action4_box_quality_soft_nms_v001';path=out/'POOLS.json'
    if path.exists():pools=json.loads(path.read_text())
    else:
        data=load_box_split(R/'outputs/action4_box_quality_candidate_v001','eval');state=torch.load(R/'outputs/action4_box_quality_nms_audit_v001/head.pt',map_location='cpu',weights_only=True);model=BoxAwareQuality().to('cuda:0');model.load_state_dict(state['head'],strict=True);model.eval();pools=[]
        with torch.inference_mode():
            for row in data:pools.append({'sample_id':row['sample_id'],'candidates':row['candidates'],'new_scores':model(row['features'].to('cuda:0'),row['geometry'].to('cuda:0')).sigmoid().cpu().tolist(),'old_scores':[p['old_product_score'] for p in row['candidates']]})
        atomic_json(path,pools)
    records=[json.loads(l) for l in (R/'data/eval/annotations.jsonl').read_text().splitlines()];reports={};predictions={}
    for score in ['old','new']:
        for method in ['soft_linear50','soft_gaussian50']:
            name=score+'_'+method;rows=[]
            for row in pools:
                candidates=[dict(p,score=s) for p,s in zip(row['candidates'],row[score+'_scores'])];rows.append({'sample_id':row['sample_id'],'instances':select(candidates,method)})
            reports[name]=score_boxes(rows,records,score_threshold=.3);predictions[name]=rows
    atomic_json(out/'REPORT.json',{'scope':'fixed200 development postprocess comparison, not independent acceptance','head_step':800,'metrics':reports,'production_changed':False});atomic_json(out/'PREDICTIONS.json',predictions)
if __name__=='__main__':run()
