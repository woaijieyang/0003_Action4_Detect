"""Compare quality-before-peak versus centre-peak-then-quality on fixed weights."""
import json,copy,torch,numpy as np
from pathlib import Path
from PIL import Image
from model.training.action4_component_train import bounded
from model.training.action4_runtime import atomic_json
from eval.action4_component_pipeline import load_trained_component
from model.action4_component_experiments import quality_decode
from eval.action4_candidate_pool import decode_candidate_pool
from eval.action4_runner import _map_detection
from eval.action4_metrics import bbox_iou
from eval.action4_bbox_metrics import score_boxes
from data.action4_dataset import prepare_full_images
from data.action4_component_data import ComponentData
from data.action4_paths import PROJECT_ROOT as R


def rerank(dense,affine,cfg):
    pool=decode_candidate_pool(dense,image_size=tuple(cfg['preprocess']['full_size']),score_threshold=.001,topk=20,max_detections=20)[0]
    quality=dense['quality_logits'].sigmoid().flatten().detach().cpu()
    mapped=[]
    for item in pool:
        row=_map_detection(item,np.linalg.inv(affine));row['score']*=float(quality[item['grid_index']]);mapped.append(row)
    selected=[]
    for row in sorted(mapped,key=lambda r:-r['score']):
        if row['score']<.001:continue
        duplicate=False
        for other in selected:
            diagonal=min(np.linalg.norm(np.array(x['bbox_full_xyxy'][2:])-x['bbox_full_xyxy'][:2]) for x in (row,other))
            if bbox_iou(row['bbox_full_xyxy'],other['bbox_full_xyxy'])>=.95 and np.linalg.norm(np.array(row['center_px'])-other['center_px'])<=.05*diagonal:duplicate=True;break
        if not duplicate:selected.append(row)
        if len(selected)==2:break
    return selected


def audit(output):
    torch.set_num_threads(2);out=bounded(output);model,cfg=load_trained_component(out/'B0.pt','B0','cpu');cfg=copy.deepcopy(cfg);cfg['decode']['score_threshold']=.001
    data=ComponentData(cfg['validation_receipt'],'eval','B0');old=[];new=[];changed=[]
    with torch.inference_mode():
        for i in range(len(data)):
            item=data[i];x,aff=prepare_full_images([item['image']],size=tuple(cfg['preprocess']['full_size']),device='cpu');dense=model(x)
            a=quality_decode(dense,aff,cfg,'giou_quality')[0];b=rerank(dense,aff[0],cfg);sid=item['record']['sample_id'];old.append({'sample_id':sid,'instances':a});new.append({'sample_id':sid,'instances':b})
            if len(a)!=len(b) or any(not np.allclose(p['bbox_full_xyxy'],q['bbox_full_xyxy']) for p,q in zip(a,b)):changed.append(sid)
            atomic_json(out/'STATUS.json',{'completed':i+1,'total':len(data)})
    report={'scope':'fixed model paired development decoding audit; not a threshold change or deployment', 'changed_box_images':changed,'product_before_peak':score_boxes(old,data.records),'centre_peak_then_quality':score_boxes(new,data.records)}
    atomic_json(out/'REPORT.json',report);atomic_json(out/'PREDICTIONS.json',{'old':old,'new':new})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();audit(a.output)
