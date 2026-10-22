"""Fixed B0 full200 score ablation; no model, threshold or training updates."""
import json,copy
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images
from eval.action4_component_pipeline import load_trained_component
from model.action4_component_experiments import quality_decode
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json


def run():
    torch.set_num_threads(2);out=R/'outputs/action4_quality_fusion_comparison_v001';model,cfg=load_trained_component(out/'B0.pt','B0','cuda:0');cfg=copy.deepcopy(cfg);threshold=cfg['decode']['score_threshold'];cfg['decode']['score_threshold']=.001;data=ComponentData(cfg['validation_receipt'],'eval','B0');predictions={k:[] for k in ['center','quality','product']}
    with torch.inference_mode():
        for i in range(len(data)):
            item=data[i];x,aff=prepare_full_images([item['image']],size=tuple(cfg['preprocess']['full_size']),device='cuda:0');dense=model(x)
            for name in predictions:
                output=dict(dense)
                if name=='quality':output['center_logits']=dense['quality_logits']
                rows=quality_decode(output,aff,cfg,'giou_quality' if name=='product' else 'baseline')[0]
                predictions[name].append({'sample_id':item['record']['sample_id'],'instances':rows})
            atomic_json(out/'STATUS.json',{'completed':i+1,'total':len(data)})
    reports={k:score_boxes(v,data.records,score_threshold=threshold) for k,v in predictions.items()}
    atomic_json(out/'REPORT.json',{'scope':'fixed200 development scoring ablation; quality-only is diagnostic not approved inference','threshold':threshold,'metrics':reports,'production_switched':False});atomic_json(out/'PREDICTIONS.json',predictions)
if __name__=='__main__':
    import traceback
    try:run()
    except BaseException:
        atomic_json(R/'outputs/action4_quality_fusion_comparison_v001/FAILURE.json',{'traceback':traceback.format_exc()});raise
