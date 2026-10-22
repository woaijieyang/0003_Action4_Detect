"""Score fixed synthetic interventions; never treat them as held-out acceptance."""
import json,torch,numpy as np
from PIL import Image
from eval.action4_component_pipeline import load_trained_component
from model.action4_component_experiments import quality_decode
from data.action4_dataset import prepare_full_images
from model.training.action4_component_train import bounded
from model.training.action4_runtime import atomic_json
from eval.action4_metrics import bbox_iou,maximum_matching


def evaluate(output):
    out=bounded(output);torch.set_num_threads(2);model,cfg=load_trained_component(out/'B0_frozen.pt','B0','cuda:0');metadata=json.loads((out/'RENDER_COMPLETE.json').read_text());rows=[]
    with torch.inference_mode():
        for case in metadata['cases']:
            with Image.open(bounded(case['image'],True)) as im:rgb=np.asarray(im.convert('RGB')).copy()
            full,aff=prepare_full_images([torch.from_numpy(rgb).permute(2,0,1)],size=tuple(cfg['preprocess']['full_size']),device='cuda:0')
            predictions=quality_decode(model(full),aff,cfg,cfg['variant'])[0]
            matches=maximum_matching([[j for j,g in enumerate(case['reference_boxes']) if bbox_iou(p['bbox_full_xyxy'],g)>=.5] for p in predictions],len(case['reference_boxes']))
            rows.append({**case,'predictions':predictions,'tp':len(matches),'fp':len(predictions)-len(matches),'fn':len(case['reference_boxes'])-len(matches)})
    atomic_json(out/'EVALUATION.json',{'status':'SCORED_VISUAL_REVIEW_REQUIRED','scope':'synthetic training-source counterfactual diagnostic, not independent acceptance','score_threshold':cfg['decode']['score_threshold'],'rows':rows,'model_changed':False,'training_data_modified':False})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.output)
