"""Read-only actual gradients and real-frame centre/quality score decomposition."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images,build_detection_targets
from model.action4_component_experiments import box_loss,quality_decode
from eval.action4_component_pipeline import load_trained_component
from eval.action4_candidate_pool import decode_candidate_pool
from eval.action4_runner import _map_detection
from eval.action4_metrics import bbox_iou
from model.training.action4_runtime import atomic_json


def run():
    torch.set_num_threads(2);out=R/'outputs/action4_b0_signal_audit_v001';model,cfg=load_trained_component(out/'B0.pt','B0','cuda:0');data=ComponentData(cfg['validation_receipt'],'train','B0');grads=[]
    selected=[]
    for heavy in [True,False]:
        selected.append(next(i for i,r in enumerate(data.records) if i not in selected and any((o['occlusion_inframe']>=.4 if heavy else o['occlusion_inframe']<.1) and not o['ignore'] for o in r['instances'])))
    model.eval()
    backbone_parameter=next(p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad)
    for index in selected:
        item=data[index];x,aff=prepare_full_images([item['image']],size=tuple(cfg['preprocess']['full_size']),device='cuda:0');dense=model(x);targets=build_detection_targets([item['record']],aff,full_size=tuple(cfg['preprocess']['full_size']),device='cuda:0');losses=box_loss(dense,targets,cfg['variant']);row={'sample_id':item['record']['sample_id'],'losses':{}}
        keys=['center_logits','offset','ltrb','quality_logits']
        for name,loss in losses.items():
            values=torch.autograd.grad(loss,[dense[k] for k in keys]+[backbone_parameter],allow_unused=True,retain_graph=True)
            norms={k:None if v is None else float(v.norm()) for k,v in zip(keys+['shared_backbone_parameter'],values)}
            row['losses'][name]={'value':float(loss.detach()),'gradient_norms':norms}
        assert row['losses']['giou']['gradient_norms']['ltrb']>0
        assert row['losses']['quality']['gradient_norms']['quality_logits']>0
        assert row['losses']['quality']['gradient_norms']['ltrb'] is None
        assert row['losses']['quality']['gradient_norms']['offset'] is None
        assert row['losses']['quality']['gradient_norms']['shared_backbone_parameter']>0
        grads.append(row);del dense,losses,x
    atomic_json(out/'GRADIENTS.json',{'scope':'two actual training images, no optimizer step','rows':grads,'training_changed':False})
    manifest=json.loads((R/'data/action4_real/cache_v001/frames_manifest.json').read_text());refs=json.loads((R/'outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json').read_text())['frames'];rows=[]
    with torch.inference_mode():
        for ref in refs:
            frame=manifest['frames'][ref['frame_id']]
            with Image.open(R/frame['image']) as im:rgb=np.array(im.convert('RGB'))
            x,aff=prepare_full_images([torch.from_numpy(rgb).permute(2,0,1)],size=tuple(cfg['preprocess']['full_size']),device='cuda:0');dense=model(x);c=dense['center_logits'].sigmoid().flatten();q=dense['quality_logits'].sigmoid().flatten();pools={}
            for name in ['center_peaks','product_peaks']:
                value=dict(dense)
                if name=='product_peaks':value['center_logits']=torch.logit((dense['center_logits'].sigmoid()*dense['quality_logits'].sigmoid()).clamp(1e-7,1-1e-7))
                pool=decode_candidate_pool(value,image_size=tuple(cfg['preprocess']['full_size']),score_threshold=.000001,topk=20,max_detections=20)[0];mapped=[]
                for item in pool:
                    proposal=_map_detection(item,np.linalg.inv(aff[0]));idx=item['grid_index'];proposal.update(center_score=float(c[idx]),quality_score=float(q[idx]),product_score=float(c[idx]*q[idx]),grid_index=idx)
                    box=(np.array(proposal['bbox_full_xyxy']).reshape(2,2)*[950/rgb.shape[1],720/rgb.shape[0]]).reshape(4)
                    proposal['demo_reference_iou']=[bbox_iou(box,g['bbox_demo_px']) for g in ref['reference_overlay_components']];mapped.append(proposal)
                pools[name]=mapped
            rows.append({'frame_id':ref['frame_id'],'candidate_pools':pools,'final_predictions':quality_decode(dense,aff,cfg,cfg['variant'])[0]})
    atomic_json(out/'REAL_CANDIDATES.json',{'scope':'demo-reference diagnostic, not GT accuracy; no threshold changes','frames':rows})
if __name__=='__main__':
    import traceback
    try:run()
    except BaseException:
        atomic_json(R/'outputs/action4_b0_signal_audit_v001/FAILURE.json',{'traceback':traceback.format_exc()});raise
