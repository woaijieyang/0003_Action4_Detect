"""Cache box-content features and IACS labels from frozen selected B0."""
from pathlib import Path
import json,traceback
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images
from eval.action4_component_pipeline import load_trained_component
from eval.action4_box_quality_pool import candidate_union
from eval.action4_metrics import bbox_iou,is_ignored
from model.action4_box_quality import box_features
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic

def run(config):
    torch.set_num_threads(1);cfg=json.loads((R/config).read_text());out=R/cfg['output'];cache=out/'cache';cache.mkdir(exist_ok=False);model,base_cfg=load_trained_component(R/cfg['base_checkpoint'],'B0','cuda:0');model.eval();model.requires_grad_(False);captured={}
    def capture(module,args,output):captured['feature']=output.detach()
    hook=model.decoder.register_forward_hook(capture);splits={}
    with torch.inference_mode():
        for split in ['train','eval']:
            data=ComponentData(cfg['validation_receipt'],split,'B0');parts=[]
            for index in range(len(data)):
                item=data[index];image,affine=prepare_full_images([item['image']],size=tuple(base_cfg['preprocess']['full_size']),device='cuda:0');dense=model(image);rows,boxes=candidate_union(dense,affine[0],base_cfg);pooled,geometry=box_features(captured['feature'],boxes,tuple(base_cfg['preprocess']['full_size']))
                valid_gt=[g for g in item['record']['instances'] if not is_ignored(g)];ignored=[g for g in item['record']['instances'] if is_ignored(g)];ious=[];valid=[]
                for row in rows:
                    overlap=max((bbox_iou(row['bbox_full_xyxy'],g['bbox_full_xyxy']) for g in valid_gt),default=0.);ignored_overlap=max((bbox_iou(row['bbox_full_xyxy'],g['bbox_full_xyxy']) for g in ignored),default=0.)
                    ious.append(overlap);valid.append(not(ignored_overlap>=.5 and overlap<.5))
                actual=torch.tensor(ious);target=torch.where(actual>=.5,actual,torch.zeros_like(actual));path=cache/f'{split}_{index:05d}.pt'
                save_checkpoint_atomic(path,{'features':pooled.cpu(),'geometry':geometry.cpu(),'target':target,'actual_iou':actual,'supervision_valid':torch.tensor(valid,dtype=torch.bool),'candidates':rows,'sample_id':item['record']['sample_id'],'split':split})
                parts.append({'path':str(path.relative_to(R)),'sample_id':item['record']['sample_id'],'count':len(rows),'positive':int((target>0).sum()),'bytes':path.stat().st_size})
                if index%25==0 or index+1==len(data):atomic_json(out/'CACHE_STATUS.json',{'split':split,'completed_images':index+1,'total':len(data)})
            splits[split]={'images':len(data),'parts':parts}
    hook.remove();atomic_json(out/'CACHE_COMPLETE.json',{'splits':splits,'target_semantics':'IoU with valid full body if IoU>=0.5 else0; ignored matches excluded','candidate_policy':'deduplicated union of20center and20product peaks; identical at inference','new_head_does_not_read_old_center_or_quality_as_input':True,'base_frozen':True,'training_started':False})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg['output']/'CACHE_FAILURE.json',{'traceback':traceback.format_exc()});raise
