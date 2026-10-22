"""Cache frozen-best A0 features once; labels remain separate from model inputs."""
from pathlib import Path
import json,os,traceback
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_surface_data import prepare_surface_batch,regions
from data import DataAdapter
from data.action4_dataset import virtual_roi
from model.action4_component_experiments import Candidate
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic

def run(config):
    torch.set_num_threads(2);cfg=json.loads((R/config).read_text());out=R/cfg['output'];cache=out/'cache';cache.mkdir(exist_ok=False)
    state=torch.load(R/cfg['base_checkpoint'],map_location='cpu',weights_only=True);assert state['role']=='A0' and state['variant']=='baseline';model=Candidate('A0',state['config']['network'],'baseline').to('cuda:0');model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False);preprocess=state['config']['preprocess'];all_splits={}
    regions()
    for split in ['train','eval']:
        data=ComponentData(cfg['validation_receipt'],split,'A0');parts=[]
        for start in range(0,len(data),16):
            features=[];labels={};metadata=[]
            for mini in range(start,min(start+16,len(data)),2):
                items=[data[i] for i in range(mini,min(mini+2,len(data)))];x,target=prepare_surface_batch(items,preprocess,split,'cuda:0')
                with torch.inference_mode():feature=model.backbone(x)
                if not torch.isfinite(feature).all():raise ValueError('Nonfinite cached features')
                features.append(feature.cpu())
                for k,v in target.items():labels.setdefault(k,[]).append(v.cpu())
                for item in items:
                    record=item['record'];obj=record['instances'][item['instance']];aff=DataAdapter.crop_affine(virtual_roi(obj['bbox_full_xyxy'],2.25),(112,112));box=np.array(obj['bbox_full_xyxy']);diag=np.linalg.norm(box[2:]-box[:2])*112/(max(box[2:]-box[:2])*2.25)
                    metadata.append({'sample_id':record['sample_id'],'instance_id':obj['instance_id'],'K_hm':(aff@np.array(record['camera']['K'])).tolist(),'canonical_m':(np.array(record['corners_3d_mm'])/1000).tolist(),'corners_hm':DataAdapter.transform_points(obj['corners_px'],aff).tolist(),'visible':obj['corner_visible'],'heavy':obj['occlusion_inframe']>=.4,'diagonal':float(diag)})
            path=cache/f'{split}_{start:05d}.pt';save_checkpoint_atomic(path,{'features':torch.cat(features),'targets':{k:torch.cat(v) for k,v in labels.items()},'metadata':metadata,'input_scope':'frozen backbone features only; targets and metadata never input to head'})
            parts.append({'path':str(path.relative_to(R)),'count':len(metadata),'bytes':path.stat().st_size});atomic_json(out/'CACHE_STATUS.json',{'split':split,'completed':start+len(metadata),'total':len(data)})
        all_splits[split]={'count':len(data),'parts':parts}
    atomic_json(out/'CACHE_COMPLETE.json',{'splits':all_splits,'base_step':state['step'],'float32_features':True,'crop_policy':'fixed GT ROI no jitter; diagnostic training, not full detector pipeline','source_RGB_changed':False})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg['output']/'CACHE_FAILURE.json',{'traceback':traceback.format_exc()});raise
