"""Independent frozen B0 and GDR-derived A0 inference from RGB only."""
import json,hashlib
import numpy as np
import torch
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import prepare_full_images,prepare_rois
from eval.action4_component_pipeline import load_trained_component
from model.action4_component_experiments import quality_decode
from model.action4_surface_head import SurfaceHead
from model.action4_surface_pnp import decode_surface_pose
from model.action4_decode import decode_corners
from model.training.action4_runtime import canonical_hash

class SurfacePipeline:
    def __init__(self,b0_checkpoint,a0_checkpoint,head_checkpoint,device='cuda:0'):
        self.device=device;self.detector,self.b_cfg=load_trained_component(b0_checkpoint,'B0',device);self.base,self.a_cfg=load_trained_component(a0_checkpoint,'A0',device)
        cfg=json.loads((R/'configs/model/action4_gdr_surface_train_v001.json').read_text());cache_cfg=json.loads((R/'configs/model/action4_gdr_surface_v001.json').read_text())
        state=torch.load(contained_path(R,head_checkpoint,exists=True),map_location='cpu',weights_only=True)
        if state['schema']!='action4.surface_eval.v1' or state['config_id']!=canonical_hash(cfg):raise ValueError('Surface checkpoint identity mismatch')
        digest=hashlib.sha256(contained_path(R,a0_checkpoint,exists=True).read_bytes()).hexdigest()
        if digest!=cache_cfg['base_sha256']:raise ValueError('Frozen surface backbone differs from training parent')
        self.surface=SurfaceHead().to(device);self.surface.load_state_dict(state['head'],strict=True);self.surface.eval();self.head_step=state['step']
    @torch.inference_mode()
    def predict(self,rgb,K,canonical,seed=23):
        full,full_affine=prepare_full_images([rgb],size=tuple(self.b_cfg['preprocess']['full_size']),device=self.device);detections=quality_decode(self.detector(full),full_affine,self.b_cfg,self.b_cfg['variant'])[0]
        if not detections:return {'detections':[],'heatmap':[],'surface':[]}
        rois,affines=prepare_rois([rgb],[p['bbox_full_xyxy'] for p in detections],[0]*len(detections),roi_size=448,expansion=2.25,device=self.device)
        features=self.base.backbone(rois);raw=decode_corners(self.base.corner_head(self.base.decoder(features))).cpu().numpy();output=self.surface(features);xyz=output['xyz'].cpu().numpy();mask=output['masks'][:,0].sigmoid().cpu().numpy();down,_=DataAdapter.letterbox_affine((448,448),(112,112));old=[];new=[]
        for i,pred in enumerate(detections):
            affine=down@affines[i];inverse=np.linalg.inv(affine);pose=decode_surface_pose(xyz[i],mask[i],affine@K,canonical,seed=seed+i)
            common=dict(pred,instance_id=i,object_id=1,corner_ids=list(range(8)))
            old.append(dict(common,corners_px=DataAdapter.transform_points(raw[i],inverse).tolist()))
            corners=DataAdapter.transform_points(pose['corners_hm'],inverse).tolist() if pose['valid'] else None
            new.append(dict(common,corners_px=corners,pose_valid=pose['valid'],pose_diagnostic=pose))
        return {'detections':detections,'heatmap':old,'surface':new}
