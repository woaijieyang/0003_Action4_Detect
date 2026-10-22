"""Isolated analytic solver candidate; original running pipeline unchanged. Candidate composition: independent B0/A0 plus existing rigid residual geometry.

No reference boxes, corner labels, visibility labels or poses enter prediction.
Real-camera K must be identified separately; synthetic K is not real calibration.
"""
import copy
import numpy as np
import torch
from torch import nn
from data import DataAdapter
from data.action4_dataset import virtual_roi
from model.action4_residual_geometry import ResidualGeometry
from model.action4_rigid_projection_analytic import fit_rigid_cuboid
from eval.action4_component_pipeline import load_trained_component,predict_components
from model.action4_components import load_component


class FrozenFeatureView(nn.Module):
    role='A0'
    def __init__(self,base):
        super().__init__();self.base=base;self.features=None
    def forward(self,images):
        self.features=self.base.decoder(self.base.backbone(images))
        return {'corner_logits':self.base.corner_head(self.features)}


class GeometryPipeline:
    def __init__(self,b0_checkpoint,a0_checkpoint,residual_checkpoint,device):
        self.device=device;self.detector,self.b_cfg=load_trained_component(b0_checkpoint,'B0',device)
        base,metadata=load_component(a0_checkpoint,role='A0',device=device)
        state=torch.load(residual_checkpoint,map_location='cpu',weights_only=True)
        if state.get('schema')!='action4.residual_predicted_roi.v1' or state['source_checkpoint']!=metadata['provenance']['source']:raise ValueError('Residual feature parent mismatch')
        self.a_cfg={'preprocess':metadata['preprocess'],'decode':{'corner_topk':20},'variant':'baseline'}
        if metadata['preprocess']['roi_size']!=448 or metadata['preprocess']['roi_expansion']!=2.25:raise ValueError('Residual requires original448/2.25 ROI contract')
        self.corners=FrozenFeatureView(base).to(device).eval();self.residual=ResidualGeometry().to(device).eval();self.residual.load_state_dict(state['model'],strict=True)
        self.fit_missing=0;self.nonfinite_fallback=0

    @torch.inference_mode()
    def predict(self,rgb,K,canonical):
        self.corners.features=None
        raw=predict_components(self.detector,self.corners,[rgb],self.b_cfg,self.a_cfg,self.device)[0]
        variants={name:copy.deepcopy(raw) for name in ('raw','rigid','visibility_residual')}
        if not raw:return variants
        features=torch.nn.functional.adaptive_avg_pool2d(self.corners.features,(4,4)).flatten(1)
        for index,prediction in enumerate(raw):
            affine=DataAdapter.crop_affine(virtual_roi(prediction['bbox_full_xyxy'],2.25),(448,448));obs=DataAdapter.transform_points(prediction['corners_px'],affine)/448;kr=affine@K
            fit=fit_rigid_cuboid(obs*448,canonical,kr,prediction['corner_peak_scores'])
            if fit.get('rotation') is None:self.fit_missing+=1;continue
            tensor=lambda x:torch.tensor(np.asarray(x),device=self.device,dtype=torch.float32)[None]
            args={'features':features[index:index+1],'observed_uv':tensor(obs),'peak_scores':tensor(prediction['corner_peak_scores']),'K_roi':tensor(kr),'canonical_m':tensor(canonical),'initial_rotation':tensor(fit['rotation']),'initial_translation':tensor(fit['translation_m'])}
            output=self.residual(**args)
            weights=np.asarray(prediction['corner_peak_scores'])*(.25+.75*output['visibility_logits'][0].sigmoid().cpu().numpy())
            weighted=fit_rigid_cuboid(obs*448,canonical,kr,weights)
            if weighted.get('rotation') is not None:
                args.update(initial_rotation=tensor(weighted['rotation']),initial_translation=tensor(weighted['translation_m']));output=self.residual(**args)
            variants['rigid'][index]['corners_px']=DataAdapter.transform_points(fit['fitted_corners_px'],np.linalg.inv(affine)).tolist()
            if not bool(torch.isfinite(output['corners_uv']).all()) or not bool((output['camera_corners'][...,2]>0).all()):self.nonfinite_fallback+=1;continue
            variants['visibility_residual'][index]['corners_px']=DataAdapter.transform_points(output['corners_uv'][0].cpu().numpy()*448,np.linalg.inv(affine)).tolist()
            variants['visibility_residual'][index]['predicted_visibility']=output['visibility_logits'][0].sigmoid().cpu().tolist()
        return variants
