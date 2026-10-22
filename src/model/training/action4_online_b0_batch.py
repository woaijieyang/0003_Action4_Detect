"""Train/infer share predicted proposal construction; labels enter loss only."""
import numpy as np
import torch
from torch.nn import functional as F
from data.action4_dataset import prepare_full_images,build_detection_targets,is_ignored
from model.action4_component_experiments import box_loss
from model.action4_box_refinement import loss as refinement_loss
from model.action4_online_b0 import proposals
from model.action4_box_quality import varifocal_loss


def match_targets(boxes,record):
    gt=boxes.new_tensor([g["bbox_full_xyxy"] for g in record["instances"] if not is_ignored(g)]).reshape(-1,4)
    ignored=boxes.new_tensor([g["bbox_full_xyxy"] for g in record["instances"] if is_ignored(g)]).reshape(-1,4)
    def overlaps(gs):
        inter=(torch.minimum(boxes[:,None,2:],gs[None,:,2:])-torch.maximum(boxes[:,None,:2],gs[None,:,:2])).clamp_min(0).prod(-1)
        union=(boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+(gs[:,2:]-gs[:,:2]).prod(-1)[None]-inter
        return inter/union.clamp_min(1e-8)
    if len(gt):
        best,index=overlaps(gt).max(-1);targets=gt[index];matched=best>=.3
    else:best=boxes.new_zeros(len(boxes));targets=boxes.clone();matched=torch.zeros(len(boxes),device=boxes.device,dtype=torch.bool)
    valid=torch.ones_like(matched)
    if len(ignored):valid=~((overlaps(ignored).max(-1).values>=.3)&(~matched))
    return targets,matched,valid

def batch_losses(model,items,base_cfg,device,augmentation_seeds=None):
    x,affines=prepare_full_images([item["image"] for item in items],size=tuple(base_cfg["preprocess"]["full_size"]),device=device)
    if augmentation_seeds is not None:
        from data.action4_photometric import augment_prepared
        sizes=[(int(item["image"].shape[2]),int(item["image"].shape[1])) for item in items]
        x,_=augment_prepared(x,sizes,augmentation_seeds,base_cfg["photometric"])
    dense_batch,features=model(x);aggregate={}
    for index,item in enumerate(items):
        dense={k:v[index:index+1] for k,v in dense_batch.items()};feature=features[index:index+1];affine=affines[index]
        targets=build_detection_targets([item["record"]],[affine],full_size=tuple(base_cfg["preprocess"]["full_size"]),stride=4,device=device)
        losses={"dense_"+k:v for k,v in box_loss(dense,targets,"giou_quality").items()}
        rows,boxes,output=proposals(model.module if hasattr(model,"module") else model,dense,feature,affine,base_cfg)
        if len(rows):
            target,matched,valid=match_targets(boxes,item["record"])
            if valid.any():
                refine=refinement_loss(output,boxes,target,matched,valid)
                from model.action4_box_refinement import paired_iou_giou
                iou,_=paired_iou_giou(output[0],target)
                quality_target=torch.where(matched&valid,iou.detach().clamp_min(1e-6),torch.zeros_like(iou))
                refine["quality"]=varifocal_loss(output[1][valid],quality_target[valid])
                losses.update({"refined_"+k:v for k,v in refine.items()})
        for key in ("quality","giou","regression"):losses.setdefault("refined_"+key,feature.sum()*0)
        for key,value in losses.items():aggregate[key]=aggregate.get(key,0)+value/len(items)
    return aggregate

def image_loss(model,item,base_cfg,device):
    return batch_losses(model,[item],base_cfg,device)
