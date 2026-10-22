"""Optional frozen B0 scoring policies; no parameter or box geometry changes."""
import torch
from data.action4_dataset import prepare_full_images
from model.action4_online_b0 import proposals
from eval.action4_soft_nms import select

@torch.inference_mode()
def predict(model,image,cfg,device,policy):
    if policy not in ("center","product","learned"):
        raise ValueError("Explicit B0 scoring policy required")
    x,aff=prepare_full_images([image],size=tuple(cfg["preprocess"]["full_size"]),device=device)
    dense,feature=model(x)
    rows,_,output=proposals(model,dense,feature,aff[0],cfg)
    refined=output[0].cpu().tolist();learned=output[1].sigmoid().cpu().tolist()
    candidates=[]
    for index,row in enumerate(rows):
        score=row["center_score"] if policy=="center" else row["old_product_score"] if policy=="product" else learned[index]
        candidates.append(dict(row,bbox_full_xyxy=refined[index],score=float(score)))
    return select(candidates,"soft_gaussian50")
