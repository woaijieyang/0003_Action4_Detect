"""Online B0 inference with the exact same predicted proposal policy."""
import torch
from data.action4_dataset import prepare_full_images
from model.action4_online_b0 import proposals
from eval.action4_soft_nms import select
@torch.inference_mode()
def predict(model,image,base_cfg,device):
    x,affines=prepare_full_images([image],size=tuple(base_cfg["preprocess"]["full_size"]),device=device)
    dense,feature=model(x);rows,boxes,output=proposals(model,dense,feature,affines[0],base_cfg)
    ps=[dict(p,bbox_full_xyxy=b,score=float(s)) for p,b,s in zip(rows,output[0].cpu().tolist(),output[1].sigmoid().cpu().tolist())]
    return select(ps,"soft_gaussian50")
