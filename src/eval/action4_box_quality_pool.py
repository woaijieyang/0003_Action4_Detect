"""One RGB-derived candidate union for both score-head training and inference."""
import numpy as np
import torch
from eval.action4_candidate_pool import decode_candidate_pool
from eval.action4_runner import _map_detection

def candidate_union(dense,affine,config):
    product=dict(dense);product['center_logits']=torch.logit((dense['center_logits'].sigmoid()*dense['quality_logits'].sigmoid()).clamp(1e-7,1-1e-7))
    unique={}
    for outputs in [dense,product]:
        pool=decode_candidate_pool(outputs,image_size=tuple(config['preprocess']['full_size']),score_threshold=0.,topk=20,max_detections=20)[0]
        for item in pool:unique.setdefault(item['grid_index'],item)
    rows=[];boxes=[];center=dense['center_logits'].sigmoid().flatten();quality=dense['quality_logits'].sigmoid().flatten()
    for index,item in unique.items():
        row=_map_detection(item,np.linalg.inv(affine));row.update(grid_index=index,center_score=float(center[index]),old_quality_score=float(quality[index]),old_product_score=float(center[index]*quality[index]));rows.append(row);boxes.append(item['bbox_full_xyxy'])
    tensor=torch.stack(boxes) if boxes else dense['ltrb'].new_empty((0,4))
    return rows,tensor
