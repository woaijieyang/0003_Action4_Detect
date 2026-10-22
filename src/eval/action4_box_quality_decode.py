"""Shared candidate suppression for score-head evaluation and future inference."""
import numpy as np
from eval.action4_metrics import bbox_iou

def select_boxes(candidates,scores,threshold=.001):
    scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(candidates),) or not np.isfinite(scores).all():raise ValueError('Invalid score vector')
    selected=[]
    for index in np.argsort(-scores,kind='stable'):
        score=float(scores[index])
        if score<threshold:continue
        row=dict(candidates[index],score=score);duplicate=False
        for kept in selected:
            diagonal=min(np.linalg.norm(np.array(x['bbox_full_xyxy'][2:])-x['bbox_full_xyxy'][:2]) for x in [row,kept])
            distance=np.linalg.norm(np.array(row['center_px'])-kept['center_px'])
            if bbox_iou(row['bbox_full_xyxy'],kept['bbox_full_xyxy'])>=.95 and distance<=.05*diagonal:duplicate=True;break
        if not duplicate:selected.append(row)
        if len(selected)==2:break
    return selected
