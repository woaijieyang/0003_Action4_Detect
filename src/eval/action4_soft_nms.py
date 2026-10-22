"""Reuse of the previously verified postprocessing selector, with no import side effects.
Extracted unchanged from outputs/action4_b0_postprocess_v001/run.py.
Soft-NMS default linear Nt=.5 / Gaussian sigma=.5; no threshold search here.
"""
import copy,math
import numpy as np
from eval.action4_metrics import bbox_iou

def select(pool,method,threshold=.001):
 pending=copy.deepcopy(pool);out=[]
 while pending and len(out)<2:
  pending.sort(key=lambda x:-x['score']);p=pending.pop(0)
  if p['score']<threshold:break
  out.append(p);remaining=[]
  for q in pending:
   iou=bbox_iou(p['bbox_full_xyxy'],q['bbox_full_xyxy'])
   if method.startswith('legacy'):
    limit=.95 if method=='legacy95' else .85
    diag=min(np.linalg.norm(np.array(x['bbox_full_xyxy'][2:])-x['bbox_full_xyxy'][:2]) for x in (p,q))
    if iou>=limit and np.linalg.norm(np.array(p['center_px'])-q['center_px'])<=.05*diag:continue
   elif method=='hard50':
    if iou>=.5:continue
   elif method=='soft_linear50':
    if iou>=.5:q['score']*=1-iou
   elif method=='soft_gaussian50':q['score']*=math.exp(-iou*iou/.5)
   else:raise ValueError(method)
   remaining.append(q)
  pending=remaining
 return out

