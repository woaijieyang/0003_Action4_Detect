"""Shared box-only maximum-cardinality assignment; no corner-error input."""
import numpy as np

def iou(a,b):
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union>0 else 0.

def match_geometry(pred_boxes,gt_boxes,pred_ids,gt_ids):
    """Max cardinality, total IoU, then stable IDs; never corner-error matching."""
    if len(gt_boxes)>16:raise ValueError('Unsupported crowded-scene protocol')
    order=sorted(range(len(pred_ids)),key=lambda i:str(pred_ids[i]))
    def tie(pairs):return tuple((str(pred_ids[p]),str(gt_ids[g])) for p,g in pairs)
    states={0:(0.,())}
    for p in order:
        nxt=dict(states)
        for mask,(score,pairs) in states.items():
            for g,b in enumerate(gt_boxes):
                v=iou(pred_boxes[p],b)
                if mask&(1<<g) or v<.5:continue
                key=mask|(1<<g);candidate=(score+v,pairs+((p,g),));old=nxt.get(key)
                if old is None or candidate[0]>old[0] or (candidate[0]==old[0] and tie(candidate[1])<tie(old[1])):
                    nxt[key]=candidate
        states=nxt
    winner=min(states.values(),key=lambda x:(-len(x[1]),-x[0],tie(x[1])))
    return list(winner[1])
