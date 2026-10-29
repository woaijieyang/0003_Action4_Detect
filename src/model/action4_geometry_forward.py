"""A fixed per-instance geometry-head schedule independent of microbatch partition.

The frozen RGB backbone may batch. No weights or network architecture change;
surface/pose arithmetic uses the same instance batch shape in train and eval.
"""
import torch

def instancewise_geometry(head,features,K,canonical):
    if len(features)==0:raise ValueError('Geometry needs a nonempty ROI batch')
    outputs=[head(features[i:i+1],K[i:i+1],canonical[i:i+1]) for i in range(len(features))]
    def combine(values):
        if torch.is_tensor(values[0]):return torch.cat(values,dim=0)
        if isinstance(values[0],dict):return {k:combine([v[k] for v in values]) for k in values[0]}
        raise TypeError('Undeclared geometry output type')
    return combine(outputs)
