"""Immutable in-memory surface feature split; supervision stored separately."""
import json
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path

def load_split(cache_output,split):
    if split not in ('train','eval'):raise ValueError('Only train/dev feature splits admitted')
    root=contained_path(R,cache_output,exists=True);manifest=json.loads((root/'CACHE_COMPLETE.json').read_text());info=manifest['splits'][split]
    features=[];targets={};metadata=[]
    for part in info['parts']:
        path=contained_path(R,part['path'],exists=True)
        if path.stat().st_size!=part['bytes']:raise ValueError('Cache file changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if len(state['features'])!=part['count']:raise ValueError('Cache count mismatch')
        features.append(state['features'])
        for key,value in state['targets'].items():targets.setdefault(key,[]).append(value)
        metadata.extend(state['metadata'])
    if len(metadata)!=info['count'] or len({(m['sample_id'],m['instance_id']) for m in metadata})!=len(metadata):raise ValueError('Duplicate/missing cache objects')
    return torch.cat(features),{k:torch.cat(v) for k,v in targets.items()},metadata
