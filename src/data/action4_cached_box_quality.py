"""Cached box features with separate supervision and frame-local candidates."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R,contained_path

def load_box_split(output,split):
    root=contained_path(R,output,exists=True);manifest=json.loads((root/'CACHE_COMPLETE.json').read_text());parts=manifest['splits'][split]['parts'];images=[]
    for part in parts:
        path=contained_path(R,part['path'],exists=True)
        if path.stat().st_size!=part['bytes']:raise ValueError('Cache file identity changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['sample_id']!=part['sample_id'] or state['split']!=split or len(state['features'])!=part['count']:raise ValueError('Cache identity mismatch')
        if len({p['grid_index'] for p in state['candidates']})!=len(state['candidates']):raise ValueError('Duplicate candidates')
        images.append(state)
    if len({x['sample_id'] for x in images})!=len(images):raise ValueError('Duplicate images')
    return images
