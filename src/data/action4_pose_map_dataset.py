"""Load immutable predicted-map cache; labels remain a separate structure."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R,contained_path

def load_maps(root,split):
    root=contained_path(R,root,exists=True);manifest=json.loads((root/"MAPS_COMPLETE.json").read_text())
    if not manifest["predicted_maps_only"] or manifest["frozen_surface_step"]!=11300:
        raise ValueError("Expected frozen predicted-surface input")
    info=manifest["splits"][split];inputs=[];targets={};metadata=[]
    for part in info["parts"]:
        p=contained_path(R,part["path"],exists=True)
        if p.stat().st_size!=part["bytes"]:raise ValueError("Map cache changed")
        state=torch.load(p,map_location="cpu",weights_only=True)
        if len(state["predicted_maps"])!=part["count"]:raise ValueError("Map count mismatch")
        inputs.append(state["predicted_maps"])
        for key,value in state["geometry_targets"].items():targets.setdefault(key,[]).append(value)
        metadata.extend(state["metadata"])
    if len(metadata)!=info["count"] or len({(m["sample_id"],m["instance_id"]) for m in metadata})!=len(metadata):
        raise ValueError("ROI identity coverage mismatch")
    return torch.cat(inputs),{k:torch.cat(v) for k,v in targets.items()},metadata
