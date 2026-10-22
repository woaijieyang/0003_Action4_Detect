"""Supervision binding by sample_id; evaluation inputs remain label-free."""
import json
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_box_quality import load_box_split

def load_inputs(cache,split):
    rows=load_box_split(R/cache,split)
    for row in rows:
        row["boxes"]=torch.tensor([p["bbox_full_xyxy"] for p in row["candidates"]],dtype=torch.float32)
        row["priors"]=torch.tensor([[p["center_score"],p["old_quality_score"],p["old_product_score"]]
                                    for p in row["candidates"]],dtype=torch.float32)
    return rows

def attach_train_targets(rows):
    refs={r["sample_id"]:r for r in map(json.loads,(R/"data/train/annotations.jsonl").read_text().splitlines())}
    if len(rows)!=1600 or {r["sample_id"] for r in rows}!=set(refs):raise ValueError("Exact train coverage required")
    for row in rows:
        record=refs[row["sample_id"]]
        gt=torch.tensor([g["bbox_full_xyxy"] for g in record["instances"] if not g.get("ignore",False)],dtype=torch.float32).reshape(-1,4)
        boxes=row["boxes"]
        if len(gt):
            inter=(torch.minimum(boxes[:,None,2:],gt[None,:,2:])-torch.maximum(boxes[:,None,:2],gt[None,:,:2])).clamp_min(0).prod(-1)
            union=(boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+(gt[:,2:]-gt[:,:2]).prod(-1)[None]-inter
            best,index=(inter/union.clamp_min(1e-8)).max(-1)
            row["targets"]=gt[index]; row["matched"]=best>=0.3
        else:
            row["targets"]=boxes.clone(); row["matched"]=torch.zeros(len(boxes),dtype=torch.bool)
    return rows
