"""CPU-only frozen surface predictions; targets stored separately, no RGB rerender."""
import json,time,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.action4_surface_head import SurfaceHead
from model.action4_pose_maps import surface_to_maps
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic,canonical_hash

def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=R/"outputs/action4_a0_frozen_surface_pose_v001";cache=out/"maps";cache.mkdir(exist_ok=False)
    source=R/"outputs/action4_gdr_surface_candidate_v001"
    manifest=json.loads((source/"CACHE_COMPLETE.json").read_text())
    state=torch.load(source/"train_run_v001/best.pt",map_location="cpu",weights_only=True)
    cfg=json.loads((R/"configs/model/action4_gdr_surface_train_v001.json").read_text())
    if state["step"]!=11300 or state["config_id"]!=canonical_hash(cfg):raise ValueError("Frozen surface identity mismatch")
    head=SurfaceHead().eval();head.load_state_dict(state["head"],strict=True);head.requires_grad_(False)
    records={split:{x["sample_id"]:x for x in map(json.loads,(R/f"data/{split}/annotations.jsonl").read_text().splitlines())} for split in ["train","eval"]}
    outputs={};begin=time.monotonic()
    for split in ["train","eval"]:
        parts=[];count=0;identities=set()
        for index,part in enumerate(manifest["splits"][split]["parts"]):
            path=contained_path(R,part["path"],exists=True)
            if path.stat().st_size!=part["bytes"]:raise ValueError("Source cache changed")
            batch=torch.load(path,map_location="cpu",weights_only=True);meta=batch["metadata"]
            K=torch.tensor([m["K_hm"] for m in meta],dtype=torch.float32)
            with torch.inference_mode():maps=surface_to_maps(head(batch["features"]),K)
            if not torch.isfinite(maps).all():raise ValueError("Nonfinite predicted maps")
            geometry=[]
            for m in meta:
                key=(m["sample_id"],m["instance_id"])
                if key in identities:raise ValueError("Duplicate ROI identity")
                identities.add(key)
                record=records[split][m["sample_id"]]
                gt=[g for g in record["instances"] if g["instance_id"]==m["instance_id"]]
                if len(gt)!=1:raise ValueError("Ambiguous ROI identity")
                T=gt[0]["canonical8_geometry"]["object_to_capture_cv_m"]
                geometry.append({"K":m["K_hm"],"canonical":m["canonical_m"],"rotation":[r[:3] for r in T[:3]],"translation":[r[3] for r in T[:3]],
                    "corners":m["corners_hm"],"diagonal":m["diagonal"],"visible":m["visible"]})
            target={k:torch.tensor([g[k] for g in geometry],dtype=torch.bool if k=="visible" else torch.float32) for k in geometry[0]}
            target_path=cache/f"{split}_{index:04d}.pt"
            save_checkpoint_atomic(target_path,{"predicted_maps":maps,"geometry_targets":target,"metadata":meta,
                "input_contract":"predicted XYZ/regions/masks plus known camera rays; no GT xyz/mask/pose in input"})
            count+=len(meta);parts.append({"path":str(target_path.relative_to(R)),"bytes":target_path.stat().st_size,"count":len(meta)})
            atomic_json(out/"CACHE_STATUS.json",{"split":split,"completed":count,"total":manifest["splits"][split]["count"],"cpu_threads":1,"seconds":time.monotonic()-begin})
        if count!=manifest["splits"][split]["count"]:raise ValueError("Missing source ROI")
        outputs[split]={"count":count,"parts":parts}
    atomic_json(out/"MAPS_COMPLETE.json",{"splits":outputs,"frozen_surface_step":11300,"backbone_step":6300,"predicted_maps_only":True,
        "cpu_threads":1,"seconds":time.monotonic()-begin,"data_files_changed":False,"training_started":False})
if __name__=="__main__":run()
