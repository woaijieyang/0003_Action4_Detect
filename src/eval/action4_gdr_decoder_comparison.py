"""Paired learned-pose vs predicted-correspondence PnP on one frozen checkpoint."""
import json,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_cached_surface_dataset import load_split
from model.action4_gdr_pose import GDRPose
from model.action4_surface_pnp import decode_surface_pose
from model.action4_pose_refinement import refine_network_pose
from model.training.action4_runtime import canonical_hash,atomic_json

def run(config,checkpoint,output,surface_checkpoint=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg=json.loads((R/config).read_text())
    out=R/output;out.mkdir(exist_ok=False)
    state=torch.load(contained_path(R,checkpoint,exists=True),map_location="cpu",weights_only=True)
    if state["config_id"]!=canonical_hash(cfg):raise ValueError("Checkpoint config mismatch")
    model=GDRPose().cuda();model.load_state_dict(state["head"],strict=True);model.eval()
    composition=None
    if surface_checkpoint is not None:
        surf=torch.load(contained_path(R,surface_checkpoint,exists=True),map_location="cpu",weights_only=True)
        surf_cfg=json.loads((R/"configs/model/action4_gdr_surface_train_v001.json").read_text())
        if surf["config_id"]!=canonical_hash(surf_cfg):raise ValueError("Surface parent configuration mismatch")
        model.surface.load_state_dict(surf["head"],strict=True)
        composition={"pose_checkpoint":checkpoint,"pose_step":state["step"],"surface_checkpoint":surface_checkpoint,
                     "surface_step":surf["step"],"surface_config_id":surf["config_id"],"scope":"explicit frozen module composition, not a single training-step checkpoint"}
    features,_,meta=load_split(R/cfg["cache_output"],"eval");rows=[]
    with torch.inference_mode():
        for start in range(0,len(features),16):
            ms=meta[start:start+16]
            K=torch.tensor([m["K_hm"] for m in ms],device="cuda:0",dtype=torch.float32)
            cad=torch.tensor([m["canonical_m"] for m in ms],device="cuda:0",dtype=torch.float32)
            prediction=model(features[start:start+16].cuda(),K,cad)
            corners=prediction["corners"].cpu().numpy();camera=prediction["camera_corners"].cpu().numpy();rot=prediction["rotation"].cpu().numpy();trans=prediction["translation"].cpu().numpy()
            xyz=prediction["surface"]["xyz"].cpu().numpy();mask=prediction["surface"]["masks"][:,0].sigmoid().cpu().numpy()
            for i,m in enumerate(ms):
                pnp=decode_surface_pose(xyz[i],mask[i],m["K_hm"],m["canonical_m"],seed=20261022+start+i)
                valid=bool(np.isfinite(corners[i]).all() and np.isfinite(camera[i]).all() and (camera[i,:,2]>0).all())
                iterative=refine_network_pose(xyz[i],mask[i],m["K_hm"],m["canonical_m"],rot[i],trans[i])
                candidates={"network":corners[i] if valid else None,"pnp":pnp["corners_hm"] if pnp["valid"] else None,"network_iterative":iterative["corners_hm"] if iterative["valid"] else None}
                row={"sample_id":m["sample_id"],"instance_id":m["instance_id"],"heavy":m["heavy"],"visible":m["visible"],"methods":{}}
                for name,uv in candidates.items():
                    errors=(np.linalg.norm(np.asarray(uv)-m["corners_hm"],axis=1)/m["diagonal"]).tolist() if uv is not None else None
                    row["methods"][name]={"valid":uv is not None,"all8":errors is not None and max(errors)<=.05,"errors":errors}
                rows.append(row)
    summary={}
    for name in ("network","pnp","network_iterative"):
        hidden=[e for row in rows if row["methods"][name]["errors"] is not None for e,v in zip(row["methods"][name]["errors"],row["visible"]) if not v]
        summary[name]={"ROIs":len(rows),"all8":sum(r["methods"][name]["all8"] for r in rows),"heavy_all8":sum(r["heavy"] and r["methods"][name]["all8"] for r in rows),"valid":sum(r["methods"][name]["valid"] for r in rows),"hidden_mean_valid_only":float(np.mean(hidden))}
    changes={"pnp_recovers_network_failure":sum(not r["methods"]["network"]["all8"] and r["methods"]["pnp"]["all8"] for r in rows),
             "pnp_loses_network_success":sum(r["methods"]["network"]["all8"] and not r["methods"]["pnp"]["all8"] for r in rows)}
    report={"scope":"paired GTROI development diagnostic, no production change","step":state["step"],"checkpoint":checkpoint,"config_id":state["config_id"],"composition":composition,"summary":summary,"changes":changes,"rows":rows}
    atomic_json(out/"REPORT.json",report);print(json.dumps({k:v for k,v in report.items() if k!="rows"}))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--surface-checkpoint");a=p.parse_args();run(a.config,a.checkpoint,a.output,a.surface_checkpoint)
