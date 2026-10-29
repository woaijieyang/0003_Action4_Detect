"""Actual Blender preflight: coupled sampling, arm occlusion and changed-pair collision."""
from pathlib import Path
import sys,json,time,collections,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))

def run():
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_sampling import propose,occlusion_bin
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    cfg=json.loads((R/"configs/data/action4_phrase2_v001.json").read_text());plan=json.loads((R/cfg["output_root"]/"SLOTS.json").read_text());catalog=json.loads((R/cfg["output_root"]/"OCCLUSION_CATALOG_v005.json").read_text())
    ids=set(plan["preview_slot_ids"]);allrows=[]
    for name in cfg["scene_counts"]:
        runtime=Phrase2Runtime(name);trees={s:LocalArmVisibility(runtime,s) for s in ["left","right"]};checker=Phrase2CollisionChecker(runtime);reasons=collections.Counter();found=[];first_collision=None;first_unknown=None
        wanted=[x for x in plan["slots"] if x["slot_id"] in ids and x["scene_id"]==name][:4]
        for slot in wanted:
            for attempt in range(120):
                if attempt%20==0:print(name,slot["slot_id"],"attempt",attempt,flush=True)
                p,error=propose(runtime,cfg,slot,attempt,catalog)
                if error:reasons[error]+=1;continue
                estimates={}
                for side in ["left","right"]:
                    pose=p["actor_pose_capture_by_side"][side];o=trees[side].estimate(np.array(pose["brace_B_capture"]),np.array(pose["core_C_capture"]),28);estimates[side]=o
                if any(o is None or occlusion_bin(o)!=slot["occlusion_bins"][side] for side,o in estimates.items()):reasons["estimated_occlusion_bin"]+=1;continue
                runtime.apply(p);collision=checker.check()
                if not collision["accepted"]:
                    reasons[collision["status"]]+=1
                    if first_collision is None:first_collision=collision
                    if collision["unknown"] and first_unknown is None:first_unknown=collision
                    continue
                corners=runtime.export_corners();row={"slot":slot,"attempt":attempt,"parameters":p,"occlusion_estimate":estimates,"collision":collision,"projection_error_px":corners["max_proposal_vs_actual_error_px"]}
                found.append(row);break
            else:found.append({"slot":slot,"unfilled":True})
        allrows.append({"template":name,"found":found,"rejection_counts":dict(reasons),"first_collision":first_collision,"first_unknown":first_unknown})
        out=R/cfg["output_root"]/"PREFLIGHT_GEOMETRY.json";out.write_text(json.dumps({"templates":allrows,"candidate_only":True},indent=2));print(name,[(x["slot"]["slot_id"],x.get("attempt"),x.get("unfilled",False)) for x in found],dict(reasons),flush=True)
if __name__=="__main__":run()
