from pathlib import Path
import sys,json,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
from data.action4_phrase2_runtime import Phrase2Runtime
from data.action4_phrase2_geometry import upperarm_reenters,junction_outside_frame
from data import action4_phrase2_queue as queue

def run():
    db=queue.connect();rejected=[];retained=0
    for scene in ["wall","table_full_garment","cloth_lift10_full_garment"]:
        runtime=Phrase2Runtime(scene)
        rows=db.execute("SELECT * FROM slots WHERE status='preview_checked' AND scene=?",(scene,)).fetchall()
        for row in rows:
            checked=json.loads(row["record"]);payload=json.loads((R/checked["candidate"]).read_text());p=payload["proposal"]
            bad=[s for s in ["left","right"] if upperarm_reenters(runtime,s,np.array(p["actor_pose_capture_by_side"][s]["brace_B_capture"]),np.array([3248,2464])) or not junction_outside_frame(runtime,s,np.array(p["actor_pose_capture_by_side"][s]["brace_B_capture"]),np.array([3248,2464]))]
            if bad:
                db.execute("UPDATE slots SET status='pending',record=NULL,error=? WHERE slot=?",("elbow_junction_or_upperarm_visual_continuity",row["slot"]))
                rejected.append({"slot":row["slot"],"sides":bad,"candidate":checked["candidate"]})
            else:retained+=1
    db.close();(R/"outputs/phrase2_generation_v001/CONTINUITY_REVIEW_v003.json").write_text(json.dumps({"retained_existing_RGB":retained,"resample_same_slots":rejected,"no_asset_geometry_changed":True},indent=2));print("retained",retained,"resample",len(rejected))
if __name__=="__main__":run()
