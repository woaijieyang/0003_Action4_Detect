"""Supplement immutable initial preview packets with scene-ray source attribution, no RGB render."""
from pathlib import Path
import sys,json,hashlib,copy
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))

def run(template):
    import numpy as np
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_scene_visibility import SceneVisibility,eligible_sources
    from data import action4_phrase2_queue as queue
    runtime=Phrase2Runtime(template);trees={s:LocalArmVisibility(runtime,s) for s in ["left","right"]};measure=SceneVisibility(runtime,trees)
    db=queue.connect();rows=db.execute("SELECT * FROM slots WHERE scene=? AND status IN ('preview_checked','rendered') ORDER BY slot",(template,)).fetchall();summary=[]
    for row in rows:
        old=json.loads(row["record"]);pointer=old["candidate"];path=R/pointer;payload=json.loads(path.read_text());proposal=payload["proposal"];slot=payload["slot"]
        runtime.apply(proposal);actual=runtime.export_corners()
        for i,obj in enumerate(payload["record"]["instances"],1):
            if np.max(np.abs(np.array(obj["corners_px"])-actual["canonical8_by_instance"][str(i)]["points_native_px"]))>.005:raise ValueError("Recovered frame pose changed")
        sources=measure.measure(proposal);ok,reason=eligible_sources(sources,slot["occlusion_bins"])
        if not ok:
            db.execute("UPDATE slots SET status='pending',record=NULL,error=?,owner=NULL WHERE slot=?",(reason,slot["slot_id"]))
            summary.append({"slot_id":slot["slot_id"],"decision":"resample_same_slot","reason":reason,"sources":sources});continue
        new=copy.deepcopy(payload);new["record"]["occlusion_sources"]=sources;new["parent_candidate"]={"path":pointer,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        target=path.with_name("ATTRIBUTED_CANDIDATE_v002.json")
        with target.open("x") as f:json.dump(new,f,allow_nan=False)
        old["candidate"]=str(target.relative_to(R));db.execute("UPDATE slots SET record=? WHERE slot=?",(json.dumps(old),slot["slot_id"]))
        summary.append({"slot_id":slot["slot_id"],"decision":"retain_existing_RGB","sources":sources})
    db.close();out=R/"outputs/phrase2_generation_v001/recovery";out.mkdir(exist_ok=True);(out/(template+".json")).write_text(json.dumps({"rows":summary,"no_RGB_rerender":True},indent=2));print(template,"retained",sum(r["decision"]=="retain_existing_RGB" for r in summary),"resample",sum(r["decision"]!="retain_existing_RGB" for r in summary),flush=True)
if __name__=="__main__":run(sys.argv[sys.argv.index("--")+1])
