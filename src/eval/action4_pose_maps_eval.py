"""CPU-only asynchronous evaluation of the complete hidden/visible pose head."""
import json,time,os,fcntl,traceback,numpy as np,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_pose_map_dataset import load_maps
from model.action4_pose_maps import PoseFromMaps
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import current_attempt,is_current_terminal,recover_claims,commit_result

@torch.inference_mode()
def evaluate(model,maps,geometry,metadata):
    model.eval();rows=[]
    for start in range(0,len(maps),16):
        g={k:v[start:start+16] for k,v in geometry.items()}
        out=model(maps[start:start+16],g["K"],g["canonical"])
        corners=out["corners"].numpy();camera=out["camera_corners"].numpy();det=torch.linalg.det(out["rotation"]).numpy()
        for i,m in enumerate(metadata[start:start+16]):
            valid=bool(np.isfinite(corners[i]).all() and np.isfinite(camera[i]).all() and (camera[i,:,2]>0).all() and abs(det[i]-1)<.001)
            error=(np.linalg.norm(corners[i]-m["corners_hm"],axis=1)/m["diagonal"]).tolist() if valid else None
            rows.append({"sample_id":m["sample_id"],"instance_id":m["instance_id"],"heavy":m["heavy"],"visible":m["visible"],"errors":error,"all8":error is not None and max(error)<=.05,"valid":valid})
    hidden=[e for row in rows if row["errors"] is not None for e,v in zip(row["errors"],row["visible"]) if not v]
    return {"scope":"GTROI diagnostic only; no B0/real-video accuracy claim","rois":len(rows),"all8":sum(r["all8"] for r in rows),"heavy_all8":sum(r["heavy"] and r["all8"] for r in rows),"valid":sum(r["valid"] for r in rows),"hidden_mean_valid_only":float(np.mean(hidden)) if hidden else None,"rows":rows}

def worker(config):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"];attempt=current_attempt()
    lock=(out/"eval.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    maps,geometry,meta=load_maps(cfg["cache_output"],"eval");model=PoseFromMaps()
    queue=out/"eval";queue.mkdir(exist_ok=True);recover_claims(queue,attempt)
    atomic_json(out/"EVAL_PROCESS.json",{"pid":os.getpid(),"attempt_id":attempt,"CPU_only":True,"threads":1})
    while True:
        request=None
        for p in sorted(queue.glob("step_*.json")):
            if p.name.count(".")!=1 or p.with_suffix(".done.json").exists() or p.with_suffix(".failed.json").exists():continue
            try:fd=os.open(p.with_suffix(".claim"),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"attempt_id":attempt},f)
            request=p;break
        if request is None:
            if is_current_terminal(out,attempt):
                if (out/"TRAIN_COMPLETE.json").exists():atomic_json(out/"EVAL_COMPLETE.json",{"evaluations":len(list(queue.glob("*.done.json"))),"attempt_id":attempt})
                return
            time.sleep(3);continue
        try:
            item=json.loads(request.read_text());snapshot=R/item["snapshot"]
            state=torch.load(snapshot,map_location="cpu",weights_only=True)
            if state["config_id"]!=canonical_hash(cfg) or state["step"]!=item["step"]:raise ValueError("Snapshot identity mismatch")
            model.load_state_dict(state["model"],strict=True);report=evaluate(model,maps,geometry,meta)
            commit_result(out,request,snapshot,{"step":state["step"],"config_id":state["config_id"],"report":report},lambda x:(x["report"]["all8"],x["report"]["valid"],-(x["report"]["hidden_mean_valid_only"] or 0)))
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"attempt_id":attempt,"traceback":traceback.format_exc()});raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);worker(p.parse_args().config)
