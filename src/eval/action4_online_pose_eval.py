"""Select pose weights using fixed B0 proposal ROIs; misses/FP stay in denominators."""
import json,time,os,fcntl,traceback,numpy as np,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_pose_map_dataset import load_maps
from model.action4_pose_maps import PoseFromMaps
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import current_attempt,recover_claims,commit_result

def joint_summary(rows,contract):
    successes=sum(r["all8"] for r in rows);total=contract["total_GT"];pred=contract["fixed_B0_prediction_count"]
    return {"all8":successes,"total_GT":total,"matched_ROIs":len(rows),"fixed_B0_missed_GT":contract["fixed_B0_missed_GT"],
        "fixed_B0_false_positives":contract["fixed_B0_false_positives"],"joint_precision":successes/pred if pred else None,
        "joint_recall":successes/total,"joint_f1":2*successes/(pred+total) if pred+total else None}

@torch.inference_mode()
def evaluate(model,maps,geometry,metadata,contract):
    model.eval();rows=[]
    for start in range(0,len(maps),32):
        g={k:v[start:start+32] for k,v in geometry.items()}
        output=model(maps[start:start+32],g["K"],g["canonical"])
        uv=output["corners"].cpu().numpy();camera=output["camera_corners"].cpu().numpy();det=torch.linalg.det(output["rotation"]).cpu().numpy()
        for i,m in enumerate(metadata[start:start+32]):
            valid=bool(np.isfinite(uv[i]).all() and np.isfinite(camera[i]).all() and (camera[i,:,2]>0).all() and abs(det[i]-1)<.001)
            error=(np.linalg.norm(uv[i]-m["corners_hm"],axis=1)/m["diagonal"]).tolist() if valid else None
            rows.append({"sample_id":m["sample_id"],"instance_id":m["instance_id"],"visible":m["visible"],"heavy":m["heavy"],
                         "errors":error,"valid":valid,"all8":error is not None and max(error)<=.05})
    hidden=[e for row in rows if row["errors"] is not None for e,v in zip(row["errors"],row["visible"]) if not v]
    return dict(joint_summary(rows,contract),scope="fixed B0 predicted-ROI pose evaluation; not GTROI or true real acceptance",
        heavy_all8=sum(r["heavy"] and r["all8"] for r in rows),heavy_GT=contract["heavy_GT"],
        valid=sum(r["valid"] for r in rows),hidden_mean_valid_only=float(np.mean(hidden)) if hidden else None,rows=rows)

def worker(config):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"];attempt=current_attempt()
    lock=(out/"eval.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    maps,geometry,meta=load_maps(cfg["cache_output"],"eval")
    manifest=json.loads((R/cfg["cache_output"]/"MAPS_COMPLETE.json").read_text());contract=dict(manifest["evaluation_contract"])
    contract["heavy_GT"]=cfg["evaluation_heavy_GT"]
    if contract["mode"]!="fixed_detector_predicted_ROI" or len(maps)+contract["fixed_B0_missed_GT"]!=contract["total_GT"]:
        raise ValueError("Predicted-ROI coverage contract broken")
    if contract["B0_step"]!=3220:raise ValueError("Frozen validation detector changed")
    device=cfg["eval_device"];maps=maps.to(device);geometry={k:v.to(device) for k,v in geometry.items()}
    model=PoseFromMaps().to(device);queue=out/"eval";queue.mkdir(exist_ok=True);recover_claims(queue,attempt)
    atomic_json(out/"EVAL_PROCESS.json",{"pid":os.getpid(),"attempt_id":attempt,"device":device,"validation":contract})
    while True:
        request=None
        for p in sorted(queue.glob("step_*.json")):
            if p.name.count(".")!=1 or p.with_suffix(".done.json").exists() or p.with_suffix(".failed.json").exists():continue
            try:fd=os.open(p.with_suffix(".claim"),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"attempt_id":attempt},f)
            request=p;break
        if request is None:
            if (out/"TRAIN_COMPLETE.json").exists() and json.loads((out/"TRAIN_COMPLETE.json").read_text())["attempt_id"]==attempt:
                atomic_json(out/"EVAL_COMPLETE.json",{"evaluations":len(list(queue.glob("*.done.json"))),"attempt_id":attempt});return
            failures=list(out.glob("TRAIN_FAILURE_rank*.json"))
            if any(json.loads(p.read_text()).get("attempt_id")==attempt for p in failures):return
            time.sleep(2);continue
        try:
            req=json.loads(request.read_text());snapshot=R/req["snapshot"];state=torch.load(snapshot,map_location="cpu",weights_only=True)
            if state["step"]!=req["step"] or state["config_id"]!=canonical_hash(cfg):raise ValueError("Checkpoint identity mismatch")
            model.load_state_dict(state["model"],strict=True);report=evaluate(model,maps,geometry,meta,contract)
            commit_result(out,request,snapshot,{"step":state["step"],"config_id":state["config_id"],"report":report},
                lambda x:(x["report"]["all8"],x["report"]["valid"],-(x["report"]["hidden_mean_valid_only"] or 0)))
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"attempt_id":attempt,"traceback":traceback.format_exc()});raise
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);worker(p.parse_args().config)
