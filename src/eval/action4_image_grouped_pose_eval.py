"""Async pose evaluation on deployment-identical per-image GPU feature batches."""
import json,os,time,fcntl,traceback
import numpy as np
import torch
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.action4_pose_maps import PoseFromMaps
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import current_attempt,recover_claims,commit_result

def load_groups(cache,device,evaluation_manifest="data/eval/annotations.jsonl"):
    root=contained_path(R,cache,exists=True)
    manifest=json.loads((root/"EXACT_CACHE.json").read_text())
    if manifest["schema"]!="action4.image_grouped_pose_cache.v1" or not manifest["preserve_image_batch_shape"]:
        raise ValueError("Deployment image-grouped input contract required")
    reference={r["sample_id"]:r for r in map(json.loads,contained_path(R,evaluation_manifest,exists=True).read_text().splitlines())}
    objects=[g for row in reference.values() for g in row["instances"] if not g.get("ignore",False)]
    if len(objects)!=manifest["total_GT"] or sum(g["occlusion_inframe"]>=.4 for g in objects)!=manifest["heavy_GT"]:
        raise ValueError("Manifest/GT denominator identity mismatch")
    groups=[];seen=set();seen_targets=set()
    for part in manifest["parts"]:
        state=torch.load(contained_path(R,part["path"],exists=True),map_location="cpu",weights_only=True)
        sid=state["sample_id"]
        if sid in seen:raise ValueError("Duplicate cached image")
        seen.add(sid)
        if len(state["inputs"]["maps"])!=part["predictions"]:raise ValueError("Batch shape changed")
        gt={g["instance_id"]:g for g in reference[sid]["instances"]}
        indices=[label["prediction_index"] for label in state["labels_for_scoring_only"]]
        if len(set(indices))!=len(indices) or any(i<0 or i>=part["predictions"] for i in indices):
            raise ValueError("Matching must be one-to-one within the original image batch")
        for label in state["labels_for_scoring_only"]:
            key=(sid,label["instance_id"])
            if key in seen_targets:raise ValueError("A GT target was scored twice")
            seen_targets.add(key)
            g=gt[label["instance_id"]]
            if not np.array_equal(g["corners_px"],label["corners_px"]):raise ValueError("Scoring label identity changed")
            label["visible"]=g["corner_visible"]
        state["inputs"]={k:v.to(device) for k,v in state["inputs"].items()}
        groups.append(state)
    if sum(len(g["inputs"]["maps"]) for g in groups)!=manifest["prediction_count"]:
        raise ValueError("Predictions/FP omitted")
    if len(seen_targets)+manifest["missed"]!=manifest["total_GT"]:
        raise ValueError("Missing targets were excluded from the denominator")
    return groups,manifest

@torch.inference_mode()
def evaluate(model,groups,manifest):
    model.eval();rows=[];valid_total=0
    for group in groups:
        inputs=group["inputs"] # No scoring label is passed into the model.
        output=model(inputs["maps"],inputs["K"],inputs["canonical"])
        uv=output["corners"].cpu().numpy();camera=output["camera_corners"].cpu().numpy()
        valid=np.isfinite(uv).all((1,2)) & np.isfinite(camera).all((1,2)) & (camera[:,:,2]>0).all(1)
        valid_total+=int(valid.sum())
        native=[DataAdapter.transform_points(p,A) for p,A in zip(uv,group["inverse_affines"])]
        for label in group["labels_for_scoring_only"]:
            index=label["prediction_index"]
            error=(np.linalg.norm(native[index]-label["corners_px"],axis=1)/label["diagonal"]).tolist() if valid[index] else None
            rows.append({"sample_id":group["sample_id"],"instance_id":label["instance_id"],"heavy":label["heavy"],"visible":label["visible"],"errors":error,"all8":error is not None and max(error)<=.05})
    successes=sum(r["all8"] for r in rows)
    hidden=[e for r in rows if r["errors"] is not None for e,v in zip(r["errors"],r["visible"]) if not v]
    return {"scope":"fixed detector full-image grouped pose eval; misses/FP retained; no GT input",
        "all8":successes,"total_GT":manifest["total_GT"],"heavy_all8":sum(r["all8"] and r["heavy"] for r in rows),
        "heavy_GT":manifest["heavy_GT"],"valid":valid_total,"prediction_count":manifest["prediction_count"],
        "joint_precision":successes/manifest["prediction_count"],"joint_recall":successes/manifest["total_GT"],
        "hidden_mean_valid_only":float(np.mean(hidden)) if hidden else None,"rows":rows}

def worker(config):
    cfg=json.loads(contained_path(R,config,exists=True).read_text());out=contained_path(R,cfg["output"],exists=True)
    if (out/"EVAL_COMPLETE.json").exists():
        raise FileExistsError("Completed evaluation may not restart; create a new explicitly scoped run")
    attempt=current_attempt();guard=(out/"eval.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    groups,manifest=load_groups(cfg["cache_output"],cfg["eval_device"],cfg.get("evaluation_manifest","data/eval/annotations.jsonl"))
    if manifest["identity"]!=cfg["cache_identity"]:raise ValueError("Frozen detector/feature identity mismatch")
    model=PoseFromMaps().to(cfg["eval_device"]);queue=out/"eval";queue.mkdir(exist_ok=True);recover_claims(queue,attempt)
    atomic_json(out/"EVAL_PROCESS.json",{"pid":os.getpid(),"attempt_id":attempt,"input_contract":"per-image GPU batches","data_role":"development"})
    while True:
        request=None
        for path in sorted(queue.glob("step_*.json")):
            if path.name.count(".")!=1 or path.with_suffix(".done.json").exists() or path.with_suffix(".failed.json").exists():continue
            try:fd=os.open(path.with_suffix(".claim"),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"attempt_id":attempt},f)
            request=path;break
        if request is None:
            terminals=[out/"TRAIN_COMPLETE.json",out/"EVAL_INPUT_COMPLETE.json"]
            if any(p.exists() and json.loads(p.read_text()).get("attempt_id")==attempt for p in terminals):
                atomic_json(out/"EVAL_COMPLETE.json",{"attempt_id":attempt,"evaluations":len(list(queue.glob("*.done.json"))),"scope":"evaluation worker completed; not proof of new training"});return
            if any(json.loads(p.read_text()).get("attempt_id")==attempt for p in out.glob("TRAIN_FAILURE*.json")):return
            time.sleep(2);continue
        try:
            req=json.loads(request.read_text());snapshot=contained_path(R,req["snapshot"],exists=True)
            state=torch.load(snapshot,map_location="cpu",weights_only=True)
            if state["step"]!=req["step"] or state["config_id"]!=canonical_hash(cfg):raise ValueError("Snapshot/config identity mismatch")
            model.load_state_dict(state["model"],strict=True)
            report=evaluate(model,groups,manifest)
            commit_result(out,request,snapshot,{"step":state["step"],"config_id":state["config_id"],"report":report},
                lambda x:(x["report"]["all8"],x["report"]["valid"],-(x["report"]["hidden_mean_valid_only"] or 0)))
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"attempt_id":attempt,"traceback":traceback.format_exc()});raise
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);worker(p.parse_args().config)
