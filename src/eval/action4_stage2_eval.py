"""Multiple independent workers, full RGB A0 evaluation, bbox-only B0 selection."""
import json,os,time,fcntl,traceback
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_component_data import ComponentData
from model.action4_release_bundle import load_bundle
from eval.action4_b0_scoring import predict
from eval.action4_bbox_metrics import score_boxes
from eval.action4_staged_eval import run as run_staged
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import current_attempt,commit_result
from model.training.action4_best_only_retention import prune_old_selected

def rank_result(result):
    r=result["report"]
    if result["role"]=="B0":return (r["mAP50_95"],r["overall"]["f1"] or 0.,-result["step"])
    return (r["corner_all8_success"],-r["invalid_pose_detections"],-(r["hidden_mean_valid_only"] or 0.),-result["step"])

def evaluate(cfg,snapshot,destination,data):
    destination.mkdir(exist_ok=False)
    if cfg["role"]=="B0":
        model,_,state=load_bundle(str(snapshot),"B0","cuda:0");rows=[]
        for i in range(len(data)):
            item=data[i]
            with torch.inference_mode():ps=predict(model,item["image"],state["config"],"cuda:0",cfg["evaluation_scoring_policy"])
            rows.append({"sample_id":item["record"]["sample_id"],"instances":ps})
        report=score_boxes(rows,data.records,.3)
        report.update(scope=f"full{len(data)} development complete-body B0 only",scoring_policy=cfg["evaluation_scoring_policy"],
                      AP_candidate_policy={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3})
        atomic_json(destination/"PREDICTIONS.json",rows);atomic_json(destination/"REPORT.json",report)
    else:
        profile={"a0_bundle":str(snapshot.relative_to(R)),"b0_bundle":cfg["evaluation_counterpart_bundle"],
                 "b0_scoring_policy":cfg["b0_scoring_policy"],"scope":"dynamic surface/backbone RGB evaluation; no feature cache"}
        atomic_json(destination/"PROFILE.json",profile)
        run_staged(str((destination/"PROFILE.json").relative_to(R)),str((destination/"rgb").relative_to(R)),
                   validation_receipt=cfg["validation_receipt"],validated_data=data)
        report=json.loads((destination/"rgb/REPORT.json").read_text())
        visibility={(r["sample_id"],g["instance_id"]):g["corner_visible"] for r in data.records for g in r["instances"]}
        hidden=[e for row in report["corner_rows"] if row["errors"] is not None
                for e,visible in zip(row["errors"],visibility[row["sample_id"],row["instance_id"]]) if not visible]
        report["hidden_mean_valid_only"]=sum(hidden)/len(hidden) if hidden else None
        atomic_json(destination/"REPORT.json",report)
    return report

def worker(config,worker_id):
    cfg=json.loads(contained_path(R,config,exists=True).read_text());out=contained_path(R,cfg["output"],exists=True);attempt=current_attempt()
    if (out/"EVAL_COMPLETE.json").exists():raise FileExistsError("Terminal evaluation cannot restart")
    guard=(out/f"eval_worker_{worker_id}.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    # Manifest validation occurs once per worker, not for every checkpoint.
    data=ComponentData(cfg["validation_receipt"],"eval","B0")
    data._validation_receipt=contained_path(R,cfg["validation_receipt"],exists=True)
    if len(data)!=cfg["expected_eval_images"]:raise ValueError("Evaluation coverage changed")
    queue=out/"eval";queue.mkdir(exist_ok=True);runs=out/"eval_runs";runs.mkdir(exist_ok=True)
    atomic_json(out/f"EVAL_PROCESS_{worker_id}.json",{"pid":os.getpid(),"attempt_id":attempt,"role":cfg["role"],"worker_id":worker_id})
    while True:
        request=None
        for path in sorted(queue.glob("step_*.json")):
            if path.name.count(".")!=1 or path.with_suffix(".done.json").exists() or path.with_suffix(".failed.json").exists():continue
            try:fd=os.open(path.with_suffix(".claim"),os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"attempt_id":attempt,"worker_id":worker_id},f)
            request=path;break
        if request is None:
            terminal=out/"TRAIN_COMPLETE.json"
            if terminal.exists() and json.loads(terminal.read_text()).get("attempt_id")==attempt:
                atomic_json(out/f"WORKER_COMPLETE_{worker_id}.json",{"attempt_id":attempt});return
            if any(json.loads(p.read_text()).get("attempt_id")==attempt for p in out.glob("TRAIN_FAILURE*.json")):return
            time.sleep(2);continue
        try:
            req=json.loads(request.read_text());snapshot=contained_path(R,req["snapshot"],exists=True)
            if req["role"]!=cfg["role"] or req["config_id"]!=canonical_hash(cfg):raise ValueError("Queue identity mismatch")
            state=torch.load(snapshot,map_location="cpu",weights_only=True)
            if state["selected_step"]!=req["step"] or state["config_id"]!=canonical_hash(cfg) or state["role"]!=cfg["role"]:raise ValueError("Snapshot identity mismatch")
            del state
            destination=runs/f"step_{req['step']:08d}_{attempt}_{worker_id}"
            report=evaluate(cfg,snapshot,destination,data)
            result={"step":req["step"],"config_id":canonical_hash(cfg),"role":cfg["role"],"training_mode":cfg["mode"],
                    "report":report,"raw_report":str((destination/"REPORT.json").relative_to(R))}
            # Claim protects work; this separate lock serializes best selection and alias updates.
            with (out/"selection.lock").open("a") as selection:
                fcntl.flock(selection,fcntl.LOCK_EX);commit_result(out,request,snapshot,result,rank_result)
                prune_old_selected(str(out))
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"attempt_id":attempt,"traceback":traceback.format_exc()});raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--worker-id",required=True);a=p.parse_args();worker(a.config,a.worker_id)
