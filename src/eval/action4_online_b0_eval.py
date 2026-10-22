"""B0-only async evaluation with immutable selected checkpoints."""
import json,os,time,fcntl,traceback
import torch
from torch.utils.data import DataLoader
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from model.action4_online_b0 import OnlineB0
from eval.action4_online_b0_pipeline import predict as legacy_predict
from eval.action4_b0_scoring import predict as policy_predict
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import current_attempt,is_current_terminal,recover_claims,commit_result

def worker(config):
    attempt=current_attempt();cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    guard=(out/"eval.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    if cfg.get("deterministic_eval",False):torch.use_deterministic_algorithms(True)
    data=ComponentData(cfg["validation_receipt"],"eval","B0")
    # Cache decoded200 RGB images once in RAM; no files or features regenerated.
    loader=DataLoader(data,batch_size=1,collate_fn=list,num_workers=2,prefetch_factor=2)
    items=[x[0] for x in loader];assert len(items)==200
    model=OnlineB0(cfg["network"]).cuda()
    queue=out/"eval";queue.mkdir(exist_ok=True);recover_claims(queue,attempt)
    atomic_json(out/"EVAL_PROCESS.json",{"pid":os.getpid(),"attempt_id":attempt})
    while True:
        request=None
        for p in sorted(queue.glob("step_*.json")):
            if p.name.count(".")!=1 or p.with_suffix(".done.json").exists() or p.with_suffix(".failed.json").exists():continue
            try:fd=os.open(p.with_suffix(".claim"),os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"attempt_id":attempt},f)
            request=p;break
        if request is None:
            if is_current_terminal(out,attempt):return
            time.sleep(3);continue
        try:
            req=json.loads(request.read_text());state=torch.load(R/req["snapshot"],map_location="cpu",weights_only=True)
            if state["role"]!="B0" or state["config_id"]!=canonical_hash(cfg) or state["step"]!=req["step"]:raise ValueError("Eval checkpoint identity mismatch")
            model.load_state_dict(state["model"],strict=True);model.eval()
            policy=cfg.get("evaluation_scoring_policy","learned")
            predictions=[{"sample_id":item["record"]["sample_id"],"instances":policy_predict(model,item["image"],cfg,"cuda:0",policy) if policy!="learned" else legacy_predict(model,item["image"],cfg,"cuda:0")} for item in items]
            report=score_boxes(predictions,[x["record"] for x in items],.3)
            report["scoring_policy"]=policy
            report["AP_candidate_policy"]={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3}
            result={"step":state["step"],"config_id":state["config_id"],"report":report,"scope":"full200 development B0 only, not real acceptance"}
            commit_result(out,request,R/req["snapshot"],result,lambda x:(x["report"]["mAP50_95"],x["report"]["overall"]["f1"]))
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"attempt_id":attempt,"traceback":traceback.format_exc()});raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);worker(p.parse_args().config)
