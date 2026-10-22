"""Additional B0 eval worker; exclusive queue claims, no concurrent BEST writer."""
import json,os,time,traceback,torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_component_data import ComponentData
from model.action4_online_b0 import OnlineB0
from eval.action4_b0_scoring import predict
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json,canonical_hash

def run(config):
    cfg=json.loads(contained_path(R,config,exists=True).read_text());out=contained_path(R,cfg["output"],exists=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    data=ComponentData(cfg["validation_receipt"],"eval","B0")
    items=[data[i] for i in range(len(data))];assert len(items)==200
    model=OnlineB0(cfg["network"]).cuda().eval();queue=out/"eval"
    while True:
        if (out/"CONTROLLER_FAILURE.json").exists() or list(out.glob("TRAIN_FAILURE_rank*.json")):
            raise RuntimeError("Original training failed; preserve queue")
        request=None
        for candidate in sorted(queue.glob("step_*.json")):
            if candidate.name.count(".")!=1 or candidate.with_suffix(".done.json").exists() or candidate.with_suffix(".failed.json").exists():continue
            try:fd=os.open(candidate.with_suffix(".claim"),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"role":"B0_eval_assist"},f)
            request=candidate;break
        if request is None:
            if (out/"TRAIN_COMPLETE.json").exists():return
            time.sleep(3);continue
        started=time.monotonic()
        try:
            req=json.loads(request.read_text());source=contained_path(R,req["snapshot"],exists=True)
            state=torch.load(source,map_location="cpu",weights_only=True)
            if state["step"]!=req["step"] or state["role"]!="B0" or state["config_id"]!=canonical_hash(cfg):
                raise ValueError("Frozen snapshot identity mismatch")
            model.load_state_dict(state["model"],strict=True);model.eval()
            policy=cfg["evaluation_scoring_policy"]
            predictions=[{"sample_id":x["record"]["sample_id"],"instances":predict(model,x["image"],cfg,"cuda:0",policy)} for x in items]
            report=score_boxes(predictions,[x["record"] for x in items],.3)
            report.update(scoring_policy=policy,AP_candidate_policy={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3})
            result={"step":state["step"],"config_id":state["config_id"],"report":report,"snapshot":str(source.relative_to(R)),
                    "scope":"full200 development B0 only; shared queue assist, not final BEST selection"}
            atomic_json(request.with_suffix(".done.json"),result)
            atomic_json(out/f"ASSIST_STATUS_{os.getpid()}.json",{"step":state["step"],"seconds":time.monotonic()-started,"snapshot_retained":True})
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"pid":os.getpid(),"traceback":traceback.format_exc()})
            raise
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);run(p.parse_args().config)
