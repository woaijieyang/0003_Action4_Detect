"""Async full 200-image development scoring; GT is used only after prediction."""
import json,os,time,fcntl,traceback
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_box_refinement_data import load_inputs
from model.action4_box_refinement import BoxRefinement
from eval.action4_soft_nms import select
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json,canonical_hash

def predict(model,rows,device):
    predictions=[]
    model.eval()
    with torch.inference_mode():
        for row in rows:
            args=[row[k].to(device) for k in ("features","geometry","priors","boxes")]
            boxes,logits,_=model(*args)
            ps=[dict(p,bbox_full_xyxy=b,score=float(s)) for p,b,s in
                zip(row["candidates"],boxes.cpu().tolist(),logits.sigmoid().cpu().tolist())]
            predictions.append({"sample_id":row["sample_id"],"instances":select(ps,"soft_gaussian50")})
    return predictions

def worker(config):
    cfg=json.loads((R/config).read_text());out=R/cfg["output"]
    guard=(out/"eval.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    rows=load_inputs(cfg["cache_output"],"eval")
    refs=[json.loads(x) for x in (R/"data/eval/annotations.jsonl").read_text().splitlines()]
    model=BoxRefinement().cuda();queue=out/"eval";queue.mkdir(exist_ok=True)
    atomic_json(out/"EVAL_PROCESS.json",{"pid":os.getpid()})
    while True:
        requests=[p for p in sorted(queue.glob("step_*.json")) if p.name.count(".")==1 and not p.with_suffix(".done.json").exists() and not p.with_suffix(".failed.json").exists()]
        if not requests:
            if (out/"TRAIN_COMPLETE.json").exists() or (out/"TRAIN_FAILURE.json").exists():return
            time.sleep(3);continue
        p=requests[0]
        try:
            request=json.loads(p.read_text());state=torch.load(R/request["snapshot"],map_location="cpu",weights_only=True)
            if state["config_id"]!=canonical_hash(cfg):raise ValueError("Checkpoint configuration mismatch")
            model.load_state_dict(state["head"],strict=True)
            predictions=predict(model,rows,"cuda:0")
            report=score_boxes(predictions,refs,.3)
            result={"step":request["step"],"report":report,"scope":"development only; not real acceptance","dual_target_pass":report["overall"]["dual_both_detection_rate"]>.90}
            best_path=out/"BEST.json";best=json.loads(best_path.read_text()) if best_path.exists() else None
            rank=lambda r:(r["report"]["mAP50_95"],r["report"]["overall"]["f1"])
            if best is None or rank(result)>rank(best):
                temp=out/"best.tmp";os.link(R/request["snapshot"],temp);os.replace(temp,out/"best.pt")
                atomic_json(best_path,result);atomic_json(out/"BEST_PREDICTIONS.json",predictions)
            atomic_json(p.with_suffix(".done.json"),result)
            (R/request["snapshot"]).unlink()
        except BaseException:
            atomic_json(p.with_suffix(".failed.json"),{"traceback":traceback.format_exc()});raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);worker(p.parse_args().config)
