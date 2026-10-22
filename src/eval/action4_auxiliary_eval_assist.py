"""Claim legacy auxiliary eval work without racing its original BEST writer."""
import os,json,time,traceback,torch
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_auxiliary_train import build
from model.training.action4_component_train import config_id
from eval.action4_auxiliary_feature_cache import evaluate_cached
from model.training.action4_runtime import atomic_json

def run(config,cache_root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cfg=json.loads(contained_path(R,config,exists=True).read_text());out=contained_path(R,cfg["output"],exists=True);queue=out/"eval"
    root=contained_path(R,cache_root,exists=True);proof=json.loads((root/"PARITY_PASS.json").read_text())
    if not proof["exact_full_report_equal"]:raise ValueError("Cache not validated")
    info=json.loads((root/"CACHE_COMPLETE.json").read_text())
    if cfg["base_checkpoint"]!=info["base_checkpoint"] or cfg["validation_receipt"]!=info["validation_receipt"]:
        raise ValueError("Frozen cache source mismatch")
    path=contained_path(R,info["cache"],exists=True);s=path.stat()
    if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]!=info["stat"]:raise ValueError("Cache changed")
    cache=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    model,preprocess=build(cfg,"cpu");model.evidence.cuda()
    if preprocess!=cache["preprocess"]:raise ValueError("Preprocessing differs")
    journal=root/"workers"/str(os.getpid());journal.mkdir(parents=True,exist_ok=False)
    atomic_json(journal/"PROCESS.json",{"pid":os.getpid(),"config":config,"physical_gpu":4,"policy":"claims only; original BEST never written; completed snapshots retained"})
    done=0
    while True:
        request=None
        for p in sorted(queue.glob("step_*.json")):
            if p.name.count(".")!=1 or p.with_suffix(".done.json").exists() or p.with_suffix(".failed.json").exists():continue
            try:fd=os.open(p.with_suffix(".claim"),os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"time":time.time(),"assisted":True},f)
            request=p;break
        if request is None:
            if (out/"TRAIN_COMPLETE.json").exists() or (out/"FAILURE.json").exists():
                atomic_json(journal/"COMPLETE.json",{"completed_evals":done});return
            time.sleep(3);continue
        try:
            before=time.monotonic();req=json.loads(request.read_text())
            state=torch.load(contained_path(R,req["snapshot"],exists=True),map_location="cpu",weights_only=True)
            if state["schema"]!="action4.aux_eval.v1" or state["config_id"]!=config_id(cfg) or state["step"]!=req["step"]:
                raise ValueError("Snapshot identity mismatch")
            model.evidence.load_state_dict(state["head"],strict=True)
            report=evaluate_cached(model,cache["batches"],preprocess)
            result={"step":state["step"],"snapshot":req["snapshot"],"report":report,
                    "assisted_eval":{"cache_root":cache_root,"worker_pid":os.getpid(),"seconds":time.monotonic()-before}}
            atomic_json(request.with_suffix(".done.json"),result);done+=1
            atomic_json(journal/"STATUS.json",{"completed_evals":done,"last_step":state["step"],"seconds":result["assisted_eval"]["seconds"]})
        except BaseException:
            atomic_json(journal/"FAILURE.json",{"request":str(request.relative_to(R)),"traceback":traceback.format_exc(),"snapshot_retained":True})
            # Return unfinished work intact to its primary evaluator; never invent done.
            claim=request.with_suffix(".claim")
            if claim.exists() and json.loads(claim.read_text()).get("pid")==os.getpid() and not request.with_suffix(".done.json").exists():claim.unlink()
            raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--cache-root",required=True);a=p.parse_args();run(a.config,a.cache_root)
