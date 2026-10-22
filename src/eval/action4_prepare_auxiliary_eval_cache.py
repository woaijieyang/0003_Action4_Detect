"""Build once and prove cached evaluation identical on all378ROIs."""
import json,os,time,hashlib,torch
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_auxiliary_train import build
from model.training.action4_component_train import config_id
from eval.action4_component_eval_cached import CachedComponentData
from eval.action4_auxiliary_eval import evaluate
from eval.action4_auxiliary_feature_cache import prepare_cache,evaluate_cached
from model.training.action4_runtime import atomic_json,save_checkpoint_atomic
def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cfg=json.loads((R/"configs/model/action4_direction_pooled_control_v001.json").read_text())
    out=R/"outputs/action4_aux_eval_acceleration_v001";out.mkdir(exist_ok=False)
    os.link(R/cfg["output"]/"best.pt",out/"validation_head.pt")
    head=torch.load(out/"validation_head.pt",map_location="cpu",weights_only=True)
    assert head["config_id"]==config_id(cfg)
    model,preprocess=build(cfg,"cuda:0");model.evidence.load_state_dict(head["head"],strict=True)
    data=CachedComponentData(cfg["validation_receipt"],"eval","A0")
    before=time.monotonic();reference=evaluate(model,data,preprocess);reference_seconds=time.monotonic()-before
    before=time.monotonic();batches=prepare_cache(model,data,preprocess);prepare_seconds=time.monotonic()-before
    before=time.monotonic();actual=evaluate_cached(model,batches,preprocess);cached_seconds=time.monotonic()-before
    if actual!=reference:
        atomic_json(out/"PARITY_FAILURE.json",{"reference":reference,"cached":actual});raise AssertionError("Cached eval changed report")
    cache=out/"features.pt";save_checkpoint_atomic(cache,{"batches":batches,"preprocess":preprocess})
    stat=cache.stat()
    manifest={"base_checkpoint":cfg["base_checkpoint"],"validation_receipt":cfg["validation_receipt"],"count":len(data),
              "cache":str(cache.relative_to(R)),"stat":[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns],
              "base_sha256":hashlib.sha256((R/cfg["base_checkpoint"]).read_bytes()).hexdigest()}
    atomic_json(out/"CACHE_COMPLETE.json",manifest)
    atomic_json(out/"PARITY_PASS.json",{"passed":True,"exact_full_report_equal":True,"ROIs":len(data),"head_step":head["step"],
                "reference_seconds":reference_seconds,"prepare_seconds":prepare_seconds,"cached_seconds":cached_seconds,
                "speedup_after_preparation":reference_seconds/cached_seconds,"scope":"same frozen head/full378ROI scoring equivalence, not model improvement"})
    print(json.dumps(json.loads((out/"PARITY_PASS.json").read_text())))
if __name__=="__main__":run()
