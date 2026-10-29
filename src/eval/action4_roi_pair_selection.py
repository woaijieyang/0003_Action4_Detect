"""Bind completed adaptation best and its B0 for development, without promotion."""
import json,subprocess,os
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import binding,read_binding,write_new,SCORE_SCHEMA

def _validate_sam_pair(a,cfg,b,bcfg):
    protocol="action4.sam_prediction_roi_enhancement.native_v1"
    if (a.get("role")!="A0" or b.get("role")!="B0" or bcfg.get("role")!="B0"
        or a.get("strict_load") is not True or b.get("strict_load") is not True
        or a.get("frozen_backbone_verified") is not True
        or a.get("epochs")!=80 or cfg.get("epochs")!=80 or cfg.get("mode")!="full_training"
        or cfg.get("training_protocol")!=protocol or cfg.get("a0_variant")!="a0_sam_image"
        or cfg.get("sam_enabled") is not True or cfg.get("corner_enabled",False) is not False
        or cfg.get("train_backbone") is not False or cfg.get("roi_source")!="frozen_b0_prediction"
        or cfg.get("roi_expansion")!=1.5 or cfg.get("roi_size")!=448
        or cfg.get("preprocess",{}).get("roi_expansion")!=1.5
        or cfg.get("evaluation_counterpart_sha256")!=b.get("sha256")
        or a.get("score_schema")!=SCORE_SCHEMA
        or cfg.get("a0_selection_contract")!="full_all8_training_candidate_v1"):
        raise ValueError("Completed sealed native SAM80 and its frozen B0 required")
    completion=a.get("completion",{})
    if (completion.get("status")!="COMPLETION_METADATA_READY"
        or type(completion.get("evaluations")) is not int or completion["evaluations"]<=0
        or type(a.get("completed_optimizer_steps")) is not int or a["completed_optimizer_steps"]<=0):
        raise ValueError("Complete verified SAM training/eval receipts required")
    contract=completion.get("inference_contract",{})
    fields=("training_protocol","parent_bundle_sha256","parent_best_step","evaluation_counterpart_sha256",
        "roi_source","roi_expansion","roi_size","prediction_roi_manifest","frozen_eval_b0_predictions",
        "sam_protocol_id","sam_cache","sam_eval_cache","sam_training_associations","sam_mask_augmentation","stage3_parent_seal")
    if any(k not in cfg or contract.get(k)!=cfg[k] for k in fields):
        raise ValueError("Sealed SAM input contract differs")
    if contract.get("roi_sampling_device")!="cpu":
        raise ValueError("Unregistered SAM sampling contract")
    sources=cfg["sam_cache"].get("expected_sources",{})
    if (sources.get("b0_model_sha256")!=b["sha256"] or not sources.get("sam_model_sha256")
        or any(cfg["sam_eval_cache"].get("expected_sources",{}).get(k)!=sources.get(k)
               for k in ("b0_model_sha256","sam_model_sha256"))):
        raise ValueError("SAM train/eval model sources differ from paired B0")


def validate_pair(a,acfg,b,bcfg):
    if acfg.get("training_protocol")=="action4.sam_prediction_roi_enhancement.native_v1":
        return _validate_sam_pair(a,acfg,b,bcfg)
    extension=acfg.get("continuation_segment") is not None
    if extension:
        from model.training.action4_roi_adaptation import validate_recipe
        validate_recipe(acfg)
        total=acfg["schedule_total_updates"]
        expected_evals=len(set([0,total]+list(range(10,total+1,10))))
        if (acfg.get("mode")!="full_training" or acfg.get("epochs")!=acfg["continuation_segment"]["epochs"]
                or a.get("continuation_segment")!=acfg["continuation_segment"]
                or a.get("completed_optimizer_steps")!=total
                or a.get("completion",{}).get("evaluations")!=expected_evals
                or a.get("frozen_backbone_verified") is not True):
            raise ValueError("Extension must be fully sealed/reloaded with complete evaluation coverage")
    if (a.get("role")!="A0" or b.get("role")!="B0" or a.get("strict_load") is not True or b.get("strict_load") is not True
        or a.get("epochs")!=(acfg["continuation_segment"]["epochs"] if extension else 50) or acfg.get("training_protocol")!="action4.fixed_roi_prediction_adaptation.v1"
        or acfg.get("roi_expansion")!=1.5 or acfg.get("preprocess",{}).get("roi_expansion")!=1.5
        or acfg.get("roi_source")!="frozen_b0_prediction" or acfg.get("a0_variant")!="a0_baseline"
        or acfg.get("a0_selection_contract")!="full_all8_training_candidate_v1"
        or acfg.get("evaluation_counterpart_sha256")!=b.get("sha256")
        or a.get("score_schema")!=SCORE_SCHEMA or bcfg.get("role")!="B0"):
        raise ValueError("Completed fixed1.5 adaptation and frozen B0 counterpart required")
    contract=a.get("completion",{}).get("inference_contract",{})
    if extension and contract.get("continuation_segment")!=acfg["continuation_segment"]:
        raise ValueError("Sealed extension inference contract differs")
    if (contract.get("roi_expansion")!=1.5 or contract.get("B0_sha256")!=b["sha256"]
        or contract.get("frozen_eval_b0_predictions")!=acfg["frozen_eval_b0_predictions"]):
        raise ValueError("Sealed ROI/eval source identity differs")

def _verify_seal(path):
    script=contained_path(R,"scripts/action4_seal_completed_role.py",exists=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
             OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    process=subprocess.run([str(R/".venv/bin/python"),"-I","-B",str(script),"--verify-seal",str(contained_path(R,path,exists=True))],
        cwd=R,env=env,capture_output=True,text=True,check=True)
    return json.loads(process.stdout)

def _verify_pair_seals(a0_seal,b0_seal):
    """Independent immutable seal checks share only the caller's bounded CPU affinity."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix="action4-pair-seal") as pool:
        a=pool.submit(_verify_seal,a0_seal)
        b=pool.submit(_verify_seal,b0_seal)
        return a.result(),b.result()

def freeze_pair(a0_seal,b0_seal,baseline_input_lock,output):
    # Reuse the production immutable-source rank/load verifier; no alternate loading policy.
    a,b=_verify_pair_seals(a0_seal,b0_seal)
    validate_pair(a["seal"],a["config"],b["seal"],b["config"])
    lock_ref=binding(baseline_input_lock);lock=read_binding(lock_ref)
    if lock["video"]["track_replay"]["b0_sha256"]!=b["seal"]["sha256"]:
        raise ValueError("Video frozen B0 source differs from adaptation counterpart")
    models={role:{"path":v["seal"]["bundle"],"sha256":v["seal"]["sha256"],
        "selected_step":v["seal"]["selected_step"],"config_id":v["seal"]["config_id"],
        "sealed_role":binding(path)} for role,v,path in [("A0",a,a0_seal),("B0",b,b0_seal)]}
    result=dict(schema="action4.completed_roi15_development_pair.v1",
        status="COMPLETED_MODELS_BOUND_NOT_TRACK_ACCEPTANCE",models=models,roi_expansion=1.5,roi_size=448,
        training_roi_source="frozen_b0_prediction",baseline_input_lock=lock_ref,
        frozen_eval_B0=a["config"]["frozen_eval_b0_predictions"],raw_video_B0=lock["video"]["raw_B0_pools"],
        CAD=lock["CAD"],camera=lock["video"]["K"],canonical8_ids=list(range(8)),
        decoder="network",score_schema=SCORE_SCHEMA,defaults_changed=False,test_consumed=False,
        tracking_accuracy_accepted=False,inference_source_freeze_required=True,
        inference_source_policy="freeze tested shared RoiSpec pipeline before paired clip inference; no mutable imports")
    result["a0_variant"]=a["config"]["a0_variant"]
    if a["config"]["a0_variant"]=="a0_sam_image":
        result["video_feature_mode_required"]="current_native_v1"
        result["sam_input_contract"]=a["seal"]["completion"]["inference_contract"]
        result["track_input_distribution_accepted"]=False
    if a["config"].get("continuation_segment") is not None:
        result["continuation_segment"]=a["config"]["continuation_segment"]
    read_binding(result["CAD"]);read_binding(result["camera"])
    write_new(output,result);return result
