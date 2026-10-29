"""Multiple independent workers, full RGB A0 evaluation, bbox-only B0 selection."""
import json,os,time,fcntl,traceback,hashlib,copy
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_component_data import ComponentData
from model.action4_release_bundle import load_bundle
from eval.action4_b0_scoring import predict
from eval.action4_bbox_metrics import score_boxes
from eval.action4_staged_eval import run as run_staged
from model.training.action4_runtime import atomic_json,canonical_hash
from model.training.action4_async_recovery import current_attempt,commit_result,checked_binding,CORNER_VARIANT,CORNER_SELECTION_CONTRACT
from model.training.action4_best_only_retention import prune_old_selected

def bind_frozen_image_inputs(profile,cfg):
    """One profile contract for baseline and explicitly registered SAM image eval."""
    from data.action4_prediction_roi_data import PROTOCOL,SAM_NATIVE_PROTOCOL
    result=dict(profile)
    if cfg.get("training_protocol") in (PROTOCOL,SAM_NATIVE_PROTOCOL):
        result.update(roi_expansion=cfg["roi_expansion"],roi_source="frozen_b0_prediction",
            frozen_b0_predictions=cfg["frozen_eval_b0_predictions"],frozen_b0_sha256=cfg["evaluation_counterpart_sha256"])
        if cfg["training_protocol"]==SAM_NATIVE_PROTOCOL:
            if result.get("sam_cache")!=cfg.get("sam_eval_cache") or cfg.get("sam_enabled") is not True:
                raise ValueError("Native SAM async profile needs the bound eval cache")
            result["prediction_input_protocol"]="action4.frozen_B0_SAM_original_eval.v1"
            result["roi_sampling_device"]="cpu"
    return result


def load_evaluation_data(cfg):
    """Use the same receipt-validated full-image dataset in workers and checks."""
    data=ComponentData(cfg["validation_receipt"],"eval","B0")
    data._validation_receipt=contained_path(R,cfg["validation_receipt"],exists=True)
    if len(data)!=cfg["expected_eval_images"]:raise ValueError("Evaluation coverage changed")
    return data


def rank_result(result):
    r=result["report"]
    if result.get("evaluation_variant")==CORNER_VARIANT:
        if result.get("deployment_eligible") is not True:raise ValueError("Unadmitted corner result cannot be ranked")
        from eval.action4_pose_protocol import rank
        return rank(r,result["step"])
    if result["role"]=="B0":return (r["mAP50_95"],r["overall"]["f1"] or 0.,-result["step"])
    if r.get('score_schema')=='action4.pose_projection_all8_002.v1':
        from eval.action4_pose_protocol import rank
        contract=result.get('a0_selection_contract')
        if contract is not None and contract!='full_all8_training_candidate_v1':
            raise ValueError('Unknown A0 candidate selection contract')
        if contract=='full_all8_training_candidate_v1':return rank(r,result['step'])
        return (bool(result.get('deployment_eligible',True)),)+rank(r,result['step'])
    return (r["corner_all8_success"],-r["invalid_pose_detections"],-(r["hidden_mean_valid_only"] or 0.),-result["step"])

def evaluate(cfg,snapshot,destination,data,device="cuda:0"):
    from data.action4_image_mode_contract import ensure_image_mode
    ensure_image_mode(cfg)
    destination.mkdir(exist_ok=False)
    if cfg["role"]=="A0" and cfg.get("a0_variant")==CORNER_VARIANT:
        from model.action4_sam_joint import corner_variant
        if not corner_variant(cfg):raise ValueError("Explicit owned corner variant required")
        from eval.action4_image_guided_eval import run_corner_evaluation
        report=run_corner_evaluation(cfg,snapshot,destination,data,device=device)
        if not isinstance(report,dict):raise ValueError("Corner evaluator must return original full report")
        report_path=destination/"REPORT.json"
        if report_path.exists():
            if json.loads(report_path.read_text())!=report:raise ValueError("Corner report file/return mismatch")
        else:atomic_json(report_path,report)
    elif cfg["role"]=="B0":
        model,_,state=load_bundle(str(snapshot),"B0","cuda:0");rows=[]
        from eval.action4_prefetch import ordered_items
        for i,item in ordered_items(data,workers=cfg.get("eval_loader_workers",4)):
            pass
            with torch.inference_mode():ps=predict(model,item["image"],state["config"],"cuda:0",cfg["evaluation_scoring_policy"])
            rows.append({"sample_id":item["record"]["sample_id"],"instances":ps})
        report=score_boxes(rows,data.records,.3)
        report.update(scope=f"full{len(data)} development complete-body B0 only",scoring_policy=cfg["evaluation_scoring_policy"],
                      AP_candidate_policy={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3})
        atomic_json(destination/"PREDICTIONS.json",rows);atomic_json(destination/"REPORT.json",report)
    else:
        profile={"a0_bundle":str(snapshot.relative_to(R)),"b0_bundle":cfg["evaluation_counterpart_bundle"],
                 "b0_scoring_policy":cfg["b0_scoring_policy"],"scope":"dynamic surface/backbone RGB evaluation; no feature cache","eval_loader_workers":cfg.get("eval_loader_workers",4),"eval_threads":cfg.get("eval_threads",2)}
        from model.action4_sam_joint import sam_variant
        if sam_variant(cfg):
            if "sam_eval_cache" not in cfg:raise ValueError("Enhanced async eval SAM cache dependency missing")
            profile["sam_cache"]=cfg["sam_eval_cache"]
            profile["sam_cache_validation_workers"]=cfg.get("sam_eval_cache_validation_workers",1)
        profile=bind_frozen_image_inputs(profile,cfg)
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

def _corner_identity(cfg):
    if cfg.get("role")!="A0" or cfg.get("a0_variant")!=CORNER_VARIANT:
        raise ValueError("Corner reference requires explicit A0 corner configuration")
    source=cfg["corner_eval_source"];checked_binding(source)
    return {"schema":"action4.corner_eval_run_identity.v1","config_id":canonical_hash(cfg),
        "role":"A0","evaluation_variant":CORNER_VARIANT,"corner_eval_source":copy.deepcopy(source),
        "parent_stage_fallback":{"path":cfg["parent_bundle"],"sha256":cfg["parent_bundle_sha256"],"selected_step":cfg["parent_best_step"]}}


def prepare_corner_selection(cfg,step,snapshot,report,reference=None):
    """Bind evaluator evidence; quality AND original protocol decides admission."""
    from data.action4_sam_native_export import _v2_checked
    from eval.action4_pose_protocol import eligible,SCHEMA
    signature=_corner_identity(cfg)
    if report.get("corner_eval_source")!=cfg["corner_eval_source"] or report.get("score_schema")!=SCHEMA:
        raise ValueError("Corner evaluation source/schema mismatch")
    for key in ("corner_quality_ready","quality_gate_passed_for_selection"):
        if type(report.get(key)) is not bool:raise ValueError("Explicit whole-checkpoint quality result required")
    if report["corner_quality_ready"]!=report["quality_gate_passed_for_selection"]:
        raise ValueError("Corner quality result flags disagree")
    snap=contained_path(R,snapshot,exists=True)
    with snap.open("rb") as stream:snapshot_sha=hashlib.file_digest(stream,"sha256").hexdigest()
    evaluated={"path":str(snap.relative_to(R)),"sha256":snapshot_sha};checked_binding(evaluated)
    calibration=_v2_checked(report["calibration_receipt"])
    checked_binding(report["prediction_artifact"])
    binding=calibration.get("run_binding",{})
    if (calibration.get("schema")!="action4.corner_checkpoint_calibration.v1"
        or calibration.get("quality_ready") is not report["corner_quality_ready"]
        or calibration.get("quality_gate_passed_for_selection") is not report["quality_gate_passed_for_selection"]
        or binding.get("checkpoint_sha256")!=snapshot_sha
        or binding.get("source_sha256")!=cfg["corner_eval_source"]["sha256"]
        or binding.get("predictions_sha256")!=report["prediction_artifact"]["sha256"]):
        raise ValueError("Corner calibration/checkpoint/source/prediction binding mismatch")
    raw=report.get("raw_pose_report");candidate=report.get("candidate_pose_report")
    if (not isinstance(raw,dict) or not isinstance(candidate,dict) or raw.get("score_schema")!=SCHEMA
        or candidate.get("score_schema")!=SCHEMA or calibration["raw_report"]["pose_report"]!=raw
        or calibration["candidate_report"]["pose_report"]!=candidate):
        raise ValueError("Corner raw/candidate evidence differs from calibration receipt")
    chosen=candidate if report["quality_gate_passed_for_selection"] else raw
    if calibration.get("final_path")!=("candidate" if report["quality_gate_passed_for_selection"] else "raw"):
        raise ValueError("Corner global branch selection mismatch")
    if any(report.get(k)!=v for k,v in chosen.items()):
        raise ValueError("Corner final report differs from globally selected original pose report")
    for name,value in (("raw_pose_report_artifact",raw),("candidate_pose_report_artifact",candidate)):
        if _v2_checked(report[name])!=value:raise ValueError("Corner pose report artifact differs")
    if step==0:
        reference=dict(copy.deepcopy(raw),corner_reference_identity=signature)
    elif not isinstance(reference,dict) or reference.get("corner_reference_identity")!=signature:
        raise ValueError("Reference config/source differs from corner evaluation")
    guard=bool(eligible(report,reference))
    fields={"evaluation_variant":CORNER_VARIANT,"selection_contract":CORNER_SELECTION_CONTRACT,
        "corner_eval_source":copy.deepcopy(cfg["corner_eval_source"]),"parent_stage_fallback":signature["parent_stage_fallback"],
        "evaluated_snapshot":evaluated,"pose_protocol_eligible":guard,
        "deployment_eligible":report["quality_gate_passed_for_selection"] and guard}
    if cfg.get("roi_finetune") is not None:
        from model.training.action4_corner_research_selection import POLICY
        fields["research_selection_policy"]=POLICY
    return fields,reference


def bind_evaluation_result(cfg,request,report,destination,image_count,worker_seconds):
    """Preserve immutable corner reports; worker timing is a separate measurement."""
    if cfg.get("role")=="A0" and cfg.get("a0_variant")==CORNER_VARIANT:
        if report.get("evaluation_images")!=image_count:
            raise ValueError("Corner evaluation image count differs from actual dataset")
        path=destination/"REPORT.json"
        raw=path.read_bytes()
        if json.loads(raw)!=report:
            raise ValueError("Corner immutable report differs from returned evaluation")
        extras={"worker_evaluation_wall_seconds":worker_seconds,"worker_evaluation_images":image_count,
                "report_artifact":{"path":str(path.relative_to(R)),"sha256":hashlib.sha256(raw).hexdigest()}}
    else:
        report["evaluation_wall_seconds"]=worker_seconds
        report["evaluation_images"]=image_count
        extras={}
    return {"step":request["step"],"config_id":canonical_hash(cfg),"role":cfg["role"],
            "training_mode":cfg["mode"],"report":report,
            "raw_report":str((destination/"REPORT.json").relative_to(R)),**extras}


def worker(config,worker_id):
    cfg=json.loads(contained_path(R,config,exists=True).read_text());out=contained_path(R,cfg["output"],exists=True);attempt=current_attempt()
    if (out/"EVAL_COMPLETE.json").exists():raise FileExistsError("Terminal evaluation cannot restart")
    from model.training.action4_run_snapshot import verify_running_snapshot
    verify_running_snapshot(cfg)
    if cfg.get("role")=="A0" and cfg.get("a0_variant")==CORNER_VARIANT:
        from eval.action4_image_guided_eval import validate_corner_eval_environment
        validate_corner_eval_environment(cfg,"cuda:0")
    guard=(out/f"eval_worker_{worker_id}.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from model.training.action4_resource_policy import apply_allocator_budget
    apply_allocator_budget(cfg,"evaluation",0)
    from model.training.action4_resource_policy import evaluation_threads
    torch.set_num_threads(evaluation_threads(cfg));torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    # Manifest validation occurs once per worker, not for every checkpoint.
    data=load_evaluation_data(cfg)
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
            eval_started=time.monotonic()
            report=evaluate(cfg,snapshot,destination,data)
            worker_seconds=time.monotonic()-eval_started
            result=bind_evaluation_result(cfg,req,report,destination,len(data),worker_seconds)
            if cfg.get("a0_variant")==CORNER_VARIANT:
                ref=out/"REFERENCE_REPORT.json"
                if req["step"]!=0:
                    while not ref.exists():
                        if list(queue.glob("*.failed.json")):raise RuntimeError("Reference evaluation failed")
                        time.sleep(2)
                fields,reference=prepare_corner_selection(cfg,req["step"],snapshot,report,
                    None if req["step"]==0 else json.loads(ref.read_text()))
                if req["step"]==0:
                    if ref.exists() and json.loads(ref.read_text())!=reference:raise ValueError("Corner raw step0 reference changed")
                    if not ref.exists():atomic_json(ref,reference)
                result.update(fields)
            elif cfg['role']=='A0' and cfg.get('score_schema')=='action4.pose_projection_all8_002.v1':
                from eval.action4_pose_protocol import eligible
                ref=out/'REFERENCE_REPORT.json'
                if req['step']==0:atomic_json(ref,report)
                else:
                    while not ref.exists():
                        if list(queue.glob('*.failed.json')):raise RuntimeError('Reference evaluation failed')
                        time.sleep(2)
                result['deployment_eligible']=eligible(report,json.loads(ref.read_text()))
                if cfg.get('a0_selection_contract') is not None:
                    if cfg['a0_selection_contract']!='full_all8_training_candidate_v1':raise ValueError('Unknown A0 training candidate contract')
                    result['a0_selection_contract']=cfg['a0_selection_contract']
            # Claim protects work; this separate lock serializes best selection and alias updates.
            with (out/"selection.lock").open("a") as selection:
                fcntl.flock(selection,fcntl.LOCK_EX);commit_result(out,request,snapshot,result,rank_result)
                prune_old_selected(str(out))
        except BaseException:
            atomic_json(request.with_suffix(".failed.json"),{"attempt_id":attempt,"traceback":traceback.format_exc()});raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--worker-id",required=True);a=p.parse_args();worker(a.config,a.worker_id)
