"""Frozen full650 corner evaluation orchestration. No training or best publication."""
import copy,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,CODE_ROOT,contained_path

def _write(path,value,*,readonly=False):
    text=json.dumps(value,indent=2,allow_nan=False)+"\n"
    with path.open("x") as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    if readonly:path.chmod(0o444)
    return {"path":str(path.relative_to(R)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def _progress(out,stage,completed,total,started,timings,*,error=None):
    """Low-frequency operational telemetry, never a completion/admission receipt."""
    value={"schema":"action4.corner_eval_progress.v1","pid":os.getpid(),
        "stage":stage,"completed_images":completed,"expected_images":total,
        "elapsed_seconds":time.monotonic()-started,"timing_seconds":dict(timings),
        "complete":False,"updated_unix":time.time()}
    if error is not None:value["error_type"]=type(error).__name__
    path=out/"PROGRESS.json";temporary=out/("PROGRESS."+str(os.getpid())+".tmp")
    with temporary.open("w") as stream:
        json.dump(value,stream,allow_nan=False);stream.write("\n")
    os.replace(temporary,path)

def _is_run_launcher_alias(name,module,root):
    """multiprocessing may retain runpy's original readonly bootstrap as __mp_main__."""
    if name!="__mp_main__" or getattr(module,"__name__",None)!="__main__" or getattr(module,"__spec__",None) is not None:
        return False
    root=Path(root).resolve();launch=root.parent/"launch.py";spec=root.parent/"RUN_SPEC.json"
    try:
        return (Path(module.__file__).resolve()==launch and launch.is_file() and not (launch.stat().st_mode & 0o222)
            and spec.is_file() and Path(module.spec_path).resolve()==spec and Path(module.code).resolve()==root)
    except (AttributeError,TypeError,OSError):return False


def _is_project_origin(name,origin):
    namespace=name.split(".")[0] in ("data","model","eval","scripts")
    real_project_file=origin.is_relative_to(R) and not origin.is_relative_to(R/".venv") and origin.is_file()
    return namespace or real_project_file


def _is_verified_torch_proxy(name,module):
    """Recognize the exact installed framework template, never arbitrary scratch code."""
    if name!="_remote_module_non_scriptable" or getattr(module,"__name__",None)!=name:return False
    generator=sys.modules.get("torch.distributed.nn.jit.instantiator")
    if generator is None:return False
    try:
        generator_file=Path(generator.__file__).resolve()
        if not generator_file.is_relative_to(R/".venv"):return False
        directory=Path(generator.INSTANTIATED_TEMPLATE_DIR_PATH).resolve()
        origin=Path(module.__file__).resolve()
        if not directory.is_relative_to(R) or origin!=directory/(name+".py") or not origin.is_file():return False
        fields=dict(assign_module_interface_cls="module_interface_cls = None",args="*args",kwargs="**kwargs",
            arg_types="*args, **kwargs",arrow_and_return_type="",arrow_and_future_return_type="",jit_script_decorator="")
        expected=generator.get_remote_module_template(True).format(**fields)
        return origin.read_text()==expected
    except (AttributeError,TypeError,KeyError,OSError):return False


def _origins(root):
    if not Path(__file__).resolve().is_relative_to(root):raise ValueError("Evaluator itself is outside selected source root")
    origins={};foreign=[]
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,"__file__",None)
        if not path or name=="__main__":continue # readonly launcher bootstrap is outside its source subdirectory
        origin=Path(path).resolve()
        if _is_run_launcher_alias(name,module,root) or _is_verified_torch_proxy(name,module):continue
        project_namespace=name.split(".")[0] in ("data","model","eval","scripts")
        if _is_project_origin(name,origin):
            if not origin.is_relative_to(root):
                foreign.append(name+"="+str(origin));continue
            if project_namespace and not origin.is_file():raise ValueError("Project module has no real source file: "+name)
            origins[name]=str(origin.relative_to(root))
    if foreign:raise ValueError("Stale/mixed project module origin: "+"; ".join(foreign))
    return origins

def _normalized_device_uuid(value):
    import re,uuid
    if isinstance(value,bytes):
        value=str(uuid.UUID(bytes=value)) if len(value)==16 else value.decode("ascii")
    text=str(value)
    if not text.startswith("GPU-"):text="GPU-"+text
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",text):
        raise ValueError("CUDA runtime did not expose a verifiable device UUID")
    return text.lower()

def _resource_guard(cfg,device,*,check_headroom=True):
    """Read-only live admission before model construction; never selects a GPU."""
    import re,subprocess
    from model.training.action4_resource_policy import validate_device_policy,process_cpu_affinity
    target=str(device);affinity=set(os.sched_getaffinity(0))
    if target not in ("cpu","cuda:0"):raise ValueError("Explicit cpu or single logical cuda:0 required")
    if target=="cpu":
        policy=validate_device_policy({"training_gpus":[],"eval_gpus":[]},R)
        allowed=set(policy["cpu_policy"]["allowed_logical_cpu_ids"])&set(range(24,96))
        if not affinity or not affinity<=allowed or len(affinity)>min(72,policy["cpu_policy"]["max_logical_cpus"]):
            raise ValueError("CPU evaluator affinity outside current allowed pool")
        return {"device":"cpu","affinity":sorted(affinity),"GPU_queried":False,"authority":policy["authority"]}
    uuid=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",uuid):
        raise ValueError("One explicit GPU UUID in CUDA_VISIBLE_DEVICES required; no default/numeric/multi GPU")
    configured=cfg.get("eval_gpus")
    if not isinstance(configured,list) or not configured or any(type(g) is not int or g not in (4,5,6,7) for g in configured):
        raise ValueError("Only configured physical eval GPUs4-7 permitted")
    policy=validate_device_policy({"training_gpus":[],"eval_gpus":sorted(set(configured))},R)
    if not {0,1,2,3}<=set(policy["forbidden_gpu_ids"]):raise ValueError("Live authority must forbid GPUs0-3")
    physical=None;free=None
    # Query only already-authorized configured indices, never a guessed/forbidden UUID.
    for gpu in sorted(set(configured)):
        text=subprocess.check_output(["nvidia-smi","--id="+str(gpu),"--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip()
        values=[x.strip() for x in text.split(",")]
        if len(values)!=3 or not values[0].isdigit() or int(values[0])!=gpu or not values[2].isdigit():
            raise ValueError("Actual GPU index/UUID/free-memory metadata unavailable")
        if values[1].lower()==uuid.lower():physical=gpu;free=int(values[2])
    if physical is None:raise ValueError("Visible UUID is not an authorized configured physical eval GPU")
    matching=[]
    for index,gpu in enumerate(configured):
        if gpu==physical:
            cpus=process_cpu_affinity(cfg,"eval_"+str(index),R)
            if cpus is not None and set(cpus)==affinity:matching.append(index)
    allowed=set(policy["cpu_policy"]["allowed_logical_cpu_ids"])&set(range(24,96))
    if len(matching)!=1 or not affinity<=allowed or len(affinity)>72:
        raise ValueError("Actual affinity must match exactly one configured eval worker allocation")
    if type(cfg.get("project_total_reserved_cpu")) is not int or cfg["project_total_reserved_cpu"]>min(72,policy["cpu_policy"]["max_logical_cpus"]):
        raise ValueError("Project CPU reservation exceeds current budget")
    budget=cfg.get("evaluation_allocator_budget_mib")
    if type(budget) is not int or budget<=0:raise ValueError("Explicit evaluation allocator budget required")
    import torch
    initialized=torch.cuda.is_initialized();reusable=0
    if initialized:
        actual=_normalized_device_uuid(getattr(torch.cuda.get_device_properties(0),"uuid",""))
        if actual!=uuid.lower() or torch.cuda.current_device()!=0:
            raise ValueError("Initialized CUDA logical0 UUID/current device differs from explicit mapping")
        reusable=max(0,int((torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))/2**20))
    margin=max(8192,int(cfg.get("uncommitted_gpu_margin_mib",8192)))
    if check_headroom and free+reusable<budget+margin:
        raise ValueError("Insufficient assigned GPU headroom; do not displace other work")
    return {"device":target,"physical_gpu":physical,"GPU_UUID":uuid,"authority":policy["authority"],
        "affinity":sorted(affinity),"eval_worker_index":matching[0],"initialized_mapping_checked":initialized,
        "fresh_free_mib":free,"reusable_own_mib":reusable,"allocator_budget_mib":budget,"margin_mib":margin}

def validate_corner_eval_environment(cfg,device="cuda:0"):
    from model.training.action4_run_snapshot import verify_running_snapshot
    from data.action4_sam_native_export import _v2_checked,_v2_sha
    from eval.action4_corner_scoring_inputs import _source
    if (cfg.get("role")!="A0" or cfg.get("a0_variant")!="a0_corner_image" or cfg.get("corner_enabled") is not True
            or cfg.get("sam_enabled") is not True or cfg.get("corner_eval_engineering_fixture") is True
            or cfg.get("eval_split","eval")!="eval" or cfg.get("eval_indices") is not None):
        raise ValueError("Explicit full-domain unadmitted corner eval configuration required")
    if CODE_ROOT==R or cfg.get("require_immutable_source") is not True or not cfg.get("run_spec"):
        raise ValueError("Corner evaluation requires actual frozen-source launcher")
    _origins(CODE_ROOT)
    verify_running_snapshot(cfg)
    _resource_guard(cfg,device)
    source_binding=cfg["corner_eval_source"];source=_source(source_binding,False)
    if contained_path(R,source["code_root"],exists=True)!=CODE_ROOT or source.get("run_spec")!=cfg["run_spec"]:
        raise ValueError("Corner eval source manifest must belong to same RunSpec/CODE_ROOT")
    spec=_v2_checked({"path":cfg["run_spec"],"sha256":_v2_sha(cfg["run_spec"])})
    manifest_path=contained_path(R,source_binding["path"],exists=True)
    if not manifest_path.is_relative_to(CODE_ROOT) or spec["files"].get(str(manifest_path.relative_to(CODE_ROOT)))!=source_binding["sha256"]:
        raise ValueError("RunSpec does not bind corner source manifest")
    return source_binding

def _preflight(cfg,snapshot,data,device):
    from data.action4_component_data import ComponentData
    from data.action4_sam_native_export import _v2_checked,_v2_read,_v2_sha,canonical
    source_binding=validate_corner_eval_environment(cfg,device)
    workers=cfg.get("eval_loader_workers",4)
    if type(workers) is not int or not 0<=workers<=8:raise ValueError("Bounded explicit eval loader workers required")
    if type(data) is not ComponentData or data.role!="B0" or len(data)!=650 or len(data.records)!=650:
        raise ValueError("Production evaluation requires actual ComponentData full650 image domain")
    if data.units!=[(i,None) for i in range(650)]:raise ValueError("Image evaluation cannot use instance/subset units")
    receipt=_v2_checked({"path":cfg["validation_receipt"],"sha256":cfg["validation_receipt_sha256"]})
    annotations={"path":receipt["splits"]["eval"]["manifest"],"sha256":receipt["splits"]["eval"]["sha256"]}
    records=_v2_read(annotations,lambda f:[json.loads(line) for line in f if line.strip()])
    if records!=data.records or any(r["split"]!="eval" for r in records):raise ValueError("Actual evaluator data differs from validated eval manifest")
    if len({r["sample_id"] for r in records})!=650:raise ValueError("Duplicate eval image")
    cad=np.asarray(records[0]["corners_3d_mm"],float)
    if cad.shape!=(8,3) or not np.isfinite(cad).all() or any(not np.array_equal(r["corners_3d_mm"],cad) for r in records):
        raise ValueError("Fixed canonical8 must be shared across the eval domain")
    path=contained_path(R,snapshot,exists=True);snap={"path":str(path.relative_to(R)),"sha256":_v2_sha(path)}
    import torch
    state=_v2_read(snap,lambda f:torch.load(f,map_location="cpu",weights_only=True))
    if (state.get("role")!="A0" or state.get("schema")!="action4.stage_bundle.corner_candidate.v1"
            or state.get("config_id")!=canonical(cfg) or state.get("config")!=cfg):
        raise ValueError("Evaluation checkpoint differs from immutable effective config")
    del state
    b0=contained_path(R,cfg["evaluation_counterpart_bundle"],exists=True)
    if _v2_sha(b0)!=cfg["evaluation_counterpart_sha256"]:raise ValueError("Frozen B0 counterpart changed")
    if cfg.get("b0_full_box_quality_receipt") is None:raise ValueError("Explicit report-bound full-box quality receipt required")
    from eval.action4_corner_prediction import validate_full_box_quality_gate
    cache_index=_v2_checked(cfg["sam_eval_cache"]["manifest"])
    producer=_v2_checked(cache_index["producer_receipt"])
    from eval.action4_corner_quality_bridge import validate_eval_quality_dependencies
    validate_eval_quality_dependencies(cfg,producer["source_contract"]["predictions"])
    _origins(CODE_ROOT)
    return snap,annotations,source_binding

def _global_predictions(payloads,ready):
    field="candidate_instances" if ready else "raw_instances"
    return [{"sample_id":p["image_id"],"instances":copy.deepcopy(p[field])} for p in payloads]

def _execute(cfg,snapshot,destination,data,*,device,engineering,factory,scorer,calibrator,bindings):
    from eval.action4_prefetch import ordered_items
    from data.action4_sam_cache_provider import native_rgb
    snapshot_binding,annotations,source_binding=bindings
    out=contained_path(R,destination,exists=False)
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):raise FileExistsError("Refuse reuse of completed/partial eval output")
    else:out.mkdir(parents=True)
    stage="initialization";started=time.time();clock_started=time.monotonic()
    timings={"input_wait":0.0,"context":0.0,"prediction":0.0};completed=0
    _progress(out,stage,completed,len(data),clock_started,timings)
    _write(out/"EVALUATION_STARTED.json",{"schema":"action4.corner_eval_transaction.v1","engineering_fixture":engineering,
        "snapshot":snapshot_binding,"corner_eval_source":source_binding,"complete":False})
    try:
        pipeline=factory(cfg,snapshot,device)
        if not engineering:
            from data.action4_sam_native_export import _v2_sha
            _origins(CODE_ROOT)
            _resource_guard(cfg,device,check_headroom=False)
            if pipeline.identity["models"]["A0"]["sha256"]!=snapshot_binding["sha256"] or _v2_sha(snapshot_binding["path"])!=snapshot_binding["sha256"]:
                raise ValueError("Loaded A0 differs from pinned eval snapshot")
            if pipeline.identity.get("engineering_fixture") is not False:raise ValueError("Production cannot use fixture inference")
            if cfg.get("b0_scoring_policy",pipeline.identity["B0_scoring_policy"])!=pipeline.identity["B0_scoring_policy"]:
                raise ValueError("Requested and actual B0 scoring policy differ")
        if cfg.get("capacity_barrier") is not None:
            from model.training.action4_capacity_barrier import wait_ready
            stage="capacity_barrier"
            _progress(out,stage,completed,len(data),clock_started,timings)
            wait_ready(cfg,"eval",int(os.environ["ACTION4_CAPACITY_WORKER_ID"]))
        stage="prediction";payloads=[];common=None
        canonical=np.asarray(data.records[0]["corners_3d_mm"],float)/1000
        tick=time.monotonic()
        _progress(out,stage,completed,len(data),clock_started,timings)
        for index,item in ordered_items(data,workers=cfg.get("eval_loader_workers",4)):
            timings["input_wait"]+=time.monotonic()-tick
            if index!=len(payloads) or item["record"]["sample_id"]!=data.records[index]["sample_id"]:
                raise ValueError("Ordered image evaluator changed sample identity")
            record=item["record"];rgb=native_rgb(item["image"])
            tick=time.monotonic()
            context=pipeline.corner_context(record["sample_id"],rgb)
            timings["context"]+=time.monotonic()-tick
            tick=time.monotonic()
            # No GT instance, pose, box, mask, error or calibration decision enters inference.
            payload=pipeline.predict_corner(rgb,np.asarray(record["camera"]["K"],float),canonical,
                sam_context=context,diagnostic_candidate=True)
            timings["prediction"]+=time.monotonic()-tick
            if payload["image_id"]!=record["sample_id"] or payload.get("quality_ready") is not False or payload["instances"]!=payload["raw_instances"]:
                raise ValueError("Prediction bridge identity/formal-raw contract violated")
            if common is None:common=copy.deepcopy(payload["identity"])
            elif common!=payload["identity"]:raise ValueError("Effective inference config changed across eval images")
            payloads.append(payload);completed=len(payloads)
            if completed%25==0 or completed==len(data):
                _progress(out,stage,completed,len(data),clock_started,timings)
            tick=time.monotonic()
        if len(payloads)!=len(data) or (not engineering and len(payloads)!=650):raise ValueError("Incomplete prediction image coverage")
        if not engineering:
            from data.action4_sam_native_export import _v2_sha
            if _v2_sha(snapshot_binding["path"])!=snapshot_binding["sha256"]:raise ValueError("Eval checkpoint changed during inference")
            _origins(CODE_ROOT)
        stage="freeze_predictions"
        _progress(out,stage,completed,len(data),clock_started,timings)
        execution=None
        if not engineering:
            execution=_write(out/"INFERENCE_EXECUTION_RECEIPT.json",{"schema":"action4.corner_inference_execution.v1",
                "status":"COMPLETE_PREDICTION_ONLY_FULL650","run_spec":cfg["run_spec"],"code_root":str(CODE_ROOT.relative_to(R)),
                "source_manifest":copy.deepcopy(source_binding),"module_origins":_origins(CODE_ROOT),"snapshot":snapshot_binding,
                "images":len(payloads),"device":str(device),"pid":os.getpid(),"GT_passed_to_inference":False},readonly=True)
        artifact=_write(out/"PREDICTION_ARTIFACT.json",{"schema":"action4.corner_prediction_artifact.v1",
            "source_manifest":copy.deepcopy(source_binding),"effective_identity":common,"results":payloads,
            "inference_execution_receipt":execution},readonly=True)
        raw_binding=_write(out/"RAW_PREDICTIONS.json",_global_predictions(payloads,False),readonly=True)
        candidate_binding=_write(out/"CANDIDATE_PREDICTIONS.json",_global_predictions(payloads,True),readonly=True)
        stage="scoring"
        _progress(out,stage,completed,len(data),clock_started,timings)
        scoring=scorer(artifact,annotations,source_binding,out/"scoring",diagnostic=engineering)
        from eval.action4_direct_corner_metrics import score_direct_head
        direct_report=score_direct_head(payloads,data.records)
        direct_binding=_write(out/"DIRECT_HEAD_REPORT.json",direct_report,readonly=True)
        stage="calibration"
        _progress(out,stage,completed,len(data),clock_started,timings)
        calibration=calibrator(scoring["domain"],scoring["run_binding"],scoring["records"],scoring["raw_report"],scoring["candidate_report"])
        calibration=copy.deepcopy(calibration)
        if engineering:
            calibration.update(quality_ready=False,quality_gate_passed_for_selection=False,
                final_path="raw",final_report=copy.deepcopy(scoring["raw_report"]),engineering_fixture=True)
            calibration["identity"]=hashlib.sha256(json.dumps({k:v for k,v in calibration.items() if k!="identity"},
                sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        ready=calibration.get("quality_ready") is True and calibration.get("quality_gate_passed_for_selection") is True and not engineering
        raw_report=calibration["raw_report"]["pose_report"];candidate_report=calibration["candidate_report"]["pose_report"]
        final=copy.deepcopy(candidate_report if ready else raw_report)
        if final is None:raise ValueError("Final original pose report is unmeasured")
        stage="commit"
        _progress(out,stage,completed,len(data),clock_started,timings)
        raw_report_binding=_write(out/"RAW_REPORT.json",raw_report,readonly=True)
        candidate_report_binding=_write(out/"CANDIDATE_REPORT.json",candidate_report,readonly=True)
        calibration_binding=_write(out/"CALIBRATION_RECEIPT.json",calibration,readonly=True)
        final_binding=_write(out/"FINAL_PREDICTIONS.json",_global_predictions(payloads,ready),readonly=True)
        final.update(direct_head_report=direct_report,direct_head_report_artifact=direct_binding,
            corner_quality_ready=ready,quality_gate_passed_for_selection=ready,
            selection_status="pending_original_pose_protocol_reference_guard" if ready else "not_eligible_quality_gate_failed",
            calibration_receipt=calibration_binding,prediction_artifact=artifact,inference_execution_receipt=execution,corner_eval_source=copy.deepcopy(source_binding),
            raw_predictions=raw_binding,candidate_predictions=candidate_binding,final_predictions=final_binding,
            raw_pose_report=raw_report,candidate_pose_report=candidate_report,
            raw_pose_report_artifact=raw_report_binding,candidate_pose_report_artifact=candidate_report_binding,
            corner_eval_engineering_fixture=engineering,evaluation_images=len(payloads),
            corner_final_path="candidate" if ready else "raw",evaluation_wall_seconds=time.time()-started)
        report_binding=_write(out/"REPORT.json",final,readonly=True)
        _write(out/"COMPLETE.json",{"status":"CORNER_EVAL_COMPLETE","engineering_fixture":engineering,"images":len(payloads),
            "report":report_binding,"prediction_artifact":artifact,"quality_ready":ready,"best_or_default_published":False},readonly=True)
        return final
    except BaseException as error:
        _progress(out,"failed:"+stage,completed,len(data),clock_started,timings,error=error)
        _write(out/"FAILED.json",{"status":"CORNER_EVAL_FAILED","stage":stage,"error_type":type(error).__name__,
            "error":str(error),"complete":False,"traceback":traceback.format_exc()})
        raise

def run_corner_evaluation(cfg,snapshot,destination,data,*,device="cuda:0"):
    """Production full650 frozen-source evaluation; no injectable inference/scoring."""
    bindings=_preflight(cfg,snapshot,data,device)
    from eval.action4_staged_pipeline import StagedPipeline
    from eval.action4_corner_scoring_inputs import build_scoring_inputs
    from eval.action4_corner_calibration import calibrate
    def factory(config,checkpoint,target_device):
        if str(target_device)=="cuda:0":
            from model.training.action4_resource_policy import apply_allocator_budget
            apply_allocator_budget(config,"evaluation",0)
            _resource_guard(config,target_device,check_headroom=False)
        return StagedPipeline.from_corner_bundles(checkpoint,config["evaluation_counterpart_bundle"],
            config["sam_eval_cache"],target_device,engineering_fixture=False,
            full_box_quality_receipt=config["b0_full_box_quality_receipt"])
    return _execute(cfg,snapshot,destination,data,device=device,engineering=False,factory=factory,
        scorer=build_scoring_inputs,calibrator=calibrate,bindings=bindings)

def run_engineering_evaluation_fixture(cfg,snapshot,destination,data,*,fixture_factory,fixture_scorer,fixture_calibrator):
    """Explicit CPU-only test seam. It cannot publish a ready or deployment result."""
    if cfg.get("mode")!="integration" or cfg.get("corner_eval_engineering_fixture") is not True:
        raise ValueError("Explicit integration engineering fixture permission required")
    bindings=({"path":str(snapshot),"sha256":"fixture_not_production"},cfg.get("fixture_annotations"),cfg.get("corner_eval_source"))
    return _execute(cfg,snapshot,destination,data,device="cpu",engineering=True,factory=fixture_factory,
        scorer=fixture_scorer,calibrator=fixture_calibrator,bindings=bindings)

def export_unchanged_engineering_snapshot(cfg):
    """Evaluation-only config rebinding; preserve every inherited model tensor."""
    if cfg.get("mode")!="integration" or cfg.get("evaluation_only_engineering") is not True:
        raise ValueError("Explicit engineering evaluation-only export required")
    import torch
    from model.training.action4_stage2_core import construct,bundle_snapshot
    from model.training.action4_runtime import save_checkpoint_atomic
    model,parent=construct(cfg,"cpu")
    if parent.get("schema")!="action4.stage_bundle.corner_candidate.v1":raise ValueError("Existing full corner engineering bundle required")
    state=bundle_snapshot(model,parent,cfg,0)
    for group in ("model","backbone","sam_adapter","corner_head"):
        if set(state[group])!=set(parent[group]):raise ValueError("Engineering export changed state domain")
        for key,value in state[group].items():
            old=parent[group][key]
            if not (torch.equal(value,old) if torch.is_tensor(value) else value==old):raise ValueError("Engineering export changed weights: "+group+"."+key)
    out=contained_path(R,cfg["output"],exists=True);path=out/"ENGINEERING_INITIAL.pt"
    if path.exists():raise FileExistsError("Preserve prior engineering export")
    save_checkpoint_atomic(path,state)
    _write(out/"EXPORT.json",dict(status="EXACT_PARENT_WEIGHTS_EXPORTED_FOR_FULL_EVAL_NOT_TRAINING",snapshot=str(path.relative_to(R)),parent=cfg["parent_bundle"],parent_step=parent["selected_step"],optimizer_updates=0,formal_training_complete=False))
    return str(path.relative_to(R))


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--snapshot");choice.add_argument("--export-initial",action="store_true")
    args=parser.parse_args()
    from data.action4_sam_native_export import _v2_checked,_v2_sha
    from data.action4_component_data import ComponentData
    config=_v2_checked({"path":args.config,"sha256":_v2_sha(args.config)})
    # Source checks run before constructing any model; data reads do not invoke inference.
    validate_corner_eval_environment(config,"cuda:0")
    if args.export_initial:args.snapshot=export_unchanged_engineering_snapshot(config)
    data=ComponentData(config["validation_receipt"],"eval","B0")
    data._validation_receipt=contained_path(R,config["validation_receipt"],exists=True)
    report=run_corner_evaluation(config,args.snapshot,args.output,data)
    print(json.dumps({"status":"CORNER_EVAL_COMPLETE","corner_quality_ready":report["corner_quality_ready"],"output":args.output}))

if __name__=="__main__":main()
