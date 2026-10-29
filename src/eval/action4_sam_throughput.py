"""One-shot, unscored full-domain throughput diagnostic; never publishes eval selection."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path

def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def configure_compute(torch,cv2,allocator_gib):
    # CUDA properties initialize CUDA too: configure determinism before any such call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(allocator_gib*1024**3/total,0)

def worker(config,worker_id):
    from data.action4_paths import PROJECT_ROOT as R,CODE_ROOT,contained_path
    from model.training.action4_runtime import atomic_json
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    out=contained_path(R,cfg["output"],exists=True)
    allocation=cfg["workers"][worker_id]
    result={"schema":"action4.unscored_throughput_worker.v1","worker_id":worker_id,
        "pid":os.getpid(),"published":False,"GT_scored":False,"completed_ids":[],"image_seconds":[],
        "allocation":allocation,"status":"INITIALIZING","started_unix":time.time()}
    started=time.perf_counter()
    try:
        if sorted(os.sched_getaffinity(0))!=allocation["cpus"]:raise ValueError("CPU allocation differs")
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=allocation["gpu_uuid"]:raise ValueError("GPU binding differs")
        for name,sha in cfg["source_files"].items():
            if digest(CODE_ROOT/name)!=sha:raise ValueError("Frozen source changed: "+name)
        for ref in cfg["bindings"].values():
            if digest(contained_path(R,ref["path"],exists=True))!=ref["sha256"]:raise ValueError("Input binding changed")
        import torch,numpy as np,cv2
        configure_compute(torch,cv2,cfg["allocator_gib"])
        from eval.action4_staged_pipeline import load_pipeline
        from eval.action4_sam_pipeline_inputs import sam_eval_provider
        from data.action4_component_data import ComponentData
        from data.action4_sam_native_export import _v2_checked
        from data.action4_frozen_detection_context import FrozenDetectionContext
        profile=cfg["profile"]
        tick=time.perf_counter()
        pipeline=load_pipeline(profile)
        if pipeline.a0_kind!="joint_pose_sam" or pipeline.roi_expansion!=1.5:raise ValueError("Required SAM/ROI1.5 pipeline differs")
        from eval.action4_surface_snapshot import model_identities
        models=model_identities(pipeline._surface_model_bindings)
        if models["A0"]["sha256"]!=cfg["bindings"]["A0"]["sha256"] or models["B0"]["sha256"]!=profile["frozen_b0_sha256"]:
            raise ValueError("Loaded models differ")
        result["model_load_seconds"]=time.perf_counter()-tick
        tick=time.perf_counter()
        data=ComponentData(cfg["validation_receipt"],"eval","B0")
        ids=[r["sample_id"] for r in data.records]
        if ids!=cfg["image_ids"] or len(ids)!=650 or len(set(ids))!=650:raise ValueError("Full image domain differs")
        provider=sam_eval_provider(pipeline,profile,"eval")
        provider.cache.validate_image_coverage(data.records)
        index=_v2_checked(profile["sam_cache"]["manifest"])
        receipt=_v2_checked(index["producer_receipt"])
        if receipt["source_contract"]["predictions"]["sha256"]!=profile["frozen_b0_predictions"]["sha256"]:
            raise ValueError("SAM/B0 saved source differs")
        rows=_v2_checked(profile["frozen_b0_predictions"])
        frozen={r["sample_id"]:r["instances"] for r in rows}
        if len(frozen)!=len(rows) or set(frozen)!=set(ids):raise ValueError("Frozen B0 domain differs")
        result["validation_seconds"]=time.perf_counter()-tick
        selected=list(range(worker_id,len(data),len(cfg["workers"])))
        if [ids[i] for i in selected]!=allocation["image_ids"]:raise ValueError("Shard changed")
        class Shard:
            def __len__(self):return len(selected)
            def __getitem__(self,i):
                item=data[selected[i]]
                # Only image identity and camera are passed to inference.
                item["record"]={k:item["record"][k] for k in ("sample_id","rgb_sha256","camera")}
                return item
        cad=np.asarray(json.loads(contained_path(R,cfg["bindings"]["cad"]["path"],exists=True).read_text())["corners_3d_mm"])/1000
        calls=[0]
        pipeline.b0.eval().requires_grad_(False)
        hook=pipeline.b0.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        result["initialization_seconds"]=time.perf_counter()-started
        result["status"]="READY"
        atomic_json(out/f"worker_{worker_id:02d}_READY.json",result)
        deadline=time.monotonic()+cfg["ready_timeout_seconds"]
        while not (out/"GO.json").exists():
            if (out/"ABORT.json").exists():raise RuntimeError("Controller aborted before inference")
            if time.monotonic()>deadline:raise TimeoutError("Inference barrier timed out")
            time.sleep(.2)
        from eval.action4_prefetch import ordered_items
        torch.cuda.synchronize();loop_start=time.perf_counter();result["inference_started_unix"]=time.time()
        for _,item in ordered_items(Shard(),workers=1,ahead=2):
            tick=time.perf_counter()
            context=provider.context(item);record=item["record"]
            if context.frame.image_id!=record["sample_id"] or context.frame.source_sha256!=record["rgb_sha256"]:
                raise ValueError("SAM/current RGB identity differs")
            bound=FrozenDetectionContext.from_sam_frame(context.frame,record["camera"]["K"],frozen[record["sample_id"]],
                sam_predictions=context.predictions,sam_sources=context.expected_sources,
                b0_sha256=profile["frozen_b0_sha256"],scoring_policy=profile["b0_scoring_policy"],
                source_cache_identity=profile["frozen_b0_predictions"]["sha256"])
            with torch.inference_mode():
                prediction=pipeline.predict(context.frame.pixels,np.asarray(record["camera"]["K"]),cad,
                    return_box_candidates=True,sam_context=context,frozen_detections=bound)
            torch.cuda.synchronize()
            result["completed_ids"].append(record["sample_id"])
            result["image_seconds"].append(time.perf_counter()-tick)
            result["last_output_instance_count"]=len(prediction["instances"])
            atomic_json(out/f"worker_{worker_id:02d}_PROGRESS.json",
                {"completed_ids":result["completed_ids"],"worker_id":worker_id,"published":False})
        result["inference_wall_seconds"]=time.perf_counter()-loop_start
        hook.remove()
        if calls[0]!=0:raise ValueError("Frozen B0 forward unexpectedly executed")
        result.update(status="PERFORMANCE_SHARD_COMPLETE_NOT_EVALUATION",B0_forward_calls=calls[0],
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException:
        result.update(status="FAILED",traceback=traceback.format_exc())
        raise
    finally:
        result["total_wall_seconds"]=time.perf_counter()-started
        result["ended_unix"]=time.time()
        atomic_json(out/f"worker_{worker_id:02d}_PERFORMANCE.json",result)

def persistent_official_service_probe(config,output,snapshot,*,rounds=2,loader_workers=3,validation_workers=4):
    """Two full scored calls through the actual evaluator; never commits a BEST."""
    from data.action4_paths import PROJECT_ROOT as R,contained_path
    from model.training.action4_runtime import atomic_json
    import torch,cv2
    if type(rounds) is not int or rounds not in (1,2):raise ValueError("Bounded one/two service rounds required")
    if type(loader_workers) is not int or loader_workers not in (1,3):raise ValueError("Registered loader budget required")
    if type(validation_workers) is not int or validation_workers not in (1,4):raise ValueError("Registered validation budget required")
    configure_compute(torch,cv2,2.0)
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    if cfg.get("role")!="A0" or cfg.get("sam_enabled") is not True or cfg.get("roi_expansion")!=1.5:
        raise ValueError("Explicit SAM A0 predictionROI1.5 configuration required")
    if cfg.get("expected_eval_images")!=650:raise ValueError("Full650 service required")
    # Allocation only; exact evaluation/prediction/scoring policies are retained.
    cfg["eval_threads"]=1
    cfg["eval_loader_workers"]=loader_workers
    cfg["sam_eval_cache_validation_workers"]=validation_workers
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False,parents=True)
    snap=contained_path(R,snapshot,exists=True)
    from eval.action4_stage2_eval import load_evaluation_data,evaluate
    result={"schema":"action4.persistent_official_eval_service_probe.v1",
        "published":False,"formal_eval_complete":False,"selection_committed":False,
        "snapshot":{"path":str(snap.relative_to(R)),"sha256":digest(snap)},
        "pid":os.getpid(),"CPU_affinity":sorted(os.sched_getaffinity(0)),
        "pipeline_rebuilt_each_call":True,"payload_proof_cache_process_persistent":True,
        "loader_workers":loader_workers,"validation_workers":validation_workers,"planned_rounds":rounds,"rounds":[]}
    started=time.perf_counter()
    try:
        tick=time.perf_counter();data=load_evaluation_data(cfg)
        result["one_time_data_validation_seconds"]=time.perf_counter()-tick
        for round_id in range(rounds):
            tick=time.perf_counter()
            report=evaluate(cfg,snap,out/f"round_{round_id}",data)
            torch.cuda.synchronize()
            expected_ids=[r["sample_id"] for r in data.records]
            for artifact in ("PREDICTIONS.json","BOX_PREDICTIONS.json"):
                predicted=json.loads((out/f"round_{round_id}/rgb"/artifact).read_text())
                if [r["sample_id"] for r in predicted]!=expected_ids:
                    raise ValueError("Official service output lost or duplicated full650 domain")
            from data.action4_sam_native_export import _PAYLOAD_PROOFS
            result["rounds"].append({"round":round_id,"wall_seconds":time.perf_counter()-tick,
                "images":len(data),"report_path":str((out/f"round_{round_id}/REPORT.json").relative_to(R)),
                "report_sha256":digest(out/f"round_{round_id}/REPORT.json"),
                "payload_proof_count":len(_PAYLOAD_PROOFS),
                "GT_scored":True,"not_a_new_best_selection":True})
            atomic_json(out/"SERVICE_PROGRESS.json",result)
        result["status"]="FULL650_SCORED_SERVICE_CALLS_COMPLETE_NO_SELECTION"
    except BaseException:
        result.update(status="FAILED",traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"]=time.perf_counter()-started
        atomic_json(out/"SERVICE_RECEIPT.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--worker-id",required=True,type=int)
    a=p.parse_args();worker(a.config,a.worker_id)
