"""Bounded full650 continuation using authenticated pilot32 and unchanged inference source."""
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,CODE_ROOT,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_read,_v2_sha,_fingerprint,canonical
from eval.action4_refined_quality_sidecar import (
    RefinedQualitySidecar,_read_manifest,_domain,capture_one,make_sidecar,quality_rows)

def reuse_capture(consumer,image,rgb,base_instances,b0_sha256):
    accepted=consumer.consume(sample_id=image["sample_id"],rgb=rgb,K=image["K"],
        base_instances=base_instances,b0_sha256=b0_sha256)
    if accepted.get("scope")!="pilot32" or accepted.get("admitted") is not False:
        raise ValueError("Only uncalibrated authenticated pilot32 may be reused")
    qualities=accepted["qualities"]
    if len(qualities)!=len(base_instances):raise ValueError("Pilot reuse candidate domain differs")
    for base,q in zip(base_instances,qualities):
        if (q["base_row_sha256"]!=canonical(base) or q["grid_index"]!=base["grid_index"]
            or q["prediction_id"]!=f"b0:{image['sample_id']}:grid:{base['grid_index']}"):
            raise ValueError("Reused quality is not owned by the same base candidate")
    enriched=[dict(base,refined_quality_score=q["refined_quality_score"],
        refined_quality_protocol=copy.deepcopy(q["refined_quality_protocol"])) for base,q in zip(base_instances,qualities)]
    quality_rows(image["sample_id"],base_instances,enriched)
    from data.action4_sam_roi_inputs import array_digest
    return {"sample_id":image["sample_id"],"image":copy.deepcopy(image),"RGB_pixel_sha256":array_digest(rgb),"enriched":enriched}

def validate_ledger(expected_ids,ledger,reuse_ids,*,complete=False):
    for row in ledger:
        ref=row.get("source_receipt")
        if (not isinstance(ref,dict) or set(ref)!={"path","sha256"} or not isinstance(ref["path"],str)
            or not isinstance(ref["sha256"],str) or len(ref["sha256"])!=64
            or type(row.get("candidates")) is not int or row["candidates"]<0):
            raise ValueError("Explicit per-image capture provenance required")
    ids=[row["sample_id"] for row in ledger]
    if len(ids)!=len(set(ids)) or ids!=expected_ids[:len(ids)]:
        raise ValueError("Continuation image domain duplicated/reordered/foreign")
    if complete and ids!=expected_ids:raise ValueError("Incomplete continuation cannot finish")
    if any(row["method"]!=("pilot_reuse" if row["sample_id"] in reuse_ids else "new_B0_forward") for row in ledger):
        raise ValueError("Continuation call/reuse provenance differs")
    return {"completed_images":len(ids),"reused_images":sum(row["method"]=="pilot_reuse" for row in ledger),
        "new_forward_images":sum(row["method"]=="new_B0_forward" for row in ledger)}

def run(config_binding):
    import subprocess,torch
    from PIL import Image
    from model.action4_release_bundle import load_bundle
    cfg=_v2_checked(config_binding)
    out=contained_path(R,cfg["output"],exists=False);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        path=out/name;temp=path.with_suffix(path.suffix+".tmp")
        temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
    captures=[];ledger=[];attempts=0;paused=[];resources=[];start=time.monotonic()
    try:
        if cfg.get("schema")!="action4.refined_quality_continue_config.v1" or cfg.get("scope")!="full650":
            raise ValueError("Explicit full650 continuation required")
        if cfg.get("calibrated") is not False or cfg.get("threshold") is not None:raise ValueError("Continuation cannot calibrate")
        entry=contained_path(R,cfg["orchestrator"]["path"],exists=True)
        if Path(__file__).resolve()!=entry or _v2_sha(entry)!=cfg["orchestrator"]["sha256"]:
            raise ValueError("Continuation orchestrator identity differs")
        source=_v2_checked(cfg["bindings"]["source"])
        if CODE_ROOT==R or contained_path(R,source["source_root"],exists=True)!=CODE_ROOT:
            raise ValueError("Must use identical pilot frozen inference source")
        for name,sha in source["files"].items():
            if _v2_sha(CODE_ROOT/name)!=sha:raise ValueError("Frozen inference source changed")
        if sorted(os.sched_getaffinity(0))!=[28,29] or os.environ.get("CUDA_VISIBLE_DEVICES")!=cfg["gpu_uuid"] or cfg["physical_gpu"]!=5:
            raise ValueError("Approved GPU5/CPU28–29 allocation differs")
        bindings=cfg["bindings"];proofs=[]
        for ref in [*bindings.values(),cfg["pilot_sidecar"],cfg["orchestrator"]]:
            path=contained_path(R,ref["path"],exists=True);before=_fingerprint(path.stat())
            if _v2_sha(path)!=ref["sha256"] or _fingerprint(path.stat())!=before:raise ValueError("Continuation dependency changed")
            proofs.append((path,before,ref["sha256"]))
        pilot_doc=_v2_checked(cfg["pilot_sidecar"])
        if pilot_doc.get("bindings")!=bindings:raise ValueError("Pilot and continuation model/image/base/inference source bindings differ")
        consumer=RefinedQualitySidecar(cfg["pilot_sidecar"],bindings=bindings,
            model_config_id=cfg["model_config_id"],selected_step=cfg["selected_step"],expected_scope="pilot32")
        base=_v2_checked(bindings["base_predictions"]);manifest=_read_manifest(bindings["data_manifest"])
        rows,images=_domain(base,manifest,"full650");ids=[r["sample_id"] for r in rows]
        reuse_ids=[r["image"]["sample_id"] for r in pilot_doc["rows"]]
        if reuse_ids!=ids[:32]:raise ValueError("Pilot must be exact same first32 in full650")
        formal=contained_path(R,cfg["formal_output"],exists=True)
        if cfg["pending_pause_threshold"]!=20:raise ValueError("Registered formal backpressure threshold required")
        def health():
            failed=[str(x.relative_to(formal)) for x in formal.glob("*FAIL*")]+[str(x.relative_to(formal)) for x in (formal/"eval").glob("*.failed.json")]
            pending=[x for x in (formal/"eval").glob("step_*.json") if x.name.count(".")==1 and not x.with_suffix(".done.json").exists()]
            return {"failed":failed,"pending":len(pending)}
        def resource():
            vals=list(map(str.strip,subprocess.check_output(["nvidia-smi","--id=5","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")))
            raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True)
            memory=[int(x.split(",")[1]) for x in raw.splitlines() if int(x.split(",")[0])==os.getpid()]
            if vals[0]!=cfg["gpu_uuid"] or int(vals[1])<16384 or (memory and max(memory)>1536):
                raise RuntimeError("Owned continuation GPU resource margin exceeded")
            value={"elapsed":time.monotonic()-start,"free_mib":int(vals[1]),"own_memory_mib":memory}
            resources.append(value);return value
        def await_dispatch():
            since=None
            while True:
                resource();state=health();resources[-1]["formal"]=state
                if not state["failed"] and state["pending"]<20:
                    if since is not None:paused.append({"start_elapsed":since,"end_elapsed":time.monotonic()-start,"resume_state":state})
                    return
                if since is None:since=time.monotonic()-start
                write("PAUSE_STATE.json",{"status":"PAUSED_NEW_IMAGES_ONLY","formal":state,"completed":len(ledger)})
                if time.monotonic()-start-since>cfg["pause_timeout_seconds"]:raise TimeoutError("Formal failure/backlog stayed blocked; no new images dispatched")
                time.sleep(2)
        await_dispatch()
        if resource()["free_mib"]<1536+16384:raise RuntimeError("Insufficient fresh model-allocation margin")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        model,_,state=load_bundle(bindings["B0"]["path"],"B0","cuda:0",expected_sha256=bindings["B0"]["sha256"])
        model.eval().requires_grad_(False)
        if state["selected_step"]!=9500 or state["config_id"]!=cfg["model_config_id"]:raise ValueError("B0 model identity differs")
        policy=state.get("inference_scoring_policy",state["config"].get("evaluation_scoring_policy","learned"))
        if policy!=cfg["scoring_policy"] or policy!="center":raise ValueError("B0 numerical policy differs")
        versions=tuple(p._version for p in list(model.parameters())+list(model.buffers()))
        write("STARTED.json",{"expected_images":650,"reuse_images":32,"new_forward_images":618,"bindings":bindings,
            "pilot_sidecar":cfg["pilot_sidecar"],"orchestrator":cfg["orchestrator"],"model_eval_mode":not model.training})
        for row,image in zip(rows,images):
            await_dispatch()
            if model.training or any(m.training for m in model.modules()):raise RuntimeError("B0 entered training mode")
            if tuple(p._version for p in list(model.parameters())+list(model.buffers()))!=versions:raise RuntimeError("B0 parameters/buffers changed")
            def decode(stream):
                with Image.open(stream) as im:return np.asarray(im.convert("RGB")).copy()
            rgb=_v2_read({"path":image["image"],"sha256":image["RGB_file_sha256"]},decode)
            if image["sample_id"] in reuse_ids:
                capture=reuse_capture(consumer,image,rgb,row["instances"],bindings["B0"]["sha256"])
                method="pilot_reuse";receipt=cfg["pilot_sidecar"]
            else:
                attempts+=1
                with torch.inference_mode():capture=capture_one(model,state["config"],rgb,image,row["instances"],device="cuda:0",scoring_policy=policy)
                method="new_B0_forward";receipt=bindings["source"]
            resource()
            captures.append(capture)
            ledger.append({"sample_id":image["sample_id"],"method":method,"source_receipt":receipt,
                "RGB_pixel_sha256":capture["RGB_pixel_sha256"],"candidates":len(row["instances"])})
            counts=validate_ledger(ids,ledger,reuse_ids)
            write("PROGRESS.json",{**counts,"actual_forward_attempts":attempts,"expected_images":650})
        counts=validate_ledger(ids,ledger,reuse_ids,complete=True)
        if counts["reused_images"]!=32 or counts["new_forward_images"]!=618 or attempts!=618:raise ValueError("Actual call/reuse mass mismatch")
        for path,before,sha in proofs:
            if _fingerprint(path.stat())!=before or _v2_sha(path)!=sha:raise ValueError("Continuation source changed before finish")
        doc=make_sidecar(bindings,base,manifest,captures,scope="full650",model_config_id=state["config_id"],selected_step=9500)
        doc["production"]={"pilot_sidecar":cfg["pilot_sidecar"],"orchestrator":cfg["orchestrator"],
            "actual_B0_forward_calls":attempts,"reused_images":32,"new_images":618,"ledger":ledger}
        torch.cuda.synchronize();del model;torch.cuda.empty_cache()
        candidate=out/"CANDIDATE_SIDECAR.json";write(candidate.name,doc)
        candidate_ref={"path":str(candidate.relative_to(R)),"sha256":_v2_sha(candidate)}
        full=RefinedQualitySidecar(candidate_ref,bindings=bindings,model_config_id=cfg["model_config_id"],selected_step=9500)
        for row,image in zip(rows,images):
            rgb=_v2_read({"path":image["image"],"sha256":image["RGB_file_sha256"]},decode)
            q=full.consume(sample_id=image["sample_id"],rgb=rgb,K=image["K"],base_instances=row["instances"],b0_sha256=bindings["B0"]["sha256"])
            if q["admitted"] or q["threshold"] is not None:raise ValueError("Unexpected quality admission")
        final=out/"SIDECAR.json";candidate.rename(final)
        values=np.asarray([q["refined_quality_score"] for row in doc["rows"] for q in row["qualities"]])
        write("COMPLETE.json",{"status":"FULL650_CAPTURE_CONSUMER_VERIFIED_UNCALIBRATED","images":650,"candidates":len(values),
            "actual_B0_forward_calls":attempts,"reused_images":32,"new_images":618,"consumer_images_verified":650,
            "sidecar":{"path":str(final.relative_to(R)),"sha256":_v2_sha(final)},
            "quantiles":dict(zip(["0","1","5","25","50","75","95","99","100"],np.percentile(values,[0,1,5,25,50,75,95,99,100]).tolist())),
            "mean":float(values.mean()),"std":float(values.std()),"calibrated":False,"threshold":None,"admitted":False,
            "wall_seconds":time.monotonic()-start,"pause_events":paused})
    except BaseException as error:
        import traceback
        write("PARTIAL_CAPTURE.json",{"captures":captures,"ledger":ledger,"complete":False})
        if hasattr(error,"capture_evidence"):
            def diagnostic(value):
                if isinstance(value,float) and not np.isfinite(value):return {"nonfinite":repr(value)}
                if isinstance(value,dict):return {k:diagnostic(v) for k,v in value.items()}
                if isinstance(value,(tuple,list)):return [diagnostic(v) for v in value]
                return value
            write("FAILED_FRAME.json",diagnostic(error.capture_evidence))
        write("FAILURE.json",{"traceback":traceback.format_exc(),"actual_forward_attempts":attempts,"completed_images":len(ledger),"complete":False,"admitted":False})
        raise
    finally:
        write("RESOURCE_SAMPLES.json",resources)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--config-sha256",required=True)
    a=p.parse_args();run({"path":a.config,"sha256":a.config_sha256})
