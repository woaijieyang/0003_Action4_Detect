"""Explicit frozen-source seed + at most15-frame native face pilot; no accuracy claim."""
import argparse,json,os,hashlib,traceback
from dataclasses import asdict
import numpy as np
from eval.action4_sam_b0_sequence import context,sha,sequence_direction
from eval.action4_sam_image_eval import validate_live_authority,save_raw,atomic_json
from eval.action4_sam_sequence_backend import SequenceBackend
from eval.action4_sam_full_review import ReceiptVerifier,checked_path,ROOT
from model.action4_sam_session_contracts import RegionSpec,SessionLedger
from model.action4_sam_response_adapter import ResponseEnvelope,ingest_response,MaskRegistrationReceipt
from model.action4_sam_face_contract import FaceSeedBinding,FaceMaskThresholds,face_point_request,validate_face_raw
from data.action4_surface_snapshot import load_snapshot
from data.action4_sam_roi_inputs import array_digest

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _mask(verifier,ref,shape):
    with np.load(verifier.verify(ref["path"],ref["sha256"]),allow_pickle=False) as z:
        if set(z.files)!={"mask"}:raise ValueError("Explicit mask-only NPZ required")
        mask=z["mask"].copy()
    if mask.dtype!=np.bool_ or mask.shape!=shape:raise ValueError("Native bool support required")
    return mask

def load_jobs(cfg,doc,region):
    """Authenticate real disk receipts; no synthetic fallback or generated masks."""
    verifier=ReceiptVerifier();pilot=cfg["face_pilot"]
    proposal=verifier.bound_json(pilot["proposal_ref"]);dep=proposal["dependencies"]
    if (proposal.get("schema")!="action4.face_prompt_hypotheses.v1"
        or proposal.get("semantic_gt") is not False or proposal.get("native_inference_executed") is not False):
        raise ValueError("Unexecuted pilot face hypotheses required")
    snapshot=load_snapshot(verifier.verify(pilot["surface_snapshot_ref"]["path"],pilot["surface_snapshot_ref"]["sha256"]))
    m,a=snapshot.metadata,snapshot.arrays
    if m["models"]!=pilot["expected_models"]:
        raise ValueError("Surface snapshot is not the explicitly frozen selected model pair")
    if (m["receipt"]!=dep["surface_receipt"] or m["models"]!=dep["models"]
        or m["roi_id"]!=dep["roi_id"] or not np.array_equal(a["K_native"],dep["K_native"])
        or not np.array_equal(a["rotation"],dep["pose"]["R"])
        or not np.array_equal(a["translation_m"],dep["pose"]["t_m"])):
        raise ValueError("Proposal differs from actual A0 surface snapshot")
    f=doc["frames"][0];body=dep["body"];video=m.get("video_context",{})
    if (dep["frame_id"]!=f["global_frame_id"] or dep["pts"]!=f["pts"]
        or tuple(dep["time_base"])!=region.time_base or dep["pixel_space_id"]!=region.pixel_identity
        or video.get("frame_id")!=f["global_frame_id"]
        or video.get("candidate",{}).get("track_id")!=body["track_id"]
        or body.get("persistent_identity_ready") is not True):
        raise ValueError("Current body/seed frame domain mismatch")
    # RGB file identity is independently bound to the verified clip first frame.
    if m["source_sha256"]!=f["sha256"]:
        raise ValueError("Snapshot RGB file identity mismatch")
    event=video.get("source_ledger",{}).get("project_body_registrations",{}).get(body["track_id"],{})
    if event.get("scoped_body_id")!=body["scoped_body_id"] or event.get("project_generation")!=body["generation"]:
        raise ValueError("Proposal body generation differs from snapshot registration")
    shape=(region.height,region.width)
    mask=_mask(verifier,pilot["body_mask_ref"],shape)
    if array_digest(mask)!=dep["body_mask_sha256"] or dep["body_mask_binding"]["mask_sha256"]!=dep["body_mask_sha256"]:
        raise ValueError("Actual current body mask differs from proposal")
    verifier.bound_json(dep["patch_candidates_ref"])
    selected=pilot["hypothesis_ids"]
    if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected) or len(selected)>3:
        raise ValueError("Explicit1..3 currently proposed visible faces per body required")
    jobs=[];seen_faces=set()
    for identity in selected:
        matches=[h for h in proposal["hypotheses"] if h["hypothesis_identity"]==identity]
        if len(matches)!=1:raise ValueError("Unique selected face hypothesis required")
        h=matches[0]
        if (h["hypothesis_identity"]!=digest({k:v for k,v in h.items() if k!="hypothesis_identity"})
            or h["parent_dependency_identity"]!=digest(dep) or h["status"]!="prompt_hypothesis"
            or h["face_id"] in seen_faces):
            raise ValueError("Face hypothesis identity/status/duplicate logical face differs")
        seen_faces.add(h["face_id"])
        native=np.asarray(h["points_native"],float);points=np.asarray(h["points_xy01"],float)
        if not np.array_equal(native/[region.width,region.height],points):
            raise ValueError("Native and normalized prompt points differ")
        box=dep["roi_spec"]["full_body_bbox"];x0,y0,x1,y1=np.clip(box,[0,0,0,0],[region.width,region.height]*2)
        seed=FaceSeedBinding(body["scoped_body_id"],body["generation"],h["face_id"],pilot["face_generation"],
            region.identity,0,f["global_frame_id"],f["pts"],region.time_base,region.pixel_identity,
            dep["models"]["A0"]["sha256"],f["sha256"],tuple(np.asarray(dep["K_native"]).reshape(-1)),
            pilot["surface_snapshot_ref"],dep["patch_candidates_ref"],pilot["proposal_ref"],
            (x0/region.width,y0/region.height,(x1-x0)/region.width,(y1-y0)/region.height))
        # Optional explicitly bound same-frame support evidence; never synthesize it.
        extras=pilot.get("support_refs",{}).get(str(h["face_id"]),{})
        neighbors=[_mask(verifier,r,shape) for r in extras.get("neighbors",[])]
        others=[_mask(verifier,r,shape) for r in extras.get("other_bodies",[])]
        jobs.append({"seed":seed,"points":points.tolist(),"labels":h["labels"],"body":mask,
                     "neighbors":neighbors,"other_bodies":others,"hypothesis_id":identity})
    return jobs

def run_sessions(cfg,source,doc,region,jobs,backend):
    """Internal execution function; public run() performs all authority checks."""
    out=checked_path(cfg["output"]);ledger=SessionLedger();results=[];accepted_masks=[]
    thresholds=FaceMaskThresholds(**cfg["face_pilot"]["thresholds"])
    for ordinal,job in enumerate(jobs):
        seed=job["seed"];folder=out/("face_%02d"%seed.face_id);folder.mkdir()
        record=ledger.plan(region,seed,worker_id=cfg["resources"]["worker_id"],
                          device_identity=cfg["resources"]["gpu_uuid"],dependency_identity=source["identity"])
        sid=None;stream=None;registrations={}
        try:
            atomic_json(folder/"FACE_SEED.json",asdict(seed))
            ledger.begin_create(record.record_id)
            created=backend.call({"type":"start_session","resource_path":str(checked_path(doc["staged_frames"])),"offload_video_to_cpu":True})
            sid=created["session_id"];ledger.created(record.record_id,sid)
            request=face_point_request(ledger,record.record_id,seed,points_xy01=job["points"],point_labels=job["labels"])
            atomic_json(folder/"PROMPT_REQUEST.json",request)
            raw=backend.call(request)
            atomic_json(folder/"RAW_RECEIPT.json",save_raw(raw,folder))
            ref={"path":str((folder/"RAW_RECEIPT.json").relative_to(ROOT)),"sha256":sha(folder/"RAW_RECEIPT.json")}
            decision=validate_face_raw(raw,seed=seed,record=ledger.get(record.record_id),region=region,
                envelope=ResponseEnvelope(sid,region.identity,0,0,"seed","seed",0),
                points_xy01=job["points"],point_labels=job["labels"],body_support=job["body"],
                neighbor_supports=job["neighbors"],other_body_supports=job["other_bodies"],
                existing_face_masks=accepted_masks,thresholds=thresholds,raw_receipt_ref=ref)
            decision["hypothesis_id"]=job["hypothesis_id"]
            atomic_json(folder/"SEED_DECISION.json",decision)
            if decision["status"]!="predicted_face_support_only":
                ledger.bind_seed(record.record_id,[])
                results.append({"face_id":seed.face_id,"status":"seed_unknown","decision":decision})
                continue
            accepted_masks.append(np.asarray(raw["outputs"]["out_binary_masks"])[0].copy())
            ledger.bind_seed(record.record_id,[0]);collector=ledger.begin_propagation(record.record_id)
            request={"type":"propagate_in_video","session_id":sid,"propagation_direction":"forward","start_frame_index":0}
            atomic_json(folder/"PROPAGATION_REQUEST.json",request)
            stream=backend.propagate_forward(request)
            for index,raw in enumerate(stream):
                directory=folder/("response_%04d"%index);directory.mkdir()
                atomic_json(directory/"RAW_RECEIPT.json",save_raw(raw,directory))
                if type(raw.get("frame_index")) is not int or raw["frame_index"]!=index or index>=len(region.frame_ids):
                    raise ValueError("Native forward response outside exact short ordered domain")
                def register(req):
                    p=directory/"selected_mask.npz"
                    with p.open("xb") as f:np.savez_compressed(f,mask=req.mask)
                    registrations[req.mask_identity]={"path":str(p.relative_to(ROOT)),"sha256":sha(p),
                        "pixel_sha256":req.pixel_sha256,"binding_identity":req.binding_identity,"binding":json.loads(req.binding_json)}
                    return MaskRegistrationReceipt(req.mask_identity,req.pixel_sha256,req.binding_identity,str(p.relative_to(ROOT)))
                ingest_response(collector,raw,envelope=ResponseEnvelope(sid,region.identity,0,0,"propagate_both","response_%04d"%index),register_mask=register)
            snapshot=ledger.finish_propagation(record.record_id,collector)
            if not snapshot.responses_complete:raise ValueError("Incomplete native face forward domain")
            atomic_json(folder/"SNAPSHOT.json",{"snapshot":asdict(snapshot),"identity":snapshot.identity,
                "registrations":registrations,"semantic_face_verified":False,"forward_quality_checked":False})
            results.append({"face_id":seed.face_id,"status":"native_identity_complete_not_face_accuracy",
                            "seed_decision":decision,"snapshot_sha256":sha(folder/"SNAPSHOT.json")})
        except BaseException as exc:
            atomic_json(folder/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc(),"accuracy_accepted":False})
            raise
        finally:
            if stream is not None:stream.close()
            if sid is not None:
                closed=backend.call({"type":"close_session","session_id":sid})
                released=isinstance(closed,dict) and closed.get("is_success") is True
                ledger.close(record.record_id,released=released)
                atomic_json(folder/"CLOSE.json",{"response":closed,"released":released})
                if not released:raise RuntimeError("Face session close failed")
            backend.assert_idle()
    result={"schema":"action4.face_native_pilot.v1","status":"PILOT_NATIVE_ATTEMPTS_COMPLETE_NOT_FACE_ACCEPTANCE",
            "results":results,"ledger":ledger.audit_log,"native_inference_executed":True,
            "config_identity":digest(cfg),"source_identity":source["identity"],
            "region_identity":region.identity,"face_pilot_inputs":cfg["face_pilot"],
            "face_tracking_accepted":False,"semantic_gt":False,"pose_accuracy_verified":False,
            "forward_quality_checked":False, "source_role":"A0_face_hypothesis",
            "temporal_mode":"causal_forward_A0_face_hypothesis"}
    atomic_json(out/"FACE_PILOT_COMPLETE.json",result)
    return result

def run(cfg):
    sequence_direction(cfg,allow_face=True)
    if (cfg.get("schema")!="action4.sam.b0_video_sequence.v1"
        or cfg.get("seed_mode")!="instance_points" or cfg.get("instance_output_bridge") is not True
        or cfg.get("temporal_mode")!="causal_forward_A0_face_hypothesis" or cfg["sequence"].get("direction")!="forward"
        or type(cfg["sequence"].get("frame_count")) is not int or not 2<=cfg["sequence"]["frame_count"]<=15):
        raise ValueError("Explicit2..15-frame forward point-instance pilot required")
    if cfg.get("discovery_seed_only") or cfg.get("input_tensor_cache") is not None:
        raise ValueError("No discovery mode or unverified input cache in face pilot")
    r=cfg["resources"]
    if r["physical_gpu"] not in (4,5,6,7) or os.environ.get("CUDA_VISIBLE_DEVICES")!=r["gpu_uuid"] or set(os.sched_getaffinity(0))!=set(r["cpu_affinity"]):
        raise ValueError("Explicit configured authorized physicalGPU and exact CPU affinity required")
    validate_live_authority(cfg)
    source,doc,region=context(cfg,allow_face=True)  # verifies actual immutable execution origin and every RGB
    jobs=load_jobs(cfg,doc,region)
    out=checked_path(cfg["output"])
    if out.exists() and any(out.iterdir()):raise FileExistsError("Preserve existing native pilot output")
    out.mkdir(parents=True,exist_ok=True)
    backend=SequenceBackend(cfg)
    try:return run_sessions(cfg,source,doc,region,jobs,backend)
    finally:backend.dispose()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--execute-native",action="store_true")
    args=parser.parse_args()
    if not args.execute_native:raise ValueError("Explicit --execute-native required; no automatic dispatch")
    cfg=json.loads(checked_path(args.config).read_text());print(json.dumps(run(cfg)))
if __name__=="__main__":main()
