"""Short-region B0 evidence and independent persistent SAM sessions; no YOLO."""
import argparse,hashlib,json,os,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_sam_image_eval import (validate_live_authority,validate_source_snapshot,bind_native_output,save_raw,atomic_json)
from eval.action4_sam_sequence_backend import SequenceBackend,SCHEMA
from model.action4_sam_session_contracts import RegionSpec,SeedBinding,SessionLedger
from model.action4_sam_response_adapter import (ResponseEnvelope,MissingOutput,validate_raw_response,ingest_response,MaskRegistrationReceipt)

def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def bound(ref):
    p=contained_path(R,ref["path"],exists=True)
    if sha(p)!=ref["sha256"]:raise ValueError("Bound sequence input changed")
    return json.loads(p.read_text())
def context(cfg, *, allow_face=False):
    sequence_direction(cfg,allow_face=allow_face)
    if cfg.get("schema")!=SCHEMA:raise ValueError("Sequence schema required")
    source=validate_source_snapshot(cfg,require_execution=True)
    doc=bound(cfg["region"])
    region=RegionSpec(doc["region_id"],tuple(f["global_frame_id"] for f in doc["frames"]),
      tuple(f["pts"] for f in doc["frames"]),tuple(doc["time_base"]),cfg["region"]["sha256"],
      doc["pixel_space_id"],doc["width"],doc["height"])
    if len(doc["frames"])!=cfg["sequence"]["frame_count"]:raise ValueError("Sequence count mismatch")
    original=bound(doc["source_manifest"])
    staged=contained_path(R,doc["staged_frames"],exists=True)
    expected_names={f"{i:06d}.png" for i in range(len(doc["frames"]))}
    if {p.name for p in staged.iterdir()}!=expected_names or any(p.is_symlink() for p in staged.iterdir()):
        raise ValueError("Isolated native frame folder contains missing/extra/symlink entries")
    for i,f in enumerate(doc["frames"]):
        source_frame=original["frames"][f["global_frame_id"]]
        if (f["local_index"]!=i or source_frame["frame_id"]!=f["global_frame_id"]
                or (f["image"],f["sha256"],f["pts"])!=(source_frame["image"],source_frame["image_sha256"],source_frame["pts"])):
            raise ValueError("Frame mapping differs from original extraction")
        source_path=contained_path(R,f["image"],exists=True)
        if sha(source_path)!=f["sha256"]:raise ValueError("Original frame changed")
        target=staged/f"{i:06d}.png"
        if not os.path.samefile(source_path,target) and sha(target)!=f["sha256"]:
            raise ValueError("Native loader frame bytes differ")
    return source,doc,region

def b0_evidence(cfg):
    validate_live_authority(cfg)
    source,doc,region=context(cfg)
    resources=cfg["resources"]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=resources["gpu_uuid"]:raise ValueError("Explicit GPU mapping required")
    import subprocess
    info=subprocess.check_output(["nvidia-smi","-i",str(resources["physical_gpu"]),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if info[0].strip()!=resources["gpu_uuid"] or int(info[1])<resources["fresh_free_mib"]:raise ValueError("B0 sequence resource headroom unavailable")
    import torch
    from PIL import Image
    from model.action4_release_bundle import load_bundle
    from eval.action4_b0_scoring import predict_candidate_pool
    torch.set_num_threads(resources["torch_threads"]);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    model,_,state=load_bundle(contained_path(R,cfg["b0"]["path"],exists=True),"B0","cuda:0",expected_sha256=cfg["b0"]["sha256"])
    if state["selected_step"]!=cfg["b0"]["selected_step"]:raise ValueError("B0 checkpoint step differs")
    out=contained_path(R,cfg["output"],exists=False);out.mkdir(parents=True,exist_ok=True)
    target=out/"B0_POOLS.jsonl"
    if target.exists():raise FileExistsError("Preserve existing B0 evidence")
    started=time.monotonic()
    with target.open("x") as stream:
        for f in doc["frames"]:
            with Image.open(contained_path(R,f["image"],exists=True)) as im:rgb=np.array(im.convert("RGB"),copy=True)
            if rgb.shape!=(region.height,region.width,3):raise ValueError("Native frame dimensions changed")
            pool=predict_candidate_pool(model,torch.from_numpy(rgb).permute(2,0,1),state["config"],"cuda:0",cfg["b0"]["scoring_policy"])
            stream.write(json.dumps({"frame":f,"pool":pool},allow_nan=False)+"\n");stream.flush()
            atomic_json(out/"B0_PROGRESS.json",{"frames":f["local_index"]+1,"total":len(doc["frames"])})
    target.chmod(0o444)
    atomic_json(out/"B0_COMPLETE.json",{"status":"B0_REGION_EVIDENCE_COMPLETE","frames":len(doc["frames"]),
       "source_identity":source["identity"],"b0":cfg["b0"],"region":cfg["region"],
       "pools":{"path":str(target.relative_to(R)),"sha256":sha(target)},"seconds":time.monotonic()-started,"GT_used":False})

def deduplicate_seed_candidates(candidates):
    """Group adjacent-grid duplicate evidence; preserve every alias in audit."""
    from model.action4_temporal_association import iou_box
    groups=[]
    for candidate in sorted(candidates,key=lambda p:(-p["score"],p["grid_index"])):
        def same(other):
            if abs(candidate["grid_index"]-other["grid_index"])!=1:return False
            a=np.asarray(candidate["bbox_full_xyxy"]);b=np.asarray(other["bbox_full_xyxy"])
            diagonal=min(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]))
            if iou_box(a,b)<.95 or np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)>.05*diagonal:return False
            av=np.asarray(candidate["appearance_128"]);bv=np.asarray(other["appearance_128"])
            return np.dot(av,bv)/(np.linalg.norm(av)*np.linalg.norm(bv))>=.995
        group=next((g for g in groups if all(same(p) for p in g)),None)
        if group is None:groups.append([candidate])
        else:group.append(candidate)
    return [g[0] for g in groups],[{"representative":g[0]["grid_index"],"aliases":[p["grid_index"] for p in g]} for g in groups if len(g)>1]

def select_seeds(rows,region,source,policy):
    """First three observed frames, strict B0-only provisional identity; no hand-selected objects."""
    from model.action4_guidance_contract import FrameIdentity,SourceIdentity
    from model.action4_temporal_association import TrackEvidence,CandidateEvidence,associate
    if (set(policy)!={"confirmations","score_min","quality_min","max_sessions"}
            or type(policy["confirmations"]) is not int or not 3<=policy["confirmations"]<=15
            or policy["score_min"]!=.9 or policy["quality_min"]!=.8 or policy["max_sessions"]!=2):
        raise ValueError("Unknown integration seed policy")
    confirmations=policy["confirmations"];seed_index=confirmations-1
    # Group only seed-eligible duplicate hypotheses. The raw association pool is unchanged.
    rows=list(rows);duplicate_audit=[]
    for ri,row in enumerate(rows[:confirmations]):
        eligible_rows=[p for p in row["pool"]["association_candidates"]
                       if p["score"]>=.9 and p["refined_quality_score"]>=.8 and p["appearance_available"]]
        representatives,groups=deduplicate_seed_candidates(eligible_rows)
        selected_ids={p["grid_index"] for p in representatives}
        eligible_ids={p["grid_index"] for p in eligible_rows}
        pool=dict(row["pool"])
        pool["association_candidates"]=[p for p in row["pool"]["association_candidates"]
            if p["grid_index"] not in eligible_ids or p["grid_index"] in selected_ids]
        rows[ri]=dict(row,pool=pool)
        if groups:duplicate_audit.append({"frame_id":row["frame"]["global_frame_id"],"groups":groups})

    if len(rows)<confirmations:raise ValueError("Requested actual seed-frame window unavailable")
    model_source=SourceIdentity("b0",source["model_sha256"],source["source_sha256"])
    def eligible(p):return p["score"]>=.9 and p["refined_quality_score"]>=.8 and p["appearance_available"]
    initial=sorted([p for p in rows[0]["pool"]["association_candidates"] if eligible(p)],key=lambda p:(-p["score"],p["grid_index"]))[:8]
    chains={f"track_{i:02d}":[p] for i,p in enumerate(initial)}
    def frame_of(row):
        f=row["frame"];return FrameIdentity(f"frame_{f['global_frame_id']:08d}",f["global_frame_id"],region.pixel_identity,region.width,region.height,f["sha256"])
    for index in range(1,confirmations):
        row=rows[index];frame=frame_of(row);previous=rows[index-1]["frame"]
        tracks=[]
        for tid,chain in chains.items():
            p=chain[-1];b=p["bbox_full_xyxy"]
            tracks.append(TrackEvidence(frame,tid,0,tuple(b),tuple(p["appearance_128"]),float(np.linalg.norm(np.asarray(b[2:])-b[:2])),
                previous["global_frame_id"],previous["pts"]*region.time_base[0]/region.time_base[1],"history_prediction",
                f"seed_chain_{tid}_frame{previous['global_frame_id']}",True,model_source))
        candidates=[CandidateEvidence(frame,str(p["grid_index"]),tuple(p["bbox_full_xyxy"]),tuple(p["appearance_128"]),p["score"],
                  eligible(p),f"frame{frame.frame_id}_grid{p['grid_index']}",model_source)
                    for p in row["pool"]["association_candidates"] if p["appearance_available"]]
        result=associate(frame,row["frame"]["pts"]*region.time_base[0]/region.time_base[1],tracks,candidates)
        bygrid={str(p["grid_index"]):p for p in row["pool"]["association_candidates"]}
        chains={m["track_id"]:chains[m["track_id"]]+[bygrid[m["candidate_id"]]] for m in result["matches"]}
    ordered=sorted(chains.items(),key=lambda item:(-min(x["score"] for x in item[1]),item[0]))
    seeds=[]
    for tid,chain in ordered[:2]:
        p=chain[-1];b=np.asarray(p["bbox_full_xyxy"],float);b=np.clip(b,[0,0,0,0],[region.width,region.height]*2)
        if not (b[2:]>b[:2]).all():continue
        xywh=np.r_[b[:2]/[region.width,region.height],(b[2:]-b[:2])/[region.width,region.height]]
        seeds.append({"track_id":tid,"candidate_id":f"b0:frame{region.frame_ids[seed_index]}:grid:{p['grid_index']}",
            "local_index":seed_index,"global_frame_id":region.frame_ids[seed_index],"box_xywh01":xywh.tolist(),"bbox_native":b.tolist(),
            "evidence_ids":[f"frame{region.frame_ids[i]}:grid:{x['grid_index']}" for i,x in enumerate(chain)],
            "initial_appearance128":p["appearance_128"],"confirmation_scores":[x["score"] for x in chain]})
    return {"seeds":seeds,"duplicate_seed_hypotheses":duplicate_audit,"stable_chains":len(ordered),"budget_unprocessed":max(0,len(ordered)-2),
        "identity_scope":"B0-only high-score multi-frame provisional seed, not certified semantic correctness"}

def load_instance_seed_points(cfg,seed,all_seeds):
    """Reuse verified discovery mask; no ground-truth or guessed box-centre prompt."""
    from model.action4_sam_instance_points import predicted_instance_points
    refs=cfg["instance_seed_sources"][seed["track_id"]]
    prior_cfg=bound(refs["config"]);prior_seeds=bound(refs["seeds"]);load=bound(refs["native_load"])
    if (prior_cfg["region"]["sha256"]!=cfg["region"]["sha256"] or prior_cfg["b0"]!=cfg["b0"]
            or prior_cfg["native"]["weight"]!=cfg["native"]["weight"]
            or prior_cfg["native"]["source_commit"]!=cfg["native"]["source_commit"]
            or load["status"]!="LOADED_ZERO_FINAL_MISSING_UNEXPECTED"):
        raise ValueError("Discovery model/frame source differs")
    old=next(x for x in prior_seeds["seeds"] if x["track_id"]==seed["track_id"])
    for key in ("candidate_id","local_index","global_frame_id","box_xywh01","bbox_native"):
        if old[key]!=seed[key]:raise ValueError("Discovery seed differs")
    binding=bound(refs["binding"]);raw_receipt=bound(refs["raw_receipt"])
    if sequence_direction(cfg) == "forward":
        validate_causal_discovery(cfg,seed,prior_cfg,prior_seeds,refs,binding,raw_receipt)
    if binding["obj_id"] is None or binding["reason"]!="bound_unique_prediction_support":
        raise ValueError("Discovery target was not uniquely bound")
    header=bound({"path":raw_receipt["header"],"sha256":raw_receipt["header_sha256"]})
    if header["frame_index"]!=seed["local_index"]:raise ValueError("Discovery mask frame differs")
    arrays=contained_path(R,raw_receipt["arrays"],exists=True)
    if sha(arrays)!=raw_receipt["sha256"]:raise ValueError("Discovery raw arrays changed")
    with np.load(arrays,allow_pickle=False) as z:
        ids=z["raw__outputs__out_obj_ids"];positions=np.flatnonzero(ids==binding["obj_id"])
        if len(positions)!=1:raise ValueError("Discovery object row ambiguous")
        index=int(positions[0]);prob=float(z["raw__outputs__out_probs"][index])
        if not 0<=prob<=1:raise ValueError("Removed/invalid discovery object")
        mask=z["raw__outputs__out_binary_masks"][index].copy()
    points,labels,evidence=predicted_instance_points(mask,seed["bbox_native"],
       [x["bbox_native"] for x in all_seeds if x["track_id"]!=seed["track_id"]])
    return points,labels,dict(evidence,discovery_refs=refs)




def validate_causal_discovery(cfg, seed, prior_cfg, prior_seeds, refs, binding, raw_receipt):
    """Causal point seeds may only consume authenticated prefix seed responses.

    Requires new explicit refs to EXISTING discovery artifacts. Old both-mode
    consumers keep their prior contract. No point/mask sampling is changed.
    """
    if (prior_cfg.get("discovery_seed_only") is not True
            or prior_cfg.get("discovery_decode_prefix") is not True
            or prior_cfg.get("seed_mode", "box_text") != "box_text"):
        raise ValueError("Causal point seed requires seed-only prefix discovery")
    if prior_cfg.get("input_tensor_cache") is not None:
        raise ValueError("Causal prefix discovery cache lacks a prefix-only decode proof")
    required = ("discovery_resource", "discovery_complete", "discovery_close")
    if any(key not in refs for key in required):
        raise ValueError("Causal discovery requires bound resource/complete/close proof")
    out = contained_path(R,prior_cfg["output"],exists=True)
    folder = out/seed["track_id"]
    fixed = {"seeds":out/"SEEDS.json", "binding":folder/"SEED_BINDING.json",
             "raw_receipt":folder/"RAW_RECEIPT.json", "discovery_resource":out/"DISCOVERY_RESOURCE.json",
             "discovery_complete":out/"NATIVE_COMPLETE.json", "discovery_close":folder/"CLOSE.json"}
    for key, expected in fixed.items():
        if contained_path(R,refs[key]["path"],exists=True) != expected:
            raise ValueError("Causal discovery evidence must belong to fixed seed directory: "+key)
    if (contained_path(R,raw_receipt["header"],exists=True) != folder/"raw.json"
            or contained_path(R,raw_receipt["arrays"],exists=True) != folder/"raw.npz"):
        raise ValueError("Causal discovery raw data is not the actual seed response")
    source=validate_source_snapshot(prior_cfg)
    doc=bound(prior_cfg["region"]);resource=bound(refs["discovery_resource"])
    stop=resource.get("decoded_frames")
    if (type(stop) is not int or stop != seed["local_index"]+1
            or resource.get("operation")!="seed_only" or resource.get("propagation_permitted") is not False
            or resource.get("full_region_frames")!=len(doc["frames"])
            or resource.get("source_frames")!=doc["frames"][:stop]):
        raise ValueError("Causal discovery prefix includes future/unbound frames")
    prefix=contained_path(R,str((out/"discovery_seed_frames").relative_to(R)),exists=True)
    if {p.name for p in prefix.iterdir()}!={f"{i:06d}.png" for i in range(stop)}:
        raise ValueError("Causal discovery actual prefix folder differs")
    for i, frame in enumerate(doc["frames"][:stop]):
        if (frame["local_index"]!=i or frame["global_frame_id"]>seed["global_frame_id"]
                or sha(contained_path(R,frame["image"],exists=True))!=frame["sha256"]
                or (prefix/f"{i:06d}.png").is_symlink()
                or sha(prefix/f"{i:06d}.png")!=frame["sha256"]):
            raise ValueError("Causal discovery prefix RGB/frame identity differs")
    done=bound(refs["discovery_complete"]);close=bound(refs["discovery_close"])
    region=RegionSpec(doc["region_id"],tuple(f["global_frame_id"] for f in doc["frames"]),
        tuple(f["pts"] for f in doc["frames"]),tuple(doc["time_base"]),prior_cfg["region"]["sha256"],
        doc["pixel_space_id"],doc["width"],doc["height"])
    targets=[t for t in done.get("tracks",[]) if t.get("track_id")==seed["track_id"]]
    if (done.get("status")!="DISCOVERY_SEEDS_ONLY_COMPLETE_NOT_TRACKING"
            or done.get("source_identity")!=source["identity"] or done.get("region_identity")!=region.identity
            or len(targets)!=1 or targets[0].get("seed_only") is not True
            or targets[0].get("propagation_executed") is not False
            or targets[0].get("obj_id")!=binding["obj_id"]
            or close.get("released") is not True or close.get("response",{}).get("is_success") is not True):
        raise ValueError("Causal discovery is not completed closed seed-only output")
    # Reconstruct the original full-domain seed request and revalidate raw
    # native row/object binding; a same-frame propagation mask cannot substitute.
    original=next(s for s in prior_seeds["seeds"] if s["track_id"]==seed["track_id"])
    sb=SeedBinding(original["track_id"],region.identity,original["candidate_id"],original["local_index"],
        original["global_frame_id"],tuple(original["box_xywh01"]),tuple(original["evidence_ids"]),"frozen_b0")
    ledger=SessionLedger()
    record=ledger.plan(region,sb,worker_id="sam_sequence_worker",
        device_identity=prior_cfg["resources"]["gpu_uuid"],dependency_identity=source["identity"])
    ledger.begin_create(record.record_id)
    request=binding["request"];sid=request.get("session_id")
    ledger.created(record.record_id,sid)
    expected=ledger.seed_request(record.record_id);expected["text"]=prior_cfg["native"]["text"]
    if request!=expected:
        raise ValueError("Causal discovery raw source request is not the actual seed request")
    from eval.action4_sam_full_review import ReceiptVerifier
    raw=ReceiptVerifier().raw(raw_receipt,folder)
    parsed=validate_raw_response(raw,record=ledger.get(record.record_id),region=region,
        envelope=ResponseEnvelope(sid,region.identity,0,0,"seed","seed",sb.local_index))
    if isinstance(parsed,MissingOutput):raise ValueError("Causal discovery seed response missing")
    candidates=[{"candidate_id":s["candidate_id"],"bbox_sam_native_xyxy":s["bbox_native"],
                 "eligible_for_SAM_diagnostic_prompt":True} for s in prior_seeds["seeds"]]
    candidate=next(c for c in candidates if c["candidate_id"]==sb.candidate_id)
    oid,reason,native=bind_native_output(parsed,{"native_shape_hw":[region.height,region.width],
        "candidates":candidates},candidate,prior_cfg["binding"])
    if native!=binding["native_candidates"]:
        from eval.action4_sam_binding_replay import replay_native_diagnostics
        native=replay_native_diagnostics(native,parsed.boxes_xywh01,width=region.width,height=region.height,
            target_box=candidate["bbox_sam_native_xyxy"],producer_load=bound(refs["native_load"]),
            source_identity=source["identity"])
    if (oid!=binding["obj_id"] or reason!=binding["reason"] or native!=binding["native_candidates"]):
        raise ValueError("Causal discovery native seed binding differs")

def sequence_direction(cfg, *, allow_face=False):
    """Explicit causal opt-in; legacy configs retain the both path."""
    direction = cfg.get("sequence", {}).get("direction", "both")
    if direction not in ("both", "forward"):
        raise ValueError("Unknown sequence direction")
    face_mode=cfg.get("temporal_mode")=="causal_forward_A0_face_hypothesis"
    face_consumer=cfg.get("sequence_consumer")=="A0_face_native_pilot"
    if face_mode or face_consumer:
        if not (allow_face and face_mode and face_consumer and direction=="forward"
                and cfg.get("seed_mode")=="instance_points" and isinstance(cfg.get("face_pilot"),dict)):
            raise ValueError("Face forward mode is exclusive to explicit A0 face pilot consumer")
    elif (direction == "forward") != (cfg.get("temporal_mode") == "causal_forward_B0_assist"):
        raise ValueError("Forward direction and causal temporal mode must agree")
    if direction == "forward" and cfg.get("discovery_seed_only", False):
        raise ValueError("Causal tracking is not a seed-only discovery run")
    return direction


def forward_responsibility(region, seed_info):
    """Tail responsibility domain; native session still indexes the full clip."""
    offset = seed_info["local_index"]
    if type(offset) is not int or not 0 <= offset < len(region.frame_ids):
        raise ValueError("Invalid causal seed native index")
    if seed_info["global_frame_id"] != region.frame_ids[offset]:
        raise ValueError("Causal seed local/global binding differs")
    import re
    confirmations=[]
    for evidence in seed_info["evidence_ids"]:
        match=re.fullmatch(r"frame([0-9]+):grid:[0-9]+",evidence)
        if match is None:raise ValueError("Causal seed needs actual B0 confirmation frame IDs")
        confirmations.append(int(match.group(1)))
    if confirmations != list(region.frame_ids[:offset+1]):
        raise ValueError("Causal seed confirmation uses future/missing/nonprefix frames")
    tail = RegionSpec(region.region_id + ":forward:" + seed_info["track_id"],
        region.frame_ids[offset:], region.pts[offset:], region.time_base,
        region.manifest_identity, region.pixel_identity, region.width, region.height)
    proof = {"schema": "action4.forward_responsibility.v1", "causal": True,
        "native_direction": "forward", "native_region_identity": region.identity,
        "responsibility_region": asdict(tail), "seed_native_local_index": offset,
        "seed_confirmation_frames": [{"global_frame_id":f,"pts":p} for f,p in zip(region.frame_ids[:offset+1],region.pts[:offset+1])],
        "mapping": [{"native_local_index": i, "responsibility_local_index": i-offset,
                     "global_frame_id": region.frame_ids[i], "pts": region.pts[i]}
                    for i in range(offset, len(region.frame_ids))],
        # SessionLedger names its transport-independent collection operation
        # propagate_both. This label is not the actual native dispatch direction.
        "collector_operation": "propagate_both", "native_resource_scope": "full_clip"}
    return tail, json.loads(json.dumps(proof))


def forward_response(raw, region, tail, offset, expected_native_index):
    """Map an explicit copy for the existing collector, preserving saved raw."""
    if (not isinstance(raw, dict) or type(raw.get("frame_index")) is not int
            or raw["frame_index"] != expected_native_index
            or not offset <= raw["frame_index"] < len(region.frame_ids)):
        raise ValueError("Causal native response out of order/outside responsibility")
    frame_id, pts = region.frame(raw["frame_index"])
    for key, expected in (("global_frame_id", frame_id), ("pts", pts)):
        if key in raw and (type(raw[key]) is not int or raw[key] != expected):
            raise ValueError("Causal raw global/PTS identity differs")
    local = raw["frame_index"] - offset
    if region.frame(raw["frame_index"]) != tail.frame(local):
        raise ValueError("Causal responsibility global/PTS mismatch")
    normalized = dict(raw, frame_index=local)
    if "region_identity" in raw:
        if raw["region_identity"] != region.identity:
            raise ValueError("Native response region identity differs")
        normalized["region_identity"] = tail.identity
    return normalized

def native_tracks(cfg):
    direction=sequence_direction(cfg);causal=direction=="forward"
    source,doc,region=context(cfg);out=contained_path(R,cfg["output"],exists=True)
    if cfg.get("b0_evidence") is None:
        complete=json.loads((out/"B0_COMPLETE.json").read_text())
        if complete["source_identity"]!=source["identity"]:raise ValueError("B0 producer source differs")
    else:
        complete=bound(cfg["b0_evidence"])
        predecessor=bound(cfg["b0_source_manifest"])
        if complete["source_identity"]!=predecessor["identity"]:
            raise ValueError("Reused B0 source receipt differs")
        # The old B0 inference/source remains an explicit immutable dependency.
        from eval.action4_sam_image_eval import digest
        if digest({k:v for k,v in predecessor.items() if k!="identity"})!=predecessor["identity"]:
            raise ValueError("Reused B0 source manifest changed")
    if (complete["b0"]!=cfg["b0"] or complete["region"]["sha256"]!=cfg["region"]["sha256"]
            or bound(complete["region"])!=doc or complete["frames"]!=len(doc["frames"])):
        raise ValueError("B0 producer model/frame domain differs")
    rows_path=contained_path(R,complete["pools"]["path"],exists=True)
    if sha(rows_path)!=complete["pools"]["sha256"]:raise ValueError("B0 evidence changed")
    rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
    if [row["frame"] for row in rows]!=doc["frames"]:raise ValueError("B0 frame domain differs")
    selection=select_seeds(rows,region,{"model_sha256":cfg["b0"]["sha256"],"source_sha256":complete["source_identity"]},cfg["seed_policy"])
    atomic_json(out/"SEEDS.json",selection)
    if causal and any(s["local_index"] != cfg["seed_policy"]["confirmations"]-1 for s in selection["seeds"]):
        raise ValueError("Causal producer requires the declared common prefix confirmation boundary")
    if not selection["seeds"]:raise ValueError("No automatically confirmed provisional seed; no forced pair")
    seed_only=cfg.get("discovery_seed_only",False)
    if type(seed_only) is not bool or (seed_only and cfg.get("seed_mode","box_text")!="box_text"):
        raise ValueError("Seed-only discovery requires explicit concept bootstrap mode")
    backend=SequenceBackend(cfg);ledger=SessionLedger();results=[]
    seed_candidates=[{"candidate_id":s["candidate_id"],"bbox_sam_native_xyxy":s["bbox_native"],"eligible_for_SAM_diagnostic_prompt":True} for s in selection["seeds"]]
    native_resource=contained_path(R,doc["staged_frames"],exists=True)
    prefix=cfg.get("discovery_decode_prefix",False)
    if type(prefix) is not bool or (prefix and not seed_only):
        raise ValueError("Decode prefix is legal only for seed-only discovery")
    if prefix:
        # No propagation occurs here; only frames through the latest seed are needed.
        stop=max(s["local_index"] for s in selection["seeds"])+1
        prefix_dir=out/"discovery_seed_frames";prefix_dir.mkdir(exist_ok=False)
        for index in range(stop):
            source_frame=contained_path(R,doc["frames"][index]["image"],exists=True)
            os.link(source_frame,prefix_dir/f"{index:06d}.png")
        native_resource=prefix_dir
        atomic_json(out/"DISCOVERY_RESOURCE.json",{"decoded_frames":stop,
            "full_region_frames":len(doc["frames"]),"operation":"seed_only",
            "source_frames":doc["frames"][:stop],"propagation_permitted":False})
    requested=cfg.get("discovery_track_ids")
    if requested is not None:
        if (not seed_only or not isinstance(requested,list) or not requested
            or len(set(requested))!=len(requested)
            or not set(requested)<={s["track_id"] for s in selection["seeds"]}):
            raise ValueError("Discovery subset must explicitly name selected seeds")
    for seed_info in selection["seeds"]:
        if requested is not None and seed_info["track_id"] not in requested:continue
        point_seed=None
        if cfg.get("seed_mode","box_text")=="instance_points":
            point_seed=load_instance_seed_points(cfg,seed_info,selection["seeds"])
        elif cfg.get("seed_mode","box_text")!="box_text":raise ValueError("Unknown native seed mode")
        evidence=tuple(seed_info["evidence_ids"])
        if point_seed is not None:evidence+=("predicted_mask_points:"+point_seed[2]["identity"],)
        responsibility,proof=forward_responsibility(region,seed_info) if causal else (region,None)
        offset=seed_info["local_index"] if causal else 0
        seed=SeedBinding(seed_info["track_id"],responsibility.identity,seed_info["candidate_id"],seed_info["local_index"]-offset,
            seed_info["global_frame_id"],tuple(seed_info["box_xywh01"]),evidence,"frozen_b0")
        record=ledger.plan(responsibility,seed,worker_id="sam_sequence_worker",device_identity=cfg["resources"]["gpu_uuid"],dependency_identity=source["identity"])
        folder=out/seed.track_id;folder.mkdir(exist_ok=False);sid=None;registrations={};stream=None
        try:
            if causal:atomic_json(folder/"RESPONSIBILITY.json",proof)
            ledger.begin_create(record.record_id)
            response=backend.call({"type":"start_session","resource_path":str(native_resource),"offload_video_to_cpu":True})
            sid=response["session_id"];ledger.created(record.record_id,sid)
            if point_seed is None:
                request=ledger.seed_request(record.record_id);request["text"]=cfg["native"]["text"]
            else:
                request=ledger.seed_request(record.record_id,points_xy01=point_seed[0],point_labels=point_seed[1],
                    obj_id=0,point_evidence_id=point_seed[2]["identity"])
                atomic_json(folder/"INSTANCE_POINTS.json",point_seed[2])
            if causal:
                collector_request=dict(request)
                request=dict(request,frame_index=seed_info["local_index"])
                atomic_json(folder/"SEED_REQUEST_MAPPING.json",{"collector_request":collector_request,
                    "native_request":request,"responsibility":proof})
            pending=ledger.get(record.record_id);raw=backend.call(request);atomic_json(folder/"RAW_RECEIPT.json",save_raw(raw,folder))
            normalized=forward_response(raw,region,responsibility,offset,offset) if causal else raw
            env=ResponseEnvelope(sid,responsibility.identity,0,0,"seed","seed",seed.local_index)
            validated=validate_raw_response(normalized,record=pending,region=responsibility,envelope=env)
            if isinstance(validated,MissingOutput):raise ValueError("Native seed response missing")
            candidate=next(c for c in seed_candidates if c["candidate_id"]==seed.candidate_id)
            obj_id,reason,native=bind_native_output(validated,{"native_shape_hw":[region.height,region.width],"candidates":seed_candidates},candidate,cfg["binding"])
            atomic_json(folder/"SEED_BINDING.json",{"request":request,"obj_id":obj_id,"reason":reason,"native_candidates":native})
            ledger.bind_seed(record.record_id,[] if obj_id is None else [obj_id])
            if obj_id is None:raise ValueError("Native seed did not uniquely bind a target")
            if point_seed is not None and obj_id!=0:raise ValueError("Native instance object ID differs from request")
            if seed_only:
                results.append({"track_id":seed.track_id,"obj_id":obj_id,
                    "seed_only":True,"propagation_executed":False})
                continue
            collector=ledger.begin_propagation(record.record_id)
            request={"type":"propagate_in_video","session_id":sid,"propagation_direction":direction,
                     "start_frame_index":seed_info["local_index"]}
            if causal:atomic_json(folder/"PROPAGATION_REQUEST.json",request)
            stream=(backend.propagate_forward(request) if causal else backend.propagate(request))
            for ordinal,raw in enumerate(stream):
                response_dir=folder/f"response_{ordinal:04d}";response_dir.mkdir()
                atomic_json(response_dir/"RAW_RECEIPT.json",save_raw(raw,response_dir))
                def register(req):
                    path=response_dir/"selected_mask.npz"
                    np.savez_compressed(path,mask=req.mask)
                    registrations[req.mask_identity]={"path":str(path.relative_to(R)),"sha256":sha(path),"pixel_sha256":req.pixel_sha256,
                        "binding_identity":req.binding_identity,"binding":json.loads(req.binding_json)}
                    return MaskRegistrationReceipt(req.mask_identity,req.pixel_sha256,req.binding_identity,str(path.relative_to(R)))
                normalized=forward_response(raw,region,responsibility,offset,offset+ordinal) if causal else raw
                envelope=ResponseEnvelope(sid,responsibility.identity,0,0,"propagate_both",f"response_{ordinal:04d}")
                parsed=ingest_response(collector,normalized,envelope=envelope,register_mask=register)
                atomic_json(folder/"PROGRESS.json",{"responses":ordinal+1,"last_local_index":parsed.local_index,"status":parsed.status})
            snapshot=ledger.finish_propagation(record.record_id,collector)
            if not snapshot.responses_complete:raise ValueError("Incomplete whole-region native responses")
            # READY here is transport/object binding, never visual accuracy acceptance.
            ledger.publish(record.record_id,identity_confirmed=True)
            atomic_json(folder/"SNAPSHOT.json",{"snapshot":asdict(snapshot),"identity":snapshot.identity,"registrations":registrations,
                "accuracy_accepted":False,"identity_scope":selection["identity_scope"]})
            results.append({"track_id":seed.track_id,"snapshot":str((folder/"SNAPSHOT.json").relative_to(R)),
                            "snapshot_sha256":sha(folder/"SNAPSHOT.json"),
                            "frames":len(snapshot.frames),"available":snapshot.mask_available_count})
        except BaseException as exc:
            current=ledger.get(record.record_id)
            if current.state!="FAILED":
                kind="oom" if "out of memory" in str(exc).lower() else "schema" if isinstance(exc,ValueError) else "transient"
                ledger.fail(record.record_id,status="inference_error",failure_kind=kind)
            atomic_json(folder/"FAILED.json",{"error":str(exc),"traceback":traceback.format_exc()})
            raise
        finally:
            if stream is not None:stream.close()
            if sid is not None:
                closed=backend.call({"type":"close_session","session_id":sid})
                released=bool(isinstance(closed,dict) and closed.get("is_success") is True)
                ledger.close(record.record_id,released=released)
                atomic_json(folder/"CLOSE.json",{"response":closed,"released":released})
                if not released:raise RuntimeError("Native session not released")
            backend.assert_idle()
    atomic_json(out/"NATIVE_COMPLETE.json",{"status":("DISCOVERY_SEEDS_ONLY_COMPLETE_NOT_TRACKING" if seed_only else "CAUSAL_FORWARD_NATIVE_RESPONSIBILITY_COMPLETE_NOT_ACCURACY" if causal else "CONTINUOUS_NATIVE_SAM_RESPONSES_COMPLETE_NOT_ASSOCIATION_EFFECT_ACCEPTANCE"),
        "region_identity":region.identity,"tracks":results,"ledger":ledger.audit_log,"source_identity":source["identity"],
        "YOLO_used":False,"association_to_current_B0_complete":False,"accuracy_accepted":False})

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--mode",choices=["b0","sam"],required=True);args=p.parse_args()
    cfg=json.loads(contained_path(R,args.config,exists=True).read_text())
    if args.mode=="b0":b0_evidence(cfg)
    else:native_tracks(cfg)
if __name__=="__main__":main()
