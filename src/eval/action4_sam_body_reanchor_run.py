"""Causal B0/SAM body-scope reanchoring, with an explicit first-switch review gate.

The source run supplies reusable generation0 masks, never labels. Subsequent
generations are genuine native sessions. Only one generation owns each output
frame; replay context never replaces already committed prefix frames.
"""
import json,time,traceback
from pathlib import Path
import numpy as np
from eval.action4_sam_sequence_backend import SequenceBackend
from eval.action4_sam_image_eval import validate_source_snapshot,validate_live_authority,save_raw
from eval.action4_sam_full_review import (
    ReceiptVerifier,read_json,atomic_json,candidates_for_frame,masks_for_frame,
    validate_completion,ReplayAudit,checked_path,ROOT)
from model.action4_sam_supported_boxes import SamSupportedBoxes
from model.action4_sam_body_reanchor import BodyReanchorGate
from model.action4_sam_generation import CandidateReplay,native_masks
from model.action4_sam_seed_extent import b0_body_scope_prompt

def run(config):
    manifest=validate_source_snapshot(config,require_execution=True)
    validate_live_authority(config)
    out=checked_path(config["output"]);out.mkdir(exist_ok=False)
    verifier=ReceiptVerifier()
    baseline=checked_path(config["reanchor"]["baseline_job"])
    baseline_manifest=verifier.bound_json(config["reanchor"]["baseline_manifest"])
    oldcfg=read_json(verifier.verify(baseline/"source/config.json",baseline_manifest["files_sha256"]["config.json"]))
    region=verifier.bound_json(oldcfg["region"])
    if len(region["frames"])!=2850:raise ValueError("Original2850 frame domain required")
    oldsession=validate_completion(read_json(baseline/"run/COMPLETE.json"),read_json(baseline/"run/CLOSE.json"))
    b0=verifier.bound_json(oldcfg["b0_evidence"])
    if b0["b0"]!=config["b0"]:raise ValueError("Frozen B0 counterpart changed")
    pools_path=verifier.verify(b0["pools"]["path"],b0["pools"]["sha256"])
    pools=[json.loads(line) for line in pools_path.open()]
    if len(pools)!=2850:raise ValueError("B0 frame coverage incomplete")
    seeds=verifier.bound_json(oldcfg["instance_seed_sources"]["track_00"]["seeds"])["seeds"]
    mapping={"track_00":0,"track_01":1}
    if {s["track_id"] for s in seeds}!=set(mapping):raise ValueError("Seed map differs")
    mode=config["reanchor"].get("geometry_mode","sam_primary_v1")
    def make_tracker(items):
        if mode=="sam_b0_fallback_v2":
            for item in items:
                appearance=np.asarray(item.get("initial_appearance128",[]),float)
                if appearance.shape!=(128,) or not np.isfinite(appearance).all() or np.linalg.norm(appearance)==0:
                    raise ValueError("Fallback mode requires actual frozen B0 seed appearance")
        tracker=SamSupportedBoxes(items,window=5,geometry_mode=mode)
        if mode=="sam_b0_fallback_v2" and tracker.fallback_policy!=config["reanchor"]["fallback_policy"]:
            raise ValueError("Consumed fallback policy differs from frozen configuration")
        return tracker
    tracker=make_tracker(seeds)
    gate=BodyReanchorGate(mapping,source_session_id=oldsession)
    backend=SequenceBackend(config);stream=None;live_sid=None
    generation=0;events=[];audit=ReplayAudit();started=time.time();last_frame=-1
    selected=out/"FUSED_BOXES.partial.jsonl"
    def frame_masks(raw,fid):
        return native_masks(raw,frame_id=fid,height=region["height"],width=region["width"],native_obj_map=mapping)
    def persist(raw,sid,gen,role):
        fid=raw["frame_index"];masks=frame_masks(raw,fid)
        d=out/f"generation_{gen:03d}"/f"frame_{fid:06d}";d.mkdir(parents=True,exist_ok=False)
        history=backend.history_observations[-1];scores=backend.tracking_score_observations[-1]
        if (history["frame"]!=fid or max(history["counts"])>15
                or scores["frame"]!=fid or scores["session_id"]!=sid):
            raise ValueError("Actual native session/history identity differs")
        frame=region["frames"][fid];receipt=save_raw(raw,d)
        st={"schema":"action4.sam.generation_frame.v1","frame":fid,"global_frame_id":fid,
            "pts":frame["pts"],"time_base":region["time_base"],"pixel_space_id":region["pixel_space_id"],
            "RGB_sha256":frame["sha256"],"session_id":sid,"generation":gen,
            "raw":receipt,"history":history,"native_scores":scores,"role":role,
            "native_obj_map":mapping,"unavailable":[k for k,v in masks.items() if v is None or not v.any()],
            "accuracy_accepted":False}
        atomic_json(d/"STATUS.json",st)
        return masks,str((d/"STATUS.json").relative_to(ROOT))
    def fused_row(fid,model,masks,status_ref,sid,gen):
        frame=region["frames"][fid]
        fused=model.update(fid,frame["pts"]*region["time_base"][0]/region["time_base"][1],
            candidates_for_frame(pools[fid],frame),masks)
        return {"frame":frame,"status":"replayed","fused":fused,
            "native_unavailable":[k for k,v in masks.items() if v is None or not v.any()],
            "native_status":status_ref,"session_id":sid,"generation":gen,
            "A0_connected":False,"full_box_accuracy_verified":False}
    failure=None;closed=[]
    try:
        with selected.open("x") as sink:
            for fid in range(2850):
                frame=region["frames"][fid]
                if fid<4:
                    row={"frame":frame,"status":"initialization","fused":{"frame_id":fid,"tracks":[],"mode":"video_inference"},"native_unavailable":[]}
                else:
                    if live_sid is None:
                        directory=baseline/"run"/f"frame_{fid:06d}"
                        st=read_json(directory/"STATUS.json");raw=verifier.raw(st["raw"],directory)
                        masks,_=masks_for_frame(st,raw,frame,region,oldsession)
                        ref=str((directory/"STATUS.json").relative_to(ROOT));sid=oldsession
                    else:
                        raw=next(stream);masks,ref=persist(raw,live_sid,generation,"current")
                        if raw["frame_index"]!=fid:raise ValueError("Current native output out of order")
                        sid=live_sid
                    row=fused_row(fid,tracker,masks,ref,sid,generation)
                    proposal=gate.observe(row)
                    if proposal["action"]=="propose_reanchor":
                        if len(events)>=config["reanchor"]["maximum_attempts"]:raise ValueError("Explicit replay budget exhausted")
                        if stream is not None:stream.close();stream=None
                        trial_sid=None;trial_stream=None
                        eventdir=out/"events"/proposal["event_id"].replace(":","_");eventdir.mkdir(parents=True,exist_ok=False)
                        atomic_json(eventdir/"PROPOSAL.json",proposal)
                        # Failures stop this run; no failed generation is silently reused.
                        trial_generation=proposal["proposed_generation"]
                        attempt_root=out/f"generation_{trial_generation:03d}"
                        if attempt_root.exists():
                            raise ValueError("Failed generation retained; new unique generation policy required")
                        committed=False
                        try:
                            trial_sid=backend.call({"type":"start_session","resource_path":str(checked_path(config["joint_fullvideo"]["frame_dir"])),"offload_video_to_cpu":True})["session_id"]
                            seed_frame=proposal["seed_frame"]
                            for tid,oid in mapping.items():
                                t=proposal["seed_tracks"][tid]
                                request,proof=b0_body_scope_prompt(trial_sid,seed_frame,oid,t["raw_detection"],t["geometry_trace"]["raw_calibration_evidence"],region["width"],region["height"])
                                response=backend.call(request)
                                d=eventdir/f"prompt_obj{oid}";d.mkdir()
                                atomic_json(d/"RECEIPT.json",{"request":request,"proof":proof,"raw":save_raw(response,d)})
                            trial_tracker=make_tracker(
                                [{"track_id":tid,"bbox_native":proposal["seed_tracks"][tid]["raw_detection"]["bbox_full_xyxy"],
                                  "initial_appearance128":proposal["seed_tracks"][tid]["raw_detection"].get("appearance_128",[])} for tid in mapping])
                            validator=CandidateReplay(session_id=trial_sid,generation=trial_generation,
                                seed_frame=seed_frame,trigger_frame=fid,native_obj_map=mapping,height=region["height"],width=region["width"])
                            trial_stream=backend.propagate_forward({"type":"propagate_in_video","session_id":trial_sid,"propagation_direction":"forward","start_frame_index":seed_frame})
                            candidate=None
                            for trial_raw in trial_stream:
                                tf=trial_raw["frame_index"]
                                validator.consume(trial_sid,trial_generation,trial_raw)
                                tm,tr=persist(trial_raw,trial_sid,trial_generation,"replay_context" if tf<fid else "responsibility_start")
                                candidate=fused_row(tf,trial_tracker,tm,tr,trial_sid,trial_generation)
                                if tf==fid:break
                            trial_stream.close();trial_stream=None
                            binding=validator.binding()
                            if candidate is None or candidate["frame"]["global_frame_id"]!=fid:raise ValueError("Candidate current frame missing")
                            if any(t["status"] not in ("observed","sam_supported") for t in candidate["fused"]["tracks"]):
                                raise ValueError("Candidate joint support ambiguous/unavailable")
                            # Do not select old per-object masks to patch new generation.
                            oldclose={"historical_closed_session":oldsession} if live_sid is None else backend.call({"type":"close_session","session_id":live_sid})
                            if live_sid is not None and oldclose.get("is_success") is not True:
                                raise RuntimeError("Previous native close was not acknowledged")
                            live_sid=None  # previous close acknowledged; never close it twice
                            record=gate.resolve(proposal["event_id"],binding=binding)
                            record.update(previous_close=oldclose,new_session=trial_sid,
                                responsibility_start=fid,context_range=[seed_frame,fid],
                                native_output_validation_only=True,accuracy_accepted=False)
                            events.append(record);atomic_json(eventdir/"COMMIT.json",record)
                            live_sid=trial_sid;trial_sid=None;generation=trial_generation
                            tracker=trial_tracker;row=candidate;committed=True
                        except Exception as error:
                            if trial_stream is not None:trial_stream.close();trial_stream=None
                            if trial_sid is not None:
                                result=backend.call({"type":"close_session","session_id":trial_sid})
                                if result.get("is_success") is not True:raise RuntimeError("Failed candidate close unacknowledged") from error
                                trial_sid=None
                            record=(gate.resolve(proposal["event_id"],failure=type(error).__name__+":"+str(error))
                                    if gate.pending is not None else
                                    dict(proposal,outcome="fatal_after_native_commit",failure=str(error)))
                            record["traceback"]=traceback.format_exc();events.append(record)
                            atomic_json(eventdir/"FAILED.json",record)
                            # Fail-stop after a preserved failed attempt until generation
                            # allocation/rollback is root reviewed; no success-empty result.
                            raise RuntimeError("Candidate replay failed; original source retained") from error
                        if live_sid is not None and fid<2849:
                            stream=backend.propagate_forward({"type":"propagate_in_video","session_id":live_sid,"propagation_direction":"forward","start_frame_index":fid+1})
                    if row["generation"]!=gate.generation:raise ValueError("Generation owner differs")
                audit.add(row);sink.write(json.dumps(row,allow_nan=False)+"\n");sink.flush()
                last_frame=fid
                atomic_json(out/"PROGRESS.json",{"last_frame":fid,"frames":fid+1,"expected":2850,
                    "generation":generation,"attempts":len(events),"seconds":time.time()-started,"accuracy_accepted":False})
                if fid==config["reanchor"]["pilot_review_frame"]:
                    if stream is not None:stream.close();stream=None
                    atomic_json(out/"PILOT_READY.json",{"last_frame":fid,"events":events,"source_identity":manifest["identity"],
                        "status":"AWAIT_ROOT_INTERFACE_AND_VISUAL_REVIEW","model_accuracy_accepted":False})
                    wait_start=time.time()
                    allow=out/"ALLOW_FULL.json"
                    while not allow.exists():
                        if time.time()-wait_start>3600:raise RuntimeError("Root review gate timed out; no full run")
                        time.sleep(5)
                    approval=read_json(allow)
                    if approval.get("source_identity")!=manifest["identity"] or approval.get("after_frame")!=fid or approval.get("allow_continue") is not True:
                        raise ValueError("Root continuation identity differs")
                    if live_sid is not None:
                        stream=backend.propagate_forward({"type":"propagate_in_video","session_id":live_sid,"propagation_direction":"forward","start_frame_index":fid+1})
            summary=audit.finish()
        if stream is not None:stream.close();stream=None
        if live_sid is not None:
            result=backend.call({"type":"close_session","session_id":live_sid});closed.append(result);live_sid=None
            if result.get("is_success") is not True:raise RuntimeError("Final native close failed")
        backend.assert_idle()
        selected.replace(out/"FUSED_BOXES.jsonl")
        atomic_json(out/"AUDIT.json",summary)
        atomic_json(out/"COMPLETE.json",{"status":"FULL2850_REANCHOR_CANDIDATE_COMPLETE_NOT_ACCURACY",
            "frames":2850,"events":events,"source_identity":manifest["identity"],"seconds":time.time()-started,
            "closed":True,"new_model_calls":"native SAM only; B0 reused","A0_connected":False,"accuracy_accepted":False})
    except BaseException:
        failure={"last_frame":last_frame,"traceback":traceback.format_exc(),"events":events,"accuracy_accepted":False}
        atomic_json(out/"FAILED.json",failure)
        raise
    finally:
        if stream is not None:stream.close()
        if live_sid is not None:
            result=backend.call({"type":"close_session","session_id":live_sid});closed.append(result)
        backend.assert_idle();backend.dispose()
        atomic_json(out/"CLEANUP.json",{"closed":closed,"failed":failure is not None})
