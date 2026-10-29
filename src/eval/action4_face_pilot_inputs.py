"""Prepare actual selected-A0 face hypotheses from authenticated full-video replay."""
import json,os,hashlib
import numpy as np
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_roi_contract import RoiSpec
from data.action4_sam_roi_inputs import ImageFrame,array_digest
from data.action4_surface_snapshot import save_snapshot
from data.action4_video_sam_snapshot import VideoMaskBinding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_staged_pipeline import StagedPipeline
from eval.action4_face_inputs import prepare_face_inputs
from eval.action4_face_prompt_proposals import FacePatchCandidates,FacePromptThresholds,propose_face_prompts
from model.action4_face_geometry import FaceGeometryContract

def run(config_path):
    import torch
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=cfg["gpu_uuid"] or set(os.sched_getaffinity(0))!=set(cfg["cpus"]):
        raise ValueError("Face preparation resource differs")
    from model.training.action4_resource_policy import validate_device_policy
    validate_device_policy({"training_gpus":[],"eval_gpus":[cfg["gpu"]]},R)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.use_deterministic_algorithms(True)
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    for x in pair["models"].values():provider.verifier.verify(x["path"],x["sha256"])
    pipe=StagedPipeline.from_bundles(pair["models"]["A0"]["path"],pair["models"]["B0"]["path"],device="cuda:0")
    pipe.b0.eval().requires_grad_(False);pipe.b0_scoring_policy=lock["image_eval"]["B0_scoring_policy"]
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000
    cref=cfg["face_contract"];cb=provider.verifier.verify(cref["path"],cref["sha256"]).read_bytes()
    contract=FaceGeometryContract.from_bytes(cb,provider.verifier.verify(lock["CAD"]["path"],lock["CAD"]["sha256"]).read_bytes())
    patches=FacePatchCandidates.from_ref(cfg["patch_candidates"],contract)
    thresholds=FacePromptThresholds(**cfg["thresholds"])
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    results=[]
    for fid in cfg["frames"]:
        rgb,context,sam_features=load_current_face_inputs(pipe,provider,fid,K);row=provider.rows[fid];f=row["frame"]
        pixels=np.moveaxis(rgb.cpu().numpy(),0,-1)
        frame=ImageFrame(f"frame{fid}",pixels,provider.region["pixel_space_id"],f["sha256"],"original",array_digest(pixels))
        result=pipe.predict_video(rgb,K,cad,video_context=context,roi_expansion=1.5,return_surface=True,surface_frame=frame,sam_video_context=sam_features)
        snapshots=result.pop("surface_snapshots");ledger=json.loads(context.source_ledger_json)
        status_path=contained_path(R,row["native_status"],exists=True)
        status=json.loads(status_path.read_text())
        raw=provider.verifier.raw(status["raw"],status_path.parent)
        ids=np.asarray(raw["outputs"]["out_obj_ids"]).tolist()
        masks=np.asarray(raw["outputs"]["out_binary_masks"])
        folder=out/f"frame_{fid:08d}";folder.mkdir()
        all_masks={}
        for tid,owner in ledger["native_owners"].items():
            oid=owner["native_obj_id"]
            if oid not in ids:continue
            mask=masks[ids.index(oid)]
            path=folder/(tid+"_body_mask.npz")
            with path.open("xb") as stream:np.savez_compressed(stream,mask=mask)
            all_masks[tid]=binding(path)
        (folder/"ROI_PREDICTION.json").write_text(json.dumps(result,allow_nan=False))
        for snapshot in snapshots:
            tid=snapshot.metadata["video_context"]["candidate"]["track_id"];owner=ledger["native_owners"][tid]
            if tid not in all_masks:
                results.append({"frame_id":fid,"track_id":tid,"status":"current_body_mask_missing"});continue
            mask=masks[ids.index(owner["native_obj_id"])]
            model=snapshot.metadata["models"];spec=RoiSpec.from_full_box(snapshot.arrays["bbox_native"],1.5,448)
            face=prepare_face_inputs(snapshot,frame=frame,expected_models=model,
                expected_prediction_id=snapshot.metadata["prediction_id"],roi_spec=spec,contract=contract,
                min_depth_m=1e-6,grazing_cos_threshold=.05)
            mb=VideoMaskBinding(fid,f["pts"],tuple(provider.region["time_base"]),frame.pixel_space_id,
                tid,face.metadata["body_identity"]["generation"],owner["native_obj_id"],owner["generation"],
                owner["session_id"],context.replay_identity,array_digest(mask),model["A0"]["sha256"])
            proposal=propose_face_prompts(face,frame=frame,video_context=context,body_binding=mb,body_mask=mask,
                patch_candidates=patches,contract=contract,expected_models=model,thresholds=thresholds)
            snap_path=folder/(tid+"_surface.npz");save_snapshot(snap_path,snapshot)
            pp=folder/(tid+"_FACE_PROPOSALS.json");pp.write_text(json.dumps(proposal,indent=2,allow_nan=False))
            good=[h for h in proposal["hypotheses"] if h["status"]=="prompt_hypothesis"]
            results.append({"frame_id":fid,"track_id":tid,"surface":binding(snap_path),"proposal":binding(pp),
                "body_mask":all_masks[tid],"other_bodies":[r for t,r in all_masks.items() if t!=tid],
                "current_native_receipt":status["raw"],"prompt_count":len(good),
                "face_ids":sorted(set(h["face_id"] for h in good)),"support_summary":proposal["support_summary"]})
    final={"status":"REAL_A0_FACE_HYPOTHESES_NOT_NATIVE_FACE_ACCEPTANCE","results":results,
        "models":pair["models"],"roi_expansion":1.5,"GT_used":False,"native_face_inference":False}
    (out/"COMPLETE.json").write_text(json.dumps(final,indent=2,allow_nan=False))
    return final


def repropose(config_path):
    """Reuse unchanged actual A0 snapshots; only the face-hypothesis policy changes."""
    from data.action4_surface_snapshot import load_snapshot
    from dataclasses import asdict
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    source=read_binding(cfg["source_inputs_ref"])
    if source["models"]!=pair["models"] or source["roi_expansion"]!=1.5:
        raise ValueError("Cached surface model/ROI differs")
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    cref=cfg["face_contract"]
    contract=FaceGeometryContract.from_bytes(provider.verifier.verify(cref["path"],cref["sha256"]).read_bytes(),
        provider.verifier.verify(lock["CAD"]["path"],lock["CAD"]["sha256"]).read_bytes())
    patches=FacePatchCandidates.from_ref(cfg["patch_candidates"],contract)
    strategy=cfg.get("strategy","surface_consensus")
    if strategy=="surface_consensus":
        thresholds=FacePromptThresholds(**cfg["thresholds"])
    elif strategy=="projected_cuboid_face_prior":
        from eval.action4_face_prompt_proposals import ProjectedFacePromptThresholds
        thresholds=ProjectedFacePromptThresholds(**cfg["thresholds"])
    else:raise ValueError("Unknown explicitly registered face proposal strategy")
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    results=[];current=None
    for previous in source["results"]:
        fid=previous["frame_id"]
        if fid not in cfg["frames"]:continue
        if current!=fid:
            rgb,context=provider.load(fid,K);f=provider.rows[fid]["frame"]
            pixels=np.moveaxis(rgb.cpu().numpy(),0,-1)
            frame=ImageFrame(f"frame{fid}",pixels,provider.region["pixel_space_id"],f["sha256"],"original",array_digest(pixels))
            current=fid
        snapshot=load_snapshot(provider.verifier.verify(previous["surface"]["path"],previous["surface"]["sha256"]),frame=frame)
        old=read_binding(previous["proposal"]);model=snapshot.metadata["models"]
        spec=RoiSpec.from_full_box(snapshot.arrays["bbox_native"],1.5,448)
        face=prepare_face_inputs(snapshot,frame=frame,expected_models=model,
            expected_prediction_id=snapshot.metadata["prediction_id"],roi_spec=spec,contract=contract,
            min_depth_m=1e-6,grazing_cos_threshold=.05)
        mb=dict(old["dependencies"]["body_mask_binding"]);mb["time_base"]=tuple(mb["time_base"])
        body_binding=VideoMaskBinding(**mb)
        with np.load(provider.verifier.verify(previous["body_mask"]["path"],previous["body_mask"]["sha256"]),allow_pickle=False) as z:
            mask=z["mask"].copy()
        args=dict(frame=frame,video_context=context,body_binding=body_binding,body_mask=mask,
            patch_candidates=patches,contract=contract,expected_models=model,thresholds=thresholds)
        if strategy=="projected_cuboid_face_prior":
            from eval.action4_face_prompt_proposals import propose_projected_face_region_prompts
            proposal=propose_projected_face_region_prompts(face,**args)
        else:
            proposal=propose_face_prompts(face,**args,support_policy=cfg["support_policy"])
        folder=out/f"frame_{fid:08d}";folder.mkdir(exist_ok=True)
        pp=folder/(previous["track_id"]+"_FACE_PROPOSALS.json")
        pp.write_text(json.dumps(proposal,indent=2,allow_nan=False))
        good=[h for h in proposal["hypotheses"] if h["status"]=="prompt_hypothesis"]
        row=dict(previous);row.update(proposal=binding(pp),prompt_count=len(good),
            face_ids=sorted(set(h["face_id"] for h in good)),support_summary=proposal["support_summary"])
        results.append(row)
    final={"status":"REAL_A0_REUSED_LOGICAL_FACE_HYPOTHESES_NOT_ACCEPTANCE","models":pair["models"],
        "results":results,"roi_expansion":1.5,"source_inputs":cfg["source_inputs_ref"],
        "A0_new_calls":0,"SAM_new_calls":0,"support_policy":cfg.get("support_policy"),"strategy":strategy,"GT_used":False}
    if strategy=="projected_cuboid_face_prior":final["status"]="REAL_A0_REUSED_PROJECTED_FACE_HYPOTHESES_NOT_ACCEPTANCE"
    (out/"COMPLETE.json").write_text(json.dumps(final,indent=2,allow_nan=False))
    return final

def load_current_face_inputs(pipe,provider,frame_id,K):
    """Use the same current-mask bridge as normal video inference."""
    if pipe.a0_kind=="joint_pose_sam":
        from eval.action4_video_sam_inputs import VideoSamFeatureContext
        return VideoSamFeatureContext.load(provider,frame_id,K,
            expected_b0_sha256=pipe.identity["B0_bundle_sha256"],
            expected_sam_model_sha256=pipe.acfg["sam_cache"]["expected_sources"]["sam_model_sha256"])
    rgb,context=provider.load(frame_id,K)
    return rgb,context,None

def prepare_saved(config_path):
    """Re-use already computed selected-SAM-A0 surface snapshots, not stale A0."""
    from data.action4_surface_snapshot import load_snapshot
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    expected={role:{k:pair["models"][role][k] for k in ("path","sha256")} for role in ("A0","B0")}
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    contract=FaceGeometryContract.from_bytes(provider.verifier.verify(cfg["face_contract"]["path"],cfg["face_contract"]["sha256"]).read_bytes(),
        provider.verifier.verify(lock["CAD"]["path"],lock["CAD"]["sha256"]).read_bytes())
    patches=FacePatchCandidates.from_ref(cfg["patch_candidates"],contract)
    thresholds=FacePromptThresholds(**cfg["thresholds"])
    out=contained_path(R,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    results=[];current=None
    for ref in cfg["surface_snapshots"]:
        snapshot=load_snapshot(provider.verifier.verify(ref["path"],ref["sha256"]))
        metadata=snapshot.metadata;video=metadata.get("video_context",{});fid=video.get("frame_id")
        if metadata["models"]!=expected or video.get("SAM_mask_features_enabled") is not True:
            raise ValueError("Saved surface is not the selected SAM-enhanced A0 pair")
        if fid not in cfg["frames"]:raise ValueError("Undeclared source frame")
        if current!=fid:
            rgb,context=provider.load(fid,K);source=provider.rows[fid]["frame"]
            pixels=np.moveaxis(rgb.cpu().numpy(),0,-1)
            frame=ImageFrame(metadata["image_id"],pixels,provider.region["pixel_space_id"],source["sha256"],metadata["variant_id"],array_digest(pixels))
            masks,receipt=provider.current_native_masks(fid)
            folder=out/f"frame_{fid:08d}";folder.mkdir(exist_ok=True)
            refs={}
            for tid,mask in masks.items():
                p=folder/(tid+"_body_mask.npz")
                with p.open("xb") as stream:np.savez_compressed(stream,mask=mask)
                refs[tid]=binding(p)
            current=fid
        # Reload validates current RGB and source frame rather than trusting a file name.
        snapshot=load_snapshot(provider.verifier.verify(ref["path"],ref["sha256"]),frame=frame)
        tid=video["candidate"]["track_id"]
        if tid not in masks:
            results.append(dict(frame_id=fid,track_id=tid,status="current_body_mask_missing",surface=ref));continue
        spec=RoiSpec.from_full_box(snapshot.arrays["bbox_native"],1.5,448)
        if not np.array_equal(snapshot.arrays["K_native"],K):raise ValueError("Saved surface K differs")
        selected={d["track_id"]:d for d in context.select_for_a0(context.consume(rgb,K,b0_sha256=expected["B0"]["sha256"]))}
        if tid not in selected or list(snapshot.arrays["bbox_native"])!=selected[tid]["bbox_for_a0"]:
            raise ValueError("Saved surface no longer matches actual resolved ROI")
        face=prepare_face_inputs(snapshot,frame=frame,expected_models=expected,
            expected_prediction_id=metadata["prediction_id"],roi_spec=spec,contract=contract,
            min_depth_m=1e-6,grazing_cos_threshold=.05)
        ledger=json.loads(context.source_ledger_json);owner=ledger["native_owners"][tid]
        mb=VideoMaskBinding(fid,source["pts"],tuple(provider.region["time_base"]),frame.pixel_space_id,
            tid,face.metadata["body_identity"]["generation"],owner["native_obj_id"],owner["generation"],owner["session_id"],
            context.replay_identity,array_digest(masks[tid]),expected["A0"]["sha256"])
        proposal=propose_face_prompts(face,frame=frame,video_context=context,body_binding=mb,body_mask=masks[tid],
            patch_candidates=patches,contract=contract,expected_models=expected,thresholds=thresholds,
            support_policy=cfg["support_policy"])
        path=folder/(tid+"_FACE_PROPOSALS.json");path.write_text(json.dumps(proposal,indent=2,allow_nan=False))
        good=[h for h in proposal["hypotheses"] if h["status"]=="prompt_hypothesis"]
        results.append(dict(frame_id=fid,track_id=tid,surface=ref,proposal=binding(path),body_mask=refs[tid],
            other_bodies=[v for t,v in refs.items() if t!=tid],current_native_receipt=receipt,
            prompt_count=len(good),face_ids=sorted(set(h["face_id"] for h in good)),support_summary=proposal["support_summary"]))
    done=dict(status="CURRENT_SAM80_SAVED_SURFACE_FACE_PROPOSALS_NOT_NATIVE_ACCEPTANCE",models=pair["models"],
        roi_expansion=1.5,results=results,A0_new_calls=0,SAM_new_calls=0,GT_used=False,
        native_face_inference=False,face_tracking_accepted=False)
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2,allow_nan=False))
    return done
