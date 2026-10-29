"""Two ROI decisions through one existing A0 pipeline and one frozen B0 pool."""
import copy
import numpy as np
from data.action4_frozen_detection_context import FrozenDetectionContext
from data.action4_sam_roi_inputs import ImageFrame,array_digest

def predict_core_frame_pair(pipeline,replay,frame_id,K,canonical,*,comparison_mode="causal_forward_B0_assist"):
    if frame_id not in replay.clip["evaluation_frame_ids"]:
        raise ValueError("Only registered core frames are compared")
    if pipeline.a0_kind!="joint_pose" or pipeline.pose_method!="network":
        raise ValueError("This comparison preserves plain A0 raw network output")
    causal=comparison_mode=="causal_forward_B0_assist"
    if comparison_mode not in ("causal_forward_B0_assist","offline_bidirectional_SAM_diagnostic") or replay.causal!=causal:
        raise ValueError("Causal and offline SAM roles cannot be silently mixed")
    local=replay.by_global[frame_id];frame=replay.doc["frames"][local]
    pool=replay.pools[local]
    if pool["frame"]!=frame:raise ValueError("B0 source frame changed")
    candidates=copy.deepcopy(pool["pool"]["single_frame_selected"])
    if replay.status=="system_failure":
        # Retain B0-only observation even when the independent SAM system failed.
        from PIL import Image
        import torch
        with Image.open(replay.verifier.verify(frame["image"],frame["sha256"])) as image:
            rgb=torch.from_numpy(np.array(image.convert("RGB"),copy=True)).permute(2,0,1)
        context=None
    else:
        rgb,context=replay.load(frame_id,K)
    pixels=rgb.permute(1,2,0).numpy()
    source=ImageFrame("frame_%08d"%frame_id,pixels,replay.region.pixel_identity,frame["sha256"],
                      "native_original",array_digest(pixels))
    cached=FrozenDetectionContext.from_frame(source,K,candidates,b0_sha256=replay.cfg["b0"]["sha256"],
        scoring_policy=replay.cfg["b0"]["scoring_policy"],source_cache_identity=replay.b0_pool_identity)
    baseline=pipeline.predict(rgb,K,canonical,frozen_detections=cached,return_box_candidates=True,roi_expansion=1.5)
    if context is None:
        assisted={"identity":copy.deepcopy(pipeline.identity),"instances":[],"detections":[],
                  "system_failure":copy.deepcopy(replay.failure_evidence),"A0_called":False}
    else:
        assisted=pipeline.predict_video(rgb,K,canonical,video_context=context,roi_expansion=1.5)
    for p in baseline["instances"]:
        p["bbox_for_a0"]=copy.deepcopy(p["bbox_full_xyxy"])
        p["corners_raw_from_resolved_roi"]=copy.deepcopy(p["corners_px"])
        p["corners_final"]=copy.deepcopy(p["corners_px"])
        p["post_A0_geometry"]="none";p["roi_source"]="frozen_B0_single_frame"
    header=dict(frame_id=frame_id,pts=frame["pts"],RGB_sha256=frame["sha256"],roi_expansion=1.5,
        pixel_space_id=replay.region.pixel_identity,comparison_mode=comparison_mode,
        frozen_B0_pool_identity=replay.b0_pool_identity)
    return {"b0_only":dict(header,prediction=baseline),
            "track_assisted":dict(header,prediction=assisted,
                track_evidence=copy.deepcopy(replay.rows[frame_id]))}


def run_registered_clip(config_path):
    """Actual fixed-best two-path development comparison; no image GT or model update."""
    import json,os
    import torch
    from pathlib import Path
    from data.action4_paths import PROJECT_ROOT as R,contained_path
    from eval.action4_roi_sweep import read_binding
    from eval.action4_clip_replay import ClipReplay,freeze_metadata_refs
    from eval.action4_staged_pipeline import StagedPipeline
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=cfg["gpu_uuid"] or set(os.sched_getaffinity(0))!=set(cfg["cpus"]):
        raise ValueError("Assigned paired evaluation resources differ")
    from model.training.action4_resource_policy import validate_device_policy
    validate_device_policy({"training_gpus":[],"eval_gpus":[cfg["gpu"]]},R)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.use_deterministic_algorithms(True)
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    refs=freeze_metadata_refs(cfg["native_config"],out/"EVIDENCE_INDEX.json",producer_exit_ref=cfg["producer_exit"])
    replay=ClipReplay(registry_ref=cfg["registry"],clip_id=cfg["clip_id"],config_ref=cfg["native_config"],
        tracker_config_ref=cfg["tracker_config"],producer_exit_ref=cfg["producer_exit"],evidence_refs=refs)
    (out/"REPLAY.json").write_text(json.dumps(replay.public_result(),allow_nan=False))
    pair=read_binding(cfg["model_pair"])
    for model in pair["models"].values():replay.verifier.verify(model["path"],model["sha256"])
    if pair["models"]["B0"]["sha256"]!=replay.cfg["b0"]["sha256"]:raise ValueError("B0 counterpart differs")
    pipe=StagedPipeline.from_bundles(pair["models"]["A0"]["path"],pair["models"]["B0"]["path"],device="cuda:0")
    pipe.b0.eval().requires_grad_(False);pipe.b0_scoring_policy=replay.cfg["b0"]["scoring_policy"]
    K=np.asarray(read_binding(cfg["K"])["K_native"])
    canonical=np.asarray(read_binding(cfg["CAD"])["corners_3d_mm"])/1000
    counts={"b0_only":0,"track_assisted":0,"track_SAM_supported":0,"initializing_frames":0};calls={"B0":0}
    def hook(*args):calls["B0"]+=1
    handle=pipe.b0.register_forward_hook(hook)
    try:
        with (out/"PAIRED_PREDICTIONS.jsonl").open("x") as stream:
            for fid in replay.clip["evaluation_frame_ids"]:
                pair_prediction=predict_core_frame_pair(pipe,replay,fid,K,canonical)
                for role in ["b0_only","track_assisted"]:counts[role]+=len(pair_prediction[role]["prediction"]["instances"])
                counts["track_SAM_supported"]+=sum(x.get("temporal_status")=="sam_supported" for x in pair_prediction["track_assisted"]["prediction"]["instances"])
                counts["initializing_frames"]+=int(not pair_prediction["track_assisted"]["prediction"]["instances"] and fid<4)
                stream.write(json.dumps(pair_prediction,allow_nan=False)+chr(10))
        if calls["B0"]:raise ValueError("Frozen B0 pools must be reused")
        result={"status":"ACTUAL_TWO_ROI_PATHS_COMPLETE_NOT_ACCURACY_ACCEPTANCE","frames":len(replay.clip["evaluation_frame_ids"]),
            "counts":counts,"calls":calls,"models":pair["models"],"roi_expansion":1.5,"ROI_source_is_only_changed_factor":True,
            "GT_used":False,"geometry_refinement":False,"tracking_causal":replay.causal}
        (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));return result
    finally:handle.remove()
