"""Research-only current-video corner bridge and read-only saved-row checks.

The coarse decoder route consumes predicted visible SAM support, never a
calibrated full-body box. No still-image cache entries or deployment admission.
"""
import copy
import json
from fractions import Fraction
import numpy as np
import torch
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R, contained_path
from data.action4_video_roi_context import VideoROIContext
from eval.action4_corner_prediction import _plain
from eval.action4_video_sam_inputs import VideoSamFeatureContext, prepare_video_pipeline_sam
from eval.action4_sam_pipeline_inputs import device_mask_aux
from eval.action4_surface_snapshot import model_identities
from data.action4_sam_native_export import _v2_checked, _v2_sha
from model.action4_sam_joint import CornerCandidateOutput
from model.action4_single_frame_pose import POLICY, observations_from_head, decode_single_frame_pose

ROUTE_REASON = "current_predicted_visible_SAM_support_not_full_box_GT_or_calibrated_full_box"
ROUTE_DIFFERENCE = "video_coarse_current_mask_and_resolved_ROI_bounds; image_eval_full_box_quality_route_not_inherited"


def validate_corner_trial_binding(trial_binding, identity, roi_expansion):
    """Authenticate the selected research model without reading evaluation labels."""
    if not isinstance(trial_binding, dict) or set(trial_binding) != {"path", "sha256"}:
        raise ValueError("Explicit corner trial selection path/SHA required")
    selection = _v2_checked(trial_binding)
    selected = selection.get("selected", {})
    if (selection.get("status") not in ("PRESERVED_ALL8_DEVELOPMENT_CANDIDATE_NOT_FINAL_SEAL","SEALED_USER_STOPPED_ALL8_RESEARCH_TRIAL","SEALED_COMPLETED75_ALL8_RESEARCH_TRIAL")
            or selected.get("route") != "corner_candidate"
            or selection.get("default_changed") is not False
            or selection.get("test_used") is not False
            or selection.get("quality_calibrated") is not False
            or identity.get("quality_ready") is not False
            or identity.get("deployment_admitted") is not False):
        raise ValueError("Explicit unadmitted development corner selection required")
    from eval.action4_video_roi_contract import trial_ratio
    seal=None
    if selection.get("status") in ("SEALED_USER_STOPPED_ALL8_RESEARCH_TRIAL","SEALED_COMPLETED75_ALL8_RESEARCH_TRIAL"):
        seal=_v2_checked(selection["sealed_research"])
        if (seal.get("selected_step")!=selected.get("step") or seal.get("selected_route")!="corner_candidate"
            or seal.get("sha256")!=selection["preserved_snapshot"]["sha256"]):
            raise ValueError("Stopped-stage trial lacks actual sealed winner")
    expected_ratio=trial_ratio(selection,seal)
    if seal is not None and seal.get("config_id")!=identity.get("A0_config_id"):raise ValueError("Sealed model configuration differs")
    models = identity["models"]
    preserved = selection["preserved_snapshot"]
    if (selected["step"] != identity["A0_step"]
            or selected["checkpoint"]["sha256"] != preserved["sha256"]
            or models["A0"]["sha256"] != preserved["sha256"]
            or contained_path(R, models["A0"]["path"], exists=True)
               != contained_path(R, preserved["path"], exists=True)
            or contained_path(R, models["B0"]["path"], exists=True)
               != contained_path(R, selection["B0"], exists=True)
            or identity["B0_bundle_sha256"] != models["B0"]["sha256"]
            or selection["roi_expansion"] != roi_expansion or roi_expansion != expected_ratio):
        raise ValueError("Corner trial selected step/model/ROI identity differs")
    # Constructor validated the B0 counterpart; retain its binding here and in
    # saved-row validation rather than trusting a display identity alone.
    if identity["counterpart"]["B0"] != models["B0"]:
        raise ValueError("Corner trial B0 counterpart differs")
    if not all(isinstance(identity.get(k), str) and identity[k] for k in ("A0_config_id", "B0_config_id")):
        raise ValueError("Loaded corner configuration identity required")
    return selection


def _check_pipeline(pipeline, trial_binding):
    if getattr(pipeline, "corner_candidate_mode", False) is not True:
        raise ValueError("Explicit corner candidate pipeline required")
    actual = model_identities(pipeline._surface_model_bindings)
    if actual != pipeline.identity["models"]:
        raise ValueError("Loaded corner model identity changed")
    versions = tuple(p._version for module in (pipeline.a0, pipeline.b0)
                     for p in list(module.parameters()) + list(module.buffers()))
    if versions != pipeline._corner_versions or pipeline.a0.training or pipeline.b0.training:
        raise ValueError("Frozen corner inference model changed")
    validate_corner_trial_binding(trial_binding, pipeline.identity, pipeline.roi_expansion)
    for ref in (pipeline._corner_eval_spec["manifest"], pipeline._corner_box_gate_binding):
        if ref is not None and _v2_sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Corner training dependency identity changed")


def _pose(rotation, translation):
    r = np.asarray(rotation, float); t = np.asarray(translation, float)
    finite = bool(np.isfinite(r).all() and np.isfinite(t).all())
    return dict(coordinate_frame="object_to_capture_cv_m", finite=finite,
                rotation=r.tolist() if finite else None, translation_m=t.tolist() if finite else None)


def _route_record(route):
    # The authenticated full-resolution mask is consumed by the decoder, but not
    # expanded into millions of JSON booleans per instance.
    result = {k: v for k, v in route.items() if k != "support_mask_native"}
    from data.action4_sam_roi_inputs import array_digest
    mask = route["support_mask_native"]
    result["support_mask_sha256"] = None if mask is None else array_digest(mask)
    result["support_mask_pixels"] = 0 if mask is None else int(mask.sum())
    return _plain(result)


@torch.inference_mode()
def predict_corner_video(pipeline, rgb, K, canonical, *, video_context,
                         sam_video_context, trial_binding, return_surface=False):
    if type(return_surface) is not bool:raise ValueError("Explicit surface export flag required")
    _check_pipeline(pipeline, trial_binding)
    if not isinstance(video_context, VideoROIContext) or not isinstance(sam_video_context, VideoSamFeatureContext):
        raise ValueError("Authenticated current video ROI and SAM contexts required")
    K = np.asarray(K, float); cad = np.asarray(canonical, float)
    if K.shape != (3, 3) or cad.shape != (8, 3) or not np.isfinite(K).all() or not np.isfinite(cad).all():
        raise ValueError("Finite native K and canonical8 required")
    expected_sam = pipeline.acfg.get("sam_cache", {}).get("expected_sources", {}).get("sam_model_sha256")
    if expected_sam is None or sam_video_context.sam_model_sha256 != expected_sam:
        raise ValueError("Current video SAM weights differ from corner training recipe")
    b0_sha = pipeline.identity["B0_bundle_sha256"]
    candidates = video_context.consume(rgb, K, b0_sha256=b0_sha)
    detections = video_context.select_for_a0(candidates)
    masks, receipt = sam_video_context.consume(rgb, K, video_context, detections, b0_sha256=b0_sha)
    frame = sam_video_context.frame
    if receipt["frame_id"] != video_context.frame_id:
        raise ValueError("Current SAM receipt frame differs")
    for p in detections:
        if (p["prediction_id"] != f"video:{p['track_id']}:frame:{video_context.frame_id}"
                or p["frame_id"] != receipt["frame_id"] or p["pts"] != receipt["pts"]
                or Fraction(*p["time_base"]) != Fraction(*receipt["time_base"])):
            raise ValueError("Current selected track/frame/PTS differs")
    if len({p["track_id"] for p in detections}) != len(detections):
        raise ValueError("Duplicate video instance")
    result = dict(schema="action4.corner_video_research.v1", identity=copy.deepcopy(pipeline.identity),
                  trial_binding=copy.deepcopy(trial_binding), roi_expansion=pipeline.roi_expansion,
                  detections=_plain(detections), instances=[], raw_instances=[], candidate_instances=[],
                  corner_observations=[], quality_ready=False, deployment_admitted=False,
                  diagnostic_candidate=True, GT_used=False, route_difference=ROUTE_DIFFERENCE,
                  frame_id=receipt["frame_id"], pts=receipt["pts"], time_base=receipt["time_base"],
                  RGB_sha256=receipt["RGB_sha256"], track_replay_identity=video_context.replay_identity,
                  canonical_m=cad.tolist(), K_native=K.tolist(), a0_roi_specs=[],
                  video_roi_consumption=dict(frame_id=video_context.frame_id,
                      replay_identity=video_context.replay_identity, A0_mask_features_enabled=True,
                      withheld=json.loads(video_context.withheld_json),
                      SAM_feature_receipt=receipt, instances=[]))
    if not detections:
        return result
    inputs, specs = prepare_video_pipeline_sam(rgb, K, detections, sam_video_context,
        video_context=video_context, b0_sha256=b0_sha, roi_expansion=pipeline.roi_expansion)
    rois = inputs.rgb.to(pipeline.device)
    km = torch.as_tensor(np.stack(inputs.K112), device=pipeline.device, dtype=rois.dtype)
    canonical_tensor = torch.as_tensor(cad, device=pipeline.device, dtype=rois.dtype)[None].expand(len(detections), -1, -1)
    aux = device_mask_aux(inputs.mask_aux, pipeline.device, rois.dtype)
    packet = pipeline.a0(rois, km, canonical_tensor, mask_aux=aux, identities=aux.identities, return_corner=True)
    if not isinstance(packet, CornerCandidateOutput) or packet.corners.identities != aux.identities:
        raise ValueError("Owned corner video forward returned wrong instance order")
    output = packet.raw_pose
    if return_surface:
        from data.action4_surface_snapshot import make_snapshot
        result["surface_snapshots"]=[make_snapshot(frame=frame,prediction=p,output=output,index=i,
            A112=inputs.A112[i],A448=inputs.A448[i],K_native=K,K112=km[i],canonical=canonical_tensor[i],
            models=pipeline.identity["models"],sam_input=inputs.source_dependencies[i],video_context=video_context)
            for i,p in enumerate(detections)]
    for i, (p, spec) in enumerate(zip(detections, specs)):
        identity = aux.identities[i]
        if identity.image_id != frame.image_id or identity.instance_id != p["prediction_id"]:
            raise ValueError("Raw/head/resolved video ROI identity mismatch")
        obs = observations_from_head(packet.corners, i, identity=identity, pixel_space_id=frame.pixel_space_id, config=POLICY)
        rotation = output["rotation"][i].cpu().numpy(); translation = output["translation"][i].cpu().numpy()
        camera = output["camera_corners"][i].cpu().numpy(); uv112 = output["corners"][i].cpu().numpy()
        raw_valid = bool(np.isfinite(camera).all() and np.isfinite(uv112).all() and (camera[:, 2] > 0).all())
        raw_corners = DataAdapter.transform_points(uv112, np.linalg.inv(inputs.A112[i])).tolist() if raw_valid else None
        mask = masks.get(p["track_id"])
        route = dict(kind="coarse", identity=list(obs["identity"]), pixel_space_id=frame.pixel_space_id,
                     support_mask_native=mask, candidate_bounds_native=list(spec.continuous_bounds),
                     full_box_quality_passed=False, reason=ROUTE_REASON,
                     route_difference=ROUTE_DIFFERENCE, quality_gate_scope="unadmitted_current_video_visible_support")
        decoded = decode_single_frame_pose(canonical_m=cad, K_native=K, A224_from_native=inputs.A224[i],
            raw_rotation=rotation, raw_translation_m=translation, observations=obs, route=route,
            pixel_space_id=frame.pixel_space_id, raw_identity=identity, roi_identity=identity,
            quality_ready=False, config=POLICY, diagnostic_candidate=mask is not None and bool(mask.any()))
        # Missing support is an explicit raw fallback; no historical mask or box
        # quality is substituted, and no decoder candidate is promoted.
        if mask is None or not mask.any():
            decoded["reason"] = "current_video_mask_unavailable_raw_fallback"
        trace = dict(available=bool(inputs.mask_aux.available[i].item()), reason=inputs.reasons[i],
                     dependencies=inputs.source_dependencies[i], q_mask=float(packet.sam_aux.q_mask[i].item()),
                     q_mask_is_calibrated=False, still_image_cache_used=False,
                     consumption_path="current_resolved_video_RGB_ROI+current_native_mask->shared_enhanced_C5->owned_corner_head")
        raw = dict(_plain(p), instance_id=i, object_id=1, corner_ids=list(range(8)), published=True,
                   source_detection_ids=p.get("source_detection_ids", [p["prediction_id"]]),
                   positive_depth_verified=raw_valid, pose_valid=raw_valid, corners_px=raw_corners,
                   corners_raw_from_resolved_roi=copy.deepcopy(raw_corners), corners_final=copy.deepcopy(raw_corners),
                   raw_network_pose=_pose(rotation, translation), final_pose=_pose(rotation, translation),
                   rotation=rotation.tolist(), translation_m=translation.tolist(), sam_input=_plain(trace),
                   post_A0_geometry="corner_candidate_raw_fallback", quality_ready=False, deployment_admitted=False,
                   pose_diagnostic=dict(method="raw_network", valid=raw_valid))
        candidate = copy.deepcopy(raw)
        accepted = decoded["geometric_candidate_accepted"]
        if accepted:
            candidate.update(corners_px=decoded["candidate_corners_native"],
                corners_final=decoded["candidate_corners_native"],
                rotation=decoded["candidate_rotation"], translation_m=decoded["candidate_translation_m"],
                final_pose=_pose(decoded["candidate_rotation"], decoded["candidate_translation_m"]),
                pose_valid=True, positive_depth_verified=True, post_A0_geometry="corner_candidate_research_only")
        candidate["pose_diagnostic"] = dict(method="corner_candidate_or_raw", reason=decoded["reason"],
            geometric_candidate_accepted=accepted, deployed=False, valid=candidate["pose_valid"])
        record = dict(prediction_id=p["prediction_id"], track_id=p["track_id"], frame_id=p["frame_id"], pts=p["pts"],
            roi_identity=list(obs["identity"]), observations=_plain(obs), A112_from_native=inputs.A112[i].tolist(),
            A224_from_native=inputs.A224[i].tolist(), K112=inputs.K112[i].tolist(), K_native=K.tolist(),
            canonical_m=cad.tolist(), roi_spec=spec.record(), bbox_for_a0=p["bbox_for_a0"],
            roi_decision_id=p["roi_decision_id"], raw_rotation=rotation.tolist(),
            raw_translation_m=translation.tolist(), raw_corners_native=raw_corners,
            route=_route_record(route), decoder_diagnostics=_plain(decoded), quality_ready=False, deployed=False)
        candidate["corner_observations"] = record["observations"]
        candidate["decoder_diagnostics"] = record["decoder_diagnostics"]
        candidate["corner_route"] = record["route"]
        result["raw_instances"].append(raw); result["candidate_instances"].append(candidate)
        result["corner_observations"].append(record)
        result["a0_roi_specs"].append(dict(spec.record(), prediction_id=p["prediction_id"], track_id=p["track_id"],
            roi_source=p["roi_source"], A448=inputs.A448[i].tolist(), A112=inputs.A112[i].tolist(),
            A224=inputs.A224[i].tolist(), K112=inputs.K112[i].tolist()))
        result["video_roi_consumption"]["instances"].append(dict(track_id=p["track_id"],
            bbox_used=p["bbox_for_a0"], roi_decision_id=p["roi_decision_id"],
            A448=inputs.A448[i].tolist(), A112=inputs.A112[i].tolist(), K112=inputs.K112[i].tolist(),
            consumption_path=trace["consumption_path"]))
    result["instances"] = copy.deepcopy(result["candidate_instances"])
    return result


def _project_saved(pose, K, cad):
    if not isinstance(pose, dict) or pose.get("coordinate_frame") != "object_to_capture_cv_m" or pose.get("finite") is not True:
        raise ValueError("Finite recorded pose required")
    r = np.asarray(pose["rotation"], float); t = np.asarray(pose["translation_m"], float)
    if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-4, rtol=0) or abs(np.linalg.det(r)-1) > 1e-4):
        raise ValueError("Recorded rigid pose invalid")
    camera = cad @ r.T + t
    if (camera[:, 2] <= 0).any():
        return None
    hom = camera @ K.T
    return hom[:, :2] / hom[:, 2:]


def _same_pixels(saved, projected):
    if projected is None:
        return saved is None
    value = np.asarray(saved, float)
    # Float32 network projection versus float64 read-only replay, not an accuracy gate.
    return value.shape == (8, 2) and np.isfinite(value).all() and np.allclose(value, projected, atol=.01, rtol=1e-6)


def validate_corner_trial_row(row, K, cad, trial_binding):
    """Read-only consistency check. Never optimize, choose or modify a saved pose."""
    pred = row.get("prediction", row)
    K = np.asarray(K, float); cad = np.asarray(cad, float)
    validate_corner_trial_binding(trial_binding, pred["identity"], pred["roi_expansion"])
    if (pred.get("schema") != "action4.corner_video_research.v1" or pred.get("trial_binding") != trial_binding
            or pred.get("quality_ready") is not False or pred.get("deployment_admitted") is not False
            or pred.get("route_difference") != ROUTE_DIFFERENCE
            or not np.array_equal(pred["canonical_m"], cad) or not np.array_equal(pred["K_native"], K)):
        raise ValueError("Saved corner trial identity or geometry domain differs")
    for key in ("frame_id", "pts", "RGB_sha256", "track_replay_identity"):
        if key in row and row[key] != pred[key]:
            raise ValueError("Saved video frame/PTS/RGB/replay differs")
    receipt = pred["video_roi_consumption"]["SAM_feature_receipt"]
    if any(receipt[k] != pred[k] for k in ("frame_id", "pts", "time_base", "RGB_sha256")):
        raise ValueError("Saved native receipt frame domain differs")
    n = len(pred["instances"])
    if any(len(pred[k]) != n for k in ("detections", "raw_instances", "candidate_instances", "corner_observations", "a0_roi_specs")):
        raise ValueError("Saved corner instance domain differs")
    if len({p["track_id"] for p in pred["instances"]}) != n:
        raise ValueError("Duplicate saved corner track")
    for p, det, raw, candidate, record, spec in zip(pred["instances"], pred["detections"],
            pred["raw_instances"], pred["candidate_instances"], pred["corner_observations"], pred["a0_roi_specs"]):
        pid = f"video:{p['track_id']}:frame:{pred['frame_id']}"
        for item in (p, det, raw, candidate, record):
            if item["prediction_id"] != pid or item["track_id"] != p["track_id"]:
                raise ValueError("Saved corner instance identity differs")
        if p != candidate or p["corner_ids"] != list(range(8)) or raw["corner_ids"] != list(range(8)):
            raise ValueError("Saved final must equal inference candidate-or-raw fixed0-7 output")
        if (p["frame_id"] != pred["frame_id"] or p["pts"] != pred["pts"]
                or Fraction(*p["time_base"]) != Fraction(*pred["time_base"])):
            raise ValueError("Saved instance frame/PTS differs")
        for key in ("bbox_b0_raw", "bbox_sam_visible", "bbox_for_a0", "roi_source", "roi_decision_id"):
            if p[key] != det[key] or raw[key] != det[key]:
                raise ValueError("Saved resolved ROI decision differs")
        from data.action4_video_roi_context import _digest
        fields = ("track_id", "frame_id", "bbox_b0_raw", "bbox_sam_visible", "bbox_for_a0", "roi_source",
                  "current_support_status", "last_trusted_full_frame_id", "last_trusted_full_pts", "last_prediction_calibration")
        if _digest({k: det[k] for k in fields}) != det["roi_decision_id"]:
            raise ValueError("Saved ROI decision digest differs")
        diag = record["decoder_diagnostics"]; route = record["route"]
        if (diag != p["decoder_diagnostics"] or record["observations"] != p["corner_observations"]
                or route != p["corner_route"] or diag["quality_ready"] is not False or diag["accepted"] is not False
                or diag["config"] != POLICY or route["kind"] != "coarse"
                or route["full_box_quality_passed"] is not False or route["reason"] != ROUTE_REASON):
            raise ValueError("Saved decoder contract differs")
        if (record["roi_identity"] != list(record["observations"]["identity"])
                or record["roi_identity"][1] != pid or diag["identity"] != record["roi_identity"]
                or route["identity"] != record["roi_identity"] or record["observations"]["corner_ids"] != list(range(8))):
            raise ValueError("Saved observation identity differs")
        from data.action4_roi_contract import RoiSpec
        expected_spec=RoiSpec.from_full_box(det["bbox_for_a0"],pred["roi_expansion"],448)
        if record["roi_spec"]!=expected_spec.record():
            raise ValueError("Saved ROI must be reconstructed from actual decided box")
        for size,key in ((448,"A448"),(112,"A112"),(224,"A224")):
            if not np.allclose(spec[key],expected_spec.affine(size),atol=1e-10,rtol=0):
                raise ValueError("Saved ROI affine differs from actual crop")
        if (not np.allclose(record["A112_from_native"],expected_spec.affine(112),atol=1e-10,rtol=0)
            or not np.allclose(record["A224_from_native"],expected_spec.affine(224),atol=1e-10,rtol=0)
            or not np.allclose(record["K112"],expected_spec.affine(112)@K,atol=1e-8,rtol=0)
            or not np.allclose(spec["K112"],expected_spec.affine(112)@K,atol=1e-8,rtol=0)):
            raise ValueError("Saved A0 camera differs from resolved ROI")
        if (record["bbox_for_a0"] != det["bbox_for_a0"] or record["roi_decision_id"] != det["roi_decision_id"]
                or record["roi_spec"] != {k: spec[k] for k in record["roi_spec"]}
                or route["candidate_bounds_native"] != record["roi_spec"]["continuous_bounds"]):
            raise ValueError("Saved decoder ROI bounds differ")
        if (p["raw_network_pose"] != raw["raw_network_pose"] or p["corners_raw_from_resolved_roi"] != raw["corners_px"]
                or record["raw_corners_native"] != raw["corners_px"]
                or record["raw_rotation"] != raw["raw_network_pose"]["rotation"]
                or record["raw_translation_m"] != raw["raw_network_pose"]["translation_m"]):
            raise ValueError("Saved raw corner branch differs")
        if not _same_pixels(raw["corners_px"], _project_saved(p["raw_network_pose"], K, cad)):
            raise ValueError("Raw corners do not project from saved raw pose")
        accepted = diag["geometric_candidate_accepted"]
        if p["pose_diagnostic"]["geometric_candidate_accepted"] is not accepted:
            raise ValueError("Saved candidate acceptance differs")
        if accepted:
            if (p["final_pose"]["rotation"] != diag["candidate_rotation"]
                    or p["final_pose"]["translation_m"] != diag["candidate_translation_m"]
                    or p["corners_final"] != diag["candidate_corners_native"]
                    or p["post_A0_geometry"] != "corner_candidate_research_only"):
                raise ValueError("Saved final differs from decoded candidate")
        elif (p["final_pose"] != p["raw_network_pose"] or p["corners_final"] != p["corners_raw_from_resolved_roi"]
              or p["post_A0_geometry"] != "corner_candidate_raw_fallback"):
            raise ValueError("Saved fallback differs from raw")
        if (p["corners_px"] != p["corners_final"]
                or p["rotation"] != p["final_pose"]["rotation"] or p["translation_m"] != p["final_pose"]["translation_m"]
                or not _same_pixels(p["corners_final"], _project_saved(p["final_pose"], K, cad))):
            raise ValueError("Saved final corners do not project from actual final pose")
    return True
