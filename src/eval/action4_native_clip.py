"""Existing native region/texture/prior diagnostic, parameterized by a source-bound clip.
No final pose publication; baseline predictions remain unchanged.
"""
import os,sys,json,time,hashlib,traceback
from contextlib import ExitStack
from pathlib import Path
import numpy as np
ROOT=Path("/data/zy/vln/a-test")


def native_pose_key(manifest):
 kind=manifest.get("native_pose_input_kind","raw_A0")
 if kind not in ("raw_A0","selected_A0"):raise ValueError("Unregistered native A0 input kind")
 return "raw_A0_pose" if kind=="raw_A0" else "selected_A0_pose"

def runtime_options(manifest):
 frames=manifest["frame_ids"]
 mode=manifest.get("A0_prior_mode","continuous")
 if mode not in ("continuous","disabled_continuous","translation_only","history_rotation"):raise ValueError("Unknown native prior mode")
 debug=manifest.get("native_debug_frame_ids",frames)
 if (not isinstance(debug,list) or any(type(x) is not int for x in debug)
     or len(set(debug))!=len(debug) or not set(debug)<=set(frames)):
  raise ValueError("Explicit unique debug frame subset required")
 return mode,set(debug)

def region_input_mode(manifest):
 value=manifest.get("region_input_mode","raw_RGB")
 if value not in ("raw_RGB","SAM_visible_support_image"):raise ValueError("Unknown region input mode")
 return value


def texture_keyframe_budget(manifest):
 value=manifest.get("texture_keyframes",1)
 if type(value) is not int or not 1<=value<=3:
  raise ValueError("Registered bounded texture keyframe budget must be1..3")
 return value

def keyframe_protection(manifest):
 value=manifest.get("retain_nonempty_keyframe",False)
 if type(value) is not bool:raise ValueError("Explicit bool keyframe protection required")
 return value

def pose_cache_refresh(manifest):
 value=manifest.get("refresh_result_pose_cache",False)
 if type(value) is not bool:raise ValueError("Explicit bool pose-cache policy required")
 return value

def keyframe_host_options(manifest):
 deferred=manifest.get("defer_keyframe_commit",False)
 if type(deferred) is not bool:raise ValueError("Explicit bool deferred keyframe mode required")
 recovery=manifest.get("recovery_mode","current_evidence_optional_reseed_and_SAM_only_continuation")
 if recovery not in ("current_evidence_optional_reseed_and_SAM_only_continuation","record_only_no_reseed"):
  raise ValueError("Unknown native recovery mode")
 policy=None
 if deferred:
  from model.action4_native_keyframe_policy import validate_policy
  policy=validate_policy(manifest.get("keyframe_host_policy"))
 return deferred,recovery,policy


def region_line_audit_frames(manifest):
 frames=manifest.get("region_line_audit_frames",[])
 if not isinstance(frames,list) or len(frames)!=len(set(frames)) or any(type(f) is not int or f not in manifest["frame_ids"] for f in frames):raise ValueError("Explicit current diagnostic frame IDs required")
 if manifest.get("region_input_mode","raw_RGB")=="SAM_visible_support_image" and manifest.get("recovery_mode")!="record_only_no_reseed":raise ValueError("Separate SAM region camera recovery is not yet admitted")
 return set(frames)


def external_region_options(manifest):
 direct=manifest.get("external_mask_likelihood",False);unknown=manifest.get("region_unknown_full_box",False);scales=manifest.get("region_search_scales")
 if type(direct) is not bool or type(unknown) is not bool:raise ValueError("Explicit external likelihood switches required")
 if (direct or unknown) and manifest.get("region_input_mode")!="SAM_visible_support_image":raise ValueError("External likelihood requires separate SAM observation camera")
 if unknown and not direct:raise ValueError("Unknown pixels require direct 0.5 likelihood, not learned RGB bins")
 if scales is not None and scales not in ([6,4,2,1],[12,8,4,2,1]):raise ValueError("Unregistered bounded coarse-scale diagnostic")
 return direct,unknown,scales


def region_gradient_audit(region, *, tracking_step):
 if type(tracking_step) is not bool:raise ValueError("Explicit native tracking-step state required")
 if not tracking_step:return dict(gradient_computed=False,actual_gradient_norm=None,actual_hessian_norm=None,gradient_reason="StartModalities_does_not_compute_gradient")
 gradient=np.asarray(region.gradient);hessian=np.asarray(region.hessian)
 if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():raise RuntimeError("Nonfinite actual region gradient/Hessian")
 return dict(gradient_computed=True,actual_gradient_norm=float(np.linalg.norm(gradient)),actual_hessian_norm=float(np.linalg.norm(hessian)),gradient_reason="ExecuteTrackingStep")


def bank_summary(texture):
 return dict(point_counts=list(texture.audit_keyframe_point_counts()),age=int(texture.audit_keyframe_age()),
  sha256=hashlib.sha256(texture.audit_keyframe_bank_bytes()).hexdigest() if hasattr(texture,"audit_keyframe_bank_bytes") else None)


def resolve_keyframe_transaction(*,obj,inst,frame,before_pose,native_ok,canonical,K,owned_mask,body_mask,support,policy):
 from model.action4_native_keyframe_policy import decide_keyframe
 tex=obj["texture"];fid=frame["frame_id"];tid=obj["track"]
 entry=dict(frame_id=fid,track_id=tid,pending=bool(tex.pending_keyframe),bank_before=bank_summary(tex),
  commit_called=False,commit_result=False,reject_called=False,reject_result=False,geometry_accepted=False)
 if not tex.pending_keyframe:
  entry.update(reason="initial_seed_not_host_accepted" if tex.last_initialization_frame==fid else "no_pending_candidate",bank_after=bank_summary(tex))
  return entry
 entry.update(pending_frame=int(tex.pending_keyframe_frame),pending_support_frame=int(tex.pending_keyframe_support_frame),
  pending_point_count=int(tex.pending_keyframe_point_count),pending_pose=np.asarray(tex.pending_keyframe_pose).tolist())
 native_pose=np.array(obj["body"].body2world_pose,dtype=float,copy=True)
 observed=np.asarray(tex.observation_audit(),dtype=float).reshape(-1,10)
 decision=decide_keyframe(policy=policy,track_id=tid,frame_id=fid,pending_frame=entry["pending_frame"],
  pending_support_frame=entry["pending_support_frame"],native_ok=bool(native_ok),pose=native_pose,
  before_pose=before_pose,canonical=canonical,K=K,observations=observed,owned_mask=owned_mask,body_mask=body_mask,
  support_audit=support["audit"].get(tid),expected_mask_sha256=inst.get("SAM_mask_sha256"),
  current_sam=inst["recovery_evidence"]["current_support_status"].get("SAM"),predicted_box=inst.get("bbox_for_a0"))
 # Full source-frame identity is checked separately from numeric evidence.
 receipt=support.get("source_receipt",{})
 if (inst.get("track_id")!=tid or support.get("frame_id")!=fid or receipt.get("frame_id")!=fid or
     receipt.get("pts")!=frame["pts"] or receipt.get("RGB_sha256")!=frame["source_png"]["sha256"]):
  decision.update(commit_recommended=False,reason="current_frame_track_RGB_identity")
 if not np.array_equal(np.asarray(entry["pending_pose"]),native_pose):
  decision.update(commit_recommended=False,reason="pending_pose_changed_before_host_review")
 entry["host_decision"]=decision
 if decision["commit_recommended"]:
  entry["commit_called"]=True;entry["commit_result"]=bool(tex.CommitPendingKeyframe(fid))
  if not entry["commit_result"]:entry["reason"]="native_commit_identity_rejected"
 if not entry["commit_result"]:
  entry["reject_called"]=True;entry["reject_result"]=bool(tex.RejectPendingKeyframe(fid))
  if not entry["reject_result"]:raise RuntimeError("Pending candidate rejection failed; stop before recovery")
 entry["bank_after"]=bank_summary(tex);entry["reason"]=entry.get("reason",decision["reason"])
 if not entry["commit_result"] and entry["bank_before"]!=entry["bank_after"]:
  raise RuntimeError("Rejected keyframe mutated active bank")
 if tex.pending_keyframe:raise RuntimeError("Unresolved pending candidate before recovery")
 return entry


def keyframe_state(objects):
 result=[]
 for obj in objects:
  tex=obj["texture"]
  available=hasattr(tex,"audit_keyframe_age") and hasattr(tex,"audit_keyframe_point_counts")
  result.append(dict(track_id=obj["track"],available=available,
   age=int(tex.audit_keyframe_age()) if available else None,
   point_counts=list(tex.audit_keyframe_point_counts()) if available else None))
  if hasattr(tex,"pending_keyframe"):
   result[-1].update(pending=bool(tex.pending_keyframe),pending_frame=int(tex.pending_keyframe_frame),bank_sha256=bank_summary(tex)["sha256"])
  if hasattr(tex,"rejected_empty_keyframes"):
   result[-1]["rejected_empty_keyframes"]=int(tex.rejected_empty_keyframes)
  if hasattr(tex,"audit_pose_cache_difference"):
   initialized=bool(result[-1]["point_counts"])
   result[-1]["pose_cache_max_abs_difference"]=float(tex.audit_pose_cache_difference()) if initialized else None
   result[-1]["cache_difference_before_results"]=float(tex.result_pose_cache_before) if initialized else None
 return result

def run_clip(output_directory):
 import pym3t
 out=(ROOT/output_directory).resolve(strict=True)
 if not out.is_relative_to(ROOT):raise ValueError("Clip output outside project")
 if (out/"POSES.jsonl").exists():raise FileExistsError("Existing native predictions must not be overwritten")
 from eval.action4_native_m3t_adapter import OwnedNativeCamera,pixel_digest,native_float32_se3_for_projection
 from model.action4_face_geometry import FaceGeometryContract,project_six_faces
 support_rows=json.loads((out/"SUPPORT_INPUTS.json").read_text())
 support_by_frame={r["frame_id"]:r for r in support_rows}
 PRIOR_ROTATION_SCALE=0.035;PRIOR_TRANSLATION_SCALE=0.003
 manifest=json.loads((out/"INPUT_MANIFEST.json").read_text());deferred,recovery_mode,host_policy=keyframe_host_options(manifest);prior_mode,debug_frames=runtime_options(manifest);model_receipt=json.loads((out/"MODEL_COMPLETE.json").read_text());modelpath=ROOT/model_receipt["model_path"];assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==model_receipt["sha256"]
 assert modelpath.resolve().is_relative_to(ROOT)
 g=json.loads((ROOT/"outputs/action4_m3t_native_readiness_v001/geometry/GEOMETRY.json").read_text());c=np.array(g["canonical8_m"]);contractbytes=(ROOT/"configs/model/action4_six_face_anchor_contract_v001.json").read_bytes();cdoc=json.loads(contractbytes);contract=FaceGeometryContract.from_bytes(contractbytes,(ROOT/cdoc["parent_asset"]["path"]).read_bytes())
 region_mode=region_input_mode(manifest);region_owners={};audit_frames=region_line_audit_frames(manifest);direct_mask,unknown_full_box,region_scales=external_region_options(manifest)
 cam=pym3t.DummyColorCamera("real_native_camera");K=np.array(manifest["rows"][0]["K"]);cam.intrinsics=pym3t.Intrinsics(K[0,0],K[1,1],K[0,2],K[1,2],3248,2464);cam.camera2world_pose=np.eye(4);owner=OwnedNativeCamera(cam,width=3248,height=2464)
 scene=pym3t.RendererGeometry("two_fixed_camera_bodies");tracker=pym3t.Tracker("native_RGB_region_texture");objects=[]
 for idx,track in enumerate(manifest["tracks"],1):
  body=pym3t.Body(track,str(out/(track+"_body.yaml")));assert scene.AddBody(body)
  model=pym3t.RegionModel(track+"_region_model",body,str(modelpath))
  region_cam=cam
  if region_mode=="SAM_visible_support_image":
   region_cam=pym3t.DummyColorCamera(track+"_SAM_visible_support_camera");region_cam.intrinsics=cam.intrinsics;region_cam.camera2world_pose=np.eye(4)
   region_owners[track]=OwnedNativeCamera(region_cam,width=3248,height=2464)
  region=pym3t.RegionModality(track+"_region",body,region_cam,model)
  if direct_mask:
   if not hasattr(region,"external_mask_likelihood"):raise RuntimeError("External mask likelihood native binary unavailable")
   region.external_mask_likelihood=True
   assert region.external_mask_likelihood
  if region_scales is not None:region.scales=region_scales

  depth=pym3t.FocusedBasicDepthRenderer(track+"_modeled_occlusion",scene,cam);assert depth.AddReferencedBody(body)
  region.ModelOcclusions(depth)
  silhouette=pym3t.FocusedSilhouetteRenderer(track+"_silhouette",scene,cam);assert silhouette.AddReferencedBody(body)
  texture=pym3t.TextureModality(track+"_texture",body,cam,silhouette);texture.ModelOcclusions(depth)
  texture.n_keyframes=texture_keyframe_budget(manifest)
  assert texture.n_keyframes==texture_keyframe_budget(manifest)
  protection=keyframe_protection(manifest)
  if hasattr(texture,"retain_nonempty_keyframe"):
   texture.retain_nonempty_keyframe=protection
   assert texture.retain_nonempty_keyframe==protection
  elif protection:raise ValueError("Native binary lacks requested keyframe protection")
  refresh=pose_cache_refresh(manifest)
  if hasattr(texture,"refresh_result_pose_cache"):
   texture.refresh_result_pose_cache=refresh
   assert texture.refresh_result_pose_cache==refresh
  elif refresh:raise ValueError("Native binary lacks result-time pose refresh")
  if hasattr(texture,"defer_keyframe_commit"):
   texture.defer_keyframe_commit=deferred
   assert texture.defer_keyframe_commit==deferred
  elif deferred:raise ValueError("Native binary lacks keyframe transaction interface")
  weight_mode=manifest.get("texture_weight_mode","legacy")
  if weight_mode=="tukey_derivative":texture.tukey_derivative_weights=True
  elif weight_mode!="legacy":raise ValueError("Unregistered texture weight mode")
  link=pym3t.Link(track+"_link",body);link.maintain_rigid_pose=True;assert link.AddModality(region);assert link.AddModality(texture);optimizer=pym3t.Optimizer(track+"_optimizer",link);assert tracker.AddOptimizer(optimizer)
  anchor=pym3t.Link(track+"_A0_anchor",None,free_directions=[False]*6);assert anchor.SetUp()
  fixed_initializer=pym3t.Optimizer(track+"_fixed_anchor_initializer",anchor);assert fixed_initializer.SetUp()
  prior=pym3t.SoftConstraint(track+"_current_A0_prior",anchor,link,np.eye(4),np.eye(4),[True]*6,
   .035,.003,PRIOR_ROTATION_SCALE,PRIOR_TRANSLATION_SCALE)
  assert optimizer.AddSoftConstraint(prior)
  objects.append(dict(fixed_initializer=fixed_initializer,anchor=anchor,prior=prior,track=track,native_id=idx,body=body,model=model,region=region,texture=texture,depth=depth,silhouette=silhouette,link=link,optimizer=optimizer))
 params=dict(texture_weight_mode=manifest.get("texture_weight_mode","legacy"),tracker_correlation_iterations=tracker.n_corr_iterations,tracker_update_iterations=tracker.n_update_iterations,texture={k:getattr(objects[0]["texture"],k) for k in ["focused_image_size","descriptor_distance_threshold","orb_n_features","orb_scale_factor","orb_n_levels","n_keyframes","max_keyframe_age","tukey_norm_constant"]},optimizer=dict(rotation=objects[0]["optimizer"].tikhonov_parameter_rotation,translation=objects[0]["optimizer"].tikhonov_parameter_translation),region_model=model_receipt["parameters"],modeled_peer_occlusions=True,measured_depth=False)
 launch=dict(region_input_mode=region_mode,external_mask_likelihood=direct_mask,region_unknown_full_box=unknown_full_box,region_search_scales=list(objects[0]["region"].scales),status="REAL_NATIVE_TRACK_STARTED_NOT_COMPLETE",pid=os.getpid(),start_unix=time.time(),CPU_affinity=sorted(os.sched_getaffinity(0)),frames=manifest["frame_ids"],tracks=manifest["tracks"],parameters=params,model_cache_sha256=model_receipt["sha256"],RGB_input_manifest_sha256=hashlib.sha256((out/"INPUT_MANIFEST.json").read_bytes()).hexdigest(),SAM_consumed_by_native=True,initialize_only_first_frame=True,A0_prior_mode=prior_mode,defer_keyframe_commit=deferred,recovery_mode=recovery_mode,keyframe_host_policy=host_policy,native_debug_frame_ids=sorted(debug_frames),smoothing=False,published=False)
 (out/"TRACK_LAUNCH.json").write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
 from model.action4_native_recovery import NativeRecoveryController,DEFAULT_POLICY
 controllers={obj["track"]:NativeRecoveryController(obj["track"],0,dict(DEFAULT_POLICY),support_count_kind="matched_within_native_scale",allow_sam_only_continuation=True) for obj in objects}
 from model.action4_region_observation_gate import RegionObservationGate
 region_gates={obj["track"]:RegionObservationGate(obj["track"]) for obj in objects}
 default_region_lines={obj["track"]:obj["region"].n_lines_max for obj in objects}
 rotation_histories={}
 if prior_mode=="history_rotation":
  from model.action4_native_rotation_history import NativeRotationHistory
  rotation_histories={obj["track"]:NativeRotationHistory(obj["track"]) for obj in objects}
 start_frame=manifest["frame_ids"][0]
 render=out/"native_render";render.mkdir(exist_ok=True);viewer=None;allrows=[];events=[];setup_seconds=None
 with (out/"POSES.jsonl").open("x") as stream,(out/"FRAME_EVENTS.jsonl").open("x") as event_stream:
  for frame in manifest["rows"]:
   fid=frame["frame_id"]
   from data.action4_native_png_io import read_native_bgr
   pixels=read_native_bgr(ROOT,frame["source_png"],shape_hw=frame["shape"][:2])

   for obj in objects:
    if audit_frames:
     if not hasattr(obj["region"],"set_observation_audit"):raise RuntimeError("Requested native region audit binary unavailable")
     obj["region"].set_observation_audit(fid in audit_frames)
   event=dict(frame_id=fid,pts=frame["pts"],time_base=frame["time_base"],calls=[],exceptions=[]);start=time.perf_counter();native_ok=True;poses=[]
   support=support_by_frame[fid];sp=ROOT/support["support"]["path"]
   assert sp.resolve().is_relative_to(ROOT) and hashlib.sha256(sp.read_bytes()).hexdigest()==support["support"]["sha256"]
   with np.load(sp,allow_pickle=False) as arrays:
    masks={obj["track"]:arrays[obj["track"]].copy() for obj in objects}
    raw_masks={obj["track"]:arrays["raw_"+obj["track"]].copy() for obj in objects}
   derived_images={};derived_audit={}
   if region_mode=="SAM_visible_support_image":
    from model.action4_region_support_image import region_support_images
    derived_images,derived_audit=region_support_images(masks,track_ids=manifest["tracks"],ambiguous_pixels=sum(support["audit"][tid]["ambiguous_pixels_excluded"] for tid in manifest["tracks"]),unknown_boxes={i["track_id"]:i["bbox_for_a0"] for i in frame["instances"]} if unknown_full_box else None)
   with ExitStack() as leases:
    leases.enter_context(owner.frame(pixels,frame_id=fid,source_sha256=frame["source_png"]["sha256"]))
    for tid,region_owner in region_owners.items():
     leases.enter_context(region_owner.frame(derived_images[tid],frame_id=fid,source_sha256=support["support"]["sha256"]))
    event["region_input_mode"]=region_mode;event["region_support_input"]=derived_audit
    del pixels,derived_images
    try:
     if fid==start_frame:
      ts=time.perf_counter();ok=tracker.SetUp();setup_seconds=time.perf_counter()-ts;event["calls"].append(dict(name="SetUp",ok=bool(ok),seconds=setup_seconds));assert ok
      for obj,inst in zip(objects,frame["instances"]):
       T=np.eye(4);T[:3,:3]=np.array(inst[native_pose_key(manifest)]["rotation"]);T[:3,3]=inst[native_pose_key(manifest)]["translation_m"];obj["body"].body2world_pose=T
       assert np.allclose(owner.pose(obj["body"]),T,atol=1e-7,rtol=0)
       event["calls"].append(dict(name="initialize_raw_A0_once" if native_pose_key(manifest)=="raw_A0_pose" else "initialize_selected_A0_once",track_id=obj["track"],native_body_id_configured=obj["native_id"]))
      viewer=pym3t.NormalColorViewer("native_two_bodies",cam,scene,.5);viewer.display_images=False;viewer.StartSavingImages(str(render),"png");assert viewer.SetUp()
     from model.action4_current_pose_prior import current_pose_prior,native_prior_directions
     event["A0_prior_consumption"]=[]
     for obj,inst in zip(objects,frame["instances"]):
      packet=current_pose_prior(inst,track_id=obj["track"],frame_id=fid,pts=frame["pts"],canonical8=c,pose_key=native_pose_key(manifest))
      directions=native_prior_directions("translation_only" if prior_mode=="history_rotation" else prior_mode,packet["available"])
      history_prediction=None
      if prior_mode=="history_rotation":
       history_prediction=rotation_histories[obj["track"]].predict(obj["track"],fid,frame["pts"],frame["time_base"])
       directions[:3]=[packet["available"] and history_prediction["available"]]*3
       packet["rotation_history"]=history_prediction
       packet["source"]="history_rotation_and_current_A0_translation"
      packet["constraint_enabled"]=any(directions)
      packet["constraint_directions_rotation_then_translation"]=directions
      packet["prior_mode"]=prior_mode
      obj["prior"].directions=directions
      assert obj["prior"].SetUp()
      assert obj["anchor"].CalculateGradientAndHessian()
      if packet["available"]:
       anchor_pose=np.asarray(packet["pose"]).copy()
       if history_prediction is not None and history_prediction["available"]:
        anchor_pose[:3,:3]=np.asarray(history_prediction["rotation"])
       obj["anchor"].link2world_pose=anchor_pose
      packet["rotation_scale_rad"]=PRIOR_ROTATION_SCALE;packet["translation_scale_m"]=PRIOR_TRANSLATION_SCALE
      packet["anchor_pose_consumed"]=np.asarray(obj["anchor"].link2world_pose).tolist()
      event["A0_prior_consumption"].append(packet)
     event["region_observation_gate"]=[]
     for obj,inst,prior_packet in zip(objects,frame["instances"],event["A0_prior_consumption"]):
      pose=owner.pose(obj["body"]);xyz=c@pose[:3,:3].T+pose[:3,3];h=xyz@K.T
      native_box=None
      if np.isfinite(h).all() and (xyz[:,2]>1e-6).all():
       uv=h[:,:2]/h[:,2:];native_box=np.r_[uv.min(0),uv.max(0)].tolist()
      ev=inst["recovery_evidence"]
      gate=region_gates[obj["track"]].observe(track_id=obj["track"],frame_id=fid,native_box=native_box,
       b0_box=inst["raw_B0_box"],current_B0=ev["current_support_status"]["B0"],
       current_SAM=ev["current_support_status"]["SAM"],prior_available=prior_packet["available"],
       resolved_box=inst.get("bbox_for_a0"),resolved_current_support=bool(inst.get("resolved_ROI_current_support",False)))
      if region_mode=="SAM_visible_support_image" and not derived_audit[obj["track"]]["available"]:
       gate.update(enabled=False,reason="current_SAM_support_missing_no_region_evidence")
      obj["region"].n_lines_max=default_region_lines[obj["track"]] if gate["enabled"] else 0
      gate["actual_line_budget"]=obj["region"].n_lines_max
      event["region_observation_gate"].append(gate)
     before=[owner.pose(obj["body"]) for obj in objects]
     ts=time.perf_counter();ok=tracker.UpdateCameras(True);event["calls"].append(dict(name="UpdateCameras",ok=bool(ok),seconds=time.perf_counter()-ts));native_ok &= bool(ok)
     for obj in objects:
      assert obj["texture"].SetCurrentSupport(masks[obj["track"]],fid)
     event["texture_support_policy"]="current_SAM_visible_unique_owner_pixels"
     event["keyframe_state_before"]=keyframe_state(objects)
     ts=time.perf_counter()
     if fid==start_frame:ok=tracker.StartModalities(fid);name="StartModalities_reference_frame"
     else:ok=tracker.ExecuteTrackingStep(fid);name="ExecuteTrackingStep"
     event["calls"].append(dict(name=name,ok=bool(ok),seconds=time.perf_counter()-ts));native_ok &= bool(ok)
     event["keyframe_state_after"]=keyframe_state(objects)
     for obj,gate in zip(objects,event["region_observation_gate"]):
      gate.update(region_gradient_audit(obj["region"],tracking_step=fid!=start_frame))
      if gate["gradient_computed"] and not gate["enabled"]:
       assert gate["actual_gradient_norm"]==0 and gate["actual_hessian_norm"]==0
     if fid in audit_frames:
      event["region_native_observations"]=[]
      for obj in objects:
       reg=obj["region"];event["region_native_observations"].append(dict(track_id=obj["track"],frame_id=fid,
        correspondences_computed=fid!=start_frame,
        correspondence_pose=None if fid==start_frame else np.asarray(reg.audit_correspondence_pose()).tolist(),
        lines=[] if fid==start_frame else reg.audit_lines(),
        line_columns=["iteration","corr_iteration","center_u","center_v","normal_u","normal_v","mean_segment","measured_variance","delta_r_px","normal_component_to_scale","scale","line_length_px","handle_occlusions","continuous_distance"],
        histogram_pixels=reg.audit_histogram_pixels(),histogram_columns=["u","v","foreground_label","b","g","r","center_u","center_v","normal_u","normal_v","handle_occlusions"],
        histogram_note="last histogram update, potentially at a later pose than last correspondence linearization",
        sampling_is_not_GT=True))
     event["native_texture_consumption"]=[]
     for obj in objects:
      tex=obj["texture"];points=np.asarray(tex.support_keypoints_native(),dtype=float).reshape(-1,2)
      xy=np.rint(points).astype(int);mask=masks[obj["track"]]
      assert not len(xy) or ((xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])).all()
      assert not len(xy) or (mask[xy[:,1],xy[:,0]]==255).all()
      event["native_texture_consumption"].append(dict(track_id=obj["track"],support_frame_id=tex.support_frame_id,
       before=tex.support_keypoints_before,after=tex.support_keypoints_after,points_native=points.tolist(),
       all_retained_points_in_current_support=True,source=support["audit"][obj["track"]]))
     for packet,obj in zip(event["A0_prior_consumption"],objects):
      packet["anchor_unchanged_after_updates"]=np.array_equal(np.asarray(obj["anchor"].link2world_pose),
        np.asarray(packet["anchor_pose_consumed"]))
      assert packet["anchor_unchanged_after_updates"]
     # Resolve EVERY body transaction before any recovery controller can reseed.
     event["keyframe_host_transactions"]=[]
     if deferred:
      for obj,inst,previous in zip(objects,frame["instances"],before):
       event["keyframe_host_transactions"].append(resolve_keyframe_transaction(
        obj=obj,inst=inst,frame=frame,before_pose=previous,native_ok=native_ok,canonical=c,K=K,
        owned_mask=masks[obj["track"]],body_mask=raw_masks[obj["track"]],support=support,policy=host_policy))
     event["keyframe_state_after_host"]=keyframe_state(objects)
     event["recovery_decisions"]=[]
     event["texture_correspondence_audit"]=[]
     for obj,inst in zip(objects,frame["instances"]):
      observed=np.asarray(obj["texture"].observation_audit(),dtype=float).reshape(-1,10)
      if not np.isfinite(observed).all():raise RuntimeError("Nonfinite actual native correspondence/weight")
      within=(observed[:,4]<=obj["texture"].tukey_norm_constant)&(observed[:,5]>0) if len(observed) else np.zeros(0,bool)
      matched_count=int(within.sum())
      event["texture_correspondence_audit"].append(dict(track_id=obj["track"],frame_id=fid,
       last_linearization=True,total_correspondences=len(observed),within_native_scale_count=matched_count,
       native_scale_px=obj["texture"].tukey_norm_constant,total_native_weight=float(observed[:,5].sum()) if len(observed) else 0.,
       columns=["observed_u","observed_v","projected_u","projected_v","error_px","objective_weight","variance","body_x_m","body_y_m","body_z_m"],
       observations=observed.tolist()))

      before_pose=owner.pose(obj["body"])
      xyz=c@before_pose[:3,:3].T+before_pose[:3,3];uv=xyz@K.T
      box=None
      if np.isfinite(uv).all() and (xyz[:,2]>1e-6).all():
       uv=uv[:,:2]/uv[:,2:];box=np.r_[uv.min(0),uv.max(0)].tolist()
      ev=inst["recovery_evidence"];bb=inst["raw_B0_box"]
      from eval.action4_native_recovery_inputs import current_recovery_inputs
      recovery_args=current_recovery_inputs(dict(ev,track_id=obj["track"],bbox_b0_raw=bb),
       raw_masks[obj["track"]],frame_id=fid,pts=frame["pts"],track_id=obj["track"])
      seed=np.eye(4);seed[:3,:3]=inst[native_pose_key(manifest)]["rotation"];seed[:3,3]=inst[native_pose_key(manifest)]["translation_m"]
      decision=controllers[obj["track"]].observe(frame_id=fid,pts=frame["pts"],track_id=obj["track"],generation=0,
       texture_points=matched_count,native_box=box,seed_pose=seed,canonical8=c,K=K,**recovery_args)
      decision["native_pose_before_decision"]=before_pose.tolist()
      decision["current_evidence"]=ev
      decision["recovery_mode"]=recovery_mode
      if decision["action"]=="reseed" and recovery_mode=="record_only_no_reseed":
       controllers[obj["track"]].acknowledge(fid,False)
       decision.update(proposed_action="reseed",action="record_only_no_reseed",native_restart_succeeded=False,
        action_suppressed=True,region_histogram_rebuilt_at_current_seed=False)
      if decision["action"]=="reseed":
       assert ev["pose_valid"] and obj["texture"].support_frame_id==fid
       obj["body"].body2world_pose=seed
       # A qualified reinitialization may rebuild histograms from the current seed.
       obj["region"].n_lines_max=default_region_lines[obj["track"]]
       decision["region_histogram_rebuilt_at_current_seed"]=True
       success=bool(pym3t.RestartSupportedBody(obj["region"],obj["texture"],fid))
       controllers[obj["track"]].acknowledge(fid,success)
       decision["native_restart_succeeded"]=success
       if not success:raise RuntimeError("Native per-body restart failed; keep failure, no silent continuation")
       event["calls"].append(dict(name="RestartSupportedBody_current_evidence",track_id=obj["track"],ok=True))
      decision["native_pose_after_decision"]=owner.pose(obj["body"]).tolist()
      event["recovery_decisions"].append(decision)
     if prior_mode=="history_rotation":
      event["rotation_history_observations"]=[]
      for obj,inst,decision in zip(objects,frame["instances"],event["recovery_decisions"]):
       matched=next(x["within_native_scale_count"] for x in event["texture_correspondence_audit"] if x["track_id"]==obj["track"])
       current=inst["recovery_evidence"]["current_support_status"]
       history_event=rotation_histories[obj["track"]].observe(obj["track"],fid,frame["pts"],frame["time_base"],
        owner.pose(obj["body"])[:3,:3],matches=matched,current_support=bool(current.get("B0") or current.get("SAM")),
        native_ok=bool(native_ok),reseeded=decision["action"]=="reseed")
       history_event["track_id"]=obj["track"];event["rotation_history_observations"].append(history_event)
     if viewer is not None and fid in debug_frames:
      ts=time.perf_counter();ok=viewer.UpdateViewer(fid);event["calls"].append(dict(name="UpdateViewer",ok=bool(ok),seconds=time.perf_counter()-ts));native_ok &= bool(ok)
    except Exception as e:
     # Explicitly discard pending proposals on failed native/host review; never auto-commit.
     event["pending_exception_rejections"]=[]
     for failed_obj in objects:
      failed_tex=failed_obj["texture"]
      if hasattr(failed_tex,"pending_keyframe") and failed_tex.pending_keyframe:
       rejected=bool(failed_tex.RejectPendingKeyframe(int(failed_tex.pending_keyframe_frame)))
       event["pending_exception_rejections"].append(dict(track_id=failed_obj["track"],rejected=rejected))
     native_ok=False;event["exceptions"].append(type(e).__name__+":"+str(e));(out/f"exception_{fid}.txt").write_text(traceback.format_exc())
    poses=[owner.pose(obj["body"]) for obj in objects]
   event["region_camera_audit"]={tid:camera.audit[-1] for tid,camera in region_owners.items()}
   event.update(wall_seconds=time.perf_counter()-start,owner_audit=owner.audit[-1],native_calls_ok=native_ok);events.append(event)
   for obj,inst,T in zip(objects,frame["instances"],poses):
    posefile=out/f"pose_{fid}_{obj['track']}.npy";np.save(posefile,T,allow_pickle=False)
    record=dict(frame_id=fid,pts=frame["pts"],time_base=frame["time_base"],track_id=obj["track"],native_body_id_configured=obj["native_id"],body_metafile=str((out/(obj["track"]+"_body.yaml")).relative_to(ROOT)),native_body_id_readback_available=False,source_RGB=frame["source_png"],source_SAM_status=inst["SAM_status"],source_SAM_mask_sha256=inst["SAM_mask_sha256"],source_B0_box=inst["raw_B0_box"],SAM_consumed_by_native=True,raw_A0_reset_this_frame=fid==start_frame,native_call_ok=native_ok,pose_native_file=str(posefile.relative_to(ROOT)),pose_native_sha256=hashlib.sha256(posefile.read_bytes()).hexdigest(),native_pose=[[float(v) if np.isfinite(v) else None for v in row] for row in T],raw_baseline_pose=inst["raw_A0_pose"],raw_baseline_corners=inst["raw_corners"],final_corners=inst["raw_corners"] if native_pose_key(manifest)=="raw_A0_pose" else inst["selected_corners"],native_pose_input_kind=manifest.get("native_pose_input_kind","raw_A0"),selected_A0_input_pose=inst[native_pose_key(manifest)],published=False,frame_wall_seconds=event["wall_seconds"],status="native_numeric_candidate_not_accuracy")
    decision=next((d for d in event.get("recovery_decisions",[]) if d["track_id"]==obj["track"]),None)
    record["rigid_state_maintenance"]={"enabled":obj["link"].maintain_rigid_pose,"count":obj["link"].rigid_projection_count,"max_element_change":obj["link"].rigid_projection_max_change}
    record["recovery_decision"]=decision
    record["raw_A0_reset_this_frame"]=fid==start_frame or (decision is not None and decision["action"]=="reseed" and decision.get("native_restart_succeeded"))
    try:
     conversion=native_float32_se3_for_projection(T,c,K)
     record["representation_conversion"]={k:v for k,v in conversion.items() if not isinstance(v,np.ndarray)}
     converted=conversion["converted_matrix"];record["converted_pose"]=converted.tolist()
     record["six_faces"]=project_six_faces(c,contract,converted[:3,:3],converted[:3,3],K,units="m",pixel_space_id="raw_video_native_integer_centers",image_size_wh=[3248,2464],min_depth_m=1e-6,grazing_cos_threshold=.05)
     record["corners_candidate"]=[v["native_uv"] for v in record["six_faces"]["corners"]]
     record["all8_positive_depth"]=all(v["positive_depth"] for v in record["six_faces"]["corners"])
     if not record["all8_positive_depth"]:record["status"]="native_numeric_invalid"
    except Exception as e:record.update(status="native_numeric_invalid",numeric_error=type(e).__name__+":"+str(e),corners_candidate=None)
    if not native_ok:record["status"]="native_call_failure_retained"
    allrows.append(record);stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
   (out/"TRACK_PROGRESS.json").write_text(json.dumps(dict(frame_id=fid,records=len(allrows),wall_seconds=event["wall_seconds"],native_ok=native_ok)))
   event_stream.write(json.dumps(event)+"\n");event_stream.flush()
   if not native_ok:
    (out/"FRAME_EVENTS.json").write_text(json.dumps(events,indent=2))
    raise RuntimeError("Actual native engineering failure retained; stop this diagnostic before further frames")

 (out/"FRAME_EVENTS.json").write_text(json.dumps(events,indent=2))
 assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==model_receipt["sha256"]
 summary=dict(texture_support_enabled=True,region_SAM_consumption=region_mode=="SAM_visible_support_image",region_input_mode=region_mode,status="REAL_NATIVE_CLIP_DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE",records=len(allrows),frames=len(events),native_call_failed_frames=[e["frame_id"] for e in events if not e["native_calls_ok"]],numeric_invalid=[dict(frame_id=r["frame_id"],track_id=r["track_id"]) for r in allrows if r["status"]!="native_numeric_candidate_not_accuracy"],default_model_loaded_for_both=True,model_cache_unchanged=True,model_generation_seconds=model_receipt["seconds"],setup_cached_seconds=setup_seconds,frame_wall_seconds=[e["wall_seconds"] for e in events],actual_tracking_steps=sum(any(c["name"]=="ExecuteTrackingStep" for c in e["calls"]) for e in events),initializations=2,recovery_counts={k:v.recovery_count for k,v in controllers.items()},SAM_consumed_by_native=True,raw_final_preserved=True,published=False,real_accuracy_accepted=False,finish_unix=time.time())
 (out/"TRACK_COMPLETE.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
