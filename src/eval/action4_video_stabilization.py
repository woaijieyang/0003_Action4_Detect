"""Apply the same video-only pose operator to saved real inference; no model calls."""
import copy,hashlib,json,math
from pathlib import Path
from fractions import Fraction
from contextlib import ExitStack
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_native_rotation_history import rotation_log

def bound(path):
 p=contained_path(R,path,exists=True)
 return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(ref):
 p=contained_path(R,ref["path"],exists=True);data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=ref["sha256"]:raise ValueError("Bound source changed")
 return json.loads(data)

def validate_processed_row(raw,candidate,K,cad,config_id):
 for k in ("frame_id","pts","RGB_sha256","roi_expansion"):
  if raw[k]!=candidate[k]:raise ValueError("Stabilized frame identity changed")
 a=raw["prediction"]["instances"];b=candidate["prediction"]["instances"]
 if [(x["track_id"],x["prediction_id"]) for x in a]!=[(x["track_id"],x["prediction_id"]) for x in b]:raise ValueError("Track domain changed")
 meta=candidate["prediction"]["video_pose_stabilization"]
 if meta["config_id"]!=config_id:raise ValueError("Stabilizer identity changed")
 for x,y in zip(a,b):
  for k in ("raw_network_pose","corners_raw_from_resolved_roi","bbox_for_a0","bbox_b0_raw","bbox_sam_visible","roi_decision_id","corner_ids","pose_valid","published"):
   if x[k]!=y[k]:raise ValueError("Raw pose/ROI/publication changed: "+k)
  if y["post_A0_geometry"]==x["post_A0_geometry"]:
   if y["corners_final"]!=x["corners_final"]:raise ValueError("Unprocessed corners changed")
   continue
  if y["post_A0_geometry"]!="video_pose_stabilization_v1" or y["pose_stabilization"]["config_id"]!=config_id:
   raise ValueError("Unknown geometry stage")
  if y.get("pose_output_contract") is not None:
   from model.action4_final_pose_record import resolve_final_pose
   resolve_final_pose(y,cad,K)
  pose=y["final_pose"];rot=np.asarray(pose["rotation"]);t=np.asarray(pose["translation_m"])
  if pose["coordinate_frame"]!="object_to_capture_cv_m" or not np.allclose(rot.T@rot,np.eye(3),atol=1e-5,rtol=0):raise ValueError("Nonrigid final pose")
  camera=cad@rot.T+t
  if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):raise ValueError("Invalid projected body")
  h=camera@K.T;uv=h[:,:2]/h[:,2,None]
  if not np.allclose(uv,y["corners_final"],atol=1e-7,rtol=0) or y["corners_px"]!=y["corners_final"]:raise ValueError("Final corners not from saved single rigid pose")
 return True


def merge_saved_image_hint(row,hint_row,*,image_policy_id,mask_policy_id):
 if hint_row.get("schema")!="action4.mask_motion_hint.frame.v1":raise ValueError("Unknown mask-hint frame schema")
 for key in ("frame_id","pts","RGB_sha256"):
  if row[key]!=hint_row.get(key):raise ValueError("Saved image/hint frame identity differs")
 items=row["prediction"]["instances"];ids=[i["track_id"] for i in items]
 if len(set(ids))!=len(ids) or set(hint_row["observations"])!=set(ids):raise ValueError("Hint/instance domain differs")
 result={}
 for item in items:
  tid=item["track_id"];ev=copy.deepcopy(item["pose_stabilization"]["image_orientation_evidence"]);hint=copy.deepcopy(hint_row["observations"][tid])
  if ev["policy_id"]!=image_policy_id or hint["policy_id"]!=mask_policy_id:raise ValueError("Saved evidence policy differs")
  for key,value in (("track_id",tid),("frame_id",row["frame_id"]),("pts",row["pts"]),("current_RGB_sha256",row["RGB_sha256"])):
   if ev.get(key)!=value or hint.get(key)!=value:raise ValueError("Per-instance evidence identity differs")
  if hint.get("eligible_for_gain") is True and (ev.get("available") is not False or ev.get("hold_rotation") is not False):raise ValueError("Hint overrides known evidence")
  ev["mask_motion_hint"]=hint;result[tid]=ev
 return result


def saved_image_hint_stream(cfg):
 saved_ref=cfg["saved_image_observations_receipt"];saved=read(saved_ref)
 root=contained_path(R,saved["processed_predictions"]["path"],exists=True).parent.parent
 validate_render_receipt(str(root.relative_to(R)),cfg["input_lock"],saved_ref)
 if saved["raw_predictions"]!=cfg["raw_predictions"]:raise ValueError("Saved image evidence used another baseline")
 hint=read(cfg["mask_motion_hints_receipt"])
 if (hint.get("schema")!="action4.mask_motion_hint.receipt.v1" or hint.get("status")!="COMPLETE2850_MASK_MOTION_HINT_READY_FOR_DEV_GAIN_ONLY"
     or hint.get("frames")!=2850 or hint.get("observations")!=5700 or hint.get("input_lock")!=cfg["input_lock"]
     or hint.get("v3_stabilization_receipt")!=saved_ref or hint.get("v3_predictions")!=saved["processed_predictions"]
     or hint.get("raw_predictions")!=cfg["raw_predictions"] or hint.get("policy_id")!=cfg["filter"]["mask_motion_policy_id"]):
  raise ValueError("Incomplete or mismatched weak-motion receipt")
 path=contained_path(R,hint["hints"]["path"],exists=True)
 if bound(path)!=hint["hints"]:raise ValueError("Mask-hint stream changed")
 with contained_path(R,saved["processed_predictions"]["path"],exists=True).open() as a,path.open() as b:
  for fid in range(2850):
   la,lb=next(a,None),next(b,None)
   if la is None or lb is None:raise ValueError("Missing evidence frame")
   row,hrow=json.loads(la),json.loads(lb)
   if row["frame_id"]!=fid:raise ValueError("Out-of-order saved evidence")
   result=merge_saved_image_hint(row,hrow,image_policy_id=cfg["filter"]["image_orientation_policy_id"],mask_policy_id=hint["policy_id"])
   if fid==2849 and (next(a,None) is not None or next(b,None) is not None):raise ValueError("Extra evidence frames")
   yield result

def generate(config_path,*,source_ref):
 source_identity=read(source_ref)
 if not isinstance(source_identity.get("files"),dict):raise ValueError("Immutable postprocessor source required")
 cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
 if cfg["schema"]!="action4.video_stable_v1.config" or cfg["mode"]!="video_development":raise ValueError("Explicit video development config")
 out=contained_path(R,cfg["output"],exists=False);out.mkdir(exist_ok=False)
 lock=read(cfg["input_lock"]);manifest=read(lock["video"]["frames_manifest"]);K=np.asarray(read(lock["video"]["K"])["K_native"],float);cad=np.asarray(read(lock["CAD"])["corners_3d_mm"],float)/1000
 tb=Fraction(manifest["source"]["time_base"]);time_base=[tb.numerator,tb.denominator]
 raw_path=contained_path(R,cfg["raw_predictions"]["path"],exists=True);data=raw_path.read_bytes()
 if hashlib.sha256(data).hexdigest()!=cfg["raw_predictions"]["sha256"]:raise ValueError("Raw prediction identity changed")
 raw=[json.loads(line) for line in data.splitlines()]
 frames=manifest["frames"]
 if len(raw)!=2850 or len(frames)!=2850:raise ValueError("Full2850 domain required")
 from eval.action4_video_roi_contract import ratio_tag
 selected_ratio=1.5
 engine=VideoPoseStabilizer(StabilizerConfig(**cfg["filter"]))
 if engine.config.measurement_source=="selected_A0_pose":
  from eval.action4_corner_video_receipt import validate_trial_render
  receipt=validate_trial_render(str(raw_path.parent.parent.relative_to(R)),cfg["input_lock"],cfg["corner_trial_receipt"])
  if receipt["predictions"]!=cfg["raw_predictions"]:raise ValueError("Selected-A0 input receipt differs")
  selected_ratio=receipt.get("roi_expansion",1.5)
 observer=None
 if engine.config.image_orientation_enabled and not cfg.get("saved_image_observations_receipt"):
  from eval.action4_video_image_orientation import VideoImageOrientationObserver
  observer=VideoImageOrientationObserver(cfg["input_lock"],enabled=True,policy=cfg.get("image_orientation_policy"))
 orientation_iter=None if observer is None else observer.iter_observe((dict(frame_id=r['frame_id'],pts=r['pts'],RGB_sha256=r['RGB_sha256'],track_ids=tuple(x['track_id'] for x in r['prediction']['instances'])) for r in raw),workers=cfg.get('image_evidence_prefetch_workers',0))
 if cfg.get("saved_image_observations_receipt"):
  if not engine.config.image_orientation_enabled or not engine.config.mask_motion_gain_enabled:raise ValueError("Unused saved hint source")
  orientation_iter=saved_image_hint_stream(cfg)
 elif engine.config.mask_motion_gain_enabled:raise ValueError("Weak gain requires its frozen saved evidence")
 image_counts={"available_instances":0,"held_instances":0,"unknown_instances":0,"maximum_hold_run":0,"weak_gain_instances":0}
 image_reasons={}
 destination=out/ratio_tag(selected_ratio);destination.mkdir();previous={};events={"raw":0,"stable":0};angles={"raw":[],"stable":[]};ages=[];processed=0
 with ExitStack() as cleanup, (destination/"predictions.jsonl").open("x") as stream:
  if orientation_iter is not None:cleanup.callback(orientation_iter.close)
  for index,(row,frame) in enumerate(zip(raw,frames)):
   if row["frame_id"]!=index or row["frame_id"]!=frame["frame_id"] or row["pts"]!=frame["pts"] or row["RGB_sha256"]!=frame["image_sha256"] or row["roi_expansion"]!=selected_ratio:raise ValueError("Raw full-domain mismatch")
   result=copy.deepcopy(row)
   orientation=None if orientation_iter is None else next(orientation_iter)
   result["prediction"]=engine.apply_prediction(row["prediction"],frame_id=index,pts=row["pts"],time_base=time_base,K_native=K,canonical_m=cad,mode="video",image_orientation_observations=orientation)
   result["prediction_origin"]="same_frozen_A0_forward_then_video_pose_stabilization"
   validate_processed_row(row,result,K,cad,engine.config.identity)
   for a,b in zip(row["prediction"]["instances"],result["prediction"]["instances"]):
    tid=a["track_id"];state=b.get("pose_stabilization",{})
    if orientation is not None:
     ev=orientation[tid];image_counts["available_instances"]+=int(ev["available"])
     image_counts["unknown_instances"]+=int(not ev["available"])
     image_counts["held_instances"]+=int(state.get("image_rotation_held",False))
     image_counts["weak_gain_instances"]+=int(state.get("weak_silhouette_gain_applied",False))
     image_counts["maximum_hold_run"]=max(image_counts["maximum_hold_run"],state.get("image_hold_run",0))
     image_reasons[ev["reason"]]=image_reasons.get(ev["reason"],0)+1
    if state.get("status") not in ("initialized","filtered"):continue
    processed+=1;ages.append(index-state["selected_rotation_frame"])
    box=np.asarray(a["bbox_for_a0"]);center=(box[:2]+box[2:])/2;diagonal=np.linalg.norm(box[2:]-box[:2])
    matrices={"raw":np.asarray(a["raw_network_pose" if engine.config.measurement_source=="raw_network_pose" else "final_pose"]["rotation"]),"stable":np.asarray(b["final_pose"]["rotation"])}
    if tid in previous and previous[tid]["frame"]==index-1:
     old=previous[tid];motion=float(np.linalg.norm(center-old["center"])/max(diagonal,1))
     for name in matrices:
      angle=float(np.linalg.norm(rotation_log(matrices[name]@old[name].T))*180/np.pi);angles[name].append(angle)
      events[name]+=int(angle>30 and motion<.1)
    previous[tid]=dict(frame=index,center=center,**matrices)
   stream.write(json.dumps(result,allow_nan=False)+"\n")
 (destination/"COMPLETE.json").write_text(json.dumps(dict(status="FULL2850_STABILIZED_DEVELOPMENT_PREDICTIONS",frames=2850,roi_expansion=selected_ratio,tracking=True),indent=2))
 metrics=dict(frames=2850,processed_instances=processed,measurement_source=engine.config.measurement_source,
  large_angle_small_ROI_motion_events_not_GT=events,
  adjacent_rotation_degrees={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in angles.items()},
  selected_rotation_source_age_frames=dict(median=float(np.median(ages)),p95=float(np.percentile(ages,95)),maximum=max(ages)),
  real_pose_accuracy_measured=False,real_video_latency_measured=False,image_orientation_evidence=image_counts if engine.config.image_orientation_enabled else None,image_orientation_reasons=image_reasons if engine.config.image_orientation_enabled else None)
 (out/"DIAGNOSTICS.json").write_text(json.dumps(metrics,indent=2))
 complete=dict(status="POSE_STABILIZED_DEVELOPMENT_COMPLETE",frames_per_ratio=2850,ratios=[selected_ratio],
  video_stage=cfg.get("video_stage","stage5-A0-stable-v1"),video_label=cfg.get("video_label","Stable V1 / causal 5-frame pose / review"),
  raw_predictions=cfg["raw_predictions"],processed_predictions=bound(destination/"predictions.jsonl"),
  config=bound(config_path),filter_id=engine.config.identity,input_lock=cfg["input_lock"],model_calls=0,
  accuracy_accepted=False,renderer_changes_geometry=False,postprocess_source=source_ref)
 (out/"COMPLETE.json").write_text(json.dumps(complete,indent=2))
 receipt=dict(schema="action4.video_pose_stabilization.receipt.v1",complete=bound(out/"COMPLETE.json"),raw_predictions=cfg["raw_predictions"],
  processed_predictions=complete["processed_predictions"],config=complete["config"],input_lock=cfg["input_lock"],
  filter_id=engine.config.identity,frames=2850,roi_expansion=selected_ratio,validated_all_rows=True,model_calls=0,accuracy_accepted=False,postprocess_source=source_ref)
 (out/"STABILIZATION_RECEIPT.json").write_text(json.dumps(receipt,indent=2))
 return dict(receipt=bound(out/"STABILIZATION_RECEIPT.json"),diagnostics=metrics)

def validate_render_receipt(video_root,input_lock,reference):
 receipt=read(reference)
 if receipt.get("schema")!="action4.video_pose_stabilization.receipt.v1" or receipt.get("frames")!=2850 or receipt.get("validated_all_rows") is not True:
  raise ValueError("Complete validated stabilized domain required")
 root=contained_path(R,video_root,exists=True)
 if receipt["input_lock"]!=input_lock or contained_path(R,receipt["complete"]["path"],exists=True)!=root/"COMPLETE.json":raise ValueError("Stabilization receipt belongs to another input")
 complete=read(receipt["complete"]);config=read(receipt["config"])
 read(receipt["postprocess_source"])
 if receipt["postprocess_source"]!=complete["postprocess_source"]:raise ValueError("Postprocessor source identity differs")
 if complete["status"]!="POSE_STABILIZED_DEVELOPMENT_COMPLETE" or complete["processed_predictions"]!=receipt["processed_predictions"] or complete["filter_id"]!=receipt["filter_id"] or StabilizerConfig(**config["filter"]).identity!=receipt["filter_id"]:
  raise ValueError("Stabilized config/predictions binding differs")
 from eval.action4_video_roi_contract import ratio_tag
 selected_ratio=receipt.get("roi_expansion",1.5)
 if complete["ratios"]!=[selected_ratio]:raise ValueError("Stabilized ratio differs from receipt")
 if selected_ratio==1.15:
  if config['filter'].get('measurement_source')!='selected_A0_pose':raise ValueError('ROI1.15 stable output requires selected corner input')
  from eval.action4_corner_video_receipt import validate_trial_render
  raw_root=contained_path(R,receipt['raw_predictions']['path'],exists=True).parent.parent
  original=validate_trial_render(str(raw_root.relative_to(R)),input_lock,config['corner_trial_receipt'])
  if original.get('roi_expansion')!=1.15 or original['predictions']!=receipt['raw_predictions']:raise ValueError('ROI1.15 stabilization input source differs')

 if contained_path(R,receipt["processed_predictions"]["path"],exists=True)!=root/ratio_tag(selected_ratio)/"predictions.jsonl":raise ValueError("Wrong stabilized file")
 for ref in (receipt["processed_predictions"],receipt["raw_predictions"]):
  p=contained_path(R,ref["path"],exists=True)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=ref["sha256"]:raise ValueError("Saved predictions changed before render")
 return receipt
