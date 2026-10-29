"""Replay saved predicted XYZ with current SAM support; no model forward or GT."""
import os,sys,json
from pathlib import Path
def main():
 R=Path("/data/zy/vln/a-test");assert Path.cwd().resolve()==R
 out=R/"outputs/action4_sam80_surface_support_v001";parent=R/"outputs/action4_sam80_surface_pnp_probe_v001"
 source=R/"outputs/action4_sam80_fixed_video_v001/source"
 sys.path[:0]=[str(source/"runtime/surface_deps"),str(source/"src")];os.environ["ACTION4_PROJECT_ROOT"]=str(R)
 import numpy as np,torch
 torch.set_num_threads(2)
 from eval.action4_roi_video_job import verify_source
 from eval.action4_roi_sweep import read_binding
 from eval.action4_reanchor_roi_provider import ReanchorROIProvider
 from eval.action4_video_sam_inputs import VideoSamFeatureContext
 from data.action4_sam_roi_inputs import _native_nearest_grid,_sample_native_nearest_on_grid,array_digest
 from data.action4_adapter import DataAdapter
 from model.action4_surface_pnp import decode_surface_pose
 verify_source(source)
 cfg=json.loads((R/"outputs/action4_sam80_fixed_video_v001/CONFIG.json").read_text())
 lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
 provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
 K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
 # Source SAM identity comes from the existing frozen video input lock.
 result=json.loads((parent/"RESULT.json").read_text());rows=[]
 for fid in result["frames"]:
  old=[r for r in result["rows"] if r["frame_id"]==fid]
  # Reuse exactly the source identities consumed in the existing model probe.
  # No model load: SAM source is bound by the input lock.
  sam_sha=lock["video"]["sam_feature_context"]["expected_sam_model_sha256"] if "sam_feature_context" in lock["video"] else None
  if sam_sha is None:
   model_cfg=json.loads((R/"runtime/action4_runs/p2_a0_sam_prediction15_formal80_v001/source/resolved_config.json").read_text())
   sam_sha=model_cfg["sam_cache"]["expected_sources"]["sam_model_sha256"]
  rgb,context,features=VideoSamFeatureContext.load(provider,fid,K,expected_b0_sha256=pair["models"]["B0"]["sha256"],expected_sam_model_sha256=sam_sha)
  detections=context.select_for_a0(context.consume(rgb,K,b0_sha256=pair["models"]["B0"]["sha256"]))
  if [r["track_id"] for r in old]!=[d["track_id"] for d in detections]:raise ValueError("Saved and current instance identities differ")
  masks,receipt=features.consume(rgb,K,context,detections,b0_sha256=pair["models"]["B0"]["sha256"])
  for r,d in zip(old,detections):
   if list(d["bbox_for_a0"])!=r["bbox_for_a0"]:raise ValueError("Resolved ROI changed")
   with np.load(parent/r["surface_file"],allow_pickle=False) as z:
    xyz=z["xyz"];prob=z["visible_probability"];A=z["A112"];km=z["K112"];cad=z["canonical_m"]
   if not np.array_equal(A,np.asarray(r["roi_spec"]["A112"])) or not np.array_equal(km,np.asarray(r["roi_spec"]["K112"])):raise ValueError("Saved affine/camera mismatch")
   mask=masks.get(r["track_id"])
   if mask is None:
    rows.append(dict(frame_id=fid,track_id=r["track_id"],status="current_SAM_unavailable",raw_corners=r["raw_corners"],surface_corners_native=None));continue
   grid=_native_nearest_grid(mask.shape,A)[:112,:112]
   support=_sample_native_nearest_on_grid(mask,grid)[0].numpy().astype(bool)
   eligible=(prob>=.5)&np.isfinite(prob)&np.isfinite(xyz).all(0)&(xyz.min(0)>=-.05)&(xyz.max(0)<=1.05)
   filtered=np.where(support,prob,0)
   pose=decode_surface_pose(xyz,filtered,km,cad)
   native=DataAdapter.transform_points(np.asarray(pose["corners_hm"]),np.linalg.inv(A)).tolist() if pose["valid"] else None
   row=dict(r,surface_pnp=pose,surface_corners_native=native,original_surface_pnp=r["surface_pnp"],
    current_SAM_mask_sha256=array_digest(mask),support112_sha256=array_digest(support),
    eligible_before=int(eligible.sum()),eligible_after=int((eligible&support).sum()),
    excluded_outside_current_SAM=int((eligible&~support).sum()),status="same_XYZ_current_SAM_intersection",
    actual_support_receipt=receipt,hidden_corner_heatmaps_modified=False)
   rows.append(row)
 (out/"RESULT.json").write_text(json.dumps(dict(status="SAVED_XYZ_SAM_SUPPORT_REPLAY_COMPLETE",rows=rows,models=result["models"],frames=result["frames"],new_model_calls=0,weights_updated=False,GT_used=False,source=str(parent.relative_to(R))),indent=2,allow_nan=False))
if __name__=="__main__":main()
