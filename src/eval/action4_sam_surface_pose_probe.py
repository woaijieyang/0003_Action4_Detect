"""Paired surface-PnP diagnosis from one frozen SAM80 forward per original frame."""
import json,os,sys
from pathlib import Path
def main():
 R=Path("/data/zy/vln/a-test");assert Path.cwd().resolve()==R
 out=R/"outputs/action4_sam80_surface_pnp_probe_v001"
 source=R/"outputs/action4_sam80_fixed_video_v001/source"
 os.environ["ACTION4_PROJECT_ROOT"]=str(R)
 sys.path[:0]=[str(source/"runtime/surface_deps"),str(source/"src")]
 import numpy as np,torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 from eval.action4_roi_video_job import verify_source
 verify_source(source)
 from eval.action4_roi_sweep import read_binding
 from eval.action4_reanchor_roi_provider import ReanchorROIProvider
 from eval.action4_staged_pipeline import StagedPipeline
 from eval.action4_video_sam_inputs import VideoSamFeatureContext
 from model.action4_surface_pnp import decode_surface_pose
 from data.action4_adapter import DataAdapter
 cfg=json.loads((R/"outputs/action4_sam80_fixed_video_v001/CONFIG.json").read_text())
 lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
 provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
 pipe=StagedPipeline.from_bundles(pair["models"]["A0"]["path"],pair["models"]["B0"]["path"],device="cpu")
 pipe.b0_scoring_policy=lock["image_eval"]["B0_scoring_policy"]
 K=np.asarray(read_binding(lock["video"]["K"])["K_native"]);cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000
 captured=[]
 def observe(module,args,result):
  for xyz,mask in zip(result["xyz"].detach().cpu().numpy(),result["masks"][:,0].detach().sigmoid().cpu().numpy()):
   captured.append((xyz.copy(),mask.copy()))
 hook=pipe.a0.geometry.surface.register_forward_hook(observe)
 rows=[];frames=[4,104,108,120,180,232,260,355]
 try:
  for fid in frames:
   captured.clear()
   rgb,context,features=VideoSamFeatureContext.load(provider,fid,K,expected_b0_sha256=pipe.identity["B0_bundle_sha256"],
    expected_sam_model_sha256=pipe.acfg["sam_cache"]["expected_sources"]["sam_model_sha256"])
   pred=pipe.predict_video(rgb,K,cad,video_context=context,roi_expansion=1.5,sam_video_context=features)
   if len(captured)!=len(pred["instances"]):raise ValueError("Surface output order/coverage mismatch")
   specs=pred["a0_roi_specs"]
   if len(specs)!=len(captured):raise ValueError("ROI count mismatch")
   for inst,spec,(xyz,prob) in zip(pred["instances"],specs,captured):
    if inst["track_id"]!=spec["track_id"]:raise ValueError("Instance ROI identity mismatch")
    if xyz.shape!=(3,112,112) or prob.shape!=(112,112):raise ValueError("Unexpected surface grid")
    diagnostic=decode_surface_pose(xyz,prob,np.asarray(spec["K112"]),cad)
    native=None
    if diagnostic["valid"]:
     native=DataAdapter.transform_points(np.asarray(diagnostic["corners_hm"]),np.linalg.inv(np.asarray(spec["A112"]))).tolist()
    name=f"frame_{fid}_{inst['track_id']}.npz"
    np.savez_compressed(out/name,xyz=xyz,visible_probability=prob,K112=np.asarray(spec["K112"]),A112=np.asarray(spec["A112"]),canonical_m=cad)
    rows.append(dict(frame_id=fid,track_id=inst["track_id"],bbox_for_a0=inst["bbox_for_a0"],roi_spec=spec,
     raw_network_pose=inst["raw_network_pose"],raw_corners=inst["corners_px"],surface_pnp=diagnostic,
     surface_corners_native=native,surface_file=name,corner_ids=list(range(8)),
     no_pose_derived_XYZ=True,no_ground_truth=True))
   (out/"PROGRESS.json").write_text(json.dumps(dict(frame_id=fid,records=len(rows))))
 finally:hook.remove()
 (out/"RESULT.json").write_text(json.dumps(dict(status="ACTUAL_FROZEN_SAM80_SURFACE_PNP_DIAGNOSTIC_COMPLETE",
  models=pair["models"],inference_source=str(source.relative_to(R)),frames=frames,forward_calls=len(frames),
  surface_is_model_prediction=True,XYZ_not_backprojected_from_pose=True,weights_changed=False,default_changed=False,
  accuracy_accepted=False,rows=rows),indent=2,allow_nan=False))
if __name__=="__main__":main()
