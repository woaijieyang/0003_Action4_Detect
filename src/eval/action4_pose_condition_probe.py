"""Observe raw6D rotation conditioning on saved-video inputs. No weights/poses altered."""
import json,os,sys,hashlib
from pathlib import Path

def main():
    R=Path("/data/zy/vln/a-test");assert Path.cwd().resolve()==R
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    out=(R/args.output).resolve(strict=True)
    if not out.is_relative_to(R):raise ValueError("Diagnostic output outside project")
    source=R/"outputs/action4_sam80_fixed_video_v001/source"
    os.environ["ACTION4_PROJECT_ROOT"]=str(R)
    sys.path[:0]=[str(source/"runtime/surface_deps"),str(source/"src")]
    import torch,numpy as np
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from eval.action4_roi_video_job import verify_source
    verify_source(source)
    from eval.action4_roi_sweep import read_binding
    from eval.action4_reanchor_roi_provider import ReanchorROIProvider
    from eval.action4_staged_pipeline import StagedPipeline
    from eval.action4_video_sam_inputs import VideoSamFeatureContext
    cfg=json.loads((R/"outputs/action4_sam80_fixed_video_v001/CONFIG.json").read_text())
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    pipe=StagedPipeline.from_bundles(pair["models"]["A0"]["path"],pair["models"]["B0"]["path"],device="cpu")
    pipe.b0_scoring_policy=lock["image_eval"]["B0_scoring_policy"]
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000
    captured=[]
    hook=pipe.a0.geometry.pose.register_forward_hook(lambda module,args,result:captured.extend(result[0].detach().cpu().double().numpy().tolist()))
    rows=[]
    try:
        for fid in [4,259,260]:
            captured.clear()
            rgb,context,features=VideoSamFeatureContext.load(provider,fid,K,
                expected_b0_sha256=pipe.identity["B0_bundle_sha256"],
                expected_sam_model_sha256=pipe.acfg["sam_cache"]["expected_sources"]["sam_model_sha256"])
            prediction=pipe.predict_video(rgb,K,cad,video_context=context,roi_expansion=1.5,sam_video_context=features)
            if len(captured)!=len(prediction["instances"]):raise ValueError("Raw6D instance order/coverage differs")
            for inst,six in zip(prediction["instances"],captured):
                a,b=np.asarray(six[:3]),np.asarray(six[3:]);na=np.linalg.norm(a);nb=np.linalg.norm(b)
                cross=np.linalg.norm(np.cross(a/na,b)) if na else 0.
                rows.append(dict(frame_id=fid,track_id=inst["track_id"],rot6d=six,x_norm=float(na),y_norm=float(nb),
                    orthogonal_y_norm=float(cross),sin_between_vectors=float(cross/nb) if nb else None,
                    raw_pose=inst["raw_network_pose"],corners=inst["corners_px"],
                    bbox_for_a0=inst["bbox_for_a0"],sam_input=inst["sam_input"]))
    finally:hook.remove()
    result={"status":"ACTUAL_CPU_ROTATION_CONDITION_OBSERVATION_COMPLETE","models":pair["models"],
        "inference_source":str(source.relative_to(R)),"frames":[4,259,260],"pipeline_calls":3,
        "pose_changed_by_hook":False,"weights_updated":False,"device":"cpu","real_GT":False,"rows":rows}
    (out/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
if __name__=="__main__":main()
