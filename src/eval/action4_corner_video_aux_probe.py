"""Same selected A0 and resolved ROI; counterfactual auxiliary mask only."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from eval.action4_roi_sweep import binding,read_binding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_video_sam_inputs import VideoSamFeatureContext,prepare_video_pipeline_sam
from eval.action4_sam_pipeline_inputs import device_mask_aux
from model.action4_release_bundle import load_bundle
from model.action4_sam_feature_adapter import MaskAux
from model.action4_native_rotation_history import rotation_log
from data import DataAdapter
def run(config_path):
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    alloc=cfg["allocation"]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=alloc["gpu_uuid"] or set(os.sched_getaffinity(0))!=set(alloc["cpus"]):
        raise ValueError("Resource identity mismatch")
    from model.training.action4_resource_policy import validate_device_policy
    validate_device_policy({"training_gpus":[],"eval_gpus":[alloc["gpu"]]},ROOT)
    import subprocess
    state=subprocess.check_output(["nvidia-smi","-i",str(alloc["gpu"]),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if state[0].strip()!=alloc["gpu_uuid"] or int(state[1])<16384:raise ValueError("Insufficient fresh headroom")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.use_deterministic_algorithms(True)
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    a=pair["models"]["A0"]
    model,base,weights=load_bundle(a["path"],"A0","cuda:0",allow_corner_candidate=True,expected_sha256=a["sha256"])
    model.eval().requires_grad_(False);base.eval().requires_grad_(False)
    versions=lambda:tuple((name,t._version) for name,t in list(model.named_parameters())+list(model.named_buffers()))
    before=versions()
    rotation6d=[]
    rotation_hook=None
    if cfg.get("capture_rotation6d"):
        rotation_hook=model.head.pose.fc_r.register_forward_hook(
            lambda module,args,result:rotation6d.append(result.detach().cpu().numpy().copy()))
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"]);cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000
    source=read_binding(cfg["video_receipt"]);path=contained_path(ROOT,source["predictions"]["path"],exists=True)
    if binding(path)!=source["predictions"]:raise ValueError("Source prediction differs")
    wanted=set(cfg["frames"]);old={}
    for line in path.open():
        row=json.loads(line)
        if row["frame_id"] in wanted:old[row["frame_id"]]=row
    if set(old)!=wanted:raise ValueError("Missing selected actual frame")
    rows=[]
    with torch.inference_mode():
      for fid in cfg["frames"]:
        rgb,context,features=VideoSamFeatureContext.load(provider,fid,K,
            expected_b0_sha256=pair["models"]["B0"]["sha256"],
            expected_sam_model_sha256=weights["config"]["sam_cache"]["expected_sources"]["sam_model_sha256"])
        detections=context.select_for_a0(context.consume(rgb,K,b0_sha256=pair["models"]["B0"]["sha256"]))
        inputs,specs=prepare_video_pipeline_sam(rgb,K,detections,features,video_context=context,
            b0_sha256=pair["models"]["B0"]["sha256"],roi_expansion=1.5)
        rois=inputs.rgb.to("cuda:0");km=torch.as_tensor(np.stack(inputs.K112),device="cuda:0",dtype=rois.dtype)
        canonical=torch.as_tensor(cad,device="cuda:0",dtype=rois.dtype)[None].expand(len(detections),-1,-1)
        aux=device_mask_aux(inputs.mask_aux,"cuda:0",rois.dtype)
        zero=MaskAux(torch.zeros_like(aux.visible),torch.zeros_like(aux.distance),torch.zeros_like(aux.available),
            aux.candidate_overlap.clone(),torch.ones_like(aux.dropout),aux.identities,aux.protocol_id)
        capture_start=len(rotation6d)
        packets={"native":model(rois,km,canonical,mask_aux=aux,identities=aux.identities,return_corner=True)}
        if not cfg.get("capture_rotation6d"):
            packets["aux_dropout"]=model(rois,km,canonical,mask_aux=zero,identities=zero.identities,return_corner=True)
        captured=np.concatenate(rotation6d[capture_start:],axis=0) if cfg.get("capture_rotation6d") else None
        if captured is not None and captured.shape!=(len(detections),6):raise ValueError("Per-instance pose-head capture domain differs")
        for i,det in enumerate(detections):
            previous=next(x for x in old[fid]["prediction"]["raw_instances"] if x["track_id"]==det["track_id"])
            row=dict(frame_id=fid,track_id=det["track_id"],bbox_for_a0=det["bbox_for_a0"],roi_decision_id=det["roi_decision_id"],
                source_RGB_sha256=features.frame.source_sha256,variants={})
            for name,packet in packets.items():
                p=packet.raw_pose;uv=DataAdapter.transform_points(p["corners"][i].cpu().numpy(),np.linalg.inv(inputs.A112[i]))
                row["variants"][name]=dict(rotation=p["rotation"][i].cpu().tolist(),translation_m=p["translation"][i].cpu().tolist(),
                    corners_native=uv.tolist(),q_corner=packet.corners.q_corner[i].cpu().tolist(),
                    available=bool(packet.sam_aux.available[i].item()),residual_norm=float(packet.sam_aux.residual[i].norm().item()))
            delta=float(np.max(abs(np.asarray(row["variants"]["native"]["corners_native"])-previous["corners_raw_from_resolved_roi"])))
            if delta>.01:raise ValueError("Actual native forward differs from saved source: "+str(delta))
            row["native_replay_max_abs_px"]=delta
            if cfg.get("capture_rotation6d"):
                values=captured[i];x=values[:3];y=values[3:]
                nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y))
                sine=float(np.linalg.norm(np.cross(x,y))/max(nx*ny,1e-30))
                row["rotation6d"]=dict(raw=values.tolist(),x_norm=nx,y_norm=ny,
                    sine_between_axes=sine,relative_collinearity_gain=1/max(sine,1e-30))
            rows.append(row)
        (out/"STATUS.json").write_text(json.dumps(dict(frame=fid,records=len(rows))))
    if rotation_hook is not None:rotation_hook.remove()
    if cfg.get("capture_rotation6d") and sum(len(x) for x in rotation6d)!=len(rows):raise ValueError("Missing native rotation logit capture")
    if versions()!=before:raise ValueError("Inference mutated model parameters/buffers")
    (out/"PREDICTIONS.json").write_text(json.dumps(dict(records=rows),indent=2))
    jumps={};previous={}
    for row in rows:
        tid=row["track_id"]
        for name,p in row["variants"].items():
            key=(tid,name);R=np.asarray(p["rotation"])
            if key in previous and previous[key][0]==row["frame_id"]-1:
                angle=float(np.linalg.norm(rotation_log(R@previous[key][1].T))*180/np.pi)
                jumps.setdefault(name,[]).append(angle)
            previous[key]=(row["frame_id"],R)
    done=dict(status="SAME_WEIGHT_SAME_RGB_ROI_AUX_COUNTERFACTUAL_COMPLETE_NOT_ACCURACY",
      frames=cfg["frames"],instances=len(rows),network_forwards=(1 if cfg.get("capture_rotation6d") else 2)*len(cfg["frames"]),models=pair["models"],input_lock=cfg["input_lock"],
      maximum_native_replay_error_px=max(x["native_replay_max_abs_px"] for x in rows),
      rotation_step_degrees={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),over30=sum(x>30 for x in v)) for k,v in jumps.items()},
      model_parameters_buffers_unchanged=True,B0_and_SAM_tracking_ROI_unchanged=True,
      only_A0_auxiliary_mask_dropout_changed=not bool(cfg.get("capture_rotation6d")),capture_rotation6d=bool(cfg.get("capture_rotation6d")),GT_used=False,default_changed=False)
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2));return done
