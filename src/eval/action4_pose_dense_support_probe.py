"""Counterfactual dense geometry support at the existing ConvPnP input."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from data.action4_surface_snapshot import load_snapshot
from data.action4_sam_roi_inputs import sample_native_nearest
from model.action4_release_bundle import load_bundle
from model.heads.gdr_rotation import ortho6d_to_mat_batch
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_roi_sweep import read_binding,binding
def run(config_path):
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    device=cfg.get("device","cpu")
    if set(os.sched_getaffinity(0))!={60,61,62,63}:raise ValueError("Explicit bounded geometry-head-only probe")
    if device=="cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU must hide GPU")
    elif device=="cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=cfg["gpu_uuid"]:raise ValueError("GPU UUID mismatch")
        from model.training.action4_resource_policy import validate_device_policy
        validate_device_policy({"training_gpus":[],"eval_gpus":[cfg["gpu"]]},ROOT)
        import subprocess
        info=subprocess.check_output(["nvidia-smi","-i",str(cfg["gpu"]),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if info[0].strip()!=cfg["gpu_uuid"] or int(info[1])<12288:raise ValueError("GPU headroom unavailable")
        torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.use_deterministic_algorithms(True)
    else:raise ValueError("Unknown explicit device")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"]);a=pair["models"]["A0"]
    model,base,state=load_bundle(a["path"],"A0",device,expected_sha256=a["sha256"])
    model.eval().requires_grad_(False);base.eval().requires_grad_(False)
    pose=model.geometry.pose
    before=tuple((n,p._version) for n,p in list(model.named_parameters())+list(model.named_buffers()))
    provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    sources=read_binding(cfg["saved_surfaces"])["results"];rows=[]
    with torch.inference_mode():
      for entry in sources:
        snap=load_snapshot(contained_path(ROOT,entry["surface"]["path"],exists=True))
        if binding(contained_path(ROOT,entry["surface"]["path"],exists=True))!=entry["surface"]:raise ValueError("Surface bytes changed")
        z=snap.arrays;meta=snap.metadata;fid=entry["frame_id"];tid=entry["track_id"]
        if meta["models"]["A0"]["sha256"]!=a["sha256"]:raise ValueError("Wrong pose-head parent")
        masks,receipt=provider.current_native_masks(fid)
        if receipt["RGB_sha256"]!=meta["source_sha256"]:raise ValueError("Current support image differs")
        tensor=lambda v:torch.from_numpy(np.asarray(v).copy()).float()[None].to(device)
        xyz=tensor(z["xyz_normalized"]);reg=tensor(z["region_logits"])[:,1:].softmax(1);m=tensor(z["mask_logits"]).sigmoid()
        K=tensor(z["K112"]);cad=tensor(z["canonical_m"]);extent=cad.amax(1)-cad.amin(1)
        yy,xx=torch.meshgrid(torch.arange(112,device=device,dtype=xyz.dtype),torch.arange(112,device=device,dtype=xyz.dtype),indexing="ij")
        rays=torch.stack(((xx[None]-K[:,0,2,None,None])/K[:,0,0,None,None],(yy[None]-K[:,1,2,None,None])/K[:,1,1,None,None]),1)
        resize=lambda x:F.interpolate(x,size=(64,64),mode="bilinear",align_corners=False)
        X,ray,region,attention=map(resize,(xyz,rays,reg,m))
        gates={"native":torch.ones_like(attention[:,:1]),"predicted_visible":attention[:,:1]}
        if tid in masks:
            owned=masks[tid].copy()
            for other,mask in masks.items():
                if other!=tid:owned &= ~mask
            sampled=sample_native_nearest(owned,z["A448"])
            gates["current_SAM"]=resize(sampled[None].float().to(device))
        row=dict(frame_id=fid,track_id=tid,source_surface=entry["surface"],variants={})
        for name,gate in gates.items():
            # .5 is zero after the original ConvPnP (XYZ-.5)*extent transform.
            coor=torch.cat((X if name=="native" else .5+gate*(X-.5),ray),1)
            rot6d,t=pose(coor,region=region*gate,extents=extent,mask_attention=attention)
            rot=ortho6d_to_mat_batch(rot6d);centroid=(t[:,:2]+.5)*112
            depth=F.softplus(t[:,2])*((K[:,0,0]+K[:,1,1])*.5*extent.amax(-1)/112)
            translation=torch.stack(((centroid[:,0]-K[:,0,2])/K[:,0,0]*depth,(centroid[:,1]-K[:,1,2])/K[:,1,1]*depth,depth),-1)
            camera=cad@rot.transpose(1,2)+translation[:,None]
            h=camera@tensor(z["K_native"]).transpose(1,2);uv=h[:,:,:2]/h[:,:,2:]
            row["variants"][name]=dict(rotation=rot[0].tolist(),translation_m=translation[0].tolist(),corners_native=uv[0].tolist(),
                mean_gate=float(gate.mean()),raw_rot6d=rot6d[0].tolist())
        native=row["variants"]["native"]
        deltaR=float(np.max(abs(np.array(native["rotation"])-z["rotation"])))
        deltaT=float(np.max(abs(np.array(native["translation_m"])-z["translation_m"])))
        if deltaR>1e-4 or deltaT>1e-5:
            (out/"PARITY_FAILURE.json").write_text(json.dumps(dict(frame_id=fid,track_id=tid,deltaR=deltaR,deltaT_m=deltaT,
                predicted=native,expected_rotation=z["rotation"].tolist(),expected_translation=z["translation_m"].tolist()),indent=2))
            raise ValueError("Cached map pose replay mismatch: "+str((fid,tid,deltaR,deltaT)))
        row["native_replay_rotation_max"]=deltaR;row["native_replay_translation_max_m"]=deltaT;rows.append(row)
    after=tuple((n,p._version) for n,p in list(model.named_parameters())+list(model.named_buffers()))
    if before!=after:raise ValueError("Head probe mutated parameters/buffers")
    (out/"PREDICTIONS.json").write_text(json.dumps(dict(records=rows),indent=2))
    result=dict(status="CACHED_SURFACE_GEOMETRY_SUPPORT_PROBE_NOT_ACCURACY",instances=len(rows),models=pair["models"],
        variants=["native","predicted_visible","current_SAM"],RGB_backbone_calls=0,SAM_calls=0,B0_calls=0,
        training_modified=False,GT_used=False,defaults_changed=False,
        only_changed="Pose input XYZ centered amplitude and region channels outside predicted visible support; rays and original visible/amodal mask channels unchanged",
        hidden_corner_heatmaps_masked=False)
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));return result
