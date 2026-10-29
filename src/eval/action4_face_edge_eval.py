"""Actual RGB / native SAM / fixed-CAD curve pilot, same raw A0 preserved."""
import json,os,hashlib,concurrent.futures
from dataclasses import asdict
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_sam_full_review import ReceiptVerifier
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from model.action4_face_edge_pose import EdgePoseConfig,refine_face_edge_pose

def resample_closed_curve(points,spacing_m):
    p=np.asarray(points,float);n=np.roll(p,-1,axis=0)
    out=[]
    for a,b in zip(p,n):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing_m)))
        out.extend(a+(b-a)*v for v in np.arange(count)/count)
    return np.asarray(out)

def _fit(job):
    from scipy.spatial.transform import Rotation
    import cv2
    os.sched_setaffinity(0,{job["cpu"]});cv2.setNumThreads(1)
    source=contained_path(R,job["input"],exists=True)
    with np.load(source,allow_pickle=False) as z:
        rgb=z["rgb"];body=z["body"];dynamic=z["dynamic"];K=z["K_crop"]
    # Conservatively exclude all dynamic screen hull, not just fragmented pixels.
    ys,xs=np.nonzero(dynamic);exclusion=np.zeros_like(dynamic,np.uint8)
    cv2.fillConvexPoly(exclusion,cv2.convexHull(np.c_[xs,ys].astype(np.int32)),1)
    cfg=EdgePoseConfig(**job["parameters"])
    result=refine_face_edge_pose(rgb,body,exclusion.astype(bool),K,
        [np.asarray(p) for p in job["curves"]],job["raw_R"],job["raw_t"],
        cad_diagonal_m=job["cad_diagonal_m"],config=cfg)
    result.update(frame_id=job["frame_id"],track_id=job["track_id"],
        input_ref=binding(source),crop_native_xyxy=job["crop_native_xyxy"],K_native=job["K_native"],
        residual_pixel_space="native_crop_same_pixel_scale",hypothesis_only=True,face_id=3,
        edge_model_source=job["edge_model_source"],A0_identity=job["A0_identity"])
    if result["candidate_pose"] is not None:
        pose=result["candidate_pose"]
        result["candidate_rotation_change_deg"]=float(Rotation.from_matrix(np.asarray(pose["R"])@np.asarray(job["raw_R"]).T).magnitude()*180/np.pi)
        result["candidate_translation_change_m"]=float(np.linalg.norm(np.asarray(pose["t_m"])-job["raw_t"]))
    path=contained_path(R,job["output"],exists=False)
    with path.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    return {k:result.get(k) for k in ["frame_id","track_id","geometric_candidate_eligible","reasons","candidate_rotation_change_deg","candidate_translation_change_m"]}

def run(config_path):
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text());out=contained_path(R,cfg["output"],exists=False)
    out.mkdir();(out/"inputs").mkdir();(out/"fits").mkdir()
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    provider=ReanchorROIProvider(**lock["video"]["track_replay"])
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    CAD=read_binding(lock["CAD"]);canonical=np.asarray(CAD["corners_3d_mm"])/1000.
    diagonal=float(np.linalg.norm(np.ptp(canonical,axis=0)))
    audit=read_binding(cfg["edge_model_source"])
    if audit["mesh_sha256"]!=CAD["mesh_data_sha256"]:raise ValueError("Physical curve CAD identity differs")
    patch=next(p for p in audit["patches"] if p["physical_patch_id"]==cfg["physical_patch_id"])
    curves=[resample_closed_curve(np.asarray(p["xyz_object_mm"])/1000.,cfg["curve_spacing_m"]).tolist() for p in patch["boundary_loops"]]
    verifier=ReceiptVerifier();wanted=set(cfg["frames"]);preds={}
    for line in verifier.verify(cfg["predictions"]["path"],cfg["predictions"]["sha256"]).open():
        row=json.loads(line)
        if row["frame_id"] in wanted:preds[row["frame_id"]]=row
    jobs=[]
    for track in cfg["native_tracks"]:
        native=read_binding(track["config"]);doc=read_binding(native["region"])
        root=contained_path(R,native["output"],exists=True)/"face_03"
        done=read_binding(track["completion"])
        if done["status"]!="PILOT_NATIVE_ATTEMPTS_COMPLETE_NOT_FACE_ACCEPTANCE":raise ValueError("Completed native pilot required")
        for index,f in enumerate(doc["frames"]):
            fid=f["global_frame_id"]
            if fid not in wanted:continue
            rgb,context=provider.load(fid,K);rgb=np.moveaxis(rgb.cpu().numpy(),0,-1)
            rawpred=next(p for p in preds[fid]["prediction"]["instances"] if p["track_id"]==track["track_id"])
            raw=rawpred["raw_network_pose"]
            if not raw["finite"]:raise ValueError("Finite source A0 required")
            stpath=contained_path(R,provider.rows[fid]["native_status"],exists=True);st=json.loads(stpath.read_text())
            bodyraw=verifier.raw(st["raw"],stpath.parent);ids=np.asarray(bodyraw["outputs"]["out_obj_ids"]).tolist()
            oid={"track_00":0,"track_01":1}[track["track_id"]]
            body=np.asarray(bodyraw["outputs"]["out_binary_masks"])[ids.index(oid)]
            folder=root/f"response_{index:04d}";rawface=verifier.raw(json.loads((folder/"RAW_RECEIPT.json").read_text()),folder)
            if rawface["frame_index"]!=index:raise ValueError("Current face source frame differs")
            mask=np.asarray(rawface["outputs"]["out_binary_masks"])[0]
            yy,xx=np.nonzero(body);pad=cfg["image_crop_margin_px"]
            l=max(0,int(xx.min())-pad);r=min(body.shape[1],int(xx.max())+pad+1);t=max(0,int(yy.min())-pad);b=min(body.shape[0],int(yy.max())+pad+1)
            A=np.array([[1.,0,-l],[0,1.,-t],[0,0,1.]])
            name=f"{track['track_id']}_{fid:08d}";file=out/"inputs"/(name+".npz")
            np.savez_compressed(file,rgb=rgb[t:b,l:r],body=body[t:b,l:r],dynamic=mask[t:b,l:r],K_crop=A@K)
            jobs.append(dict(cpu=cfg["cpus"][len(jobs)%len(cfg["cpus"])],input=str(file.relative_to(R)),
                parameters=cfg["parameters"],curves=curves,raw_R=raw["rotation"],raw_t=raw["translation_m"],
                cad_diagonal_m=diagonal,frame_id=fid,track_id=track["track_id"],crop_native_xyxy=[l,t,r,b],
                K_native=K.tolist(),edge_model_source=cfg["edge_model_source"],A0_identity=pair["models"]["A0"],
                output=str((out/"fits"/(name+".json")).relative_to(R))))
    (out/"JOBS.json").write_text(json.dumps(jobs,indent=2))
    if len(set(cfg["cpus"]))!=len(cfg["cpus"]) or not set(cfg["cpus"])<=set(range(24,96)):raise ValueError("CPU budget")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cfg["cpus"])) as pool:
        summaries=list(pool.map(_fit,jobs))
    report=dict(status="REAL_RGB_EDGE_CANDIDATES_NOT_DEPLOYED",jobs=len(jobs),
        geometric_candidate_eligible=sum(bool(s["geometric_candidate_eligible"]) for s in summaries),
        summaries=summaries,model_updates=0,final_corners_modified=False,metric_pose_accuracy_measured=False,
        point_to_curve_semantics="actual CAD rear bezel candidate; no calibrated LCD corners")
    (out/"COMPLETE.json").write_text(json.dumps(report,indent=2))
    return report
