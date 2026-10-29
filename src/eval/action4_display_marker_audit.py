"""Read-only image-feature audit inside existing tracked body windows; no pose use."""
import json,os,hashlib,concurrent.futures,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding,binding

def _part(job):
    import cv2
    os.sched_setaffinity(0,{job["cpu"]});cv2.setNumThreads(1)
    params=cv2.aruco.DetectorParameters()
    params.detectInvertedMarker=job["parameters"]["detectInvertedMarker"]
    params.cornerRefinementMethod=job["parameters"]["cornerRefinementMethod"]
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),params)
    path=contained_path(R,job["output"],exists=False);rows=[];start=time.monotonic()
    with path.open("x") as stream:
        for frame,pred in job["inputs"]:
            file=contained_path(R,frame["image"],exists=True);payload=file.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=frame["image_sha256"]:raise ValueError("Original RGB changed")
            gray=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape!=(job["height"],job["width"]):raise ValueError("Frame decode geometry differs")
            instances=[]
            for instance in pred["prediction"]["instances"]:
                b=np.asarray(instance["bbox_for_a0"],float);cx,cy=(b[:2]+b[2:])/2;L=max(b[2]-b[0],b[3]-b[1])*1.3
                l=max(0,int(cx-L/2));t=max(0,int(cy-L/2));r=min(gray.shape[1],int(cx+L/2)+1);bottom=min(gray.shape[0],int(cy+L/2)+1)
                observations=[]
                if r>l and bottom>t:
                    corners,ids,rejected=detector.detectMarkers(gray[t:bottom,l:r])
                    if ids is not None:
                        for quad,marker in zip(corners,ids[:,0]):
                            observations.append(dict(marker_id=int(marker),corners_native=(quad[0]+[l,t]).tolist(),
                                corner_order="OpenCV"+cv2.__version__+"_DICT_APRILTAG_36h11_returned_marker_local_order_no_screen_sort"))
                instances.append(dict(track_id=instance["track_id"],bbox_for_a0=instance["bbox_for_a0"],
                    audit_crop_native=[l,t,r,bottom],observations=observations))
            row=dict(frame_id=frame["frame_id"],pts=frame["pts"],source_RGB_sha256=frame["image_sha256"],instances=instances)
            stream.write(json.dumps(row,allow_nan=False)+chr(10));rows.append(row)
    return dict(path=str(path.relative_to(R)),frames=len(rows),seconds=time.monotonic()-start,opencv_version=cv2.__version__)

def run(config_path):
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    import cv2
    cv2.setNumThreads(1)
    if cv2.__version__!=cfg["opencv_version"]:raise ValueError("Pinned native OpenCV runtime differs")
    if set(cfg["parameters"])!={"detectInvertedMarker","cornerRefinementMethod"}:
        raise ValueError("Explicit feature-extractor parameter identity required")
    manifest=read_binding(cfg["frames_manifest"]);pred_path=contained_path(R,cfg["predictions"]["path"],exists=True)
    if binding(pred_path)!=cfg["predictions"]:raise ValueError("Fixed tracked body input changed")
    preds=[json.loads(l) for l in pred_path.open()]
    frames=manifest["frames"]
    if len(preds)!=2850 or len(frames)!=2850 or any(p["frame_id"]!=i or f["frame_id"]!=i or p["pts"]!=f["pts"] for i,(p,f) in enumerate(zip(preds,frames))):
        raise ValueError("Exact2850 frame/PTS coverage required")
    cpus=cfg["cpus"]
    if len(cpus)>72 or len(set(cpus))!=len(cpus) or not set(cpus)<=set(range(24,96)):raise ValueError("CPU budget")
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    jobs=[]
    for i,cpu in enumerate(cpus):
        start=i*2850//len(cpus);end=(i+1)*2850//len(cpus)
        jobs.append(dict(cpu=cpu,parameters=cfg["parameters"],inputs=list(zip(frames[start:end],preds[start:end])),height=manifest["source"]["height"],
            width=manifest["source"]["width"],output=str((out/f"part_{i:02d}.jsonl").relative_to(R))))
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cpus)) as pool:parts=list(pool.map(_part,jobs))
    counts=defaultdict(Counter);zero=Counter();per_frame=Counter();events=[]
    with (out/"OBSERVATIONS.jsonl").open("x") as f:
        expected=0
        for part in parts:
            for line in (R/part["path"]).open():
                row=json.loads(line)
                if row["frame_id"]!=expected:raise ValueError("Missing or duplicate audit frame")
                f.write(line);expected+=1;detected=0
                for instance in row["instances"]:
                    tid=instance["track_id"];observations=instance["observations"]
                    if observations:detected+=1
                    else:zero[tid]+=1
                    for obs in observations:counts[tid][str(obs["marker_id"])]+=1
                    if len(observations)>1:events.append(dict(frame_id=row["frame_id"],track_id=tid,kind="multiple_markers_in_body_window",marker_ids=[o["marker_id"] for o in observations]))
                per_frame[str(detected)]+=1
        if expected!=2850:raise ValueError("Incomplete audit")
    result=dict(status="COMPLETE2850_IMAGE_FEATURE_AUDIT_NOT_POSE_OR_CLASS_ACCURACY",frames=2850,parts=parts,
        dictionary="DICT_APRILTAG_36h11",parameters=cfg["parameters"],detection_variant_id=cfg["detection_variant_id"],
        per_track_marker_counts={k:dict(v) for k,v in counts.items()},
        per_track_frames_without_marker=dict(zero),frames_by_body_windows_with_marker=dict(per_frame),events=events,
        fixed_to_body_verified=False,metric_CAD_binding_available=False,workflow_admitted=False,model_updates=0,
        note="Feature extraction only; existing B0/ROI/A0 predictions and all failures remain unchanged")
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));return result
