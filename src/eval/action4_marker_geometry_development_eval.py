"""Full development-only shadow comparison; workers never receive GT pose/boxes."""
import copy,json,os,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding,binding
from model.action4_planar_landmark_pose import planar_pose_candidates
from data.action4_instance_assignment import iou

def _worker(job):
    import cv2
    os.sched_setaffinity(0,{job["cpu"]});cv2.setNumThreads(1)
    params=cv2.aruco.DetectorParameters()
    if cv2.__version__!=job["opencv_version"]:raise ValueError("Detector runtime differs from CAD binding")
    if any(getattr(params,k)!=v for k,v in job["detector_parameters"].items()):
        raise ValueError("Detector defaults differ from registered material binding")
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),params)
    output=[];audit=[]
    for row in job["rows"]:
        if set(row)!={"sample_id","image","image_sha256","width","height","K_native","baseline_prediction"}:
            raise ValueError("Inference whitelist violated")
        payload=contained_path(R,row["image"],exists=True).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row["image_sha256"]:raise ValueError("Source image changed")
        gray=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_GRAYSCALE)
        if gray.shape!=(row["height"],row["width"]):raise ValueError("Pixel geometry differs")
        corners,ids,_=detector.detectMarkers(gray)
        predicted=copy.deepcopy(row["baseline_prediction"]);proposals={};events=[]
        if ids is not None:
            for quad,mid in zip(corners,ids[:,0]):
                q=quad[0].astype(float);identity=str(int(mid))
                event=dict(marker_id=int(mid),corners_native=q.tolist(),status="unassigned")
                events.append(event)
                if identity not in job["landmarks"]:event["reason"]="no_estimated_CAD_binding";continue
                area=abs(cv2.contourArea(q.astype(np.float32)));coverage=[]
                for j,p in enumerate(predicted["instances"]):
                    x0,y0,x1,y1=p["bbox_full_xyxy"]
                    box=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
                    inter,_=cv2.intersectConvexConvex(q.astype(np.float32),box)
                    coverage.append((float(inter/max(area,1e-12)),j))
                coverage.sort(reverse=True)
                if not coverage or coverage[0][0]<job["policy"]["minimum_box_coverage"] or coverage[0][0]-(coverage[1][0] if len(coverage)>1 else 0)<job["policy"]["minimum_ownership_margin"]:
                    event["reason"]="no_unique_predicted_box_support";continue
                j=coverage[0][1];event.update(prediction_id=predicted["instances"][j]["prediction_id"],match_basis="predicted_box_weak_support")
                if np.min(np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1))<job["policy"]["minimum_marker_side_px"]:
                    event["reason"]="insufficient_marker_pixel_extent";continue
                candidates=planar_pose_candidates(job["landmarks"][identity],q,row["K_native"],job["canonical"])
                event["pose_candidates"]=candidates
                good=sorted([x for x in candidates["candidates"] if x["valid"]],key=lambda x:x["correspondence_rms_native_px"])
                if not good:event["reason"]="no_positive_depth_candidate";continue
                best=good[0]
                if best["correspondence_rms_native_px"]>job["policy"]["maximum_reprojection_rms_px"]:
                    event["reason"]="marker_reprojection_gate";continue
                if len(good)>1 and good[1]["correspondence_rms_native_px"]-best["correspondence_rms_native_px"]<job["policy"]["minimum_solution_gap_px"]:
                    event["reason"]="unresolved_planar_two_solution_ambiguity";continue
                c=np.asarray(best["corners_native"]);box=np.r_[c.min(0),c.max(0)]
                if iou(box,np.asarray(predicted["instances"][j]["bbox_full_xyxy"]))<job["policy"]["minimum_geometry_box_IoU"]:
                    event["reason"]="inconsistent_complete_body_extent";continue
                event.update(status="candidate_prechecks_pass_not_deployment",reason="image_correspondence_and_predicted_box_checks")
                proposals.setdefault(j,[]).append((best,event))
        for j,choices in proposals.items():
            if len(choices)!=1:
                for _,event in choices:event.update(status="unassigned",reason="multiple_markers_for_one_prediction")
                continue
            best,event=choices[0];p=predicted["instances"][j]
            p["corners_A0_raw"]=copy.deepcopy(p["corners_px"])
            p["corners_px"]=best["corners_native"];p["corner_ids"]=list(range(8))
            p["pose_valid"]=True;p["positive_depth_verified"]=True
            p["pose_diagnostic"]=dict(method="estimated_display_landmark_candidate",valid=True,metric_GT=False)
            p["selected_geometry_pose"]=dict(rotation=best["R"],translation_m=best["t_m"])
            p["geometry_source"]="development_shadow_estimated_appearance_binding"
            p["marker_observation"]=event
        output.append(predicted);audit.append(dict(sample_id=row["sample_id"],events=events))
    return output,audit

def run(config_path):
    from eval.action4_pose_protocol import score_pose
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    baseline=read_binding(cfg["baseline_predictions"]);records=[json.loads(l) for l in contained_path(R,cfg["annotations"]["path"],exists=True).open()]
    if binding(contained_path(R,cfg["annotations"]["path"],exists=True))!=cfg["annotations"]:raise ValueError("Development annotations changed")
    if len(baseline)!=650 or len(records)!=650:raise ValueError("Full650 development domain required")
    bm={p["sample_id"]:p for p in baseline}
    if len(bm)!=650 or set(bm)!={r["sample_id"] for r in records}:raise ValueError("Sample domain differs")
    binding_doc=read_binding(cfg["landmark_binding"])
    if binding_doc["detection_variant_id"]!="opencv_4_13_0_default_NONE":raise ValueError("Detector/binding mismatch")
    landmarks={str(m["id"]):[c["xyz_object_m"] for c in m["corners"]] for m in binding_doc["markers"]}
    canonical=np.asarray(read_binding(cfg["CAD"])["corners_3d_mm"])/1000.
    # Deliberate whitelist boundary: no GT instances/pose/visibility reach workers.
    inference=[dict(sample_id=r["sample_id"],image=r["image"],image_sha256=r["rgb_sha256"],width=r["width"],height=r["height"],
                    K_native=r["camera"]["K"],baseline_prediction=bm[r["sample_id"]]) for r in records]
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    cpus=cfg["cpus"]
    if not set(cpus)<=set(range(24,96)) or len(cpus)!=len(set(cpus)):raise ValueError("CPU budget")
    jobs=[]
    for n,cpu in enumerate(cpus):
        jobs.append(dict(cpu=cpu,rows=inference[n*650//len(cpus):(n+1)*650//len(cpus)],
          landmarks=landmarks,canonical=canonical.tolist(),policy=cfg["policy"],
          opencv_version=binding_doc["opencv_version"],detector_parameters=binding_doc["detector_parameters"]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cpus)) as pool:parts=list(pool.map(_worker,jobs))
    candidate=[p for predictions,_ in parts for p in predictions];audit=[p for _,a in parts for p in a]
    (out/"PREDICTIONS.json").write_text(json.dumps(candidate,allow_nan=False))
    (out/"OBSERVATION_AUDIT.json").write_text(json.dumps(audit,allow_nan=False))
    base_report=score_pose(baseline,records);report=score_pose(candidate,records)
    if base_report["pose_all8_at_002_success"]!=cfg["expected_baseline_success"]:raise ValueError("Frozen baseline score did not reproduce")
    (out/"BASELINE_REPORT.json").write_text(json.dumps(base_report,indent=2))
    (out/"CANDIDATE_REPORT.json").write_text(json.dumps(report,indent=2))
    changed=sum("selected_geometry_pose" in p for r in candidate for p in r["instances"])
    result=dict(status="FULL650_DEVELOPMENT_SHADOW_GEOMETRY_SCORED_NOT_DEPLOYED",images=650,GT_count=report["pose_gt_count"],
        changed_predictions=changed,baseline_All8_002=base_report["pose_all8_at_002_success"],
        candidate_All8_002=report["pose_all8_at_002_success"],model_updates=0,new_B0_A0_calls=0,
        inference_GT_consumed=False,track_enabled=False,real_metric_binding_verified=False,default_changed=False)
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));return result
