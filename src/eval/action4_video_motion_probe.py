"""Paired original-video diagnosis of an opt-in image-motion veto.

Consumes frozen raw A0 and SAM outputs; no model inference, test, default
publication, or invented pose ground truth. Both variants start at the same
clip context and are scored on the same preregistered core.
"""
import hashlib,json,copy
from pathlib import Path
from dataclasses import replace
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from model.action4_video_pose_stabilizer import VideoPoseStabilizer,StabilizerConfig
from model.action4_video_motion_observation import measure_motion
from model.action4_native_rotation_history import rotation_log
from eval.action4_video_stabilization import read,bound,validate_processed_row

def run(config_path,source_ref):
    import cv2
    cv2.setNumThreads(1)
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    read(source_ref)
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    lock=read(cfg["input_lock"])
    data=contained_path(ROOT,cfg["raw_predictions"]["path"],exists=True).read_bytes()
    if hashlib.sha256(data).hexdigest()!=cfg["raw_predictions"]["sha256"]:raise ValueError("Raw source changed")
    raw=[json.loads(s) for s in data.splitlines()]
    replay=lock["video"]["track_replay"]["predictions_ref"]
    data=contained_path(ROOT,replay["path"],exists=True).read_bytes()
    if hashlib.sha256(data).hexdigest()!=replay["sha256"]:raise ValueError("SAM replay changed")
    fused=[json.loads(s) for s in data.splitlines()]
    frames=read(lock["video"]["frames_manifest"])["frames"]
    K=np.asarray(read(lock["video"]["K"])["K_native"],float)
    cad=np.asarray(read(lock["CAD"])["corners_3d_mm"],float)/1000
    def inputs(fid):
        frame=frames[fid];row=fused[fid]
        p=contained_path(ROOT,frame["image"],exists=True)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=frame["image_sha256"]:raise ValueError("RGB identity differs")
        image=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        status=json.loads(contained_path(ROOT,row["native_status"],exists=True).read_text())
        if (status["frame"]!=fid or status["pts"]!=frame["pts"]
            or status["RGB_sha256"]!=frame["image_sha256"]
            or status["session_id"]!=row["session_id"] or status["generation"]!=row["generation"]):
            raise ValueError("Current SAM/frame ownership differs")
        header=read(dict(path=status["raw"]["header"],sha256=status["raw"]["header_sha256"]))["outputs"]
        path=contained_path(ROOT,status["raw"]["arrays"],exists=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=status["raw"]["sha256"]:raise ValueError("SAM array differs")
        with np.load(path,allow_pickle=False) as arrays:
            ids=arrays[header["out_obj_ids"]["array_key"]].tolist()
            masks=arrays[header["out_binary_masks"]["array_key"]].copy()
        mapping=status.get("native_obj_map")
        if mapping is None:mapping={x["track_id"]:x["obj_id"] for x in status["instances"] if x["status"]=="available"}
        if len(set(mapping.values()))!=len(mapping):raise ValueError("Duplicated native ownership")
        masks={tid:masks[ids.index(oid)] for tid,oid in mapping.items() if oid in ids}
        return image,masks,status
    filter_cfg=StabilizerConfig(**cfg["filter"])
    if filter_cfg.motion_veto_enabled:raise ValueError("Baseline must be motion-off")
    summary=[];contact=[]
    xyz=cad;edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(np.abs(xyz[i]-xyz[j])>1e-8)==1]
    for clip in cfg["clips"]:
        baseline=VideoPoseStabilizer(filter_cfg);candidate=VideoPoseStabilizer(replace(filter_cfg,motion_veto_enabled=True))
        prior=None;last_output={};stats=dict(core_frames=0,instances=0,motion_available=0,vetoed=0,
            baseline_large_jumps=0,candidate_large_jumps=0)
        folder=out/clip["name"];folder.mkdir()
        with (folder/"PREDICTIONS.jsonl").open("x") as stream:
            for fid in range(clip["start"],clip["end"]):
                im,masks,status=inputs(fid);row=raw[fid];evidence={}
                if prior is not None:
                    prev_im,prev_masks,prev_status=prior
                    prev={x["track_id"]:x for x in raw[fid-1]["prediction"]["instances"]}
                    for inst in row["prediction"]["instances"]:
                        tid=inst["track_id"]
                        observation=dict(available=False,reason="missing_or_cross_generation_support")
                        same=prev_status["session_id"]==status["session_id"] and prev_status["generation"]==status["generation"]
                        if same and tid in prev_masks and tid in masks and tid in prev:
                            bb=np.asarray([prev[tid]["bbox_for_a0"],inst["bbox_for_a0"]])
                            lo=bb[:,:2].min(0);hi=bb[:,2:].max(0);pad=.2*(hi-lo)
                            other_prev=np.zeros_like(prev_masks[tid]);other_now=np.zeros_like(masks[tid])
                            for key,m in prev_masks.items():
                                if key!=tid:other_prev|=m
                            for key,m in masks.items():
                                if key!=tid:other_now|=m
                            observation=measure_motion(prev_im,im,prev_masks[tid],masks[tid],np.r_[lo-pad,hi+pad],
                                previous_other=other_prev,current_other=other_now)
                        observation.update(track_id=tid,frame_id=fid,previous_frame_id=fid-1,
                            pts=row["pts"],time_base=[1,11988],same_generation=same,
                            RGB_sha256=row["RGB_sha256"],previous_RGB_sha256=raw[fid-1]["RGB_sha256"],
                            generation=status["generation"],session_id=status["session_id"])
                        evidence[tid]=observation
                kwargs=dict(frame_id=fid,pts=row["pts"],time_base=[1,11988],K_native=K,canonical_m=cad,mode="video")
                a=copy.deepcopy(row);b=copy.deepcopy(row)
                a["prediction"]=baseline.apply_prediction(row["prediction"],**kwargs)
                b["prediction"]=candidate.apply_prediction(row["prediction"],motion_observations=evidence,**kwargs)
                validate_processed_row(row,a,K,cad,baseline.config.identity);validate_processed_row(row,b,K,cad,candidate.config.identity)
                in_core=clip["core_start"]<=fid<clip["core_end"]
                if in_core:stats["core_frames"]+=1
                for ia,ib in zip(a["prediction"]["instances"],b["prediction"]["instances"]):
                    tid=ib["track_id"];decision=ib["pose_stabilization"].get("image_motion_decision",{})
                    if in_core:
                        stats["instances"]+=1;stats["motion_available"]+=int(evidence.get(tid,{}).get("available",False))
                        stats["vetoed"]+=int(decision.get("reject_candidate",False))
                        if tid in last_output:
                            for name,inst in (("baseline",ia),("candidate",ib)):
                                R=np.array(inst["final_pose"]["rotation"]);old=last_output[tid][name]
                                stats[name+"_large_jumps"]+=int(np.linalg.norm(rotation_log(R@old.T))>np.pi/6)
                    last_output[tid]={k:np.asarray(x["final_pose"]["rotation"]) for k,x in (("baseline",ia),("candidate",ib))}
                stream.write(json.dumps(dict(frame_id=fid,core=in_core,baseline=a,candidate=b,motion=evidence),allow_nan=False)+"\n")
                if fid in clip.get("review_frames",[]):
                    rgb=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
                    # Native same crop and same pixels in both columns, all eight fixed IDs.
                    for j in range(len(b["prediction"]["instances"])):
                        box=np.asarray(row["prediction"]["instances"][j]["bbox_for_a0"])
                        center=(box[:2]+box[2:])/2;side=int(np.ceil(max(box[2:]-box[:2])*1.7))
                        l,t=np.floor(center-side/2).astype(int);h,w=im.shape;panels=[]
                        for label,result in (("baseline",a),("image veto",b)):
                            tile=np.zeros((side,side,3),np.uint8);x0,y0=max(l,0),max(t,0);x1,y1=min(l+side,w),min(t+side,h)
                            if x1>x0 and y1>y0:tile[y0-t:y1-t,x0-l:x1-l]=rgb[y0:y1,x0:x1]
                            uv=np.array(result["prediction"]["instances"][j]["corners_final"])-[l,t]
                            for x,y in edges:cv2.line(tile,tuple(uv[x].astype(int)),tuple(uv[y].astype(int)),(0,0,255),max(1,side//170))
                            tile=cv2.resize(tile,(300,300))
                            cv2.putText(tile,f"{label} f{fid} #{j}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);panels.append(tile)
                        contact.append(np.hstack(panels))
                prior=im,masks,status
        summary.append(dict(clip=clip,**stats))
        (folder/"COMPLETE.json").write_text(json.dumps(summary[-1],indent=2))
        print(json.dumps(summary[-1]),flush=True)
    for i in range(0,len(contact),8):
        cv2.imwrite(str(out/f"REVIEW_{i//8:02d}.jpg"),np.vstack(contact[i:i+8]))
    done=dict(status="PAIRED_DEVELOPMENT_PROBE_COMPLETE_NOT_ACCEPTANCE",source=source_ref,config=bound(config_path),
              raw_predictions=cfg["raw_predictions"],clips=summary,model_calls=0,GT_used=False,
              real_pose_accuracy_measured=False,defaults_changed=False)
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2))
    return done


def replay_preserved_evidence(config_path,source_ref):
    """One policy change against identical cached actual optical observations."""
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    read(source_ref);parent=read(cfg["parent_complete"])
    if parent["status"]!="PAIRED_DEVELOPMENT_PROBE_COMPLETE_NOT_ACCEPTANCE":raise ValueError("Invalid source probe")
    parent_cfg=read(parent["config"]);lock=read(parent_cfg["input_lock"])
    data=contained_path(ROOT,parent["raw_predictions"]["path"],exists=True).read_bytes()
    if hashlib.sha256(data).hexdigest()!=parent["raw_predictions"]["sha256"]:raise ValueError("Original A0 changed")
    raw=[json.loads(s) for s in data.splitlines()]
    K=np.asarray(read(lock["video"]["K"])["K_native"],float);cad=np.asarray(read(lock["CAD"])["corners_3d_mm"],float)/1000
    base=StabilizerConfig(**parent_cfg["filter"])
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    results=[]
    for clip,ref in zip(parent["clips"],cfg["clip_predictions"]):
        definition=clip["clip"];path=contained_path(ROOT,ref["path"],exists=True);data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=ref["sha256"]:raise ValueError("Saved optical evidence changed")
        rows=[json.loads(s) for s in data.splitlines()]
        if [r["frame_id"] for r in rows]!=list(range(definition["start"],definition["end"])):raise ValueError("Clip domain changed")
        engine=VideoPoseStabilizer(replace(base,motion_veto_enabled=True,motion_veto_hold_frames=cfg["hold_frames"]))
        folder=out/definition["name"];folder.mkdir()
        stats=dict(core_frames=0,instances=0,motion_available=0,vetoed=0,held=0,
                   baseline_large_jumps=0,candidate_large_jumps=0)
        previous={}
        with (folder/"PREDICTIONS.jsonl").open("x") as stream:
            for row in rows:
                fid=row["frame_id"];original=raw[fid];evidence=row["motion"];candidate=copy.deepcopy(original)
                candidate["prediction"]=engine.apply_prediction(original["prediction"],frame_id=fid,pts=original["pts"],
                    time_base=[1,11988],K_native=K,canonical_m=cad,mode="video",motion_observations=evidence)
                validate_processed_row(original,candidate,K,cad,engine.config.identity)
                if row["core"]:stats["core_frames"]+=1
                for a,b in zip(row["baseline"]["prediction"]["instances"],candidate["prediction"]["instances"]):
                    tid=b["track_id"];decision=b["pose_stabilization"]["image_motion_decision"]
                    if row["core"]:
                        stats["instances"]+=1;stats["motion_available"]+=int(evidence.get(tid,{}).get("available",False))
                        stats["vetoed"]+=int(decision.get("reject_candidate",False));stats["held"]+=int(decision.get("bounded_rotation_hold",False))
                        if tid in previous:
                            for key,x in (("baseline",a),("candidate",b)):
                                delta=np.linalg.norm(rotation_log(np.array(x["final_pose"]["rotation"])@previous[tid][key].T))
                                stats[key+"_large_jumps"]+=int(delta>np.pi/6)
                    previous[tid]={k:np.asarray(x["final_pose"]["rotation"]) for k,x in (("baseline",a),("candidate",b))}
                stream.write(json.dumps(dict(frame_id=fid,core=row["core"],baseline=row["baseline"],candidate=candidate,motion=evidence),allow_nan=False)+"\n")
        results.append(dict(clip=definition,**stats));print(json.dumps(results[-1]),flush=True)
    done=dict(status="PAIRED_MOTION_HOLD_REPLAY_COMPLETE_NOT_ACCEPTANCE",source=source_ref,config=bound(config_path),
              clips=results,parent=cfg["parent_complete"],model_calls=0,new_optical_flow_calls=0,GT_used=False,
              real_pose_accuracy_measured=False,defaults_changed=False)
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2))
    return done
