"""Audit reference and candidate geometry against the SAME predicted body support."""
import json,hashlib
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from eval.action4_video_stabilization import read,bound
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from model.action4_pose_support_consistency import SupportPolicy,pose_support_consistency

def run(config_path):
    import cv2
    cv2.setNumThreads(1)
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    out=contained_path(ROOT,cfg["output"],exists=False);out.mkdir(exist_ok=False)
    lock=read(cfg["input_lock"]);provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    K=np.asarray(read(lock["video"]["K"])["K_native"])
    cad=np.asarray(read(lock["CAD"])["corners_3d_mm"])/1000
    policy=SupportPolicy(**cfg["support_policy"]);rows=[];summary={}
    for ref in cfg["clip_predictions"]:
        p=contained_path(ROOT,ref["path"],exists=True);data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=ref["sha256"]:raise ValueError("Frozen diagnostic predictions changed")
        for row in map(json.loads,data.splitlines()):
            if not row["core"]:continue
            fid=row["frame_id"];masks,receipt=provider.current_native_masks(fid)
            if receipt["frame_id"]!=fid or receipt["pts"]!=row["candidate"]["pts"]:
                raise ValueError("Current support frame differs")
            a={x["track_id"]:x for x in row["baseline"]["prediction"]["instances"]}
            b={x["track_id"]:x for x in row["candidate"]["prediction"]["instances"]}
            for tid,candidate in b.items():
                record=dict(frame_id=fid,track_id=tid,mask_receipt=receipt,
                            image_motion_decision=candidate["pose_stabilization"]["image_motion_decision"],poses={})
                if tid not in masks:
                    record["status"]="current_mask_missing";rows.append(record);continue
                other=np.zeros_like(masks[tid])
                for key,m in masks.items():
                    if key!=tid:other|=m
                poses={"raw_A0":candidate["raw_network_pose"],"baseline":a[tid]["final_pose"],"motion_candidate":candidate["final_pose"]}
                for name,pose in poses.items():
                    evidence=pose_support_consistency(masks[tid],K,cad,pose["rotation"],pose["translation_m"],
                        policy=policy,other_instance_mask=other)
                    record["poses"][name]=evidence
                    summary.setdefault(name,dict(total=0,compatible=0,shape_contradiction=0,insufficient_visibility=0))
                    s=summary[name];s["total"]+=1;s["compatible"]+=int(evidence["reference_compatible"])
                    s["shape_contradiction"]+=int("visible_support_outside_projected_body" in evidence["reasons"])
                    s["insufficient_visibility"]+=int("insufficient_visible_shape_evidence" in evidence["reasons"])
                record["status"]="NECESSARY_GEOMETRY_EVIDENCE_NOT_POSE_ACCURACY"
                rows.append(record)
    (out/"INSTANCE_EVIDENCE.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
    done=dict(status="CURRENT_SUPPORT_AUDIT_COMPLETE_NOT_DEPLOYMENT",config=bound(config_path),
              summary=summary,instances=len(rows),GT_used=False,default_modified=False,model_calls=0,
              interpretation="Mask can contradict a reference, but cannot resolve canonical face/corner identity, verify absolute pose, or prove static LCD texture")
    (out/"COMPLETE.json").write_text(json.dumps(done,indent=2))
    return done
