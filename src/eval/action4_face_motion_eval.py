"""CPU replay of actual native face and body supports for RGB-border motion."""
import json,hashlib
from dataclasses import asdict
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_roi_inputs import array_digest
from eval.action4_sam_full_review import ReceiptVerifier
from eval.action4_sam_image_eval import validate_source_snapshot
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_roi_sweep import read_binding,binding
from model.action4_face_motion_evidence import FaceMotionConfig,measure_border_motion

def run(config_path):
    import cv2
    cv2.setNumThreads(1);cv2.setRNGSeed(125)
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    lock=read_binding(cfg["input_lock"]);provider=ReanchorROIProvider(**lock["video"]["track_replay"],register_project_bodies=True)
    verifier=ReceiptVerifier();K=np.asarray(read_binding(lock["video"]["K"])["K_native"])
    policy=FaceMotionConfig(**cfg["motion"])
    out=contained_path(R,cfg["output"],exists=False);out.mkdir()
    rows=[];summaries=[]
    for job in cfg["jobs"]:
        native=read_binding(job["config"]);source=validate_source_snapshot(native)
        complete=verifier.bound_json(job["completion"])
        exit_receipt=verifier.bound_json(job["exit"])
        if (exit_receipt["returncode"]!=0 or complete["source_identity"]!=source["identity"]
                or complete["status"]!="PILOT_NATIVE_ATTEMPTS_COMPLETE_NOT_FACE_ACCEPTANCE"):
            raise ValueError("Actual completed native pilot required")
        root=contained_path(R,native["output"],exists=True)
        doc=verifier.bound_json(native["region"])
        face_id=job["face_id"];folder=root/f"face_{face_id:02d}"
        seed=json.loads((folder/"FACE_SEED.json").read_text())
        decision=json.loads((folder/"SEED_DECISION.json").read_text())
        snap=json.loads((folder/"SNAPSHOT.json").read_text())
        close=json.loads((folder/"CLOSE.json").read_text())
        if (decision["status"]!="predicted_face_support_only" or not close["released"]
                or not snap["snapshot"]["responses_complete"] or seed["face_id"]!=face_id):
            raise ValueError("Current native face support has no complete closed response domain")
        previous=None;accepted=0;total=0
        for index,frame in enumerate(doc["frames"]):
            fid=frame["global_frame_id"];rgb,context=provider.load(fid,K);pixels=np.moveaxis(rgb.cpu().numpy(),0,-1)
            status_path=contained_path(R,provider.rows[fid]["native_status"],exists=True)
            body_status=json.loads(status_path.read_text());body_raw=verifier.raw(body_status["raw"],status_path.parent)
            owner=json.loads(context.source_ledger_json)["native_owners"][job["track_id"]]
            body_ids=np.asarray(body_raw["outputs"]["out_obj_ids"]).tolist()
            body=np.asarray(body_raw["outputs"]["out_binary_masks"])[body_ids.index(owner["native_obj_id"])]
            response=folder/f"response_{index:04d}"
            raw=verifier.raw(json.loads((response/"RAW_RECEIPT.json").read_text()),response)
            if raw["frame_index"]!=index or np.asarray(raw["outputs"]["out_obj_ids"]).tolist()!=[0]:
                raise ValueError("Native face frame/object changed")
            mask=np.asarray(raw["outputs"]["out_binary_masks"])[0]
            body_event=json.loads(context.source_ledger_json)["project_body_registrations"][job["track_id"]]
            if body_event["scoped_body_id"]!=seed["body_track_id"] or body_event["project_generation"]!=seed["body_generation"]:
                raise ValueError("Face and actual current body identity mismatch")
            identity=dict(body_id=body_event["scoped_body_id"],body_generation=seed["body_generation"],
                face_id=face_id,face_generation=seed["face_generation"],pixel_space_id=doc["pixel_space_id"],
                time_base=doc["time_base"],frame_id=fid,pts=frame["pts"],rgb_sha256=frame["sha256"],
                rgb_tensor_sha256=array_digest(pixels),body_mask_sha256=array_digest(body),dynamic_mask_sha256=array_digest(mask))
            current=(pixels,body,mask,identity)
            if previous is not None:
                result=measure_border_motion(previous[0],pixels,previous_body=previous[1],current_body=body,
                    previous_dynamic=previous[2],current_dynamic=mask,previous_identity=previous[3],current_identity=identity,config=policy)
                result["project_track_id"]=job["track_id"];rows.append(result);accepted+=int(result["accepted_image_registration"]);total+=1
            previous=current
        summaries.append(dict(track_id=job["track_id"],face_id=face_id,pairs=total,accepted_image_registrations=accepted))
    with (out/"MOTION.jsonl").open("x") as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+chr(10))
    report=dict(status="ACTUAL_RGB_BORDER_MEASUREMENTS_NOT_POSE_ACCEPTANCE",summaries=summaries,
        config=binding(contained_path(R,config_path,exists=True)),new_model_calls=0,geometry_fusion=False,final_corners_modified=False)
    (out/"COMPLETE.json").write_text(json.dumps(report,indent=2));return report
