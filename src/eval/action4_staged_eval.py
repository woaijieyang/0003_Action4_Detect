"""Full200 predicted-ROI diagnostic; bbox and corner results stay separate."""
import json,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import is_ignored
from eval.action4_staged_pipeline import load_pipeline
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json

def run(selection,output,split="eval",validation_receipt="outputs/required/DATA_VALIDATED_ALL_SPLITS.json"):
    torch.set_num_threads(2);cfg=json.loads((R/selection).read_text());out=R/output;out.mkdir(exist_ok=False)
    pipeline=load_pipeline(cfg)
    data=ComponentData(validation_receipt,split,"B0")
    cad=np.array(json.loads((R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"])/1000
    predictions=[];box_predictions=[];references=[]
    for i in range(len(data)):
        item=data[i];record=item["record"]
        assert all(is_ignored(g)==bool(g.get("ignore",False)) for g in record["instances"])
        result=pipeline.predict(item["image"],np.asarray(record["camera"]["K"]),cad,return_box_candidates=True)
        predictions.append({"sample_id":record["sample_id"],"instances":result["instances"]})
        box_predictions.append({"sample_id":record["sample_id"],"instances":result["box_candidates"]})
        operating=[p for p in result["box_candidates"] if p["score"]>=.3]
        if [(p["score"],p["bbox_full_xyxy"]) for p in operating] != [(p["score"],p["bbox_full_xyxy"]) for p in result["instances"]]:
            raise ValueError("Operating box and corner indices disagree")
        references.append(record)
        if i%25==0:atomic_json(out/"STATUS.json",{"completed":i+1,"total":len(data)})
    boxes=score_boxes(box_predictions,references,.3)
    boxes["AP_candidate_policy"]={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3,"threshold_applied_before_AP":False}
    per_image={r["sample_id"]:r for r in boxes["per_image"]};predmap={r["sample_id"]:r["instances"] for r in predictions}
    rows=[];heavy_total=0;GT=0;invalid=sum(not p["pose_valid"] for r in predictions for p in r["instances"])
    for record in references:
        sid=record["sample_id"];gt=[g for g in record["instances"] if not g.get("ignore",False)]
        GT+=len(gt);heavy_total+=sum(g["occlusion_inframe"]>=.4 for g in gt)
        for pi,gi in per_image[sid]["matches"]:
            prediction=predmap[sid][pi];g=gt[gi];error=None
            if prediction["pose_valid"] and prediction["corners_px"] is not None:
                diag=np.linalg.norm(np.array(g["bbox_full_xyxy"])[2:]-g["bbox_full_xyxy"][:2])
                error=(np.linalg.norm(np.array(prediction["corners_px"])-g["corners_px"],axis=1)/diag).tolist()
            rows.append({"sample_id":sid,"prediction_index":pi,"instance_id":g["instance_id"],"heavy":g["occlusion_inframe"]>=.4,
                         "errors":error,"all8":error is not None and max(error)<=.05})
    report={"scope":f"full{len(data)} {split} B0 predicted ROIs, no target box/pose input","dataset_role":"development" if split=="eval" else "historical_test_not_new_blind_acceptance","identity":pipeline.identity,
        "bbox_only":boxes,"corner_GT":GT,"corner_all8_success":sum(r["all8"] for r in rows),
        "heavy_GT":heavy_total,"heavy_all8_success":sum(r["heavy"] and r["all8"] for r in rows),
        "invalid_pose_detections":invalid,"bbox_matched_targets":len(rows),"box_missed_targets":GT-len(rows),
        "corner_protocol":"IoU0.5 one-to-one box matches, then fixedID all8 errors<=.05GTdiagonal; unmatched GT and invalid pose fail; no invalid prediction removal from bbox score",
        "corner_error_scope":"matched boxes only, no zero errors invented for misses","corner_rows":rows}
    atomic_json(out/"BOX_PREDICTIONS.json",box_predictions);atomic_json(out/"PREDICTIONS.json",predictions);atomic_json(out/"REPORT.json",report)
    print(json.dumps({k:report[k] for k in ("identity","corner_GT","corner_all8_success","heavy_GT","heavy_all8_success","invalid_pose_detections","bbox_matched_targets","box_missed_targets")}))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--output",required=True);p.add_argument("--split",choices=["eval","test"],default="eval");p.add_argument("--validation-receipt",default="outputs/required/DATA_VALIDATED_ALL_SPLITS.json");a=p.parse_args();run(a.selection,a.output,a.split,a.validation_receipt)
