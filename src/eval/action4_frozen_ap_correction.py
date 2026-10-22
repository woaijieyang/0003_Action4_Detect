"""Re-evaluate unchanged sealed B0 AP without the operating-threshold truncation."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from model.action4_release_bundle import load_bundle
from eval.action4_online_b0_pipeline import predict
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json

def run():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=R/"outputs/action4_frozen_test_ap_correction_v001";out.mkdir(exist_ok=False)
    model,_,state=load_bundle("models/B0/best.pt","B0","cuda:0")
    data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","test","B0")
    predictions=[];references=[]
    for i in range(len(data)):
        item=data[i]
        predictions.append({"sample_id":item["record"]["sample_id"],"instances":predict(model,item["image"],state["config"],"cuda:0")})
        references.append(item["record"])
        if i%25==0:atomic_json(out/"STATUS.json",{"completed":i+1,"total":len(data)})
    report=score_boxes(predictions,references,.3)
    report.update(B0_step=state["selected_step"],dataset_role="historical_test_protocol_correction_only",
        AP_candidate_policy={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3,"threshold_applied_before_AP":False},
        checkpoint_or_threshold_changed=False,new_candidate_evaluated=False)
    old=json.loads((R/"outputs/action4_final_test_v001/REPORT.json").read_text())["bbox_only"]
    report["operating_counts_equal_previous"]=all(report["overall"][k]==old["overall"][k] for k in ["tp","fp","fn","both_detected_images"])
    report["previous_truncated_AP50_95"]=old["mAP50_95"]
    atomic_json(out/"PREDICTIONS.json",predictions);atomic_json(out/"REPORT.json",report)
if __name__=="__main__":run()
