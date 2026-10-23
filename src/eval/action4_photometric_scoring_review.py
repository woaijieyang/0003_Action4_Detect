"""Recheck scoring factors on completed photometric weights; select on dev only."""
import json,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from model.action4_release_bundle import load_bundle
from eval.action4_b0_frozen_factor_review import predict_factors,METHODS
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json

def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=R/"outputs/action4_photometric_scoring_review_v001";out.mkdir(exist_ok=False)
    model,_,state=load_bundle("models/candidates/b0_photometric_v001/best.pt","B0","cuda:0")
    data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","eval","B0")
    predictions={k:[] for k in METHODS};references=[]
    for i in range(len(data)):
        item=data[i];result=predict_factors(model,item["image"],state["config"]);references.append(item["record"])
        for name,value in result.items():predictions[name].append({"sample_id":item["record"]["sample_id"],"instances":value})
        if i%25==0:atomic_json(out/"STATUS.json",{"completed":i+1,"total":len(data)})
    reports={k:score_boxes(v,references,.3) for k,v in predictions.items()}
    summary={k:{"mAP50_95":v["mAP50_95"],"F1":v["overall"]["f1"],"dual":v["overall"]["both_detected_images"]} for k,v in reports.items()}
    selected=max(summary,key=lambda k:(summary[k]["mAP50_95"],summary[k]["F1"]))
    atomic_json(out/"REPORTS.json",reports);atomic_json(out/"PREDICTIONS.json",predictions)
    atomic_json(out/"DEV_SELECTION.json",{"B0_step":state["selected_step"],"summary":summary,"selected_method":selected,
        "selection":"development mAP50:95 then F1, fixed six methods; real result not viewed to choose method","real_evaluated":False,
        "source_weights_changed":False,"current_baseline":"refined__center"})
    print(json.dumps(summary))
if __name__=="__main__":run()
