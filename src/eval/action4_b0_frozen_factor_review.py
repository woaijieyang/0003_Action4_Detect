"""Paired postprocessing factors on one frozen B0; no training or threshold fit."""
import json,torch,numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from data.action4_paths import PROJECT_ROOT as R
from data.action4_dataset import prepare_full_images
from data.action4_component_data import ComponentData
from model.action4_release_bundle import load_bundle
from model.action4_online_b0 import proposals
from eval.action4_soft_nms import select
from eval.action4_metrics import bbox_iou
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json

METHODS=[b+"__"+s for b in ("proposal","refined") for s in ("center","product","learned")]
@torch.inference_mode()
def predict_factors(model,image,cfg):
    x,aff=prepare_full_images([image],size=tuple(cfg["preprocess"]["full_size"]),device="cuda:0")
    dense,feature=model(x);rows,boxes,output=proposals(model,dense,feature,aff[0],cfg)
    refined=output[0].cpu().tolist();score=output[1].sigmoid().cpu().tolist()
    answer={}
    for method in METHODS:
        geometry,scoring=method.split("__")
        candidates=[]
        for i,row in enumerate(rows):
            probability=row["center_score"] if scoring=="center" else row["old_product_score"] if scoring=="product" else score[i]
            candidates.append(dict(row,bbox_full_xyxy=row["bbox_full_xyxy"] if geometry=="proposal" else refined[i],score=float(probability)))
        answer[method]=select(candidates,"soft_gaussian50")
    return answer

def run(output):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=R/output;out.mkdir(exist_ok=False)
    model,_,state=load_bundle("models/B0/best.pt","B0","cuda:0");cfg=state["config"]
    data=ComponentData("outputs/required/DATA_VALIDATED_ALL_SPLITS.json","eval","B0")
    predictions={m:[] for m in METHODS};references=[]
    for i in range(len(data)):
        item=data[i];result=predict_factors(model,item["image"],cfg);references.append(item["record"])
        for method,rows in result.items():predictions[method].append({"sample_id":item["record"]["sample_id"],"instances":rows})
        if i%25==0:atomic_json(out/"STATUS.json",{"images":i+1,"total":len(data)})
    scores={m:score_boxes(p,references,.3) for m,p in predictions.items()}
    atomic_json(out/"DEVELOPMENT_PREDICTIONS.json",predictions)
    atomic_json(out/"DEVELOPMENT_REPORTS.json",scores)
    manifest=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())
    refs=json.loads((R/"outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json").read_text())["frames"]
    real=[]
    for ref in refs:
        with Image.open(R/manifest["frames"][ref["frame_id"]]["image"]) as im:image=np.array(im.convert("RGB"))
        result=predict_factors(model,torch.from_numpy(image).permute(2,0,1),cfg)
        scale=np.array([950/image.shape[1],720/image.shape[0]]*2)
        rows={}
        for method,ps in result.items():
            ps=[p for p in ps if p["score"]>=.3]
            matrix=np.array([[bbox_iou(np.array(p["bbox_full_xyxy"])*scale,g["bbox_demo_px"]) for g in ref["reference_overlay_components"]] for p in ps])
            matches=[]
            if matrix.size:
                a,b=linear_sum_assignment(-matrix);matches=[float(matrix[i,j]) for i,j in zip(a,b)]
            rows[method]={"predictions":ps,"reference_ious":matches,"both_reference_ge05":len(matches)==2 and min(matches)>=.5}
        real.append({"frame_id":ref["frame_id"],"methods":rows})
    report={"scope":"frozen-weight paired factors; development and previously reviewed real-reference6; no test consumed, no threshold search",
        "B0_step":state["selected_step"],"score_threshold":.3,"methods":{m:{"mAP50_95":scores[m]["mAP50_95"],"F1":scores[m]["overall"]["f1"],"dual":scores[m]["overall"]["both_detected_images"],"dual_total":scores[m]["overall"]["dual_images"],"real_reference_both":sum(r["methods"][m]["both_reference_ge05"] for r in real)} for m in METHODS},"real_rows":real,"production_promoted":False}
    atomic_json(out/"REPORT.json",report);print(json.dumps(report["methods"]))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)
