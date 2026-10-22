"""2x2 frozen box-vs-score intervention on identical real-image proposals."""
import json,torch,numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from data.action4_paths import PROJECT_ROOT as R
from eval.action4_box_refinement_pipeline import RefinedB0
from eval.action4_box_quality_pool import candidate_union
from data.action4_dataset import prepare_full_images
from model.action4_box_quality import box_features
from eval.action4_soft_nms import select
from eval.action4_metrics import bbox_iou
from model.training.action4_runtime import atomic_json

def run(head,output):
    torch.set_num_threads(1);out=R/output;out.mkdir(exist_ok=False)
    predictor=RefinedB0(R/"outputs/action4_box_quality_candidate_v001/base_B0_best.pt",R/head)
    manifest=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())
    refs=json.loads((R/"outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json").read_text())["frames"]
    results=[]
    with torch.inference_mode():
        for ref in refs:
            with Image.open(R/manifest["frames"][ref["frame_id"]]["image"]) as im:rgb=im.convert("RGB")
            x,affines=prepare_full_images([torch.from_numpy(np.array(rgb)).permute(2,0,1)],size=tuple(predictor.config["preprocess"]["full_size"]),device="cuda:0")
            dense=predictor.base(x);ps,canvas=candidate_union(dense,affines[0],predictor.config)
            f,g=box_features(predictor.feature,canvas,tuple(predictor.config["preprocess"]["full_size"]))
            b=f.new_tensor([p["bbox_full_xyxy"] for p in ps]);priors=f.new_tensor([[p["center_score"],p["old_quality_score"],p["old_product_score"]] for p in ps])
            rb,logits,_=predictor.head(f,g,priors,b);rb=rb.cpu().tolist();scores=logits.sigmoid().cpu().tolist()
            variants={};scale=np.array([950/rgb.width,720/rgb.height]*2)
            for new_box in (False,True):
                for new_score in (False,True):
                    name=("newbox" if new_box else "oldbox")+"_"+("newscore" if new_score else "oldscore")
                    candidates=[dict(p,bbox_full_xyxy=rb[i] if new_box else p["bbox_full_xyxy"],score=scores[i] if new_score else p["old_product_score"]) for i,p in enumerate(ps)]
                    selected=select(candidates,"soft_gaussian50",threshold=.3)
                    matrix=np.array([[bbox_iou(np.array(p["bbox_full_xyxy"])*scale,r["bbox_demo_px"]) for r in ref["reference_overlay_components"]] for p in selected])
                    matches=[]
                    if matrix.size:
                        a,z=linear_sum_assignment(-matrix);matches=[float(matrix[i,j]) for i,j in zip(a,z)]
                    variants[name]={"both_reference_ge05":len(matches)==2 and min(matches)>=.5,"reference_iou":matches,"selected":selected}
            results.append({"frame_id":ref["frame_id"],"variants":variants,"pool":[dict(p,refined_box=rb[i],new_score=scores[i]) for i,p in enumerate(ps)]})
    predictor.close()
    totals={k:sum(r["variants"][k]["both_reference_ge05"] for r in results) for k in variants}
    atomic_json(out/"REPORT.json",{"scope":"same6realframes intervention; demo reference not GT","checkpoint_step":predictor.state["step"],"totals":totals,"frames":results})
    print(json.dumps({"step":predictor.state["step"],"totals":totals,"frames":[{"frame":r["frame_id"],"pass":{k:v["both_reference_ge05"] for k,v in r["variants"].items()}} for r in results]}))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--head",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.head,a.output)
