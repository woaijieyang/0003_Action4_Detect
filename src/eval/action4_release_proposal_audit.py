"""Frozen real-frame proposal/score boundary audit, demo reference only."""
import json,torch,numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_dataset import prepare_full_images
from model.action4_release_bundle import load_bundle
from model.action4_online_b0 import proposals
from eval.action4_metrics import bbox_iou
from eval.action4_online_b0_pipeline import predict
from model.training.action4_runtime import atomic_json

def run(bundle,output):
    torch.set_num_threads(1)
    out=R/output;out.mkdir(exist_ok=False)
    model,_,state=load_bundle(bundle,"B0","cuda:0");cfg=state["config"]
    frames=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())["frames"]
    refs=json.loads((R/"outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json").read_text())["frames"]
    results=[]
    with torch.inference_mode():
        for ref in refs:
            fid=ref["frame_id"]
            with Image.open(R/frames[fid]["image"]) as im:rgb=im.convert("RGB")
            image=torch.from_numpy(np.array(rgb)).permute(2,0,1)
            x,aff=prepare_full_images([image],size=tuple(cfg["preprocess"]["full_size"]),device="cuda:0")
            dense,features=model(x);rows,boxes,refined=proposals(model,dense,features,aff[0],cfg)
            scores=refined[1].sigmoid().cpu().tolist()
            raw=boxes.cpu().numpy();adjusted=refined[0].cpu().numpy()
            scale=np.array([950/rgb.width,720/rgb.height]*2)
            targets=[]
            order=np.argsort(-np.asarray(scores))
            for target in ref["reference_overlay_components"]:
                gt=target["bbox_demo_px"]
                before=[bbox_iou(box*scale,gt) for box in raw]
                after=[bbox_iou(box*scale,gt) for box in adjusted]
                eligible=[i for i in order if after[i]>=.5]
                targets.append({"best_proposal_iou":max(before,default=0),"best_refined_iou":max(after,default=0),
                    "first_good_score_rank":list(order).index(eligible[0])+1 if eligible else None,
                    "first_good_score":scores[eligible[0]] if eligible else None})
            results.append({"frame_id":fid,"proposal_count":len(rows),"reference_targets":targets,
                "top5":[{"score":scores[i],"bbox":adjusted[i].tolist()} for i in order[:5]],
                "final":predict(model,image,cfg,"cuda:0")})
    atomic_json(out/"REPORT.json",{"B0_step":state["selected_step"],"scope":"demo overlay reference diagnostic, not certified GT or tuning","rows":results})
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--bundle",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.bundle,a.output)
