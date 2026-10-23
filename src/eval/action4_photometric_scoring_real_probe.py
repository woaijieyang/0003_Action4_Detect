"""Frozen dev-selected scoring method on prespecified distributed real frames."""
import json,torch,numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from model.action4_release_bundle import load_bundle
from eval.action4_b0_scoring import predict
from eval.action4_metrics import bbox_iou,maximum_matching
from model.training.action4_runtime import atomic_json

def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=R/"outputs/action4_photometric_scoring_review_v001"
    selection=json.loads((root/"DEV_SELECTION.json").read_text())
    if selection["selected_method"]!="refined__learned":raise ValueError("This probe is bound to the preselected dev winner")
    reference=json.loads((R/"outputs/action4_full_demo_reference_v001/REFERENCE.json").read_text())
    eligible=[r for r in reference["rows"] if r["reference_candidate"]]
    chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,60,dtype=int)]
    atomic_json(root/"REAL_PROBE_SELECTION.json",{"frame_ids":[r["frame_id"] for r in chosen],"selected_before_prediction":True,"scope":"diagnostic only, no acceptance on60frames"})
    manifest=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())
    baseline={r["frame_id"]:r["prediction"]["instances"] for r in map(json.loads,(R/"outputs/action4_photometric_validation_v001/video/predictions.jsonl").read_text().splitlines())}
    model,_,state=load_bundle("models/candidates/b0_photometric_v001/best.pt","B0","cuda:0");rows=[]
    scale=np.array([950/3248,720/2464]*2)
    for ref in chosen:
        fid=ref["frame_id"]
        with Image.open(R/manifest["frames"][fid]["image"]) as im:image=torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1)
        candidate=[p for p in predict(model,image,state["config"],"cuda:0","learned") if p["score"]>=.3]
        values={}
        for name,ps in [("center",baseline[fid]),("learned",candidate)]:
            gt=[g["bbox_demo_px"] for g in ref["components"]]
            edges=[[j for j,g in enumerate(gt) if bbox_iou(np.array(p["bbox_full_xyxy"])*scale,g)>=.5] for p in ps]
            match=maximum_matching(edges,2)
            values[name]={"both":len(match)==2,"matched":len(match),"predictions":ps}
        rows.append({"frame_id":fid,"methods":values})
    summary={name:sum(r["methods"][name]["both"] for r in rows) for name in ["center","learned"]}
    atomic_json(root/"REAL_PROBE_REPORT.json",{"B0_step":state["selected_step"],"frames":len(rows),"summary":summary,"rows":rows,
        "scope":"same60 prespecified demo-reference frames; not certifiedGT, not whole-video acceptance","production_promoted":False})
    print(json.dumps(summary))
if __name__=="__main__":run()
