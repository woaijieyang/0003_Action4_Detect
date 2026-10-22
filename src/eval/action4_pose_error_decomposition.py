"""Post-prediction oracle decomposition; never part of inference or selection."""
import json,numpy as np
from data.action4_paths import PROJECT_ROOT as R

def run():
    out=R/"outputs/action4_pose_error_decomposition_v001";out.mkdir(exist_ok=True)
    if (out/"REPORT.json").exists():raise FileExistsError("Preserve previous diagnostic")
    preds={x["sample_id"]:x["instances"] for x in json.loads((R/"outputs/action4_b0_center_scoring_candidate_v001/development/PREDICTIONS.json").read_text())}
    report=json.loads((R/"outputs/action4_b0_center_scoring_candidate_v001/development/REPORT.json").read_text())
    refs={x["sample_id"]:x for x in map(json.loads,(R/"data/eval/annotations.jsonl").read_text().splitlines())}
    rows=[]
    for match in report["bbox_only"]["per_image"]:
        ref=refs[match["sample_id"]];gt=[g for g in ref["instances"] if not g.get("ignore",False)]
        K=np.array(ref["camera"]["K"]);cad=np.array(ref["corners_3d_mm"])/1000
        for pi,gi in match["matches"]:
            p=preds[match["sample_id"]][pi];g=gt[gi]
            if not p["pose_valid"]:continue
            T=np.array(g["canonical8_geometry"]["object_to_capture_cv_m"]);Rgt=T[:3,:3];tgt=T[:3,3]
            diagnostic=p["pose_diagnostic"];Rp=np.array(diagnostic["rotation"]);tp=np.array(diagnostic["translation_m"])
            diag=np.linalg.norm(np.array(g["bbox_full_xyxy"])[2:]-g["bbox_full_xyxy"][:2]);target=np.array(g["corners_px"])
            def error(rot,trans):
                camera=cad@rot.T+trans
                if (camera[:,2]<=0).any():return None
                h=camera@K.T;uv=h[:,:2]/h[:,2,None]
                return np.linalg.norm(uv-target,axis=1)/diag
            errors={"predicted":error(Rp,tp),"oracle_rotation_only":error(Rgt,tp),"oracle_translation_only":error(Rp,tgt)}
            angle=np.degrees(np.arccos(np.clip((np.trace(Rp@Rgt.T)-1)/2,-1,1)))
            rows.append({"sample_id":match["sample_id"],"instance_id":g["instance_id"],"heavy":g["occlusion_inframe"]>=.4,
                "rotation_error_deg":float(angle),"translation_error_mm":float(np.linalg.norm(tp-tgt)*1000),
                "relative_depth_error":float(abs(tp[2]-tgt[2])/tgt[2]),"pnp_median_reprojection":diagnostic["median_reprojection_px"],
                "methods":{k:{"all8":bool(v is not None and max(v)<=.05),"errors":v.tolist() if v is not None else None} for k,v in errors.items()}})
    summary={k:{"all8":sum(x["methods"][k]["all8"] for x in rows),"heavy_all8":sum(x["heavy"] and x["methods"][k]["all8"] for x in rows)}
        for k in ["predicted","oracle_rotation_only","oracle_translation_only"]}
    for label,subset in [("all",rows),("predicted_success",[x for x in rows if x["methods"]["predicted"]["all8"]]),("predicted_failure",[x for x in rows if not x["methods"]["predicted"]["all8"]])]:
        summary[label]={"count":len(subset),**{key:float(np.median([x[key] for x in subset])) for key in ["rotation_error_deg","translation_error_mm","relative_depth_error","pnp_median_reprojection"]}}
    result={"scope":"ORACLE DIAGNOSTIC ONLY: GT pose substituted after frozen predictions; NOT deployable scores or accuracy gains",
        "total_GT":378,"valid_matched_predictions":len(rows),"summary":summary,"rows":rows}
    (out/"REPORT.json").write_text(json.dumps(result,indent=2));print(json.dumps(summary))
if __name__=="__main__":run()
