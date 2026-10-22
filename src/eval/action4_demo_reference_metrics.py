"""Demo-envelope consistency, explicitly distinct from certified detection GT."""
import json,numpy as np
from eval.action4_metrics import bbox_iou,maximum_matching

def compare(reference,predictions,*,prediction_format="staged",source_wh=(3248,2464)):
    if reference.get("certified_GT") is not False or reference.get("training_labels_allowed") is not False:
        raise ValueError("Uncertified reference role must be explicit")
    refs=reference["rows"]
    if len(refs)!=len(predictions) or len(refs)!=reference["frames"]:
        raise ValueError("Full frame coverage required, including unscorable references")
    scale=np.array([950/source_wh[0],720/source_wh[1]]*2);rows=[]
    for fid,(ref,prediction) in enumerate(zip(refs,predictions)):
        if ref["frame_id"]!=fid or prediction["frame_id"]!=fid or prediction["pts"]!=400*fid:
            raise ValueError("Frame identity/timestamp mismatch")
        if not ref["reference_candidate"]:continue
        if len(ref["components"])!=2:raise ValueError("Ambiguous reference marked eligible")
        if prediction_format=="staged":instances=prediction["prediction"]["instances"]
        elif prediction_format=="component":instances=prediction["variants"]["raw"]
        else:raise ValueError("Unknown prediction format")
        boxes=[np.array(p["bbox_full_xyxy"])*scale for p in instances if p["score"]>=.3]
        if any(not np.isfinite(b).all() or np.any(b[2:]<=b[:2]) for b in boxes):
            raise ValueError("Invalid model box; never silently discard failures")
        gt=[p["bbox_demo_px"] for p in ref["components"]]
        values={}
        for threshold in [.45,.5,.55]:
            pairs=maximum_matching([[j for j,g in enumerate(gt) if bbox_iou(b,g)>=threshold] for b in boxes],2)
            values[str(threshold)]={"both":len(pairs)==2,"matched":len(pairs),"predictions":len(boxes)}
        rows.append({"frame_id":fid,"thresholds":values})
    summary={str(t):{"both":sum(r["thresholds"][str(t)]["both"] for r in rows),"frames":len(rows),
        "rate":sum(r["thresholds"][str(t)]["both"] for r in rows)/len(rows) if rows else None} for t in [.45,.5,.55]}
    return {"scope":"demo envelope consistency; not certified real recall or mAP","reference_eligible_frames":len(rows),
        "unscorable_reference_frames":len(refs)-len(rows),"summary":summary,"rows":rows}

if __name__=="__main__":
    import argparse
    from data.action4_paths import PROJECT_ROOT as R,contained_path
    p=argparse.ArgumentParser();p.add_argument("--reference",required=True);p.add_argument("--predictions",required=True)
    p.add_argument("--format",choices=["staged","component"],default="staged");p.add_argument("--output",required=True)
    a=p.parse_args()
    reference=json.loads(contained_path(R,a.reference,exists=True).read_text())
    predictions=[json.loads(l) for l in contained_path(R,a.predictions,exists=True).read_text().splitlines()]
    output=contained_path(R,a.output,exists=False)
    with output.open("x") as f:json.dump(compare(reference,predictions,prediction_format=a.format),f,indent=2)
