"""Exposed-development quality regression diagnostic; never issues calibration admission."""
import copy,json
import numpy as np
from scipy.stats import spearmanr
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_sha
from eval.action4_refined_quality_sidecar import RefinedQualitySidecar,_read_manifest
from eval.action4_metrics import bbox_iou
from eval.action4_bbox_metrics import score_boxes

TARGET="quality_target_max_nonignore_fullbody_box_IoU"

def regression_rows(base,references,sidecar):
    if ([r["sample_id"] for r in base]!=[r["sample_id"] for r in references]
        or [r["image"]["sample_id"] for r in sidecar["rows"]]!=[r["sample_id"] for r in base]):
        raise ValueError("Exact ordered development domain required")
    result=[]
    for pred,ref,quality in zip(base,references,sidecar["rows"]):
        gt=[g for g in ref["instances"] if not g.get("ignore",False)]
        if len(pred["instances"])!=len(quality["qualities"]):raise ValueError("Quality candidate domain differs")
        for p,q in zip(pred["instances"],quality["qualities"]):
            pid=f"b0:{ref['sample_id']}:grid:{p['grid_index']}"
            if q["prediction_id"]!=pid:raise ValueError("Quality candidate identity differs")
            ious=[bbox_iou(p["bbox_full_xyxy"],g["bbox_full_xyxy"]) for g in gt]
            target=max(ious,default=0.)
            result.append({"sample_id":ref["sample_id"],"prediction_id":pid,"grid_index":p["grid_index"],
                "B0_score":p["score"],"B0_operating_selected":p["score"]>=.3,"refined_quality":q["refined_quality_score"],
                TARGET:target,"quality_error_q_minus_target":q["refined_quality_score"]-target,
                "max_IoU_GT_instance_ids":[g["instance_id"] for g,iou in zip(gt,ious) if iou==target],
                "nonignore_GT_count":len(gt),"not_one_to_one_detector_assignment":True})
    return result

def regression_stats(rows):
    q=np.asarray([r["refined_quality"] for r in rows],float);y=np.asarray([r[TARGET] for r in rows],float)
    if not len(q):return {"candidates":0,"MAE":None,"bias":None,"RMSE":None,"Pearson":None,"Spearman":None}
    err=q-y;variable=len(q)>1 and np.ptp(q)>0 and np.ptp(y)>0
    return {"candidates":len(q),"MAE":float(np.abs(err).mean()),"bias_q_minus_target":float(err.mean()),
        "RMSE":float(np.sqrt(np.square(err).mean())),"mean_q":float(q.mean()),"mean_target":float(y.mean()),
        "Pearson":float(np.corrcoef(q,y)[0,1]) if variable else None,
        "Spearman":float(spearmanr(q,y).statistic) if variable else None,
        "correlation_unavailable_reason":None if variable else "insufficient_or_constant_values",
        "q_range":[float(q.min()),float(q.max())],"target_range":[float(y.min()),float(y.max())]}

def select_diagnostic_tie(threshold_reports):
    preference={.5:2,.4:1,.6:0}
    return max(threshold_reports,key=lambda r:((r["overall"]["f1"] or 0.),
        (r["overall"]["recall"] or 0.),(r["overall"]["precision"] or 0.),preference[r["quality_threshold"]]))["quality_threshold"]

def run(preregistration):
    pre=_v2_checked(preregistration);out=contained_path(R,preregistration["path"],exists=True).parent
    if pre["quality_threshold_grid"]!=[.4,.5,.6] or pre["B0_operating_score_threshold"]!=.3:
        raise ValueError("Fixed limited diagnostic threshold policy changed")
    if _v2_sha("src/eval/action4_bbox_metrics.py")!=pre["scorer_sha256"]:raise ValueError("Original bbox scorer changed")
    inputs=pre["inputs"];side=_v2_checked(inputs["sidecar"])
    bindings={k:inputs[k] for k in ("B0","base_predictions","data_manifest","source")}
    # Full-domain contract validation only; raw RGB consumption was already verified in capture.
    RefinedQualitySidecar(inputs["sidecar"],bindings=bindings,model_config_id=side["model_config_id"],selected_step=9500)
    base=_v2_checked(inputs["base_predictions"]);refs=_read_manifest(inputs["data_manifest"])
    if len(refs)!=650 or any(r["split"]!="eval" for r in refs):raise ValueError("Development eval650 only")
    rows=regression_rows(base,refs,side)
    stats=regression_stats(rows);bins=[]
    edges=pre["fixed_quality_bins"]
    for i,(lo,hi) in enumerate(zip(edges,edges[1:])):
        selected=[r for r in rows if lo<=r["refined_quality"] and (r["refined_quality"]<hi or i==len(edges)-2 and r["refined_quality"]<=hi)]
        bins.append({"lower":lo,"upper":hi,"right_inclusive":i==len(edges)-2,**regression_stats(selected)})
    baseline=score_boxes(base,refs,.3)
    reports=[];selection_signatures=[]
    for threshold in pre["quality_threshold_grid"]:
        predictions=[];selection=[]
        for source,quality in zip(base,side["rows"]):
            retained=[];records=[]
            for p,q in zip(source["instances"],quality["qualities"]):
                keep=q["refined_quality_score"]>=threshold
                if keep:retained.append(copy.deepcopy(p))
                records.append({"prediction_id":q["prediction_id"],"grid_index":p["grid_index"],
                    "B0_score":p["score"],"refined_quality":q["refined_quality_score"],
                    "quality_selected":keep,"operating_selected":keep and p["score"]>=.3})
            predictions.append({"sample_id":source["sample_id"],"instances":retained})
            selection.append({"sample_id":source["sample_id"],"predictions":records})
        report=score_boxes(predictions,refs,.3)
        signature=[[r["prediction_id"] for r in image["predictions"] if r["operating_selected"]] for image in selection]
        selection_signatures.append(signature)
        report.update(quality_threshold=threshold,quality_selection=selection,unchanged_GT_denominator=sum(x["GT"] for x in report["per_image"]),
            refined_quality_is_additional_filter=True,B0_score_ranking_unchanged=True)
        path=out/f"THRESHOLD_{threshold:.1f}.json"
        with path.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False)
        reports.append(report)
    all_same=all(v==selection_signatures[0] for v in selection_signatures)
    counts=[sum(not g.get("ignore",False) for g in r["instances"]) for r in refs]
    report={"status":"EXPOSED_DEVELOPMENT_DIAGNOSTIC_COMPLETE_NOT_CALIBRATED","target_name":TARGET,
        "quality_target_is_not_one_to_one_detector_TP":True,"quality_regression_all_candidates":stats,
        "quality_regression_operating_candidates":regression_stats([r for r in rows if r["B0_operating_selected"]]),
        "fixed_decile_bins":bins,"baseline_bbox_metrics":baseline,
        "threshold_summaries":[{"threshold":r["quality_threshold"],"overall":r["overall"],"GT":r["unchanged_GT_denominator"],
            "per_scene":r["per_scene"],"per_target_stratum":r["per_target_stratum"]} for r in reports],
        "all_thresholds_same_selected_prediction_IDs":all_same,
        "diagnostic_tie_selected_threshold":select_diagnostic_tie(reports),
        "threshold_is_not_calibrated_or_published":True,
        "negative_scope":{"targetless_images":counts.count(0),"single_target_images":counts.count(1),"two_target_images":counts.count(2),
            "real_world_negative_specificity_measured":False},
        "exposure_disclosure":pre["exposure_disclosure"],
        "limitations":["Exposed synthetic development data, not independent calibration or test.",
            "Max-IoU quality targets allow duplicate predictions to share oneGT; detector scoring is separately one-to-one.",
            "Threshold grid unchanged after seeing q distribution.",
            "No threshold discrimination on this dataset." if all_same else "Threshold differences are development-only.",
            "No targetless/single-target coverage where respective count iszero; no real negative specificity evidence."],
        "calibrated":False,"admitted":False,"gate_receipt_written":False}
    for name,value in (("QUALITY_ROWS.json",rows),("DIAGNOSTIC.json",report)):
        with (out/name).open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False)
    return report

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--preregistration",required=True);p.add_argument("--sha256",required=True)
    a=p.parse_args();run({"path":a.preregistration,"sha256":a.sha256})
