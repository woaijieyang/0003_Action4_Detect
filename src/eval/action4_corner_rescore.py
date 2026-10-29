"""Offline scoring repair using immutable complete predictions; no model forward."""
import argparse,json,os,time
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_sha
from eval.action4_corner_scoring_inputs import build_scoring_inputs
from eval.action4_corner_calibration import calibrate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    cfg=json.loads(contained_path(R,args.config,exists=True).read_text())
    from model.training.action4_run_snapshot import verify_running_snapshot
    verify_running_snapshot(cfg)
    refs=cfg["preserved_corner_scoring"]
    artifact=_v2_checked(refs["predictions"])
    execution=_v2_checked(refs["execution"])
    if (artifact["inference_execution_receipt"]!=refs["execution"]
        or artifact["source_manifest"]!=refs["inference_source"]
        or execution["source_manifest"]!=refs["inference_source"]
        or execution["status"]!="COMPLETE_PREDICTION_ONLY_FULL650"
        or execution["images"]!=650 or len(artifact["results"])!=650
        or execution["snapshot"]!=artifact["effective_identity"]["models"]["A0"]):
        raise ValueError("Preserved complete prediction execution identities differ")
    out=contained_path(R,cfg["output"],exists=False)
    if out.exists():raise FileExistsError("Preserve scoring attempts")
    out.mkdir(parents=True)
    start=time.time()
    try:
        result=build_scoring_inputs(refs["predictions"],refs["annotations"],
            refs["inference_source"],out/"scoring",diagnostic=False)
        calibration=calibrate(result["domain"],result["run_binding"],result["records"],
            result["raw_report"],result["candidate_report"])
        (out/"CALIBRATION_RECEIPT.json").write_text(json.dumps(calibration,allow_nan=False,indent=2)+"\n")
        record={"status":"PRESERVED_FULL650_RESCORE_COMPLETE","images":650,"model_forward_calls":0,
            "original_predictions_changed":False,"original_failure_preserved":True,
            "scoring_run_spec":cfg["run_spec"],"input_bindings":refs,
            "elapsed_seconds":time.time()-start,"quality_ready":calibration["quality_ready"],
            "formal_training_or_deployment_accepted":False}
        (out/"COMPLETE.json").write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps(record),flush=True)
    except BaseException as error:
        import traceback
        (out/"FAILED.json").write_text(json.dumps({"error":str(error),"traceback":traceback.format_exc()},indent=2))
        raise
if __name__=="__main__":main()
