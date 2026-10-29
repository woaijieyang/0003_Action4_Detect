"""Full200 predicted-ROI diagnostic; bbox and corner results stay separate."""
import json,torch,numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import is_ignored
from eval.action4_staged_pipeline import load_pipeline
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json

def validate_reused_data(data, validation_receipt, split):
    """Reuse only the exact worker-validated receipt and requested full-image split."""
    from data.action4_paths import contained_path
    expected = contained_path(R, validation_receipt, exists=True)
    if getattr(data, "_validation_receipt", None) != expected:
        raise ValueError("Reused evaluation data receipt identity mismatch")
    if data.role != "B0" or not data.records or any(r["split"] != split for r in data.records):
        raise ValueError("Full-image evaluation dataset role/split mismatch")


def run(selection,output,split="eval",validation_receipt="outputs/required/DATA_VALIDATED_ALL_SPLITS.json",validated_data=None):
    if validated_data is not None:
        validate_reused_data(validated_data, validation_receipt, split)
    cfg=json.loads((R/selection).read_text());out=R/output;out.mkdir(exist_ok=False)
    from model.training.action4_resource_policy import evaluation_threads
    torch.set_num_threads(evaluation_threads(cfg))
    pipeline=load_pipeline(cfg)
    from eval.action4_sam_pipeline_inputs import sam_eval_provider
    sam_provider=sam_eval_provider(pipeline,cfg,split)
    data=ComponentData(validation_receipt,split,"B0") if validated_data is None else validated_data
    if sam_provider is not None:sam_provider.cache.validate_image_coverage(data.records)
    cad=np.array(json.loads((R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"])/1000
    frozen_boxes=None;frozen_calls=[0];frozen_hook=None
    if cfg.get("frozen_b0_predictions") is not None:
        from data.action4_sam_native_export import _v2_checked
        from eval.action4_surface_snapshot import model_identities
        if sam_provider is not None:
            if cfg.get("prediction_input_protocol")!="action4.frozen_B0_SAM_original_eval.v1":
                raise ValueError("Explicit frozen B0 plus SAM eval contract required")
            index=_v2_checked(cfg["sam_cache"]["manifest"])
            receipt=_v2_checked(index["producer_receipt"])
            if receipt["source_contract"]["predictions"]["sha256"]!=cfg["frozen_b0_predictions"]["sha256"]:
                raise ValueError("SAM prompts and replayed B0 predictions do not share the same saved output identity")
        sources=model_identities(pipeline._surface_model_bindings)
        if sources["B0"]["sha256"]!=cfg["frozen_b0_sha256"]:raise ValueError("Frozen B0 differs")
        source_rows=_v2_checked(cfg["frozen_b0_predictions"])
        frozen_boxes={r["sample_id"]:r["instances"] for r in source_rows}
        if len(frozen_boxes)!=len(source_rows) or set(frozen_boxes)!={r["sample_id"] for r in data.records}:raise ValueError("Frozen eval domain differs")
        pipeline.b0.eval().requires_grad_(False)
        frozen_hook=pipeline.b0.register_forward_hook(lambda *a:frozen_calls.__setitem__(0,frozen_calls[0]+1))
    predictions=[];box_predictions=[];references=[]
    from eval.action4_prefetch import ordered_items
    for i,item in ordered_items(data,workers=cfg.get("eval_loader_workers",4)):
        record=item["record"]
        assert all(is_ignored(g)==bool(g.get("ignore",False)) for g in record["instances"])
        sam_kwargs={} if sam_provider is None else {"sam_context":sam_provider.context(item)}
        inference_rgb=item["image"] if sam_provider is None else sam_kwargs["sam_context"].frame.pixels
        if frozen_boxes is not None:
            from data.action4_frozen_detection_context import FrozenDetectionContext
            from data.action4_sam_roi_inputs import ImageFrame,array_digest
            if sam_provider is not None:
                context=sam_kwargs["sam_context"]
                if context.frame.image_id!=record["sample_id"] or context.frame.source_sha256!=record["rgb_sha256"]:
                    raise ValueError("SAM/frozen detector source image identity differs")
                sam_kwargs["frozen_detections"]=FrozenDetectionContext.from_sam_frame(context.frame,record["camera"]["K"],frozen_boxes[record["sample_id"]],
                    sam_predictions=context.predictions,sam_sources=context.expected_sources,
                    b0_sha256=cfg["frozen_b0_sha256"],scoring_policy=cfg["b0_scoring_policy"],source_cache_identity=cfg["frozen_b0_predictions"]["sha256"])
            else:
                native=item["image"].permute(1,2,0).numpy()
                frame=ImageFrame(record["sample_id"],native,"phrase2_native_integer_centers",record["rgb_sha256"],"native_original",array_digest(native))
                sam_kwargs["frozen_detections"]=FrozenDetectionContext.from_frame(frame,record["camera"]["K"],frozen_boxes[record["sample_id"]],
                    b0_sha256=cfg["frozen_b0_sha256"],scoring_policy=cfg["b0_scoring_policy"],source_cache_identity=cfg["frozen_b0_predictions"]["sha256"])
        result=pipeline.predict(inference_rgb,np.asarray(record["camera"]["K"]),cad,
                                return_box_candidates=True,**sam_kwargs)
        predictions.append({"sample_id":record["sample_id"],"instances":result["instances"]})
        box_predictions.append({"sample_id":record["sample_id"],"instances":result["box_candidates"]})
        operating=[p for p in result["box_candidates"] if p["score"]>=.3]
        if [(p["score"],p["bbox_full_xyxy"]) for p in operating] != [(p["score"],p["bbox_full_xyxy"]) for p in result["instances"]]:
            raise ValueError("Operating box and corner indices disagree")
        references.append(record)
        if i%25==0:atomic_json(out/"STATUS.json",{"completed":i+1,"total":len(data)})
    if frozen_hook is not None:
        frozen_hook.remove()
        if frozen_calls[0]!=0:raise ValueError("Frozen eval unexpectedly executed B0")
    boxes=score_boxes(box_predictions,references,.3)
    boxes["AP_candidate_policy"]={"post_nms_cutoff":.001,"max_detections":2,"operating_threshold":.3,"threshold_applied_before_AP":False}
    per_image={r["sample_id"]:r for r in boxes["per_image"]};predmap={r["sample_id"]:r["instances"] for r in predictions}
    rows=[];heavy_total=0;GT=0;invalid=sum(not p["pose_valid"] for r in predictions for p in r["instances"])
    for record in references:
        sid=record["sample_id"];gt=[g for g in record["instances"] if not g.get("ignore",False)]
        GT+=len(gt);heavy_total+=sum(g["occlusion_inframe"]>=.4 for g in gt)
        for pi,gi in per_image[sid]["matches"]:
            prediction=predmap[sid][pi];g=gt[gi];error=None
            if prediction["pose_valid"] and prediction["corners_px"] is not None:
                diag=np.linalg.norm(np.array(g["bbox_full_xyxy"])[2:]-g["bbox_full_xyxy"][:2])
                error=(np.linalg.norm(np.array(prediction["corners_px"])-g["corners_px"],axis=1)/diag).tolist()
            rows.append({"sample_id":sid,"prediction_index":pi,"instance_id":g["instance_id"],"heavy":g["occlusion_inframe"]>=.4,
                         "errors":error,"all8":error is not None and max(error)<=.05})
    report={"scope":f"full{len(data)} {split} B0 predicted ROIs, no target box/pose input","dataset_role":"development" if split=="eval" else "historical_test_not_new_blind_acceptance","identity":pipeline.identity,
        "bbox_only":boxes,"corner_GT":GT,"corner_all8_success":sum(r["all8"] for r in rows),
        "heavy_GT":heavy_total,"heavy_all8_success":sum(r["heavy"] and r["all8"] for r in rows),
        "invalid_pose_detections":invalid,"bbox_matched_targets":len(rows),"box_missed_targets":GT-len(rows),
        "corner_protocol":"IoU0.5 one-to-one box matches, then fixedID all8 errors<=.05GTdiagonal; unmatched GT and invalid pose fail; no invalid prediction removal from bbox score",
        "corner_error_scope":"matched boxes only, no zero errors invented for misses","corner_rows":rows}
    if sam_provider is not None:
        report["sam_cache_inputs"]={"manifest":cfg["sam_cache"]["manifest"],
            "engineering_fixture":sam_provider.cache.allow_synthetic,
            "performance_admissible":not sam_provider.cache.allow_synthetic,
            "GT_used_for_prediction":False}
    if frozen_boxes is not None:
        report["frozen_b0_predictions"]=cfg["frozen_b0_predictions"]
        report["B0_forward_calls"]=frozen_calls[0]
        report["roi_expansion"]=pipeline.roi_expansion
    from eval.action4_pose_protocol import score_pose
    report.update(score_pose(predictions,references))
    if cfg.get("frozen_b0_predictions") is not None:
        from eval.action4_roi_pose_diagnostics import summarize
        diagnostics=summarize(predictions,references,report)
        atomic_json(out/"POSE_DIAGNOSTICS.json",diagnostics)
        report["physical_pose_diagnostics"]={k:v for k,v in diagnostics.items() if k!="rows"}

    atomic_json(out/"BOX_PREDICTIONS.json",box_predictions);atomic_json(out/"PREDICTIONS.json",predictions);atomic_json(out/"REPORT.json",report)
    print(json.dumps({k:report[k] for k in ("identity","corner_GT","corner_all8_success","heavy_GT","heavy_all8_success","invalid_pose_detections","bbox_matched_targets","box_missed_targets")}))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--output",required=True);p.add_argument("--split",choices=["eval","test"],default="eval");p.add_argument("--validation-receipt",default="outputs/required/DATA_VALIDATED_ALL_SPLITS.json");a=p.parse_args();run(a.selection,a.output,a.split,a.validation_receipt)
