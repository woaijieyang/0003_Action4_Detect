"""Build prediction-only SAM prompt inputs for exact native RGB variants.

This is an explicit new input contract, never an alias for native_original.
The existing v1/v2 native runners deliberately do not admit this schema yet.
"""
import copy,json,os
import numpy as np
from eval.action4_rgb_variant_b0 import (
    path,read,binding,write,canonical,pack_row,validate_inputs,sealed_source)

ROW_SCHEMA="action4.sam.prediction_prompt_image.variant.v1"

def prompt_row(entry,prediction,source_receipt):
    if entry.get("variant") not in ("light","blur") or entry.get("variant_id") in (None,"","native_original","original"):
        raise ValueError("Explicit derived RGB variant required")
    if entry.get("recipe",{}).get("variant_id")!=entry["variant_id"] or entry["recipe"].get("variant_sha256")!=entry.get("actual_pixels_sha256"):
        raise ValueError("Variant recipe/pixel binding differs")
    if prediction!=pack_row(entry,prediction["instances"]):
        raise ValueError("B0 prediction source, variant, candidate IDs or raw rows changed")
    shape=entry["native_shape"]
    if (len(shape)!=3 or shape[2]!=3 or any(type(v) is not int or v<=0 for v in shape)):
        raise ValueError("Native RGB HWC dimensions required")
    height,width=shape[:2];candidates=[]
    for item,raw in zip(prediction["candidates"],prediction["instances"]):
        b=np.asarray(raw.get("bbox_full_xyxy"),float);score=raw["score"]
        if b.shape!=(4,) or not np.isfinite(b).all() or not (b[2:]>b[:2]).all() or not 0<=score<=1:
            raise ValueError("Finite original B0 full-body box and probability required")
        clipped=np.clip(b,[0,0,0,0],[width,height,width,height])
        eligible=bool(item["published"] and (clipped[2:]>clipped[:2]).all())
        x0,y0,x1,y1=clipped
        candidates.append({"candidate_id":item["prediction_id"],
            "source":"frozen_completed_B0_rgb_variant_saved_prediction",
            "bbox_full_native_xyxy":b.tolist(),"bbox_sam_native_xyxy":clipped.tolist(),
            "bbox_sam_normalized_xywh":[float(x0/width),float(y0/height),float((x1-x0)/width),float((y1-y0)/height)],
            "score":score,"published":item["published"],
            "eligible_for_SAM_diagnostic_prompt":eligible,
            "raw_saved_prediction":copy.deepcopy(raw),
            "identity_binding":{"sample_id":prediction["sample_id"],"variant_id":entry["variant_id"],
                "grid_index":item["grid_index"],"raw_saved_ordinal":item["raw_saved_ordinal"],
                "variant_pixels_sha256":entry["actual_pixels_sha256"]}})
    return {"schema":ROW_SCHEMA,"image_id":entry["image_id"],"split":"train",
        "rgb_variant":entry["variant"],"variant_id":entry["variant_id"],
        "image_path":entry["actual_input"]["path"],"image_sha256":entry["actual_input"]["sha256"],
        "original_source":copy.deepcopy(entry["original_source"]),
        "variant_pixels_sha256":entry["actual_pixels_sha256"],
        "native_shape_hw":[height,width],"pixel_space_id":"native_integer_centers",
        "source_receipt":copy.deepcopy(source_receipt),"candidates":candidates}

def verified_leaf(completion_binding):
    complete=json.loads(read(completion_binding))
    if (complete.get("status")!="COMPLETE_FROZEN_B0_RGB_VARIANT_LEAF"
            or complete.get("engineering_fixture") is not False
            or complete.get("GT_passed_to_inference") is not False):
        raise ValueError("Completed real frozen-B0 variant leaf required")
    cfg=json.loads(read(complete["config"]));inputs=json.loads(read(complete["input"]))
    if cfg.get("schema")!="action4.rgb_variant_B0_leaf.v1" or cfg.get("mode") not in ("production_cpu_pilot","production_native_variant_leaf"):
        raise ValueError("Explicit real B0 producer required")
    if cfg["input"]!=complete["input"]:
        raise ValueError("Completion/input configuration differs")
    output_root=path(cfg["output"])
    if path(completion_binding["path"])!=output_root/"COMPLETE.json" or path(complete["predictions"]["path"])!=output_root/"PREDICTIONS.json":
        raise ValueError("Cross-run completion/prediction splice")
    read(cfg["orchestrator"])
    for ref in (completion_binding,complete["config"],complete["input"],complete["predictions"],cfg["orchestrator"]):
        if path(ref["path"]).stat().st_mode & 0o222:raise ValueError("Producer artifacts must be readonly")
    validate_inputs(inputs)
    seal,spec,bcfg=sealed_source(cfg["B0_seal"])
    if cfg["scoring_policy"]!=bcfg["evaluation_scoring_policy"] or cfg["publication_threshold"]!=.3:
        raise ValueError("Frozen B0 scoring policy differs")
    predictions=json.loads(read(complete["predictions"]))
    if (len(predictions)!=complete["images"] or len(predictions)!=len(inputs["rows"])
            or [p["sample_id"] for p in predictions]!=inputs["image_ids"]):
        raise ValueError("Complete original image domain required, including zero detections")
    refs={"variant_B0_completion":completion_binding,"variant_B0_config":complete["config"],
        "variant_input":complete["input"],"predictions":complete["predictions"],"B0_seal":cfg["B0_seal"],
        "B0_bundle":{"path":seal["bundle"],"sha256":seal["sha256"]},
        "B0_run_spec":{"path":seal["run_spec"],"sha256":seal["run_spec_sha256"]}}
    rows=[prompt_row(e,p,refs) for e,p in zip(inputs["rows"],predictions)]
    if (sum(len(r["candidates"]) for r in rows)!=complete["saved_candidates"]
            or sum(c["published"] for r in rows for c in r["candidates"])!=complete["published_candidates"]):
        raise ValueError("Producer candidate counts changed")
    return rows,inputs,refs,complete

def prepare(completion_binding,output):
    rows,inputs,refs,complete=verified_leaf(completion_binding)
    out=path(output,False);out.mkdir(parents=True,exist_ok=False)
    manifest=out/"PROMPTS.jsonl"
    with manifest.open("x") as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+chr(10))
        stream.flush();os.fsync(stream.fileno())
    manifest.chmod(0o444)
    return write(out/"PREPARED.json",{"schema":"action4.sam_rgb_variant_preparation.v1",
        "status":"EXACT_VARIANT_PROMPTS_PREPARED_NOT_NATIVE_OR_TRAINING_ADMITTED",
        "manifest":binding(manifest),"source_receipt":refs,"variant":inputs["variant"],
        "images":len(rows),"saved_candidates":sum(len(r["candidates"]) for r in rows),
        "eligible_jobs":sum(c["eligible_for_SAM_diagnostic_prompt"] for r in rows for c in r["candidates"]),
        "source_B0_device":complete["device"],"contains_GT_labels":False,
        "native_SAM_calls":0,"original_6500_changed":False,
        "remaining":["variant-aware native worker lock and response binding","variant cache exporter/provider/association","image-level variant selection in Stage2Units"]})


def verify_prepared(prepared_binding):
    prepared=json.loads(read(prepared_binding))
    if (prepared.get("schema")!="action4.sam_rgb_variant_preparation.v1"
            or prepared.get("status")!="EXACT_VARIANT_PROMPTS_PREPARED_NOT_NATIVE_OR_TRAINING_ADMITTED"
            or prepared.get("native_SAM_calls")!=0 or prepared.get("contains_GT_labels") is not False):
        raise ValueError("Explicit prediction-only variant preparation required")
    rows,inputs,refs,complete=verified_leaf(prepared["source_receipt"]["variant_B0_completion"])
    actual=[json.loads(line) for line in read(prepared["manifest"]).decode().splitlines() if line.strip()]
    if (actual!=rows or prepared["source_receipt"]!=refs or prepared["variant"]!=inputs["variant"]
            or prepared["images"]!=len(rows)
            or prepared["saved_candidates"]!=sum(len(r["candidates"]) for r in rows)
            or prepared["eligible_jobs"]!=sum(c["eligible_for_SAM_diagnostic_prompt"] for r in rows for c in r["candidates"])):
        raise ValueError("Variant input domain or exact B0 evidence changed")
    return prepared,rows

def validate_native_config(config):
    from eval.action4_sam_image_eval import SCHEMA_VARIANT,validate_live_authority
    if config.get("schema")!=SCHEMA_VARIANT or config.get("split")!="train" or config.get("rgb_variant") not in ("light","blur"):
        raise ValueError("Explicit derived train-RGB native SAM contract required")
    prepared=json.loads(read(config["prepared_variant"]))
    if (config.get("manifest")!=prepared["manifest"]["path"]
            or config.get("manifest_sha256")!=prepared["manifest"]["sha256"]
            or config.get("input_contract")!=config["prepared_variant"]["path"]
            or config.get("expected_images")!=prepared["images"]
            or config.get("expected_saved_candidates")!=prepared["saved_candidates"]
            or config.get("expected_prompts")!=prepared["eligible_jobs"]
            or config["rgb_variant"]!=prepared["variant"]):
        raise ValueError("Frozen variant input and expected counts differ")
    if (config.get("upstream_role")!="frozen_completed_B0_rgb_variant_predictions"
            or config.get("native",{}).get("text")!="camera"
            or config.get("K_contract")!={"K":None,"status":"not_supplied_not_consumed_by_native_SAM"}
            or config.get("retry_attempt_limit")!=2):
        raise ValueError("Variant diagnostic must preserve native prompt, K and explicit retry policy")
    if config.get("binding")!={"bbox_iou_role":"diagnostic_only","unique_margin":.15,
        "unique_margin_basis":"target_coverage_minus_best_other_candidate_coverage",
        "mask_in_box_min":.8,"multi_target_support_min":.7,"multiple_native_policy":"require_single_eligible_mask"}:
        raise ValueError("Variant must use the same instance binding policy")
    if config["resources"].get("torch_threads") not in (1,2):
        raise ValueError("Bounded native worker threads required")
    validate_live_authority(config)
    return config

def load_native_manifest(config):
    prepared,rows=verify_prepared(config["prepared_variant"])
    if (prepared["images"]!=config["expected_images"] or prepared["variant"]!=config["rgb_variant"]
            or prepared["saved_candidates"]!=config["expected_saved_candidates"]):
        raise ValueError("Verified variant manifest differs from configured domain")
    jobs=[]
    for row in rows:
        for candidate in row["candidates"]:
            if candidate["eligible_for_SAM_diagnostic_prompt"]:
                jobs.append({"image":row,"candidate":candidate})
    if len(jobs)!=config["expected_prompts"]:raise ValueError("Native prompt coverage differs")
    return rows,jobs
