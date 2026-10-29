"""Loss-only frozen-prediction ownership. Never reads GT masks or changes SAM inputs."""
import json
import math
from pathlib import Path
from .action4_paths import PROJECT_ROOT as ROOT,contained_path
from .action4_dataset import is_ignored
from .action4_sam_native_export import (_v2_checked,_v2_read,verify_completed_export_receipt,
    canonical,sha,COMPLETED_MODE)

SCHEMA="action4.sam_training_associations.v2"
ASSIGNMENT="preregistered_unique_prediction_id_no_mask_selection"
POLICY={"schema":"action4.sam_training_association_policy.v1",
    "published_score_min":0.3,"complete_bbox_iou_min":0.5,
    "multiple_gt_candidate":"ambiguous_not_usable",
    "unique_gt_candidate_ranking":["predicted_score_desc","stable_candidate_id_asc"],
    "ignored_gt":"excluded_from_units_and_matching","invalid_gt_bbox":"unknown",
    "native_availability":"independent_of_supervision_ownership",
    "GT_masks_used":False,"GT_corners_used":False,"native_mask_selection_changed":False,
    "scope":"native_original_full_box_supervision_ownership_only"}

def _box(value):
    if not isinstance(value,(list,tuple)) or len(value)!=4:return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):return None
    a,b,c,d=map(float,value)
    return (a,b,c,d) if c>a and d>b else None

def _iou(a,b):
    area=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

def derive_rows(records,saved,images,native_statuses,*,variant_ids=None):
    """Pure deterministic box ownership; SAM availability does not affect ranking."""
    rows=[];total=ignored=0;seen_images=set()
    for record in sorted(records,key=lambda r:r["sample_id"]):
        sid=record["sample_id"]
        if sid in seen_images:raise ValueError("Duplicate annotation image")
        seen_images.add(sid)
        units={};seen_gt=set()
        for obj in record["instances"]:
            raw_id=obj["instance_id"]
            if type(raw_id) not in (int,str) or not str(raw_id).strip():raise ValueError("Explicit scalar GT instance ID required")
            gid=str(raw_id)
            if not gid or gid in seen_gt:raise ValueError("Duplicate/invalid annotation GT instance")
            seen_gt.add(gid);total+=1
            if is_ignored(obj):ignored+=1;continue
            units[gid]=_box(obj.get("bbox_full_xyxy"))
        predictions={}
        for raw in saved[sid]:
            grid=raw["grid_index"]
            if variant_ids is None:pid=f"b0:{sid}:grid:{grid}"
            else:
                variant=variant_ids[sid]
                if not isinstance(variant,str) or variant in ("","native_original","original"):
                    raise ValueError("Explicit derived prediction variant identity required")
                pid=f"b0:{sid}:{variant}:grid:{grid}"
            if type(grid) is not int or grid<0 or pid in predictions:raise ValueError("Duplicate/invalid prediction grid")
            box=_box(raw["bbox_full_xyxy"]);score=raw["score"]
            if box is None or isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score):
                raise ValueError("Invalid frozen predicted bbox/score")
            predictions[pid]=(box,float(score))
        if set(predictions)!={p["prediction_id"] for p in images[sid]["predictions"]}:
            raise ValueError("Exact cache/prediction candidate image membership required")
        qualifying={pid:[gid for gid,b in units.items() if b is not None and _iou(box,b)>=POLICY["complete_bbox_iou_min"]]
                    for pid,(box,score) in predictions.items() if score>=POLICY["published_score_min"]}
        for gid,bbox in sorted(units.items()):
            unique=sorted((pid for pid,gids in qualifying.items() if gids==[gid]),
                          key=lambda pid:(-predictions[pid][1],pid))
            ambiguous=sorted(pid for pid,gids in qualifying.items() if gid in gids and len(gids)>1)
            if bbox is None:status,pid,reason="unknown",None,"invalid_or_missing_complete_gt_bbox"
            elif unique:status,pid,reason="matched",unique[0],"unique_gt_bbox_ownership_predicted_score_then_id"
            elif ambiguous:status,pid,reason="ambiguous",None,"candidate_qualifies_multiple_gt_units"
            else:status,pid,reason="missing",None,"no_published_unique_gt_candidate"
            relevant=sorted(set(unique+ambiguous))
            rows.append({"image_id":sid,"gt_instance_id":gid,"status":status,"prediction_id":pid,"reason":reason,
                "qualified_prediction_ids":unique,"ambiguous_prediction_ids":ambiguous,
                "native_status":native_statuses[(sid,pid)] if pid is not None else None,
                "candidate_native_statuses":{p:native_statuses[(sid,p)] for p in relevant}})
    coverage={"image_count":len(records),"annotation_instance_count":total,"ignored_instance_count":ignored,
              "nonignored_gt_units":total-ignored,"association_rows":len(rows)}
    return rows,coverage

def _prepare(cache_spec,annotations,policy,index=None):
    if not isinstance(cache_spec,dict) or set(cache_spec)!={"manifest","expected_sources"}:
        raise ValueError("Explicit production cache spec required")
    loaded=_v2_checked(cache_spec["manifest"])
    if index is not None and loaded!=index:raise ValueError("Provider cache differs from bound association cache")
    index=loaded
    if (index.get("split")!="train" or index.get("origin")!="predicted_sam"
            or index.get("rgb_variant")!="native_original" or index.get("sources")!=cache_spec["expected_sources"]
            or index["sources"].get("upstream_mode")!=COMPLETED_MODE):
        raise ValueError("Production associations require complete native-original completed-B0 train cache")
    receipt=verify_completed_export_receipt(index)
    if _v2_checked(policy)!=POLICY:raise ValueError("Association policy differs from preregistered conservative policy")
    records=_v2_read(annotations,lambda f:[json.loads(line) for line in f if line.strip()])
    if not records:raise ValueError("Full annotation image domain required")
    images={row["image_id"]:row for row in index["images"]}
    if len(records)!=len(images) or {r["sample_id"] for r in records}!=set(images):
        raise ValueError("Annotation/cache full image coverage mismatch")
    for r in records:
        if r.get("split")!="train" or r.get("status")!="training_accepted" or not isinstance(r.get("instances"),list):
            raise ValueError("Accepted train annotation records required")
        e=images[r["sample_id"]]
        if (contained_path(ROOT,r["image"],exists=True)!=contained_path(ROOT,e["image_path"],exists=True)
                or r["rgb_sha256"]!=e["source_sha256"]):
            raise ValueError("Annotation/cache source image identity mismatch")
    source=receipt["source_contract"];prediction_rows=_v2_checked(source["predictions"])
    saved={r["sample_id"]:r["instances"] for r in prediction_rows}
    if len(saved)!=len(prediction_rows) or set(saved)!=set(images):raise ValueError("Prediction image domain mismatch")
    statuses={(r["image_id"],r["prediction_id"]):r["status"] for r in receipt["mapping"]}
    rows,coverage=derive_rows(records,saved,images,statuses)
    payload={"schema":SCHEMA,"assignment_policy":ASSIGNMENT,
        "cache_manifest":cache_spec["manifest"],"cache_manifest_sha256":cache_spec["manifest"]["sha256"],
        "expected_sources":cache_spec["expected_sources"],"annotations":annotations,"policy":policy,
        "predictions":source["predictions"],"source_contract":source,"coverage":coverage,"rows":rows}
    payload["identity"]=canonical(payload)
    return payload,{r["sample_id"]:canonical(r) for r in records}

def build_associations(cache_spec,annotations,policy,output):
    payload,_=_prepare(cache_spec,annotations,policy)
    path=contained_path(ROOT,output,exists=False)
    with path.open("x") as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write("\n")
    return {"path":str(path.relative_to(ROOT)),"sha256":sha(path)}

def verify_associations(association,cache_spec,index=None):
    if not isinstance(association,dict) or association.get("schema")!=SCHEMA:
        raise ValueError("Completed cache requires production association v2; fixtures are not admissible")
    expected,record_identities=_prepare(cache_spec,association["annotations"],association["policy"],index)
    if association!=expected:
        raise ValueError("Production association differs from full deterministic annotation/prediction ownership")
    return record_identities



VARIANT_SCHEMA="action4.sam_training_associations.rgb_variant.v1"
VARIANT_POLICY=dict(POLICY,scope="native_rgb_variant_original_box_supervision_ownership_only")


from collections import OrderedDict
_ANNOTATION_SUBSETS=OrderedDict()

def _variant_annotations(binding,image_ids):
    """Keep one verified parsed document, copy only requested loss-owner records."""
    import os,copy
    from .action4_sam_native_export import _v2_sha,_fingerprint
    p=contained_path(ROOT,binding["path"],exists=True).resolve(strict=True);before=_fingerprint(p.stat())
    if _v2_sha(p)!=binding["sha256"]:raise ValueError("Annotation bytes changed")
    key=(os.getpid(),str(p),binding["sha256"],before)
    if key not in _ANNOTATION_SUBSETS:
        rows=_v2_read(binding,lambda stream:[json.loads(line) for line in stream if line.strip()])
        by_id={r["sample_id"]:r for r in rows}
        if len(by_id)!=len(rows):raise ValueError("Duplicate original annotation image")
        _ANNOTATION_SUBSETS[key]=by_id
        while len(_ANNOTATION_SUBSETS)>2:_ANNOTATION_SUBSETS.popitem(last=False)
    by_id=_ANNOTATION_SUBSETS[key]
    if not set(image_ids)<=set(by_id):raise ValueError("Variant cache images absent from original annotations")
    result=[copy.deepcopy(by_id[sid]) for sid in sorted(image_ids)]
    if _fingerprint(p.stat())!=before:raise ValueError("Annotations changed while selecting records")
    _ANNOTATION_SUBSETS.move_to_end(key)
    return result,len(by_id)

def _prepare_variant(cache_spec,annotations,policy,index=None):
    """Qualify only the exact cached variant domain, using original labels for loss ownership."""
    from .action4_sam_native_export import VARIANT_MODE,verify_variant_export_receipt
    if not isinstance(cache_spec,dict) or set(cache_spec)!={"manifest","expected_sources"}:
        raise ValueError("Explicit variant cache spec required")
    loaded=_v2_checked(cache_spec["manifest"])
    if index is not None and loaded!=index:raise ValueError("Provider variant cache differs from association cache")
    index=loaded
    if (index.get("split")!="train" or index.get("origin")!="predicted_sam"
            or index.get("rgb_variant") not in ("light","blur") or index.get("sources")!=cache_spec["expected_sources"]
            or index["sources"].get("upstream_mode")!=VARIANT_MODE):
        raise ValueError("Explicit completed variant cache required; native_original is not an alias")
    receipt=verify_variant_export_receipt(index)
    if receipt.get("formal_training_admitted") is not False:raise ValueError("Variant export training gate changed")
    if _v2_checked(policy)!=VARIANT_POLICY:raise ValueError("Variant association policy differs from preregistered matching policy")
    images={row["image_id"]:row for row in index["images"]}
    if not images:raise ValueError("Empty variant annotation domain")
    records,annotation_count=_variant_annotations(annotations,images)
    for record in records:
        if record.get("split")!="train" or record.get("status")!="training_accepted" or not isinstance(record.get("instances"),list):
            raise ValueError("Accepted original train annotation records required")
        original=images[record["sample_id"]]["original_source"]
        if (contained_path(ROOT,record["image"],exists=True)!=contained_path(ROOT,original["path"],exists=True)
                or record["rgb_sha256"]!=original["sha256"]):
            raise ValueError("Variant annotation/original source image identity mismatch")
    source=receipt["source_contract"];prediction_rows=_v2_checked(source["predictions"])
    saved={row["sample_id"]:row["instances"] for row in prediction_rows}
    if len(saved)!=len(prediction_rows) or set(saved)!=set(images):raise ValueError("Variant prediction image domain mismatch")
    statuses={(row["image_id"],row["prediction_id"]):row["status"] for row in receipt["mapping"]}
    variant_ids={sid:entry["variant_id"] for sid,entry in images.items()}
    rows,coverage=derive_rows(records,saved,images,statuses,variant_ids=variant_ids)
    coverage.update(source_annotation_image_count=annotation_count,scope="exact_variant_cache_domain_only")
    image_bindings=[{key:entry[key] for key in ("image_id","image_path","source_sha256","pixel_space_id","variant_id","variant_sha256","original_source")}
                    for sid,entry in sorted(images.items())]
    payload={"schema":VARIANT_SCHEMA,"assignment_policy":ASSIGNMENT,
        "cache_manifest":cache_spec["manifest"],"cache_manifest_sha256":cache_spec["manifest"]["sha256"],
        "expected_sources":cache_spec["expected_sources"],"annotations":annotations,"policy":policy,
        "predictions":source["predictions"],"source_contract":source,"coverage":coverage,"rows":rows,
        "variant_images":image_bindings,"formal_training_admitted":False,
        "consumer_qualification":"exact_cached_variant_supervision_ownership_not_full_dataset_admission"}
    payload["identity"]=canonical(payload)
    return payload,{r["sample_id"]:canonical(r) for r in records}


def build_variant_associations(cache_spec,annotations,policy,output):
    payload,_=_prepare_variant(cache_spec,annotations,policy)
    path=contained_path(ROOT,output,exists=False)
    with path.open("x") as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write("\n")
    return {"path":str(path.relative_to(ROOT)),"sha256":sha(path)}


def verify_variant_associations(association,cache_spec,index=None):
    if not isinstance(association,dict) or association.get("schema")!=VARIANT_SCHEMA:
        raise ValueError("Typed variant supervision association required")
    expected,identities=_prepare_variant(cache_spec,association["annotations"],association["policy"],index)
    if association!=expected:raise ValueError("Variant association differs from exact original annotation/prediction ownership")
    return identities


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-spec",required=True);p.add_argument("--annotations",required=True)
    p.add_argument("--annotations-sha256",required=True);p.add_argument("--policy",required=True)
    p.add_argument("--policy-sha256",required=True);p.add_argument("--output",required=True);args=p.parse_args()
    spec_path=contained_path(ROOT,args.cache_spec,exists=True)
    spec=_v2_checked({"path":str(spec_path),"sha256":sha(spec_path)})
    print(json.dumps(build_associations(spec,{"path":args.annotations,"sha256":args.annotations_sha256},
        {"path":args.policy,"sha256":args.policy_sha256},args.output)))
