"""Full650 development-only YOLO native cache, fixed-policy scoring and audit runner.

Default modes prepare/status/score/aggregate are CPU only. run-shard is explicit
and resource-gated; never imports or starts SAM. No test/locked-set access.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path("/data/zy/vln/a-test")
DEFAULT_CONFIG = "configs/eval/action4_full_zeroshot_dev_v001.json"
NATIVE_SCHEMA = "action4.full_zeroshot.native.v1"
JOINT_CONTRACT = "visible_support_plus_prompt_joint_only_99"


def resolve(value, exists=True):
    p = ROOT / value
    if p.is_symlink() or not p.resolve(strict=exists).is_relative_to(ROOT.resolve()):
        raise ValueError("Path outside project or symlink")
    return p.resolve(strict=exists)


_ACTIVE_OUTPUT_ROOT = None


def output_root():
    return resolve(_ACTIVE_OUTPUT_ROOT or "outputs/action4_full_zeroshot_eval_v001", False)


def sha(path):
    with resolve(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(resolve(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write_json(path, value):
    p = resolve(path, False)
    if not p.is_relative_to(output_root()):
        raise ValueError("Writes are confined to owned evaluation output")
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + ".tmp_" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, p)


def fingerprint(path):
    s = resolve(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def checked_file(path, expected):
    if sha(path) != expected:
        raise ValueError("File content identity mismatch: " + str(path))
    return {"path": str(resolve(path).relative_to(ROOT)), "sha256": expected, "stat": fingerprint(path)}


def load_config(path):
    cfg = read_json(path)
    d = cfg["dataset"]
    if cfg["schema"] not in ("action4.full_zeroshot_dev.v1","action4.full_zeroshot_dev.v2") or d["split"] != "eval":
        raise ValueError("Only explicit full development eval supported")
    if d["manifest"] != "data/phrase2/eval/annotations.jsonl" or d["expected_images"] != 650 or d["expected_original_gt"] != 1300:
        raise ValueError("Full650/1300 eval binding required; no subset/test/locked override")
    if cfg["scoring"]["contract"] != JOINT_CONTRACT or cfg["scoring"]["observed_gate"] != .99:
        raise ValueError("Visible-support/prompt99 contract cannot be relaxed")
    if not 1 <= cfg["inference"]["max_cpu_threads"] <= 4:
        raise ValueError("CPU budget must be1..4")
    if len({p["id"] for p in cfg["policies"]}) != len(cfg["policies"]):
        raise ValueError("Duplicate policy IDs")
    global _ACTIVE_OUTPUT_ROOT
    if cfg["schema"]=="action4.full_zeroshot_dev.v2" and not cfg.get("profile_manifest"):
        raise ValueError("v2 requires declared profile_manifest")
    if cfg.get("profile_manifest"):
        if cfg["schema"]!="action4.full_zeroshot_dev.v2" or not cfg.get("output_root"):
            raise ValueError("Declared profiles require v2 and explicit output_root")
        destination=resolve(cfg["output_root"],False)
        if not destination.is_relative_to(ROOT/"outputs") or destination==ROOT/"outputs/action4_full_zeroshot_eval_v001":
            raise ValueError("New declared profiles cannot write the baseline output")
        _ACTIVE_OUTPUT_ROOT=str(destination)
    else:
        _ACTIVE_OUTPUT_ROOT=None
    return cfg


def box_area(box):
    return max(0., box[2]-box[0]) * max(0., box[3]-box[1])


def box_iou(a, b):
    intersection = box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union else 0.


def box_gap(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def mask_box(mask):
    import numpy as np
    yy, xx = np.nonzero(mask)
    return None if not len(xx) else [float(xx.min()-.5), float(yy.min()-.5), float(xx.max()+.5), float(yy.max()+.5)]


def boundary_points(mask):
    import numpy as np
    from scipy.ndimage import binary_erosion
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.empty((0,2))
    x0, x1, y0, y1 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
    crop = mask[y0:y1, x0:x1]
    edge = crop & ~binary_erosion(crop, structure=np.ones((3,3), bool), border_value=0)
    y, x = np.nonzero(edge)
    return np.column_stack((x+x0, y+y0)).astype(float)


def pair_geometry(a, b, ma, mb, rules):
    import numpy as np
    from scipy.spatial import cKDTree
    ba, bb = a["bbox_full_xyxy"], b["bbox_full_xyxy"]
    smaller = min(math.hypot(ba[2]-ba[0], ba[3]-ba[1]), math.hypot(bb[2]-bb[0], bb[3]-bb[1]))
    centers = math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2, (ba[1]+ba[3]-bb[1]-bb[3])/2)
    va, vb = mask_box(ma), mask_box(mb)
    result = {"gt_ids":[a["id"],b["id"]], "center_distance_px":centers,
              "center_distance_over_smaller_full_diagonal":centers/smaller,
              "full_bbox_iou":box_iou(ba,bb), "visible_bbox_iou":None,
              "visible_bbox_gap_px":None, "near_pair":None,
              "mask_min_pixel_center_distance_px":None, "mask_intersection_px":int(np.count_nonzero(ma & mb))}
    if va is not None and vb is not None:
        gap = box_gap(va,vb)
        short = min(va[2]-va[0],va[3]-va[1],vb[2]-vb[0],vb[3]-vb[1])
        result.update(visible_bbox_iou=box_iou(va,vb), visible_bbox_gap_px=gap,
                      near_pair=bool(gap<=rules["near_pair_gap_fraction"]*short or box_iou(va,vb)>0))
        if result["mask_intersection_px"]:
            distance=0.
        else:
            distance=float(cKDTree(boundary_points(mb)).query(boundary_points(ma),k=1,workers=1)[0].min())
        result["mask_min_pixel_center_distance_px"]=distance
    return result


def interval_label(value, edges, prefix):
    for lo, hi in zip(edges[:-1],edges[1:]):
        if lo <= value < hi:
            return f"{prefix}:{lo:g}_to_{hi:g}"
    raise ValueError("Value outside declared bin range")


def prepare_one(index, row, cfg):
    import numpy as np
    from PIL import Image
    d, rules = cfg["dataset"], cfg["groups"]
    if row["split"] != "eval" or "/eval/" not in row["image"]:
        raise ValueError("Non-eval sample refused")
    image = checked_file(row["image"], row["rgb_sha256"])
    with Image.open(resolve(row["image"])) as im:
        if im.size != (row["width"],row["height"]):
            raise ValueError("Image dimensions differ from manifest")
    targets=[];masks=[];groups={"all_dev", "scene:"+row["scene_family"]}
    for inst in row["instances"]:
        if inst.get("ignore") is not False:
            raise ValueError("Original1300 nonignored contract changed; never silently drop GT")
        binding=checked_file(inst["mask_visible"],inst["mask_visible_sha256"])
        mask=np.array(Image.open(resolve(inst["mask_visible"])).convert("L"))>=128
        if mask.shape != (row["height"],row["width"]):
            raise ValueError("GT mask native shape mismatch")
        pixels=int(mask.sum())
        if pixels != inst["area_visible_px"]:
            raise ValueError("GT actual visible-pixel count differs from manifest")
        occ=float(inst["occlusion_inframe"])
        amodal=int(inst["area_inframe_amodal_px"])
        actual=max(0.,min(1.,1-pixels/amodal)) if amodal else None
        if not math.isfinite(occ) or not 0<=occ<=1 or (actual is not None and abs(actual-occ)>1e-6):
            raise ValueError("Actual occlusion fraction inconsistent")
        box=list(map(float,inst["bbox_full_xyxy"]))
        if len(box)!=4 or not all(math.isfinite(x) for x in box) or box_area(box)<=0:
            raise ValueError("Invalid original full GT bbox")
        width_fraction=(box[2]-box[0])/row["width"]
        ig=[interval_label(occ,rules["occlusion_edges"],"occlusion"),
            interval_label(pixels,rules["visible_area_edges_px"],"visible_area_px")]
        if occ>=rules["heavy_occlusion_min"]:ig.append("heavy_occlusion")
        if occ>=rules["severe_occlusion_min"]:ig.append("severe_occlusion")
        if width_fraction<rules["small_target_full_bbox_width_fraction"]:ig.append("small_target")
        if pixels==0:ig.append("support_unmeasured_empty_visible")
        groups.update(ig)
        target={"id":str(inst["instance_id"]),"original_object_id":inst["object_id"],
                "class_name":"camera","semantic_note":"GT Action4; evaluation class is camera proxy, not model specificity",
                "annotation_status":"valid","bbox_full_xyxy":box,"visible_bbox_xyxy":mask_box(mask),
                "visible_mask":binding,"actual_occlusion_fraction":occ,
                "occlusion_recomputed_from_pixels":actual,"visible_pixels":pixels,"inframe_amodal_pixels":amodal,
                "full_bbox_width_pixels":box[2]-box[0],"full_bbox_width_fraction":width_fraction,
                "support_measurable":bool(pixels),"groups":ig}
        targets.append(target);masks.append(mask)
    pairs=[]
    for i in range(len(targets)):
        for j in range(i+1,len(targets)):
            pair=pair_geometry(targets[i],targets[j],masks[i],masks[j],rules)
            pairs.append(pair)
            if pair["near_pair"] is True:groups.add("near_pair")
            if pair["near_pair"] is None:groups.add("near_pair_unmeasured")
            groups.add(interval_label(pair["center_distance_over_smaller_full_diagonal"],rules["pair_center_ratio_edges"],"pair_center_ratio"))
            if pair["full_bbox_iou"]>0:groups.add("full_boxes_overlap")
            if pair["mask_intersection_px"]>0:groups.add("visible_masks_intersect")
            if pair["mask_min_pixel_center_distance_px"] is not None:
                groups.add(interval_label(pair["mask_min_pixel_center_distance_px"],rules["mask_distance_edges_px"],"mask_distance_px"))
    return {"index":index,"sample_id":row["sample_id"],"image":image,"native_shape":[row["height"],row["width"]],
            "scene_family":row["scene_family"],"original_group_id":row.get("group_id"),
            "source_group":str(row.get("background_source_id","UNKNOWN"))+":"+row["scene_family"],
            "independence_status":"NOT_ESTABLISHED","original_gt_count":len(targets),"targets":targets,
            "pairs":pairs,"groups":sorted(groups)}


def activate_snapshot(cfg):
    manifest_path=cfg["inference"]["prediction_snapshot"]
    if sha(manifest_path)!=cfg["inference"]["prediction_snapshot_sha256"]:
        raise ValueError("Prediction snapshot manifest changed")
    snapshot=read_json(manifest_path);code=resolve(snapshot["code_root"])
    for name,expected in snapshot["files"].items():
        if sha(code/name)!=expected:raise ValueError("Frozen prediction source changed: "+name)
    for module in list(sys.modules.values()):
        name=getattr(module,"__name__","");filename=getattr(module,"__file__",None)
        orchestrator = name in ("eval", "eval.action4_full_zeroshot_eval", __name__)
        if filename and name.split(".")[0] in ("model","data","eval") and not orchestrator and not resolve(filename).is_relative_to(code):
            raise ValueError("Use a fresh interpreter; mixed live/frozen project imports")
    os.environ["ACTION4_PROJECT_ROOT"]=str(ROOT);sys.dont_write_bytecode=True
    sys.path.insert(0,str(code/"src"))
    return snapshot


def prompt_split(rows,cfg):
    spec=cfg["prompt_split"]
    checked_file(spec["exposed_dev_records"],spec["exposed_dev_records_sha256"])
    exposed=read_json(spec["exposed_dev_records"])
    byid={r["sample_id"]:r for r in rows}
    exposed_ids=[r["sample_id"] for r in exposed]
    if any(sid not in byid for sid in exposed_ids):raise ValueError("Prior dev exposure not in cohort")
    dev_families={byid[sid]["scene_family"] for sid in exposed_ids}
    families={}
    for row in rows:families.setdefault(row["scene_family"],[]).append(row["sample_id"])
    dev=[r["sample_id"] for r in rows if r["scene_family"] in dev_families]
    gate=[r["sample_id"] for r in rows if r["scene_family"] not in dev_families]
    return {"schema":"action4.full_dev.prompt_split.v1","policy":spec,"group_key":"scene_family",
            "families":families,"previously_exposed_ids":exposed_ids,
            "external_exposed_dev":spec["external_exposed_dev"],"prompt_dev_ids":dev,"prompt_gate_ids":gate,
            "dev_images":len(dev),"gate_images":len(gate),
            "status":"BLOCKED_NO_UNEXPOSED_SCENE_FAMILY" if not gate else "FAMILY_SPLIT_FROZEN",
            "formal_gate_complete":False,"not_a_final_test":True}



def cohort_contract(cfg):
    contract={"dataset":cfg["dataset"],"groups":cfg["groups"],"prompt_split":cfg["prompt_split"],
              "preparer_source_sha256":sha(__file__)}
    if cfg.get("profile_manifest"):
        contract.update(profile_manifest=cfg["profile_manifest"],reuse_cohort=cfg["reuse_cohort"],
                        real_dev=cfg["real_dev"],output_root=cfg["output_root"],inference_configuration=cfg["inference"])
    return contract


def declared_profiles(cfg,asset_cfg):
    """Load pre-registered native visual profiles; never register/query here."""
    declaration=cfg["profile_manifest"]
    checked_file(declaration["path"],declaration["sha256"])
    manifest=read_json(declaration["path"])
    if manifest.get("schema")!=declaration["schema"] or manifest.get("status")!=declaration["status"]:
        raise ValueError("Declared profile manifest schema/status mismatch")
    if manifest.get("model_assets")!=asset_cfg["assets"] or manifest.get("upstream_sources")!=asset_cfg["sources"]:
        raise ValueError("Registration/inference model-source identity mismatch")
    expected=cfg["inference"]["profile_names"]
    byname={p["name"]:p for p in manifest["profiles"]}
    if len(byname)!=len(manifest["profiles"]) or set(byname)!=set(expected):
        raise ValueError("Declared profile names/count changed")
    excluded=set(cfg["real_dev"]["excluded_enrollment_frame_ids"])
    if set(manifest["enrollment_frame_ids"])!=excluded or manifest.get("forbidden_query_or_scored_enrollment") is not True:
        raise ValueError("Enrollment exclusion contract mismatch")
    profiles=[]
    for name in expected:
        p=byname[name]
        if p.get("text") is not False or p.get("target_class_ids")!=[0] or p.get("negative_class_ids")!=[]:
            raise ValueError("This declared manifest is visual single-positive-class only")
        if p.get("embedding_shape")!=[1,1,512] or p.get("embedding_dtype")!="float32" or p.get("embedding_finite") is not True:
            raise ValueError("Invalid registered embedding metadata")
        proof=p["strict_reload"]
        if not all(proof.get(k) is True for k in ("official_loader_used","npz_keys_exact","names_exact","embedding_bit_exact")):
            raise ValueError("Profile lacks strict official reload receipt")
        checked_file(p["path"],p["sha256"])
        for key in ("reference_rgb","reference_mask","raster"):
            checked_file(p[key]["path"],p[key]["sha256"])
        checked_file(p["reference_manifest_path"],p["reference_manifest_sha256"])
        if set(p["reference_scope"]["enrollment_frame_ids"])!=excluded:
            raise ValueError("Profile-specific enrollment exclusion differs")
        profiles.append(dict(p,profile_manifest_sha256=declaration["sha256"]))
    return profiles


def reused_synthetic_cohort(cfg):
    """Reuse verified rows/group definitions; do not decode GT masks again."""
    declaration=cfg["reuse_cohort"]
    checked_file(declaration["receipt"],declaration["receipt_sha256"])
    receipt=read_json(declaration["receipt"])
    if receipt["cohort_sha256"]!=declaration["cohort_sha256"]:
        raise ValueError("Inherited cohort identity mismatch")
    checked_file(receipt["cohort_path"],declaration["cohort_sha256"])
    checked_file(receipt["group_inventory_path"],declaration["groups_sha256"])
    for key in ("dataset","groups","prompt_split"):
        if receipt["contract"][key]!=cfg[key]:
            raise ValueError("Inherited data/group/split contract changed: "+key)
    if receipt["contract"]["preparer_source_sha256"]!=declaration["preparer_source_sha256"]:
        raise ValueError("Inherited source protocol identity mismatch")
    checked_file(cfg["dataset"]["manifest"],cfg["dataset"]["manifest_sha256"])
    rows=[json.loads(x) for x in resolve(receipt["cohort_path"]).read_text().splitlines()]
    if len(rows)!=650 or sum(x["original_gt_count"] for x in rows)!=1300:
        raise ValueError("Inherited original650/1300 coverage incomplete")
    for row in rows:
        verify_bound_input(row["image"])
        for target in row["targets"]:verify_bound_input(target["visible_mask"])
    return rows,read_json(receipt["group_inventory_path"]),receipt


def real_query_ids(cfg):
    spec=cfg["real_dev"]
    ids=[i for start,end in spec["frame_ranges_inclusive"] for i in range(start,end+1)]
    if len(ids)!=spec["expected_images"] or len(ids)!=len(set(ids)):
        raise ValueError("Real development frame count/uniqueness mismatch")
    registry=spec["registered_inventory"]
    checked_file(registry["path"],registry["sha256"])
    permitted=[row["frame_id"] for segment in read_json(registry["path"])["segments"] for row in segment["rows"]]
    if ids!=permitted:raise ValueError("Queries differ from registered harddev; test/locked frames forbidden")
    if set(ids)&set(spec["excluded_enrollment_frame_ids"]):
        raise ValueError("Enrollment frames must never be queries/scored")
    if spec["annotation_status"]!="unknown" or spec["metrics_allowed"] is not False:
        raise ValueError("Real unannotated frames cannot enter supervised metrics")
    return ids


def prepare_real_rows(cfg,start_index):
    from PIL import Image
    spec=cfg["real_dev"];ids=real_query_ids(cfg)
    checked_file(spec["frame_manifest"],spec["frame_manifest_sha256"])
    manifest=read_json(spec["frame_manifest"])
    if manifest.get("complete") is not True:raise ValueError("Real frame manifest incomplete")
    byid={row["frame_id"]:row for row in manifest["frames"]}
    rows=[]
    for offset,frame_id in enumerate(ids):
        source=byid[frame_id]
        binding=checked_file(source["image"],source["image_sha256"])
        with Image.open(resolve(source["image"])) as image:width,height=image.size
        if [width,height]!=[manifest["source"]["width"],manifest["source"]["height"]]:
            raise ValueError("Real native frame dimensions changed")
        rows.append({"index":start_index+offset,"sample_id":f"real_harddev_{frame_id:08d}",
                     "domain":"real_harddev_unannotated","frame_id":frame_id,"pts":source["pts"],
                     "timestamp_seconds":source["timestamp_seconds"],"image":binding,"native_shape":[height,width],
                     "annotation_status":"unknown","original_gt_count":None,"targets":None,"pairs":None,
                     "groups":["real_harddev_unannotated"],"source_group":"single_real_video",
                     "independence_status":"SAME_VIDEO_NOT_INDEPENDENT_SCENE","prompt_partition":"prompt_dev",
                     "reference_enrollment":False,"supervised_metrics_allowed":False})
    return rows


def prepare_declared_cohort(cfg):
    synthetic,groups,parent=reused_synthetic_cohort(cfg)
    real=prepare_real_rows(cfg,len(synthetic))
    lock=inference_lock(cfg);rows=synthetic+real
    if len(rows)*len(lock["profiles"])!=cfg["inference"]["expected_native_tasks"]:
        raise ValueError("Declared native query budget mismatch")
    contract=cohort_contract(cfg);cohort_id=canonical(contract)
    folder=resolve(output_root()/"cohorts"/cohort_id,False)
    if not folder.is_relative_to(output_root()):raise ValueError("Cohort output escaped")
    if (folder/"COHORT_RECEIPT.json").exists():
        receipt=read_json(folder/"COHORT_RECEIPT.json")
        if sha(receipt["cohort_path"])!=receipt["cohort_sha256"]:raise ValueError("Existing declared cohort changed")
        write_json(output_root()/"COHORT_RECEIPT.json",receipt);return receipt
    folder.mkdir(parents=True,exist_ok=False)
    path=folder/"COHORT.jsonl"
    path.write_text("".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in rows))
    inventory={"schema":"action4.full_zeroshot.mixed_domain_inventory.v1",
               "synthetic":groups,"real":{"images":len(real),"annotation_status":"unknown","gt_count":None,
               "supervised_metrics_allowed":False,"enrollment_excluded":cfg["real_dev"]["excluded_enrollment_frame_ids"]},
               "no_GT_mask_redecode":True,"source_cohort_sha256":parent["cohort_sha256"]}
    write_json(folder/"GROUPS.json",inventory);write_json(output_root()/"GROUPS.json",inventory)
    write_json(folder/"INFERENCE_LOCK.json",lock)
    receipt={"schema":"action4.full_zeroshot.cohort.v2","status":"DECLARED_COHORT_PREPARED_NO_INFERENCE",
             "cohort_id":cohort_id,"cohort_path":str(path.relative_to(ROOT)),"cohort_sha256":sha(path),
             "group_inventory_path":str((folder/"GROUPS.json").relative_to(ROOT)),
             "inference_lock_path":str((folder/"INFERENCE_LOCK.json").relative_to(ROOT)),
             "inference_identity":lock["identity"],"contract":contract,"images":len(rows),
             "synthetic_images":650,"original_gt":1300,"unknown_gt_images":len(real),"real_gt_count":None,
             "expected_native_tasks":len(rows)*len(lock["profiles"]),"synthetic_cohort_reused":True,
             "source_cohort_sha256":parent["cohort_sha256"],"GT_masks_decoded_again":0,
             "enrollment_query_overlap":False,"test_or_locked_read":False,"inference_started":False,
             "created_unix":time.time()}
    write_json(folder/"COHORT_RECEIPT.json",receipt);write_json(output_root()/"COHORT_RECEIPT.json",receipt)
    write_json(output_root()/"QUERY_SPLIT.json",{"synthetic_dev_images":650,"real_dev_unknown_GT":len(real),
               "synthetic_gate_images":0,"formal_gate_complete":False,
               "real_enrollment_excluded":cfg["real_dev"]["excluded_enrollment_frame_ids"]})
    return receipt


def legacy_command(cfg,argv):
    """Keep existing default CLI on its accepted d6 implementation/cache identity."""
    frozen=resolve("outputs/action4_full_zeroshot_eval_v001/launch_snapshot_v001/runner.py")
    if cfg.get("profile_manifest") or Path(__file__).resolve()==frozen:return None
    if sha(frozen)!="d6ea2b51b12687375cec7e91573edd84471edc02851cece7ab68effb566b06fc":
        raise ValueError("Legacy compatibility runner changed")
    return [sys.executable,"-B",str(frozen),*argv]


def inference_lock(cfg):
    snapshot=activate_snapshot(cfg)
    code=resolve(snapshot["code_root"])
    from model.action4_yoloe_adapter import PREDICT_ARGS, verify_source_tree
    if cfg["inference"]["predict_args"]!=PREDICT_ARGS:
        raise ValueError("Declared native arguments differ from actual frozen adapter")
    asset_cfg=read_json(code/cfg["inference"]["asset_config"])
    for entry in asset_cfg["assets"].values():checked_file(entry["path"],entry["sha256"])
    sources={entry["path"]:verify_source_tree(entry["path"],entry["commit"]) for entry in asset_cfg["sources"]}
    if cfg.get("profile_manifest"):
        profiles=declared_profiles(cfg,asset_cfg)
    else:
        receipt_path=cfg["inference"]["profile_receipt"];receipt=read_json(receipt_path)
        if receipt["status"]!="REGISTERED_6_PROFILES_WITH_SELF_CHECKS":raise ValueError("Unvalidated profile receipt")
        byname={p["name"]:p for p in receipt["profiles"]};profiles=[]
        for name in cfg["inference"]["visual_profiles"]:
            p=byname[name];checked_file(p["path"],p["sha256"])
            folder=resolve(p["path"]).parent
            checked_file(folder/"reference.png",p["reference_sha256"])
            checked_file(folder/"body_mask.npy",p["bodymask_sha256"])
            profiles.append({**p,"text":False,"profile_receipt_sha256":sha(receipt_path),
                             "official_visual_raster_sha256":sha(folder/"official_visual_raster.npy")})
        text=cfg["inference"]["text_profile"];checked_file(text["path"],text["sha256"]);profiles.append(dict(text))
    runtime=read_json(code/cfg["inference"]["runtime_lock"])
    lock={"schema":"action4.full_zeroshot.inference_lock.v1","asset_config":asset_cfg,
          "asset_config_sha256":sha(code/cfg["inference"]["asset_config"]),"asset_identity_sha256":canonical(asset_cfg),
          "source_identity_sha256":canonical(sources),"source_tree_identities":sources,
          "adapter_source_sha256":sha(code/cfg["inference"]["adapter"]),"snapshot":snapshot,
          "runtime_lock_path":cfg["inference"]["runtime_lock"],"runtime_identity_sha256":canonical(runtime),
          "profiles":profiles,"arguments":PREDICT_ARGS,
          **({"declared_profile_manifest":cfg["profile_manifest"]} if cfg.get("profile_manifest") else {}),
          "raw_mask_preservation":"binary packbits when exact0/1; float32 otherwise; no missing response fabrication"}
    lock["identity"]=canonical(lock)
    return lock


def prepare(cfg):
    if cfg.get("profile_manifest"):return prepare_declared_cohort(cfg)
    manifest=checked_file(cfg["dataset"]["manifest"],cfg["dataset"]["manifest_sha256"])
    rows=[json.loads(line) for line in resolve(cfg["dataset"]["manifest"]).read_text().splitlines() if line.strip()]
    if len(rows)!=650 or sum(len(r["instances"]) for r in rows)!=1300 or len({r["sample_id"] for r in rows})!=650:
        raise ValueError("Expected complete650 unique images and all1300 original GT")
    contract=cohort_contract(cfg)
    cohort_id=canonical(contract);folder=resolve(output_root()/"cohorts"/cohort_id,False)
    if not folder.is_relative_to(output_root()):raise ValueError("Cohort escaped owned output")
    receipt_path=folder/"COHORT_RECEIPT.json"
    if receipt_path.exists():
        saved=read_json(receipt_path)
        if sha(folder/"COHORT.jsonl")!=saved["cohort_sha256"]:raise ValueError("Prepared cohort changed")
        write_json(output_root()/"COHORT_RECEIPT.json",saved)
        return saved
    lock=inference_lock(cfg)
    split=prompt_split(rows,cfg)
    dev_ids=set(split["prompt_dev_ids"])
    prepared=[]
    with ThreadPoolExecutor(max_workers=cfg["inference"]["max_cpu_threads"]) as pool:
        for record in pool.map(lambda item:prepare_one(*item,cfg),enumerate(rows)):
            record["prompt_partition"]="prompt_dev" if record["sample_id"] in dev_ids else "prompt_gate"
            prepared.append(record)
            if len(prepared)%50==0:print("prepared",len(prepared),"/650",flush=True)
    folder.mkdir(parents=True,exist_ok=False)
    cohort=folder/"COHORT.jsonl"
    cohort.write_text("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in prepared))
    image_groups=Counter(g for row in prepared for g in row["groups"])
    instance_groups=Counter(g for row in prepared for t in row["targets"] for g in t["groups"])
    inventory={"schema":"action4.full_zeroshot.group_inventory.v1","images":len(prepared),"original_gt":1300,
               "gt_per_image":dict(Counter(str(r["original_gt_count"]) for r in prepared)),
               "image_groups":dict(image_groups),"instance_groups":dict(instance_groups),
               "empty_visible_gt_count":sum(not t["support_measurable"] for r in prepared for t in r["targets"]),
               "pure_negative":{"images":sum(r["original_gt_count"]==0 for r in prepared),"status":"NOT_MEASURED"},
               "other_camera":{"annotated_instances":0,"status":"NOT_MEASURED"},
               "definitions":cfg["groups"],"not_result_selected":True,"independence_status":"NOT_ESTABLISHED"}
    write_json(folder/"GROUPS.json",inventory);write_json(folder/"INFERENCE_LOCK.json",lock)
    write_json(folder/"PROMPT_SPLIT.json",split);write_json(output_root()/"PROMPT_SPLIT.json",split)
    receipt={"schema":"action4.full_zeroshot.cohort.v1","status":"FULL_COHORT_PREPARED_NO_INFERENCE",
             "cohort_id":cohort_id,"cohort_path":str(cohort.relative_to(ROOT)),"cohort_sha256":sha(cohort),
             "group_inventory_path":str((folder/"GROUPS.json").relative_to(ROOT)),
             "inference_lock_path":str((folder/"INFERENCE_LOCK.json").relative_to(ROOT)),
             "inference_identity":lock["identity"],"manifest":manifest,"contract":contract,
             "prompt_split":{"dev_images":split["dev_images"],"gate_images":split["gate_images"],"status":split["status"]},
             "images":650,"original_gt":1300,"input_rgb_and_visible_mask_bytes_verified":True,
             "test_or_locked_read":False,"inference_started":False,"created_unix":time.time()}
    write_json(receipt_path,receipt);write_json(output_root()/"COHORT_RECEIPT.json",receipt)
    write_json(output_root()/"GROUPS.json",inventory)
    return receipt


def load_cohort(cfg):
    receipt=read_json(output_root()/"COHORT_RECEIPT.json")
    if receipt["contract"]!=cohort_contract(cfg):
        raise ValueError("Cohort config/source changed; prepare a new identity")
    if sha(receipt["cohort_path"])!=receipt["cohort_sha256"] or sha(cfg["dataset"]["manifest"])!=cfg["dataset"]["manifest_sha256"]:
        raise ValueError("Cohort/manifest changed")
    rows=[json.loads(line) for line in resolve(receipt["cohort_path"]).read_text().splitlines()]
    labeled=[row for row in rows if row.get("domain","synthetic_dev")=="synthetic_dev"]
    if len(labeled)!=650 or sum(row["original_gt_count"] for row in labeled)!=1300:raise ValueError("Incomplete synthetic cohort")
    if cfg.get("profile_manifest"):
        real=[row for row in rows if row.get("domain")=="real_harddev_unannotated"]
        if [row["frame_id"] for row in real]!=real_query_ids(cfg) or any(row["targets"] is not None or row["original_gt_count"] is not None for row in real):
            raise ValueError("Real unknown-GT cohort/exclusion contract changed")
    elif len(rows)!=650:raise ValueError("Legacy cohort cannot contain other domains")
    lock=read_json(receipt["inference_lock_path"])
    if lock["identity"]!=canonical({k:v for k,v in lock.items() if k!="identity"}):raise ValueError("Inference lock changed")
    return receipt,rows,lock


def task_identity(row,profile,lock):
    return {"schema":NATIVE_SCHEMA,"sample_id":row["sample_id"],"image_sha256":row["image"]["sha256"],
            "native_shape":row["native_shape"],"profile_name":profile["name"],"profile_sha256":profile["sha256"],
            "reference_id":profile["reference_id"],"inference_identity":lock["identity"],
            "runner_source_sha256":sha(__file__)}


def cache_folder(identity):
    value=resolve(output_root()/"native"/canonical(identity),False)
    if not value.is_relative_to(output_root()):raise ValueError("Cache escaped owned output")
    return value


def save_raw(result,destination):
    import numpy as np
    from model.action4_yoloe_adapter import validate_prediction_arrays
    shape=list(result["native_shape"])
    validate_prediction_arrays(result["boxes"],result["scores"],result["classes"],result["masks"],shape)
    n=len(result["boxes"]);masks=result["masks"]
    arrays={"boxes":result["boxes"],"scores":result["scores"],"classes":result["classes"],
            "native_shape":np.asarray(shape),"bitorder":np.asarray("big")}
    if masks is None:
        arrays.update(mask_encoding=np.asarray("none"),mask_bits=np.empty((0,(math.prod(shape)+7)//8),np.uint8))
    elif np.logical_or(masks==0,masks==1).all():
        arrays.update(mask_encoding=np.asarray("binary_packbits"),
                      mask_bits=np.packbits(masks.astype(bool).reshape(n,math.prod(shape)),axis=1,bitorder="big"))
    else:
        arrays.update(mask_encoding=np.asarray("native_float32"),masks=np.asarray(masks,dtype=np.float32))
    np.savez_compressed(destination,**arrays)


def load_raw(path):
    import numpy as np
    from model.action4_yoloe_adapter import validate_prediction_arrays
    with np.load(resolve(path),allow_pickle=False) as data:
        shape=tuple(int(x) for x in data["native_shape"]);n=len(data["boxes"]);encoding=str(data["mask_encoding"])
        if encoding=="none":masks=None
        elif encoding=="binary_packbits":
            masks=np.unpackbits(data["mask_bits"],axis=1,count=math.prod(shape),bitorder="big").reshape(n,*shape).astype(bool)
        elif encoding=="native_float32":masks=data["masks"].copy()
        else:raise ValueError("Unknown native mask encoding")
        result={k:data[k].copy() for k in ("boxes","scores","classes")}
        result.update(native_shape=list(shape),masks=masks)
    validate_prediction_arrays(**result)
    return result


def completed_native(identity):
    folder=cache_folder(identity);pointer=folder/"COMPLETED.json"
    if not pointer.exists():return None
    record=read_json(pointer)
    if record["identity"]!=identity or record["status"]!="NATIVE_RESPONSE_COMPLETE":
        raise ValueError("Native completion identity mismatch")
    if sha(record["raw_path"])!=record["raw_sha256"]:raise ValueError("Native cache bytes changed")
    return record


class SampleInputFailure(RuntimeError):
    """A specific immutable image failed decode; native model was not called."""


def verify_bound_input(binding):
    # Reuse the content SHA verified at prepare only while ordinary filesystem
    # identity is unchanged; changed stat triggers a real content hash check.
    if fingerprint(binding["path"])!=binding["stat"]:
        checked_file(binding["path"],binding["sha256"])


def run_native_task(row,profile,lock,predictor,*,retry_failed=False):
    import numpy as np
    from PIL import Image
    verify_bound_input(row["image"]);checked_file(profile["path"],profile["sha256"])
    identity=task_identity(row,profile,lock);previous=completed_native(identity)
    if previous is not None:return previous,True
    folder=cache_folder(identity)
    if (folder/"FAILURE.json").exists() and not retry_failed:
        raise ValueError("Prior native failure requires explicit --retry-failed; no silent retries")
    folder.mkdir(parents=True,exist_ok=True)
    attempt_id=uuid.uuid4().hex;attempt=folder/("attempt_"+attempt_id);attempt.mkdir()
    write_json(attempt/"STARTED.json",{"identity":identity,"time":time.time(),"retry_failed":retry_failed})
    try:
        try:
            rgb=np.asarray(Image.open(resolve(row["image"]["path"])).convert("RGB"))
        except OSError as error:
            raise SampleInputFailure("Image decode failed: "+row["sample_id"]) from error
        start=time.monotonic()
        result=predictor.predict(rgb,profile_path=resolve(profile["path"]),reference_id=profile["reference_id"],text=profile.get("text",False))
        if result.get("inference_response_available")!=1:raise ValueError("Missing/incomplete native response")
        if result["profile_sha256"]!=profile["sha256"] or result["reference_id"]!=profile["reference_id"]:
            raise ValueError("Returned profile identity mismatch")
        if (canonical(result["source_identity"])!=lock["source_identity_sha256"] or
                canonical(result["runtime_identity"])!=lock["runtime_identity_sha256"] or
                canonical(result["asset_identity"])!=lock["asset_identity_sha256"] or
                result["arguments"]!=lock["arguments"] or result["native_shape"]!=row["native_shape"]):
            raise ValueError("Returned native model/source/runtime/input identity mismatch")
        raw=attempt/"native.npz";save_raw(result,raw)
        record={"status":"NATIVE_RESPONSE_COMPLETE","identity":identity,"attempt_id":attempt_id,
                "raw_path":str(raw.relative_to(ROOT)),"raw_sha256":sha(raw),
                "raw_count":len(result["scores"]),"budget_hit":len(result["scores"])==lock["arguments"]["max_det"],
                "target_evidence_available":result["target_evidence_available"],
                "negative_evidence_available":result["negative_evidence_available"],
                "native_status":result["status"],"class_names_native":result["class_names"],
                "retry_failed":retry_failed,"prior_failure_present":(folder/"FAILURE.json").exists(),
                "seconds":time.monotonic()-start,"completed_unix":time.time()}
        write_json(attempt/"COMPLETED.json",record);write_json(folder/"COMPLETED.json",record)
        return record,False
    except Exception as error:
        failure={"status":"FAILED_NOT_EMPTY_SUCCESS","identity":identity,"attempt_id":attempt_id,
                 "error":repr(error),"traceback":traceback.format_exc(),"time":time.time(),
                 "original_gt_count":row["original_gt_count"],
                 "failure_class":"SAMPLE_INPUT_DECODE" if isinstance(error,SampleInputFailure) else "FATAL_RESOURCE_IDENTITY_SCHEMA_OR_RUNTIME"}
        write_json(attempt/"FAILURE.json",failure);write_json(folder/"FAILURE.json",failure)
        raise


def shard_rows(rows,index,count):
    if type(index) is not int or type(count) is not int or count<1 or not 0<=index<count:
        raise ValueError("Invalid shard")
    return [r for r in rows if r["index"]%count==index]


def runtime_boot(cfg):
    snapshot=activate_snapshot(cfg)
    code=resolve(snapshot["code_root"])
    runtime=read_json(code/cfg["inference"]["runtime_lock"])
    if runtime.get("authority"):
        authority=runtime["authority"]
        active=read_json(authority["active_pointer"])["active_ref"]
        if active!={"path":authority["path"],"sha256":authority["sha256"]}:
            raise ValueError("Runtime authority no longer active; root review required")
        checked_file(authority["path"],authority["sha256"])
        admitted=read_json(authority["path"])
        if (admitted.get("GPU0_forbidden") is not True or
                admitted["zeroshot_eval_gpu1"]["yolo_max_workers"]<runtime["max_concurrent_yolo_workers"]):
            raise ValueError("Active authority does not admit the declared workers")
    mapping=runtime["gpu_mapping"]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=mapping["physical_uuid"] or mapping["physical_index"]==0:
        raise ValueError("Explicit reviewed non-GPU0 UUID required for run-shard")
    values=subprocess.check_output(["nvidia-smi","--id="+mapping["physical_uuid"],
           "--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if values[0].strip()!=mapping["physical_uuid"] or float(values[1])<runtime["required_minimum_free_before_start_gib"]*1024:
        raise ValueError("GPU resource admission failed; no inference started")
    for p in reversed(runtime["sys_path_additions"]):sys.path.insert(0,str(resolve(p)))
    sys.path.insert(0,str(code/"src"))
    import model.action4_yoloe_adapter as adapter
    def local_cache():
        for key,name in {"HF_HOME":"hf","TORCH_HOME":"torch","XDG_CACHE_HOME":"xdg","PIP_CACHE_DIR":"pip",
                         "TMPDIR":"tmp","CUDA_CACHE_PATH":"cuda","YOLO_CONFIG_DIR":"ultralytics"}.items():
            p=output_root()/"runtime_cache"/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
        os.environ["YOLO_AUTOINSTALL"]="False"
    # Only confine existing adapter cache paths; no model/prediction changes.
    adapter.local_environment=local_cache
    def deny(*args,**kwargs):raise RuntimeError("Network disabled for frozen local inference")
    socket.create_connection=deny;socket.socket.connect=deny
    import torch
    torch.set_num_threads(cfg["inference"]["max_cpu_threads"]);torch.set_num_interop_threads(1)
    return adapter.FrozenYOLOE(config=str(code/cfg["inference"]["asset_config"]),device="cuda:0",
                              runtime_lock=str(code/cfg["inference"]["runtime_lock"]))



def require_prior_native_release(cfg):
    if not cfg.get("profile_manifest"):return None
    gate=cfg["previous_native_release"];previous=read_json(gate["cohort_receipt"])
    launch=read_json(gate["launch"])
    if launch["runner_sha256"]!=previous["contract"]["preparer_source_sha256"]:
        raise ValueError("Previous native launch/cohort source identity mismatch")
    previous_rows=[json.loads(x) for x in resolve(previous["cohort_path"]).read_text().splitlines()]
    previous_lock=read_json(previous["inference_lock_path"])
    keys=set()
    for worker in launch["workers"]:
        index=worker["shard"];packet_path=resolve(gate["launch"]).parent/f"SHARD_{index:02d}.json"
        if not packet_path.exists():raise ValueError("Previous native tasks are not complete; no new GPU work")
        packet=read_json(packet_path)
        expected={(row["sample_id"],profile["name"]) for row in shard_rows(previous_rows,index,len(launch["workers"]))
                  for profile in previous_lock["profiles"]}
        actual=[(row["sample_id"],row["profile"]) for row in packet.get("completed_tasks",[])]
        if packet.get("status")!="COMPLETE_NATIVE_SHARD" or packet.get("failures") or set(actual)!=expected or len(actual)!=len(expected):
            raise ValueError("Previous native coverage/failures unresolved; no new GPU work")
        if packet.get("cohort_id")!=previous["cohort_id"]:raise ValueError("Previous native shard identity mismatch")
        process=subprocess.run(["ps","-p",str(worker["pid"]),"-o","args="],capture_output=True,text=True)
        if any(str(value).endswith("/runner.py") and str(value) in process.stdout for value in worker["command"]):
            raise ValueError("Previous native GPU process still present; wait for release")
        keys.update(expected)
    if len(keys)!=gate["expected_native_tasks"]:
        raise ValueError("Previous4550 native task budget mismatch")
    return {"status":"PREVIOUS_NATIVE_COMPLETE_AND_PROCESSES_EXITED","native_tasks":len(keys),
            "launch_sha256":sha(gate["launch"]),"cohort_sha256":previous["cohort_sha256"]}


def run_shard(cfg,index,retry_failed=False):
    receipt,rows,lock=load_cohort(cfg)
    prior_release=require_prior_native_release(cfg)
    if prior_release is not None:write_json(output_root()/"PRIOR_NATIVE_RELEASE.json",prior_release)
    if inference_lock(cfg)["identity"]!=lock["identity"]:raise ValueError("Native inference identity changed")
    selected=shard_rows(rows,index,cfg["inference"]["shard_count"])
    # No model starts unless the operator explicitly invokes run-shard.
    pending=[]
    for row in selected:
        verify_bound_input(row["image"])
        for profile in lock["profiles"]:
            checked_file(profile["path"],profile["sha256"])
            if completed_native(task_identity(row,profile,lock)) is None:pending.append((row["sample_id"],profile["name"]))
    predictor=runtime_boot(cfg) if pending else None
    completed=[];failures=[]
    for row in selected:
        for profile in lock["profiles"]:
            try:
                record,reused=run_native_task(row,profile,lock,predictor,retry_failed=retry_failed)
                completed.append({"sample_id":row["sample_id"],"profile":profile["name"],"reused":reused,
                                  "raw_count":record["raw_count"]})
            except SampleInputFailure as error:
                failures.append({"sample_id":row["sample_id"],"profile":profile["name"],
                                 "failure_class":"SAMPLE_INPUT_DECODE","error":repr(error)})
                break
            except Exception as error:
                fatal={"status":"FATAL_WORKER_STOPPED","cohort_id":receipt["cohort_id"],
                       "sample_id":row["sample_id"],"profile":profile["name"],"completed_tasks":completed,
                       "failures":failures,"error":repr(error),"traceback":traceback.format_exc(),
                       "remaining_tasks_not_attempted":True}
                write_json(output_root()/f"SHARD_{index:02d}.json",fatal)
                raise
        write_json(output_root()/f"SHARD_{index:02d}.json",
                   {"schema":"action4.full_zeroshot.shard.v1","cohort_id":receipt["cohort_id"],
                    "inference_identity":lock["identity"],"shard_index":index,
                    "assigned_images":len(selected),"completed_tasks":completed,"failures":failures,
                    "status":"FAILED_TASKS_PRESERVED" if failures else "IN_PROGRESS"})
        print(row["sample_id"],"native tasks",len(completed),"failures",len(failures),flush=True)
    status="COMPLETE_NATIVE_SHARD" if not failures and len(completed)==len(selected)*len(lock["profiles"]) else "INCOMPLETE_NATIVE_SHARD"
    result={"status":status,"cohort_id":receipt["cohort_id"],"shard_index":index,
            "assigned_images":len(selected),"completed_tasks":completed,"failures":failures}
    write_json(output_root()/f"SHARD_{index:02d}.json",result)
    return result


def publish(raw,profile,policy):
    target_classes=tuple(profile["target_class_ids"]) if "target_class_ids" in profile else ((0,1) if profile.get("text") else (0,))
    rows=[]
    for i,score in enumerate(raw["scores"]):
        if float(score)<policy["score_threshold"] or int(raw["classes"][i]) not in target_classes:continue
        box=raw["boxes"][i].astype(float).tolist()
        pad=policy["prompt_padding_fraction"]
        dx=(box[2]-box[0])*pad;dy=(box[3]-box[1])*pad
        rows.append({"id":profile["name"]+":"+str(i),"source":"yolo","class_id":0,
                     "native_class_id":int(raw["classes"][i]),"reference_id":profile["reference_id"],
                     "score":float(score),"bbox":box,
                     "prompt_bbox":[box[0]-dx,box[1]-dy,box[2]+dx,box[3]+dy],
                     "mask":None if raw["masks"] is None else raw["masks"][i]>=.5})
    return rows


def dedup(predictions,policy):
    import numpy as np
    kept=[]
    for p in sorted(predictions,key=lambda p:(-p["score"],p["id"])):
        match=None
        for q in kept:
            if p["mask"] is None or q["mask"] is None:continue
            a,b=p["bbox"],q["bbox"]
            distance=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)
            if distance>policy["center_distance_small_diag"]*min(math.hypot(a[2]-a[0],a[3]-a[1]),math.hypot(b[2]-b[0],b[3]-b[1])):continue
            union=np.count_nonzero(p["mask"]|q["mask"])
            iou=np.count_nonzero(p["mask"]&q["mask"])/union if union else 0.
            if iou>=policy["mask_iou_dedup"]:match=q;break
        if match is None:kept.append(dict(p,members=[p["id"]]))
        else:match["members"].append(p["id"])
    return kept


def association_diagnostics(predictions,targets,score):
    import numpy as np
    visible=[t for t in targets if t["mask"] is not None and t["mask"].any()]
    own={};merged=[];duplicates=[]
    for p in predictions:
        values={t["id"]:(float(np.count_nonzero(p["mask"]&t["mask"])/t["mask"].sum()) if p["mask"] is not None else None)
                for t in visible}
        own[p["id"]]=values
        ids=[tid for tid,value in values.items() if value is not None and value>=.5]
        if len(ids)>=2:merged.append({"prediction_id":p["id"],"gt_ids":ids})
    for t in visible:
        ids=[pid for pid,values in own.items() if values[t["id"]] is not None and values[t["id"]]>=.5]
        if len(ids)>=2:duplicates.append({"gt_id":t["id"],"prediction_ids":ids})
    matched={p for _,p in (score.get("joint_matches") or [])}
    unmatched=[p["id"] for p in predictions if p["id"] not in matched] if score.get("scoring_eligible") else None
    zero_support=[pid for pid,values in own.items() if values and all(v==0 for v in values.values())]
    return {"two_to_one_merge_candidates":merged,"one_to_two_duplicate_candidates":duplicates,
            "own_fraction_matrix":own,"unmatched_published_ids":unmatched,
            "zero_overlap_with_evaluable_visible_gt_ids":zero_support,
            "unmatched_semantics":"failure to match strict support/prompt; NOT automatically non-DJI",
            "model_specific_non_DJI_error_rate":"NOT_MEASURED_NO_OTHER_CAMERA_LABELS"}


def load_scorer(cfg):
    spec=cfg["scoring"]
    if spec.get("semantics_reviewed") is not True or not spec.get("source_sha256") or "PENDING" in spec["expected_schema"]:
        raise ValueError("Scorer root semantic/schema lock pending; no99 scoring/promotion")
    snapshot=activate_snapshot(cfg)
    source=resolve(snapshot["code_root"])/spec["source"]
    if sha(source)!=spec["source_sha256"]:raise ValueError("Scorer source identity changed")
    import importlib.util
    module_spec=importlib.util.spec_from_file_location("action4_full_dev_frozen_scorer",source)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    if resolve(module.__file__)!=source:raise ValueError("Wrong scorer import")
    declared=getattr(module,"JOINT_GATE_CONTRACT",JOINT_CONTRACT)
    if declared!=JOINT_CONTRACT:raise ValueError("Scorer gate semantic mismatch")
    return module


def score_one(row,policy,lock,cfg,scorer):
    import numpy as np
    from PIL import Image
    byname={p["name"]:p for p in lock["profiles"]}
    native=[];predictions=[]
    for name in policy["profiles"]:
        p=byname[name];identity=task_identity(row,p,lock);record=completed_native(identity)
        if record is None:raise ValueError("Native task missing; cannot fabricate empty output")
        native.append({"profile":name,"raw_sha256":record["raw_sha256"],"raw_count":record["raw_count"],
                       "budget_hit":record["budget_hit"],"native_status":record["native_status"]})
        raw=load_raw(record["raw_path"]);predictions.extend(publish(raw,p,policy))
    published=dedup(predictions,policy)
    if row.get("domain")=="real_harddev_unannotated":
        if row.get("targets") is not None or row.get("original_gt_count") is not None:
            raise ValueError("Unannotated real data must not fabricate an empty GT list")
        return {"schema":"action4.full_zeroshot.real_diagnostic.v1","sample_id":row["sample_id"],
                "frame_id":row["frame_id"],"domain":row["domain"],"original_gt_count":None,
                "annotation_status":"unknown","policy":policy,"native":native,"score":None,
                "published_predictions":[{k:v for k,v in p.items() if k!="mask"} for p in published],
                "published_pre_dedup_count":len(predictions),"published_count":len(published),
                "accuracy_status":"NOT_MEASURED_UNKNOWN_GT","formal_99_pass":False,
                "GT_merge_duplicate_FP_FN":"NOT_MEASURED_UNKNOWN_GT","reference_enrollment":False}
    # All native model responses exist before this function opens any GT mask.
    targets=[]
    for t in row["targets"]:
        checked_file(t["visible_mask"]["path"],t["visible_mask"]["sha256"])
        targets.append({"id":t["id"],"class_name":t["class_name"],"annotation_status":"valid",
                        "bbox_full_xyxy":t["bbox_full_xyxy"],
                        "mask":np.array(Image.open(resolve(t["visible_mask"]["path"])).convert("L"))>=128})
    score=scorer.score_image(published,targets,image_id=row["sample_id"],class_protocol=cfg["class_protocol"],
                             groups=row["groups"],source_group=row["source_group"])
    return {"schema":"action4.full_zeroshot.scored_image.v1","sample_id":row["sample_id"],
            "original_gt_count":row["original_gt_count"],"policy":policy,"native":native,
            "published_predictions":[{k:v for k,v in p.items() if k!="mask"} for p in published],
            "published_pre_dedup_count":len(predictions),"score":score,
            "associations":association_diagnostics(published,targets,score),
            "full_bbox_iou":"DIAGNOSTIC_ONLY","non_DJI_model_discrimination":"NOT_MEASURED"}


def scoring_identity(cfg,receipt,policy):
    return {"cohort_sha256":receipt["cohort_sha256"],"policy":policy,"scorer":cfg["scoring"],
            "class_protocol":cfg["class_protocol"],"runner_source_sha256":sha(__file__)}


def score_shard(cfg,index):
    scorer=load_scorer(cfg);receipt,rows,lock=load_cohort(cfg)
    if any(r["prompt_partition"]!="prompt_dev" for r in rows):raise ValueError("Gate scoring requires separately frozen candidate authorization; no gate exposure here")
    results=[]
    for policy in cfg["policies"]:
        identity=scoring_identity(cfg,receipt,policy);folder=output_root()/"scores"/canonical(identity)
        write_json(folder/"IDENTITY.json",identity)
        for row in shard_rows(rows,index,cfg["inference"]["shard_count"]):
            target=folder/(row["sample_id"]+".json")
            if target.exists():
                record=read_json(target)
                # Native identities/bytes are rechecked even on a scoring resume.
                previous={x["profile"]:x["raw_sha256"] for x in record["native"]}
                for name in policy["profiles"]:
                    current=completed_native(task_identity(row,next(p for p in lock["profiles"] if p["name"]==name),lock))
                    if current is None or previous.get(name)!=current["raw_sha256"]:
                        raise ValueError("Scored item lost/changed its native provenance")
                if record.get("scoring_identity")!=identity:raise ValueError("Scored identity mismatch")
                results.append({"sample_id":row["sample_id"],"policy":policy["id"],"reused":True});continue
            try:
                record=score_one(row,policy,lock,cfg,scorer);record["scoring_identity"]=identity
                write_json(target,record);results.append({"sample_id":row["sample_id"],"policy":policy["id"],"reused":False})
            except Exception as error:
                write_json(folder/(row["sample_id"]+".failure.json"),
                           {"status":"SCORING_FAILED_NOT_EMPTY_SUCCESS","scoring_identity":identity,
                            "sample_id":row["sample_id"],"original_gt_count":row["original_gt_count"],
                            "error":repr(error),"traceback":traceback.format_exc()})
                raise
    return {"status":"SCORING_SHARD_COMPLETE","shard_index":index,"records":results}


def aggregate(cfg):
    scorer=load_scorer(cfg);receipt,rows,lock=load_cohort(cfg);summaries={}
    if any(r["prompt_partition"]!="prompt_dev" for r in rows):raise ValueError("No full-score gate exposure permitted")
    for policy in cfg["policies"]:
        identity=scoring_identity(cfg,receipt,policy);folder=output_root()/"scores"/canonical(identity)
        records=[];missing=[];failed=[]
        for row in rows:
            target=folder/(row["sample_id"]+".json")
            if not target.exists():
                (failed if target.with_name(row["sample_id"]+".failure.json").exists() else missing).append(row["sample_id"]);continue
            record=read_json(target)
            if record.get("scoring_identity")!=identity or record["original_gt_count"]!=row["original_gt_count"]:
                raise ValueError("Mixed scored cohort/identity")
            records.append(record)
        if missing or failed:
            summaries[policy["id"]]={"status":"INCOMPLETE_NOT_SCORED_AS_EMPTY_SUCCESS","expected_images":len(rows),
                "original_gt":1300,"completed_images":len(records),"missing_ids":missing,"failed_ids":failed,
                "formal_99_pass":False}
            continue
        real=[record for record in records if record["schema"]=="action4.full_zeroshot.real_diagnostic.v1"]
        measured=[record for record in records if record["schema"]=="action4.full_zeroshot.scored_image.v1"]
        if len(measured)!=650:raise ValueError("Synthetic scoring coverage changed")
        if cfg.get("profile_manifest") and (len(real)!=cfg["real_dev"]["expected_images"] or any(record["score"] is not None for record in real)):
            raise ValueError("Real diagnostic coverage/unknown GT semantics changed")
        report=scorer.summarize([r["score"] for r in measured],required_groups=tuple(cfg["scoring"]["required_groups"]))
        if report.get("schema")!=cfg["scoring"]["expected_schema"]:raise ValueError("Actual scorer schema differs from lock")
        if sum(r["original_gt_count"] for r in measured)!=1300:raise ValueError("GT denominator lost")
        report.update(full_development_images=650,original_gt=1300,formal_99_pass=False,
                      evaluation_role="FULL_DEVELOPMENT_EXPOSURE_NOT_INDEPENDENT_GATE",
                      model_specificity="UNVERIFIED_NO_OTHER_CAMERA_LABELS",pure_negative="NOT_MEASURED",
                      missing_or_failed_images=0,
                      merge_image_count=sum(bool(r["associations"]["two_to_one_merge_candidates"]) for r in measured),
                      duplicate_image_count=sum(bool(r["associations"]["one_to_two_duplicate_candidates"]) for r in measured))
        summaries[policy["id"]]={"status":"COMPLETE_FULL650_DEVELOPMENT","scoring_identity":identity,"metrics":report}
        if real:
            summaries[policy["id"]].update(status="COMPLETE_650_SCORED_PLUS_REAL56_DIAGNOSTICS",
                real_diagnostics={"images":len(real),"annotation_status":"unknown","gt_count":None,
                    "precision":None,"recall":None,"TP":None,"FP":None,"FN":None,"strict_success_rate":None,
                    "published_predictions":sum(r["published_count"] for r in real),
                    "images_with_zero_published":sum(r["published_count"]==0 for r in real),
                    "enrollment_excluded":cfg["real_dev"]["excluded_enrollment_frame_ids"],
                    "scope":"same-video raw/publication diagnostics, not accuracy or negative-image scoring"})
    write_json(output_root()/"FULL_DEV_SUMMARY.json",summaries)
    return summaries


def status(cfg):
    receipt,rows,lock=load_cohort(cfg);complete=0;failed=0;pending=0;historical_failures=0
    for row in rows:
        for profile in lock["profiles"]:
            identity=task_identity(row,profile,lock)
            if (cache_folder(identity)/"FAILURE.json").exists():historical_failures+=1
            if completed_native(identity) is not None:complete+=1
            elif (cache_folder(identity)/"FAILURE.json").exists():failed+=1
            else:pending+=1
    return {"status":"COMPLETE_NATIVE" if pending+failed==0 else "NATIVE_NOT_COMPLETE",
            "images":len(rows),"synthetic_images":650,"original_gt":1300,
            "unknown_gt_images":len(rows)-650,"profiles":len(lock["profiles"]),
            "expected_native_tasks":len(rows)*len(lock["profiles"]),"completed":complete,"failed":failed,"pending":pending,
            "historical_failure_tasks":historical_failures}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","status","run-shard","score-shard","aggregate"))
    parser.add_argument("--config",default=DEFAULT_CONFIG)
    parser.add_argument("--shard-index",type=int)
    parser.add_argument("--retry-failed",action="store_true")
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise ValueError("Expected verified project root")
    cfg=load_config(args.config)
    command=legacy_command(cfg,sys.argv[1:])
    if command is not None:os.execv(sys.executable,command)
    sys.path.insert(0,str(resolve("src")))
    if args.mode=="prepare":result=prepare(cfg)
    elif args.mode=="status":result=status(cfg)
    elif args.mode=="aggregate":result=aggregate(cfg)
    else:
        if args.shard_index is None:parser.error("--shard-index required")
        result=run_shard(cfg,args.shard_index,args.retry_failed) if args.mode=="run-shard" else score_shard(cfg,args.shard_index)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
