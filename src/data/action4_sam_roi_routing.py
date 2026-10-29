"""Integration-only 50/30/20 training crop routing over frozen bbox ownership."""
import copy,hashlib,json
import numpy as np
from .action4_training_crop_specs import TrainingCropSpec
from .action4_sam_native_export import _v2_checked,_v2_sha,canonical
from .action4_sam_training_associations import _box
from .action4_dataset import is_ignored

POLICY={"schema":"action4.sam_roi_routing.v1","admission":"integration_only_not_calibrated",
    "route_probabilities":{"gt_jitter":0.5,"prediction":0.3,"extra_offset":0.2},
    "prediction_requested_full_probability":0.5,"coarse_policy_ready":False,
    "available_prediction_source":"completed_B0_full_only",
    "gt_jitter":{"center_fraction":0.15,"scale_range":[0.65,1.35]},
    "extra_offset":{"center_fraction":0.35,"scale_range":[0.65,1.0]},
    "offset_units":"fraction_of_GT_full_box_width_height","RGB_expansion":2.25,
    "candidate_ranking":["predicted_score_desc","stable_candidate_id_asc"],
    "fallback":["requested_prediction_route","other_prediction_route","gt_jitter"],
    "SAM_input":"unchanged_fixed_supervision_association","rng":"PCG64_seed_epoch_image_GTinstance_v1"}

def validate_policy(policy):
    if not isinstance(policy,dict) or canonical(policy)!=canonical(POLICY):
        raise ValueError("Unknown/unadmitted ROI routing policy")
    return json.loads(json.dumps(policy))

def validate_config(cfg,split):
    if cfg.get("sam_roi_routing") is None:return
    variant=cfg.get("a0_variant");corner=cfg.get("corner_enabled",False)
    if (cfg.get("role")!="A0" or cfg.get("sam_enabled") is not True
            or variant not in ("a0_sam_image","a0_corner_image") or type(corner) is not bool
            or corner!=(variant=="a0_corner_image") or cfg.get("mode")!="integration" or split!="train"):
        raise ValueError("ROI routing requires explicit matching SAM/corner integration train variant; no formal/eval/baseline admission")
    if cfg.get("roi_jitter")!=POLICY["gt_jitter"]:
        raise ValueError("Routing does not overwrite existing roi_jitter; exact registered settings required")

def decision(seed,epoch,image_id,instance_id,policy=POLICY):
    validate_policy(policy)
    if type(seed) is not int or seed<0 or type(epoch) is not int or epoch<0:
        raise ValueError("Explicit nonnegative routing seed/epoch required")
    if any(not isinstance(x,str) or not x for x in (image_id,instance_id)):raise ValueError("Routing identity required")
    ident=canonical({"seed":seed,"epoch":epoch,"image_id":image_id,"instance_id":instance_id,"policy":policy})
    resolved=int(ident[:16],16)
    rng=np.random.Generator(np.random.PCG64(resolved));p=rng.random()
    route="gt_jitter" if p<.5 else ("prediction_full" if rng.random()<.5 else "prediction_coarse") if p<.8 else "extra_offset"
    return {"base_seed":seed,"epoch":epoch,"seed_identity":ident,"augmentation_seed":resolved,"requested_route":route}

class SamRoiRouter:
    """Provider supplies the already verified complete association-v2 document."""
    def __init__(self,policy_binding,association,images):
        self.policy_binding=dict(policy_binding);self.policy=validate_policy(_v2_checked(policy_binding))
        mode=(association.get("schema"),association.get("expected_sources",{}).get("upstream_mode"))
        if (mode not in (("action4.sam_training_associations.v2","b0_completed_native_receipt_v2"),
                         ("action4.sam_training_associations.rgb_variant.v1","b0_completed_native_rgb_variant_receipt_v1"))
                or association.get("predictions")!=association.get("source_contract",{}).get("predictions")):
            raise ValueError("ROI routing requires verified completed-B0 original or variant association")
        variant=mode[0]=="action4.sam_training_associations.rgb_variant.v1"
        self.prediction_binding=dict(association["predictions"]);self.source_contract=association["source_contract"]
        rows=_v2_checked(self.prediction_binding);self.predictions={}
        for row in rows:
            sid=row["sample_id"]
            if sid in self.predictions or sid not in images:raise ValueError("Routing prediction image mismatch")
            candidates={}
            for raw in row["instances"]:
                grid=raw["grid_index"]
                if variant:
                    variant_id=images[sid]["variant_id"]
                    if row.get("variant_id")!=variant_id:raise ValueError("Routing RGB variant differs from native cache")
                    pid=f"b0:{sid}:{variant_id}:grid:{grid}"
                else:pid=f"b0:{sid}:grid:{grid}"
                score=raw["score"];box=_box(raw["bbox_full_xyxy"])
                if (type(grid) is not int or grid<0 or pid in candidates or box is None
                        or isinstance(score,bool) or not isinstance(score,(int,float)) or not np.isfinite(score)):
                    raise ValueError("Routing candidate duplicate/invalid grid, score or bbox")
                if raw.get("route","full") not in ("full","prediction_full"):
                    raise ValueError("Current frozen source is full-only; no fabricated coarse route")
                candidates[pid]={"prediction_id":pid,"grid_index":grid,"score":float(score),"bbox_full_xyxy":list(box),"route":"prediction_full"}
            if set(candidates)!={p["prediction_id"] for p in images[sid]["predictions"]}:raise ValueError("Routing exact cache candidate membership mismatch")
            self.predictions[sid]=candidates
        if set(self.predictions)!=set(images):raise ValueError("Routing prediction image coverage mismatch")
        self.associations={}
        for row in association["rows"]:
            key=(row["image_id"],str(row["gt_instance_id"]))
            if key in self.associations or key[0] not in self.predictions:raise ValueError("Routing association image/unit mismatch")
            eligible=row["qualified_prediction_ids"]
            if len(eligible)!=len(set(eligible)) or any(pid not in self.predictions[key[0]] or self.predictions[key[0]][pid]["score"]<.3 for pid in eligible):
                raise ValueError("Routing qualified candidate duplicate/wrong image/ID/score")
            if row["status"]!="matched" and eligible:raise ValueError("Unmatched ownership cannot supply route candidates")
            self.associations[key]=row

    def choose(self,item,*,seed,epoch):
        if _v2_sha(self.prediction_binding["path"])!=self.prediction_binding["sha256"]:
            raise ValueError("Frozen routing predictions changed")
        if _v2_sha(self.policy_binding["path"])!=self.policy_binding["sha256"]:raise ValueError("Frozen routing policy changed")
        record=item["record"];obj=record["instances"][item["instance"]]
        sid=record["sample_id"];gid=str(obj["instance_id"])
        if record.get("split")!="train" or is_ignored(obj):raise ValueError("Routing only nonignore train GT units")
        association=self.associations.get((sid,gid))
        if association is None:raise ValueError("Production routing GT unit missing; no implicit fixture fallback")
        gt=_box(obj["bbox_full_xyxy"])
        if gt is None:raise ValueError("Finite positive GT complete box required for training crop")
        info=decision(seed,epoch,sid,gid,self.policy);requested=info["requested_route"]
        pool=[self.predictions[sid][pid] for pid in association["qualified_prediction_ids"]]
        pool.sort(key=lambda c:(-c["score"],c["prediction_id"]))
        effective=requested;chosen=None;fallback=None
        if requested.startswith("prediction_"):
            chosen=next((p for p in pool if p["route"]==requested),None)
            if chosen is None:
                other="prediction_coarse" if requested=="prediction_full" else "prediction_full"
                chosen=next((p for p in pool if p["route"]==other),None)
                if chosen is not None:effective=other;fallback="requested_route_unavailable"
                else:effective="gt_jitter";fallback="predicted_roi_missing"
        params=None
        if chosen is not None:box=tuple(chosen["bbox_full_xyxy"])
        else:
            settings=self.policy["extra_offset"] if effective=="extra_offset" else self.policy["gt_jitter"]
            # Independent crop substream preserves route-first selection and ordering.
            rng=np.random.Generator(np.random.PCG64(int(info["seed_identity"][16:48],16)))
            fraction=rng.uniform(-settings["center_fraction"],settings["center_fraction"],2)
            scale=float(rng.uniform(*settings["scale_range"]))
            original=np.asarray(gt);wh=original[2:]-original[:2];centre=(original[:2]+original[2:])/2+fraction*wh
            size=wh*scale;box=tuple(map(float,np.r_[centre-size/2,centre+size/2]))
            params={"center_fraction_xy":fraction.tolist(),"scale":scale}
        spec=TrainingCropSpec(sid,gid,box,requested,effective,info["augmentation_seed"],self.policy_binding["sha256"],
            fallback or "requested_route_available",None if chosen is None else chosen["prediction_id"])
        trace=dict(info,effective_route=effective,fallback_reason=fallback,policy_binding=self.policy_binding,
            policy=self.policy,crop_parameters=params,association_status=association["status"],
            fixed_SAM_prediction_id=association["prediction_id"],selected_crop_prediction=chosen,
            prediction_source=self.prediction_binding,source_contract=self.source_contract,
            coarse_policy_ready=False,eligible_routes=sorted({p["route"] for p in pool}),
            GT_used_for_native_prompt=False,active=True)
        return spec,copy.deepcopy(trace)

ROUTE_COUNT_ROUTES=("gt_jitter","prediction_full","prediction_coarse","extra_offset")
ROUTE_COUNT_FALLBACKS=("requested_route_unavailable","predicted_roi_missing")
ROUTE_COUNT_FIELDS=("active_units",)+tuple("requested."+r for r in ROUTE_COUNT_ROUTES)+tuple("effective."+r for r in ROUTE_COUNT_ROUTES)+tuple("fallbacks."+r for r in ROUTE_COUNT_FALLBACKS)

def summarize_routes(traces):
    """Fixed 11-count domain for caller-owned all-rank reduction/persistence."""
    result={"active_units":0,"requested":dict.fromkeys(ROUTE_COUNT_ROUTES,0),
            "effective":dict.fromkeys(ROUTE_COUNT_ROUTES,0),"fallbacks":dict.fromkeys(ROUTE_COUNT_FALLBACKS,0)}
    for trace in traces:
        if type(trace.get("active")) is not bool:raise ValueError("Explicit active flag required for route counts")
        if not trace["active"]:continue
        if trace["requested_route"] not in ROUTE_COUNT_ROUTES or trace["effective_route"] not in ROUTE_COUNT_ROUTES or trace["fallback_reason"] not in (None,)+ROUTE_COUNT_FALLBACKS:
            raise ValueError("Unknown route counter domain")
        result["active_units"]+=1
        for dest,key in (("requested","requested_route"),("effective","effective_route")):
            name=trace[key];result[dest][name]+=1
        reason=trace["fallback_reason"]
        if reason is not None:result["fallbacks"][reason]+=1
    return result
