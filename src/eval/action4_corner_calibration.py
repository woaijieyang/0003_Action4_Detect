"""Pure whole-checkpoint calibration AFTER prediction freeze; never inference."""
import copy,hashlib,json,math

POLICY={"schema":"action4.corner_calibration_policy.v1","eval_images":650,
    "selected_q_min":0.6,"usable_p_valid_min":0.8,"minimum_valid_selected_points":100,
    "precision_numerator":9,"precision_denominator":10,"coverage_numerator":1,"coverage_denominator":10,
    "minimum_usable_ids":6,"error_fraction_gt_diagonal224":0.01,
    "score_schema":"action4.pose_projection_all8_002.v1",
    "selection":"whole_checkpoint_only","effect_admission":False}
RUN_KEYS=("checkpoint_sha256","config_sha256","decoder_sha256","data_sha256","source_sha256",
          "b0_model_sha256","sam_model_sha256","sam_cache_sha256","predictions_sha256")
RECORD_KEYS={"image_id","gt_instance_id","prediction_id","run_identity","corner_ids","q_corner","p_valid",
             "uv224","multimodal","label_known","gt_valid","error224","gt_diagonal224"}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or value=="0"*64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Explicit nonzero SHA256 binding required")

def _id(value):
    if type(value) not in (str,int) or not str(value).strip():raise ValueError("Explicit scalar GT identity required")
    return str(value)

def _key(row):return (row["image_id"],_id(row["gt_instance_id"]))
def _record_sort(row):
    return (row["image_id"],row["gt_instance_id"] is None,str(row["gt_instance_id"]),str(row["prediction_id"]))
def _groups(row):
    value=row.get("groups",[row["group"]] if "group" in row else [])
    if not isinstance(value,list) or not value or any(not isinstance(x,str) or not x for x in value) or len(value)!=len(set(value)):
        raise ValueError("Explicit unique nonempty group labels required")
    return sorted(value)

def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def _base_stats():
    return {"eligible_gt_instances":0,"quality_ineligible_gt":0,"pose_ignored_gt":0,"unmatched_prediction_instances":0,"missing_prediction_instances":0,"missing_record_instances":0,
        "selected_known_points":0,"selected_valid_points":0,"selected_correct_points":0,"selected_unknown_points":0,
        "unmeasured_selected_valid_points":0,"unknown_label_points":0,"instances_with_6_usable_ids":0}

def _score(report,run,units,images,predictions,policy):
    required={"schema","run_identity","data_sha256","score_schema","complete","pose_gt_count","pose_all8_at_002_success","outcomes","image_ids","pose_report"}
    if not isinstance(report,dict) or not required<=set(report) or report["schema"]!="action4.corner_calibration_pose_outcomes.v1":
        raise ValueError("Explicit full-domain original pose score wrapper required")
    if (report["run_identity"]!=run["identity"] or report["data_sha256"]!=run["data_sha256"]
            or report["score_schema"]!=policy["score_schema"] or type(report["complete"]) is not bool):
        raise ValueError("Pose report run/data/score schema mismatch")
    if len(report["image_ids"])!=len(set(report["image_ids"])) or not set(report["image_ids"])<=set(images):
        raise ValueError("Duplicate/foreign pose report image domain")
    pose_units={k:v for k,v in units.items() if v["pose_nonignore"]}
    if type(report["pose_gt_count"]) is not int or report["pose_gt_count"]!=len(pose_units):
        raise ValueError("Original pose denominator must include all nonignore GT, not quality eligible subset")
    outcomes={}
    for row in report["outcomes"]:
        if set(row)!={"image_id","gt_instance_id","success"}:raise ValueError("Only frozen GT outcomes accepted")
        key=_key(row)
        if key not in pose_units or key in outcomes or row["success"] is not None and type(row["success"]) is not bool:
            raise ValueError("Duplicate/foreign/invalid score outcome")
        outcomes[key]=row["success"]
    successes=sum(v is True for v in outcomes.values())
    if type(report["pose_all8_at_002_success"]) is not int or report["pose_all8_at_002_success"]!=successes:
        raise ValueError("Declared successes differ from frozen outcomes")
    unmeasured=sum(outcomes.get(k) is None for k in pose_units)
    full=report["pose_report"];derivable_complete=True
    def ratio(a,b):return a/b if b else None
    def same(value,expected):
        return value is None if expected is None else _finite(value) and abs(value-expected)<=1e-12
    if full is None:
        if report["complete"]:raise ValueError("Complete wrapper cannot omit original full pose report")
    else:
        fields={"score_schema","pose_gt_count","pose_output_count","pose_all8_at_002_success","pose_all8_at_002_rate",
            "pose_all8_at_005_rate","pose_all8_at_002_precision","pose_all8_at_002_f1","pose_fp","pose_fn",
            "invalid_pose_count","pose_instance_precision","pose_corner_p95_valid_matched_only",
            "strict_image_success_rate","per_gt_failure_reason","unmatched_pose_predictions","pose_groups"}
        if not isinstance(full,dict) or not fields<=set(full) or full["score_schema"]!=policy["score_schema"]:
            raise ValueError("Original action4 pose protocol full report required")
        n=len(pose_units);outputs=len(predictions)
        for key,expected in (("pose_gt_count",n),("pose_output_count",outputs),("pose_all8_at_002_success",successes),
                            ("pose_fp",outputs-successes),("pose_fn",n-successes)):
            if type(full[key]) is not int or full[key]!=expected:raise ValueError("Original pose report count mismatch: "+key)
        for key,expected in (("pose_all8_at_002_rate",ratio(successes,n)),("pose_all8_at_002_precision",ratio(successes,outputs)),
                            ("pose_all8_at_002_f1",ratio(2*successes,n+outputs))):
            if not same(full[key],expected):raise ValueError("Original pose report metric mismatch: "+key)
        source_rows={};matched=set();errors=[];success005=0
        expected_groups={};success_by_image={i:0 for i in images}
        gt_by_image={i:0 for i in images};pred_by_image={i:0 for i in images}
        for image,_ in predictions:pred_by_image[image]+=1
        for image,_ in pose_units:gt_by_image[image]+=1
        for key in ("pose_all8_at_005_rate","strict_image_success_rate"):
            value=full[key]
            if value is not None and (not _finite(value) or not 0<=value<=1):raise ValueError("Pose rate outside [0,1]: "+key)
        p95=full["pose_corner_p95_valid_matched_only"]
        if p95 is not None and (not _finite(p95) or p95<0):raise ValueError("Invalid original pose P95")
        if type(full["invalid_pose_count"]) is not int or not 0<=full["invalid_pose_count"]<=outputs:
            raise ValueError("Invalid original invalid-pose count")
        for row in full["per_gt_failure_reason"]:
            key=(row["sample_id"],_id(row["instance_id"]))
            if key not in pose_units or key in source_rows or type(row["all8_at_002"]) is not bool:
                raise ValueError("Original pose GT source domain mismatch")
            source_rows[key]=row["all8_at_002"]
            if type(row.get("all8_at_005")) is not bool or row["all8_at_002"] and not row["all8_at_005"]:
                raise ValueError("Original per-GT .002/.005 successes inconsistent")
            success005+=row["all8_at_005"];success_by_image[key[0]]+=row["all8_at_002"]
            for name in pose_units[key]["pose_groups"]:
                group=expected_groups.setdefault(name,{"gt":0,"tp002":0})
                group["gt"]+=1;group["tp002"]+=row["all8_at_002"]
            if outcomes.get(key) is not None and outcomes[key]!=row["all8_at_002"]:raise ValueError("Wrapper outcome differs from original pose scorer")
            pid=row["prediction_id"]
            if pid is not None:
                pair=(key[0],pid)
                if pair not in predictions or pair in matched:raise ValueError("Original pose prediction matching inconsistent")
                matched.add(pair)
                values=row.get("errors_px")
                if values is None:derivable_complete=False
                elif not isinstance(values,list) or len(values)!=8 or any(not _finite(v) or v<0 for v in values):
                    raise ValueError("Matched pose requires8 finite nonnegative pixel errors")
                else:errors.extend(values)
            elif row["all8_at_002"] or row["all8_at_005"] or row.get("errors_px") is not None:
                raise ValueError("Unmatched GT cannot have successes or matched errors")
        if set(source_rows)!=set(pose_units) or sum(source_rows.values())!=successes:
            raise ValueError("Original pose full GT coverage/success mismatch")
        unmatched=[(r["sample_id"],r["prediction_id"]) for r in full["unmatched_pose_predictions"]]
        if len(unmatched)!=len(set(unmatched)) or set(unmatched)!=predictions-matched:
            raise ValueError("Original pose report omits/duplicates published unmatched predictions")
        if not same(full["pose_instance_precision"],ratio(len(matched),outputs)):raise ValueError("Original instance precision inconsistent")
        invalid=0
        for row in full["unmatched_pose_predictions"]:
            if row.get("reason") not in ("invalid_pose","unmatched_pose"):raise ValueError("Unknown original unmatched-pose reason")
            invalid+=row["reason"]=="invalid_pose"
        if full["invalid_pose_count"]!=invalid:raise ValueError("Original invalid-pose count differs from unmatched reasons")
        if not same(full["pose_all8_at_005_rate"],ratio(success005,n)):raise ValueError("Original .005 rate differs from per-GT outcomes")
        strict=sum(success_by_image[i]==gt_by_image[i] and pred_by_image[i]==gt_by_image[i] for i in images)
        if not same(full["strict_image_success_rate"],ratio(strict,len(images))):raise ValueError("Original strict-image success inconsistent")
        if derivable_complete:
            # Same linear percentile definition used by numpy.percentile in original scorer.
            ordered=sorted(errors)
            if ordered:
                location=.95*(len(ordered)-1);lo=math.floor(location);hi=math.ceil(location)
                expected_p95=ordered[lo]+(ordered[hi]-ordered[lo])*(location-lo)
            else:expected_p95=None
            if not same(p95,expected_p95):raise ValueError("Original P95 differs from matched error population")
        if not isinstance(full["pose_groups"],dict) or set(full["pose_groups"])!=set(expected_groups):
            raise ValueError("Original pose group domain differs from registered memberships")
        for name,expected in expected_groups.items():
            group=full["pose_groups"][name]
            if (not isinstance(group,dict) or not {"gt","tp002","recall"}<=set(group)
                    or type(group["gt"]) is not int or type(group["tp002"]) is not int
                    or group["gt"]!=expected["gt"] or group["tp002"]!=expected["tp002"]
                    or not same(group["recall"],ratio(expected["tp002"],expected["gt"]))):
                raise ValueError("Original pose group statistics inconsistent: "+name)
    groups={}
    for key,row in pose_units.items():
        for name in _groups(row):
            group=groups.setdefault(name,{"denominator":0,"successes":0,"unmeasured":0})
            group["denominator"]+=1;group["successes"]+=outcomes.get(key) is True;group["unmeasured"]+=outcomes.get(key) is None
    return {"denominator":len(pose_units),"successes":successes,
        "all8_at_2_percent":ratio(successes,len(pose_units)) if unmeasured==0 and full is not None else None,
        "unmeasured":unmeasured,"complete":report["complete"] and set(report["image_ids"])==set(images)
            and set(outcomes)==set(pose_units) and unmeasured==0 and full is not None and derivable_complete,
        "groups":groups,"primary_score_schema":policy["score_schema"],"derived_statistics_measured":derivable_complete and full is not None}


def calibrate(domain,run_binding,records,raw_report,candidate_report,*,policy=POLICY):
    if policy!=POLICY:raise ValueError("Registered whole-checkpoint calibration policy required")
    run=copy.deepcopy(run_binding)
    if set(run)!=set(RUN_KEYS)|{"identity"}:raise ValueError("Complete checkpoint/config/decoder/data/source/B0/SAM/prediction binding required")
    for key in RUN_KEYS:_sha(run[key])
    if run["identity"]!=digest({k:v for k,v in run.items() if k!="identity"}):raise ValueError("Run identity mismatch")
    required={"schema","split","image_ids","gt_units","prediction_units","predictions_sha256","data_sha256","domain_sha256"}
    if not isinstance(domain,dict) or set(domain)!=required or domain["schema"]!="action4.corner_calibration_domain.v1" or domain["split"]!="eval":
        raise ValueError("Explicit frozen eval domain required")
    if domain["data_sha256"]!=run["data_sha256"] or domain["predictions_sha256"]!=run["predictions_sha256"]:
        raise ValueError("Calibration data/prediction identity mismatch")
    if domain["domain_sha256"]!=digest({k:v for k,v in domain.items() if k!="domain_sha256"}):raise ValueError("Domain manifest identity mismatch")
    images=domain["image_ids"]
    if not isinstance(images,list) or len(images)!=len(set(images)) or any(not isinstance(x,str) or not x for x in images):raise ValueError("Unique image domain required")
    units={}
    for row in domain["gt_units"]:
        if set(row) not in ({"image_id","gt_instance_id","eligible","pose_nonignore","pose_groups","group"},{"image_id","gt_instance_id","eligible","pose_nonignore","pose_groups","groups"}) or row["image_id"] not in images or type(row["eligible"]) is not bool or type(row["pose_nonignore"]) is not bool:
            raise ValueError("Explicit GT unit eligibility/group required")
        _groups(row)
        pg=row["pose_groups"]
        if not isinstance(pg,list) or len(pg)!=len(set(pg)) or any(not isinstance(v,str) or not v for v in pg) or (row["pose_nonignore"] and "all" not in pg):
            raise ValueError("Explicit original pose-group membership required")
        if row["eligible"] and not row["pose_nonignore"]:raise ValueError("Quality eligibility must be a subset of nonignore pose GT")
        key=_key(row)
        if key in units:raise ValueError("Duplicate GT domain unit")
        units[key]=row
    expected_predictions=set()
    for row in domain["prediction_units"]:
        if set(row)!={"image_id","prediction_id"} or row["image_id"] not in images or not isinstance(row["prediction_id"],str) or not row["prediction_id"]:
            raise ValueError("Explicit published prediction domain required")
        key=(row["image_id"],row["prediction_id"])
        if key in expected_predictions:raise ValueError("Duplicate published prediction")
        expected_predictions.add(key)
    scored={};used=set();unmatched=[]
    for row in records:
        key=None if row["gt_instance_id"] is None else _key(row)
        if row["image_id"] not in images or (key is not None and (key not in units or key in scored)) or row.get("run_identity")!=run["identity"]:
            raise ValueError("Mixed run/foreign/duplicate calibration unit")
        pid=row.get("prediction_id")
        if pid is None:
            if key is None:raise ValueError("Unmatched record must name a frozen prediction")
            if set(row)!={"image_id","gt_instance_id","prediction_id","run_identity"}:raise ValueError("Missing prediction record must be explicit and minimal")
        else:
            if not isinstance(pid,str) or not pid or (row["image_id"],pid) in used or set(row)!=RECORD_KEYS:raise ValueError("Duplicate prediction ownership or unknown point record fields")
            pair=(row["image_id"],pid)
            if pair not in expected_predictions:raise ValueError("Prediction outside frozen domain")
            used.add(pair)
            if key is None and any(row["gt_valid"]):raise ValueError("Unmatched prediction cannot be labeled GT-valid")
            if row["corner_ids"]!=list(range(8)) or any(type(x) is not int for x in row["corner_ids"]):raise ValueError("Fixed unique corner IDs required")
            for field in ("q_corner","p_valid","uv224","multimodal","label_known","gt_valid","error224"):
                if not isinstance(row[field],(list,tuple)) or len(row[field])!=8:raise ValueError("Exactly8 corner fields required")
            for i in range(8):
                if (not _finite(row["q_corner"][i]) or not 0<=row["q_corner"][i]<=1 or
                        not _finite(row["p_valid"][i]) or not 0<=row["p_valid"][i]<=1 or
                        len(row["uv224"][i])!=2 or not all(_finite(v) for v in row["uv224"][i]) or
                        any(type(row[f][i]) is not bool for f in ("multimodal","label_known","gt_valid"))):
                    raise ValueError("Invalid frozen prediction/label fields")
                error=row["error224"][i]
                if error is not None and (not _finite(error) or error<0):raise ValueError("Invalid scoring error")
        if key is None:unmatched.append(row)
        else:scored[key]=row
    stats=_base_stats();groups={};reasons=[]
    population=[(key,unit,scored.get(key)) for key,unit in sorted(units.items())]
    population += [(None,{"eligible":True,"groups":["__unmatched_predictions__"]},row) for row in sorted(unmatched,key=_record_sort)]
    for key,unit,row in population:
        local_groups=[groups.setdefault(name,_base_stats()) for name in _groups(unit)]
        def count(field,n=1):
            stats[field]+=n
            for group in local_groups:group[field]+=n
        if key is None:count("unmatched_prediction_instances")
        elif unit["eligible"]:count("eligible_gt_instances")
        else:count("quality_ineligible_gt")
        if key is not None and not unit["pose_nonignore"]:count("pose_ignored_gt")
        if row is None:
            if unit["eligible"]:count("missing_record_instances");count("missing_prediction_instances")
            continue
        if row["prediction_id"] is None:
            if unit["eligible"]:count("missing_prediction_instances")
            continue
        usable=0;diag=row["gt_diagonal224"]
        for i in range(8):
            known=row["label_known"][i];valid=row["gt_valid"][i]
            selected=row["q_corner"][i]>=policy["selected_q_min"]
            in_domain=all(0<=v<=223 for v in row["uv224"][i])
            usable+=bool(selected and row["p_valid"][i]>=policy["usable_p_valid_min"] and in_domain and not row["multimodal"][i])
            if not known:
                count("unknown_label_points")
                if selected:count("selected_unknown_points")
                continue
            if not selected:continue
            count("selected_known_points")
            if not valid:continue # Known invalid high-q points are negatives, never removed.
            error=row["error224"][i]
            if not _finite(diag) or diag<=0 or error is None:
                count("unmeasured_selected_valid_points");continue
            count("selected_valid_points")
            # Preserve the trained valid-and-error quality target; domain validity is a GT label.
            if error<=policy["error_fraction_gt_diagonal224"]*diag:count("selected_correct_points")
        if key is not None and unit["eligible"] and usable>=policy["minimum_usable_ids"]:count("instances_with_6_usable_ids")
    raw=_score(raw_report,run,units,images,expected_predictions,policy);candidate=_score(candidate_report,run,units,images,expected_predictions,policy)
    if len(images)!=policy["eval_images"]:reasons.append("full650_image_domain_required")
    if set(scored)!=set(units):reasons.append("incomplete_GT_record_coverage")
    if used!=expected_predictions:reasons.append("incomplete_prediction_record_coverage")
    if stats["eligible_gt_instances"]==0:reasons.append("no_eligible_GT_instances")
    if stats["unknown_label_points"]:reasons.append("quality_labels_unmeasured")
    if stats["selected_unknown_points"] or stats["unmeasured_selected_valid_points"]:reasons.append("selected_quality_outcomes_unmeasured")
    if stats["selected_valid_points"]<policy["minimum_valid_selected_points"]:reasons.append("insufficient_GT_valid_selected_points")
    precision_measured=not (stats["selected_unknown_points"] or stats["unmeasured_selected_valid_points"])
    if precision_measured and (stats["selected_known_points"]==0 or stats["selected_correct_points"]*10<stats["selected_known_points"]*9):
        reasons.append("selected_known_precision_below90percent")
    if stats["instances_with_6_usable_ids"]*10<stats["eligible_gt_instances"]:
        reasons.append("six_usable_ID_instance_coverage_below10percent")
    if not raw["complete"] or not candidate["complete"]:reasons.append("pose_score_domain_incomplete_or_unmeasured")
    if raw["complete"] and candidate["complete"] and candidate["successes"]<raw["successes"]:reasons.append("candidate_all8_below_raw")
    ready=not reasons
    stats["selected_known_precision"]=stats["selected_correct_points"]/stats["selected_known_points"] if precision_measured and stats["selected_known_points"] else None
    stats["six_ID_instance_coverage"]=stats["instances_with_6_usable_ids"]/stats["eligible_gt_instances"] if stats["eligible_gt_instances"] else None
    receipt={"schema":"action4.corner_checkpoint_calibration.v1","run_binding":run,"domain_sha256":domain["domain_sha256"],
        "policy":copy.deepcopy(policy),"quality_ready":ready,"quality_gate_passed_for_selection":ready,
        "selection_eligibility":"not_decided_here","selection_guard_required":"action4_pose_protocol.eligible(final.pose_report,stage_reference)","reasons":reasons,
        "coverage":{"expected_images":len(images),"GT_units":len(units),"GT_records":len(scored),
            "published_prediction_units":len(expected_predictions),"prediction_records":len(used),
            "missing_GT_units":[list(k) for k in sorted(set(units)-set(scored))],
            "missing_prediction_units":[list(k) for k in sorted(expected_predictions-used)]},
        "point_and_instance_statistics":stats,"groups":groups,"group_semantics":"overlapping labels; do not sum groups as disjoint totals","raw_statistics":raw,"candidate_statistics":candidate,
        "raw_report":copy.deepcopy(raw_report),"candidate_report":copy.deepcopy(candidate_report),
        "final_report":copy.deepcopy(candidate_report if ready else raw_report),"final_path":"candidate" if ready else "raw",
        "decision_scope":"whole_checkpoint_not_per_image_GT","effect_admission":False}
    for field in ("raw_report","candidate_report","final_report"):
        receipt[field]["outcomes"]=sorted(receipt[field]["outcomes"],key=_key)
        if receipt[field]["pose_report"] is not None:
            full=receipt[field]["pose_report"]
            full["per_gt_failure_reason"]=sorted(full["per_gt_failure_reason"],key=lambda r:(r["sample_id"],str(r["instance_id"])))
            full["unmatched_pose_predictions"]=sorted(full["unmatched_pose_predictions"],key=lambda r:(r["sample_id"],r["prediction_id"]))
    receipt["scoring_records_sha256"]=digest(sorted(records,key=_record_sort))
    receipt["identity"]=digest(receipt)
    return receipt
