"""Explicit prediction-only bridge for unadmitted corner research bundles.

No GT, scoring or calibration. Formal instances always retain the raw A0 result.
A geometric candidate may be recorded separately for whole-checkpoint assessment.
"""
import copy
from dataclasses import dataclass,field
from types import MappingProxyType
import json,os
import hashlib
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import DataAdapter
from eval.action4_sam_pipeline_inputs import SamInferenceContext,prepare_pipeline_sam,device_mask_aux
from model.action4_single_frame_pose import POLICY,observations_from_head,decode_single_frame_pose


@dataclass(frozen=True)
class EngineeringFullBoxGate:
    image_id: str
    prediction_id: str
    bbox_full_native: tuple
    passed: bool
    reason: str


def _binding_equal(a,b):
    return (isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b)=={"path","sha256"}
        and contained_path(R,a["path"],exists=True).resolve()==contained_path(R,b["path"],exists=True).resolve()
        and a["sha256"]==b["sha256"])


def validate_full_box_quality_gate(binding,*,b0_sha256,b0_config_id=None,expected_reference_predictions=None):
    """Validate bound development report/policy/source/counts, without reading GT labels."""
    from data.action4_sam_native_export import _v2_checked
    from eval.action4_b0_scoring import REFINED_QUALITY_PROTOCOL
    gate=_v2_checked(binding);scope="phrase2_synthetic_development_full650"
    if (gate.get("schema")!="action4.b0_refined_full_box_quality_gate.v1" or
        gate.get("status")!="development_calibrated" or gate.get("B0_bundle_sha256")!=b0_sha256 or
        gate.get("score_field")!="refined_quality_score" or gate.get("score_protocol")!=REFINED_QUALITY_PROTOCOL or
        type(gate.get("threshold")) not in (int,float) or not np.isfinite(gate["threshold"]) or not 0<=gate["threshold"]<=1):
        raise ValueError("Calibrated full-box gate source/field/protocol invalid")
    if gate.get("scope")!=scope or gate.get("real_world_quality_calibrated") is not False:
        raise ValueError("Full-box gate must retain synthetic-development scope")
    if "calibration_report" not in gate:raise ValueError("Bound full-box calibration report required")
    report=_v2_checked(gate["calibration_report"])
    if (report.get("schema")!="action4.b0_refined_quality_calibration.v1" or report.get("status")!="development_calibrated"
        or report.get("complete") is not True or report.get("split")!="eval" or report.get("scope")!=scope
        or report.get("B0_bundle_sha256")!=b0_sha256 or report.get("images")!=650 or report.get("GT")!=1300
        or report.get("selected_threshold")!=gate["threshold"] or report.get("original_fields_exact_full650") is not True
        or report.get("B0_publication_threshold_unchanged")!=.3 or report.get("model_weights_changed") is not False
        or report.get("real_world_quality_calibrated") is not False or report.get("non_DJI_specificity_measured") is not False):
        raise ValueError("Full-box calibration report binding/scope/threshold mismatch")
    pre=_v2_checked(report["preregistration"])
    if (pre.get("field")!="refined_quality_score" or pre.get("threshold_grid")!=[.5,.6,.7,.8,.9]
        or pre.get("minimum_selected_precision")!=.95 or pre.get("minimum_GT_coverage")!=.9
        or pre.get("positive_definition")!="one-to-one full body box IoU>=0.5 with nonignoreGT"
        or pre.get("selection")!="among eligible thresholds maximize GT coverage, then precision, then lower threshold"
        or pre.get("no_weights_or_publication_threshold_change") is not True):
        raise ValueError("Unregistered full-box quality calibration policy")
    complete=_v2_checked(report["capture_complete"]);source=complete.get("identity",{})
    if (complete.get("status")!="COMPLETE_FULL650_NEW_QUALITY_FIELD_ORIGINAL_PREDICTIONS_EXACT"
        or complete.get("images")!=650 or complete.get("candidates")!=1300
        or complete.get("predictions")!=report["predictions"] or complete.get("B0_retrained") is not False
        or complete.get("default_weights_changed") is not False or source.get("B0_bundle",{}).get("sha256")!=b0_sha256
        or source.get("selected_step")!=9500 or source.get("new_field")!="refined_quality_score"
        or source.get("GT_used_for_inference") is not False or source.get("scoring_policy_unchanged")!="center"
        or source.get("data_manifest_sha256")!=report["data_manifest"]["sha256"]
        or (b0_config_id is not None and source.get("config_id")!=b0_config_id)):
        raise ValueError("Full-box calibration capture/model/source mismatch")
    if expected_reference_predictions is not None and not _binding_equal(source["reference_predictions"],expected_reference_predictions):
        raise ValueError("Calibrated source predictions differ from actual eval SAM counterpart")
    predictions=_v2_checked(report["predictions"]);reference=_v2_checked(source["reference_predictions"])
    if (len(predictions)!=650 or len(reference)!=650 or len({p["sample_id"] for p in predictions})!=650
        or [p["sample_id"] for p in predictions]!=[p["sample_id"] for p in reference]):
        raise ValueError("Full-box calibration prediction image domain mismatch")
    q=[]
    for row,old in zip(predictions,reference):
        stripped=dict(row,instances=[{k:v for k,v in p.items() if k!="refined_quality_score"} for p in row["instances"]])
        if stripped!=old:raise ValueError("Quality capture altered original B0 fields")
        for p in row["instances"]:
            value=p["refined_quality_score"]
            if type(value) not in (int,float) or not np.isfinite(value) or not 0<=value<=1:
                raise ValueError("Invalid captured refined quality")
            if p["score"]>=.3:q.append(value)
    if len(q)!=1300 or report.get("quality_range")!=[min(q),max(q)]:
        raise ValueError("Full-box calibration candidate/quality domain mismatch")
    rows=report.get("threshold_results")
    if not isinstance(rows,list) or [r["threshold"] for r in rows]!=pre["threshold_grid"]:
        raise ValueError("Calibration finite threshold grid differs")
    for r in rows:
        selected=sum(v>=r["threshold"] for v in q)
        if any(type(r.get(k)) is not int or r[k]<0 for k in ("selected","TP","FP","GT","both_detected","dual_images")):
            raise ValueError("Calibration counts must be explicit nonnegative integers")
        if (r["selected"]!=selected or r["GT"]!=1300 or r["TP"]>min(selected,1300) or r["FP"]!=selected-r["TP"]
            or not 0<=r["both_detected"]<=r["dual_images"]<=650 or 2*r["both_detected"]>r["TP"]):
            raise ValueError("Calibration selected/count mass mismatch")
        precision=r["TP"]/selected if selected else None;coverage=r["TP"]/1300
        if r.get("precision")!=precision or r.get("GT_coverage")!=coverage:
            raise ValueError("Calibration precision/coverage inconsistent")
        eligible=precision is not None and precision>=.95 and coverage>=.9
        if r.get("eligible") is not eligible:raise ValueError("Calibration eligibility differs from policy")
        groups=r.get("groups",{})
        if groups.get("all",{}).get("GT")!=1300 or groups.get("all",{}).get("selected_true_positive")!=r["TP"]:
            raise ValueError("Calibration all-group counts inconsistent")
        for group in groups.values():
            n=group.get("GT");tp=group.get("selected_true_positive")
            if type(n) is not int or type(tp) is not int or not 0<=tp<=n<=1300 or n==0 or group.get("GT_coverage")!=tp/n:
                raise ValueError("Calibration subgroup arithmetic inconsistent")
    eligible=[r for r in rows if r["eligible"]]
    if not eligible:raise ValueError("No preregistered full-box operatingpoint passed")
    selected=max(eligible,key=lambda r:(r["GT_coverage"],r["precision"],-r["threshold"]))
    if report.get("selected")!=selected or selected["threshold"]!=gate["threshold"]:
        raise ValueError("Full-box gate selection differs from bound preregistration/report")
    return copy.deepcopy(gate)



_B0_SOURCE_PROOF_TOKEN=object()


@dataclass(frozen=True,slots=True)
class _FrozenB0Source:
    binding:object
    path:object
    fingerprint:tuple
    rows:object
    image_ids:frozenset
    pid:int
    _token:object=field(default=None,init=False,repr=False,compare=False)

    @classmethod
    def load(cls,binding,image_ids):
        from data.action4_sam_native_export import _v2_checked,_fingerprint
        ref=copy.deepcopy(binding)
        path=contained_path(R,ref["path"],exists=True).resolve(strict=True)
        before=_fingerprint(path.stat())
        source=_v2_checked(ref)
        if _fingerprint(path.stat())!=before:raise ValueError("Frozen B0 source changed during authentication")
        rows={}
        for row in source:
            image_id=row["sample_id"]
            if image_id in rows:raise ValueError("Duplicate frozen B0 image")
            rows[image_id]=json.dumps(row["instances"],sort_keys=True,separators=(",",":"),allow_nan=False)
        domain=frozenset(image_ids)
        if frozenset(rows)!=domain:raise ValueError("Corner frozen B0 image domain differs from native SAM")
        proof=cls(MappingProxyType(ref),path,before,MappingProxyType(rows),domain,os.getpid())
        object.__setattr__(proof,"_token",_B0_SOURCE_PROOF_TOKEN)
        return proof

    def current_rows(self,binding,image_id,image_ids):
        from data.action4_sam_native_export import _fingerprint,_v2_sha
        if self._token is not _B0_SOURCE_PROOF_TOKEN or self.pid!=os.getpid():
            raise ValueError("Frozen B0 source proof must be authenticated in current process")
        if dict(self.binding)!=binding or frozenset(image_ids)!=self.image_ids:
            raise ValueError("Frozen B0 source binding/domain changed")
        path=contained_path(R,binding["path"],exists=True).resolve(strict=True)
        if path!=self.path or _fingerprint(path.stat())!=self.fingerprint:
            raise ValueError("Frozen B0 source file identity changed")
        # Existing SHA proof checks bytes on misses, with stat-keyed reuse on hits.
        if _v2_sha(path)!=self.binding["sha256"] or _fingerprint(path.stat())!=self.fingerprint:
            raise ValueError("Frozen B0 source bytes/identity changed")
        if image_id not in self.rows:raise ValueError("Current frame absent from frozen B0 source")
        return json.loads(self.rows[image_id])  # independent current-image copy only

    def __reduce_ex__(self,protocol):
        raise TypeError("Frozen B0 source proof is process-local; authenticate in the consumer")



def _corner_saved_rows(pipeline,image_id):
    proof=getattr(pipeline,"_corner_frozen_source",None)
    if type(proof) is not _FrozenB0Source:
        raise ValueError("Authenticated immutable B0 source proof required")
    return proof.current_rows(pipeline._corner_prediction_source_binding,image_id,pipeline.corner_cache.images)


def _corner_input_contract(cfg):
    """Registered native replay or the historical explicitly labelled live-B0 route."""
    from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL,validate_config
    if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
        validate_config(cfg,"train")
        return {"schema":CORNER_NATIVE_PROTOCOL,"roi_source":"frozen_b0_prediction","roi_expansion":cfg["roi_expansion"],
            "B0_sha256":cfg["evaluation_counterpart_sha256"],
            "scoring_policy":cfg["b0_scoring_policy"],
            "frozen_predictions":copy.deepcopy(cfg["frozen_eval_b0_predictions"])}
    if (cfg.get("training_protocol") is not None or cfg.get("roi_source")=="frozen_b0_prediction"
        or cfg.get("roi_expansion",2.25)!=2.25 or cfg.get("preprocess",{}).get("roi_expansion",2.25)!=2.25):
        raise ValueError("Corner prediction inputs require a registered native contract or legacy2.25 contract")
    return {"schema":"action4.corner_legacy_live_B0_roi225.v1","roi_expansion":2.25,
            "roi_source":"legacy_live_B0_prediction"}


def _corner_candidates(pipeline,rgb,K,owned_context):
    contract=pipeline._corner_input_contract
    if contract!=_corner_input_contract(pipeline.acfg) or pipeline.roi_expansion!=contract["roi_expansion"]:
        raise ValueError("Corner frozen input contract changed")
    if contract["schema"]=="action4.corner_legacy_live_B0_roi225.v1":
        return pipeline._detect_corner(rgb)
    from data.action4_frozen_detection_context import FrozenDetectionContext
    ref=pipeline._corner_prediction_source_binding
    alias=getattr(pipeline,"_corner_prediction_alias",None)
    if alias is not None:
        from data.action4_prediction_roi_data import verify_prediction_binding_alias
        if verify_prediction_binding_alias(alias["producer"],ref)!=alias:raise ValueError("Corner prediction alias receipt changed")
    if ref!=contract["frozen_predictions"] or pipeline.b0_scoring_policy!=contract["scoring_policy"]:
        raise ValueError("Corner saved B0 source/policy differs from input contract")
    frame=owned_context.frame
    saved=_corner_saved_rows(pipeline,frame.image_id)
    frozen=FrozenDetectionContext.from_sam_frame(frame,K,saved,
        sam_predictions=owned_context.predictions,sam_sources=owned_context.expected_sources,
        b0_sha256=contract["B0_sha256"],scoring_policy=contract["scoring_policy"],
        source_cache_identity=ref["sha256"])
    return frozen.consume(rgb,K,b0_sha256=pipeline.corner_cache.sources["b0_model_sha256"],
                          scoring_policy=pipeline.b0_scoring_policy)


def _prepare_corner_sam(pipeline,rgb,K,detections,context):
    if pipeline._corner_input_contract!=_corner_input_contract(pipeline.acfg):
        raise ValueError("Corner input contract changed before ROI preparation")
    if pipeline.roi_expansion!=pipeline._corner_input_contract["roi_expansion"]:
        raise ValueError("Corner ROI expansion changed")
    return prepare_pipeline_sam(rgb,K,detections,context,roi_expansion=pipeline.roi_expansion)


def _corner_full_box_quality(pipeline,prediction,quality_evidence=None):
    if pipeline._corner_input_contract["roi_source"]=="frozen_b0_prediction":
        if quality_evidence is None:
            return False,"frozen_refined_quality_sidecar_not_bound","unadmitted"
        pid=prediction["prediction_id"]
        if pid not in quality_evidence:raise ValueError("Missing current-instance sidecar quality")
        q=quality_evidence[pid]
        return bool(q["passed"]),"exact_capture_bound_development_quality",q["scope"]
    calibrated=pipeline._corner_box_gate
    if calibrated is None:return False,"full_box_quality_not_calibrated","unadmitted"
    from eval.action4_b0_scoring import REFINED_QUALITY_PROTOCOL
    if prediction.get("refined_quality_protocol")!=REFINED_QUALITY_PROTOCOL:
        raise ValueError("Current refined quality diagnostic protocol missing")
    return (prediction["refined_quality_score"]>=calibrated["threshold"],
        "frozen_development_calibrated_refined_quality","declared_calibration_receipt")


def create_corner_pipeline(cls,a0_bundle,b0_bundle,eval_sam_cache,*,device,engineering_fixture,full_box_quality_receipt):
    from model.action4_release_bundle import load_bundle,CORNER_SCHEMA
    from model.action4_checkpoint_paths import resolve_checkpoint
    from model.action4_sam_joint import CORNER_PROTOCOL
    from data.action4_sam_cache_provider import SamCacheProvider
    from eval.action4_surface_snapshot import bind_model_file,model_identities,check_model_file
    if type(engineering_fixture) is not bool:raise ValueError("Explicit engineering fixture permission required")
    import os
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        if torch.cuda.is_initialized():raise RuntimeError("Set deterministic CUDA workspace before initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
    self=cls.__new__(cls);self.device=device;self.corner_candidate_mode=True
    self._surface_model_bindings={role:bind_model_file(resolve_checkpoint(p)) for role,p in (("A0",a0_bundle),("B0",b0_bundle))}
    models=model_identities(self._surface_model_bindings)
    self.a0,self.base,az=load_bundle(models["A0"]["path"],"A0",device,
        allow_corner_candidate=True,expected_sha256=models["A0"]["sha256"])
    if az.get("schema")!=CORNER_SCHEMA or az.get("corner_protocol")!=CORNER_PROTOCOL:
        raise ValueError("Explicit unadmitted full corner candidate required")
    counterpart=az["corner_counterpart"]
    if not _binding_equal(counterpart["B0"],models["B0"]):raise ValueError("Corner B0 counterpart differs from actual bundle")
    if not isinstance(eval_sam_cache,dict) or set(eval_sam_cache)!={"manifest","expected_sources"}:
        raise ValueError("Explicit eval SAM cache spec required")
    if not _binding_equal(counterpart["eval_sam_cache"],eval_sam_cache["manifest"]):
        raise ValueError("Corner eval SAM counterpart differs from actual cache")
    self.corner_cache=SamCacheProvider(eval_sam_cache,split="eval",allow_synthetic=engineering_fixture)
    from data.action4_sam_native_export import _v2_checked
    index=_v2_checked(eval_sam_cache["manifest"])
    if index["sources"].get("upstream_mode")=="b0_completed_native_receipt_v2":
        receipt=_v2_checked(index["producer_receipt"])
        self._corner_prediction_source_binding=copy.deepcopy(receipt["source_contract"]["predictions"])
        self._corner_frozen_source=None
        # Legacy live-B0 comparisons also use the authenticated per-image copies.
        self._corner_saved_predictions=None
    elif engineering_fixture and index["origin"]=="synthetic_fixture":
        self._corner_prediction_source_binding=None
        self._corner_saved_predictions={r["image_id"]:r["engineering_saved_predictions"] for r in index["images"]}
    else:raise ValueError("Corner bridge requires completed B0 cache, or explicit engineering fixture")

    if self.corner_cache.sources["b0_model_sha256"]!=models["B0"]["sha256"]:
        raise ValueError("Eval SAM prompt B0 source mismatch")
    self.b0,_,bs=load_bundle(models["B0"]["path"],"B0",device,expected_sha256=models["B0"]["sha256"])
    if bs["architecture"]!="online_b0":raise ValueError("Corner bridge currently requires actual completed baseline B0")
    self.bcfg=bs["config"];self.acfg=az["config"]
    self._corner_input_contract=_corner_input_contract(self.acfg)
    self.roi_expansion=self._corner_input_contract["roi_expansion"]
    if self._corner_input_contract["roi_source"]=="frozen_b0_prediction":
        if engineering_fixture:raise ValueError("Native corner protocol cannot use engineering fixture overrides")
        from data.action4_prediction_roi_data import verify_prediction_binding_alias
        self._corner_prediction_alias=verify_prediction_binding_alias(self._corner_prediction_source_binding,self._corner_input_contract["frozen_predictions"])
        self._corner_prediction_source_binding=copy.deepcopy(self._corner_input_contract["frozen_predictions"])
        if models["B0"]["sha256"]!=self._corner_input_contract["B0_sha256"]:
            raise ValueError("Loaded corner/native B0 differs from frozen input contract")
    if self._corner_prediction_source_binding is not None:
        self._corner_frozen_source=_FrozenB0Source.load(self._corner_prediction_source_binding,self.corner_cache.images)
    self.b0_scoring_policy=bs.get("inference_scoring_policy",self.bcfg.get("evaluation_scoring_policy","learned"))
    if self.b0_scoring_policy not in ("learned","center","product"):raise ValueError("Unknown B0 scoring policy")
    self.a0_kind=az["architecture"];self.pose_method="network_raw_corner_research_only"
    self.a0.eval().requires_grad_(False);self.b0.eval().requires_grad_(False)
    self._corner_versions=tuple(p._version for module in (self.a0,self.b0) for p in list(module.parameters())+list(module.buffers()))
    self._corner_engineering_fixture=engineering_fixture
    self._corner_eval_spec=copy.deepcopy(eval_sam_cache)
    self._corner_box_gate=None
    self._corner_box_gate_binding=copy.deepcopy(full_box_quality_receipt)
    self._corner_quality_bridge=None
    quality_bridge_ref=self.acfg.get("corner_refined_quality_bridge")
    if quality_bridge_ref is not None:
        if self._corner_input_contract["roi_source"]!="frozen_b0_prediction" or full_box_quality_receipt is None:
            raise ValueError("Quality bridge requires native frozen predictions and a bound calibration")
        from eval.action4_corner_quality_bridge import CornerQualityBridge
        self._corner_quality_bridge=CornerQualityBridge(quality_bridge_ref,
            b0_sha256=models["B0"]["sha256"],b0_config_id=bs["config_id"],b0_step=bs["selected_step"],
            base_predictions=self._corner_prediction_source_binding,quality_gate=full_box_quality_receipt)
        self._corner_box_gate=validate_full_box_quality_gate(full_box_quality_receipt,
            b0_sha256=models["B0"]["sha256"],b0_config_id=bs["config_id"])
    elif full_box_quality_receipt is not None:
        self._corner_box_gate=validate_full_box_quality_gate(full_box_quality_receipt,
            b0_sha256=models["B0"]["sha256"],b0_config_id=bs["config_id"],
            expected_reference_predictions=self._corner_prediction_source_binding)
    self.identity={"schema":"action4.corner_prediction_bridge.v1","models":models,
        "A0_config_id":az["config_id"],"B0_config_id":bs["config_id"],"A0_step":az["selected_step"],"B0_step":bs["selected_step"],
        "A0_kind":az["architecture"],"B0_bundle_sha256":models["B0"]["sha256"],"B0_scoring_policy":self.b0_scoring_policy,
        "input_contract":copy.deepcopy(self._corner_input_contract),
        "corner_protocol":copy.deepcopy(az["corner_protocol"]),"counterpart":copy.deepcopy(counterpart),
        "eval_SAM_cache":copy.deepcopy(eval_sam_cache["manifest"]),"decoder_policy":copy.deepcopy(POLICY),
        "quality_ready":False,"deployment_admitted":False,"engineering_fixture":engineering_fixture,
        "full_box_quality_receipt":copy.deepcopy(full_box_quality_receipt),
        "refined_quality_bridge":copy.deepcopy(quality_bridge_ref),
        "prediction_source_alias":copy.deepcopy(getattr(self,"_corner_prediction_alias",None)),
        "full_box_quality_policy":"not_calibrated; false unless explicit engineering fixture gate" if self._corner_box_gate is None else copy.deepcopy(self._corner_box_gate)}
    for binding in self._surface_model_bindings.values():check_model_file(binding)
    if {p.data_ptr() for p in self.b0.parameters()} & {p.data_ptr() for p in self.a0.parameters()}:
        raise ValueError("A0 and B0 parameter storage shared")
    return self


def _owned_context(pipeline,image_id,rgb):
    """Private snapshot of provider-owned immutable values; never exposed publicly."""
    frame,predictions=pipeline.corner_cache.frame_predictions(image_id,rgb)
    return SamInferenceContext(frame,dict(predictions),dict(pipeline.corner_cache.sources),pipeline._corner_engineering_fixture)


def cache_context(pipeline,image_id,rgb):
    """Public context owns its map and immutable copies, including returned metadata."""
    from data.action4_sam_roi_inputs import (ImageFrame,NativeMaskIdentity,NativeSamPrediction,
        verify_image_frame,verify_native_prediction)
    owned=_owned_context(pipeline,image_id,rgb)
    f=owned.frame
    frame=verify_image_frame(ImageFrame(f.image_id,f.pixels,f.pixel_space_id,f.source_sha256,f.variant_id,f.variant_sha256))
    public={}
    for key,pred in owned.predictions.items():
        public[key]=verify_native_prediction(NativeSamPrediction(copy.deepcopy(pred.metadata),pred.mask),
            frame,NativeMaskIdentity(*key),owned.expected_sources,allow_synthetic=owned.engineering_fixture)
    return SamInferenceContext(frame,public,dict(owned.expected_sources),owned.engineering_fixture)


def _checked_context(pipeline,rgb,context):
    from data.action4_sam_roi_inputs import verify_image_frame,verify_native_prediction,NativeMaskIdentity
    if not isinstance(context,SamInferenceContext):raise ValueError("Explicit cache-bound SAM context required")
    if context.engineering_fixture is not pipeline._corner_engineering_fixture or context.expected_sources!=pipeline.corner_cache.sources:
        raise ValueError("SAM context source/engineering permission differs from actual cache")
    frame=verify_image_frame(context.frame)
    # Do not reacquire reference through the public/caller-owned context.
    owned=_owned_context(pipeline,frame.image_id,rgb)
    if (frame.image_id,frame.pixel_space_id,frame.source_sha256,frame.variant_id,frame.variant_sha256)!=(
        owned.frame.image_id,owned.frame.pixel_space_id,owned.frame.source_sha256,owned.frame.variant_id,owned.frame.variant_sha256):
        raise ValueError("SAM frame differs from registered eval cache")
    if set(context.predictions)!=set(owned.predictions):raise ValueError("SAM candidate cache domain differs")
    for key,reference in owned.predictions.items():
        supplied=verify_native_prediction(context.predictions[key],frame,NativeMaskIdentity(*key),
            owned.expected_sources,allow_synthetic=owned.engineering_fixture)
        if supplied.metadata!=reference.metadata:
            raise ValueError("SAM candidate cache identity/assignment differs")
        if ((supplied.mask is None)!=(reference.mask is None) or
                (reference.mask is not None and not np.array_equal(supplied.mask,reference.mask))):
            raise ValueError("SAM mask payload differs from owned verified cache")
    # Forward consumes only this independent map and provider-owned verified pixels.
    return owned


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


@torch.inference_mode()
def predict_corner(pipeline,rgb,K,canonical,*,sam_context,diagnostic_candidate=False,engineering_route_gates=None):
    from eval.action4_surface_snapshot import check_model_file
    if type(diagnostic_candidate) is not bool:raise ValueError("Explicit diagnostic_candidate flag required")
    if engineering_route_gates is not None and (not pipeline._corner_engineering_fixture or not isinstance(engineering_route_gates,dict)):
        raise ValueError("Uncalibrated external route gates require explicit engineering fixture mode")

    for binding in pipeline._surface_model_bindings.values():check_model_file(binding)
    versions=tuple(p._version for module in (pipeline.a0,pipeline.b0) for p in list(module.parameters())+list(module.buffers()))
    if versions!=pipeline._corner_versions or pipeline.a0.training or pipeline.b0.training:
        raise ValueError("Frozen corner inference model changed")
    K=np.asarray(K,float);cad=np.asarray(canonical,float)
    if K.shape!=(3,3) or cad.shape!=(8,3) or not np.isfinite(K).all() or not np.isfinite(cad).all():
        raise ValueError("Finite native K and fixed canonical8 metres required")
    from data.action4_sam_native_export import _v2_sha
    manifest=pipeline._corner_eval_spec["manifest"]
    if _v2_sha(manifest["path"])!=manifest["sha256"]:raise ValueError("Bound eval cache manifest changed")
    if pipeline._corner_box_gate_binding is not None:
        ref=pipeline._corner_box_gate_binding
        if _v2_sha(ref["path"])!=ref["sha256"]:raise ValueError("Bound full-box calibration receipt changed")
        report_binding=pipeline._corner_box_gate["calibration_report"]
        if _v2_sha(report_binding["path"])!=report_binding["sha256"]:raise ValueError("Bound full-box calibration report changed")

    owned_context=_checked_context(pipeline,rgb,sam_context)
    frame=owned_context.frame
    candidates=[]
    for original in _corner_candidates(pipeline,rgb,K,owned_context):
        p=dict(original);grid=p.get("grid_index")
        if not isinstance(grid,(int,np.integer)) or isinstance(grid,(bool,np.bool_)) or grid<0:
            raise ValueError("Actual stable B0 grid index required")
        pid=f"b0:{frame.image_id}:grid:{int(grid)}"
        if p.get("prediction_id",pid)!=pid:raise ValueError("B0 prediction/grid identity mismatch")
        p["prediction_id"]=pid;candidates.append(p)
    if len({p["prediction_id"] for p in candidates})!=len(candidates):raise ValueError("Duplicate B0 candidate identity")
    if pipeline._corner_prediction_source_binding is not None:
        saved=_corner_saved_rows(pipeline,frame.image_id)
    else:
        saved=pipeline._corner_saved_predictions.get(frame.image_id)
    if saved is None:raise ValueError("Current image absent from frozen B0 candidate domain")
    expected={int(p["grid_index"]):p for p in saved}
    if len(expected)!=len(saved) or {int(p["grid_index"]) for p in candidates}!=set(expected):
        raise ValueError("Current B0 candidate domain differs from SAM prompt source")
    for p in candidates:
        source=expected[int(p["grid_index"])]
        if any(key not in p or _plain(p[key])!=_plain(value) for key,value in source.items()):
            raise ValueError("Current B0 raw fields differ from frozen SAM source")
    if engineering_route_gates is not None and not set(engineering_route_gates)<={p["prediction_id"] for p in candidates}:
        raise ValueError("Engineering gate points to another candidate")
    quality_evidence=None
    bridge=getattr(pipeline,"_corner_quality_bridge",None)
    if bridge is not None:
        quality_evidence=bridge.consume(sample_id=frame.image_id,rgb=frame.pixels,K=K,
            base_instances=saved,b0_sha256=pipeline.identity["B0_bundle_sha256"])
    detections=[p for p in candidates if p["score"]>=.3]
    result={"schema":"action4.corner_prediction_bridge.v1","image_id":frame.image_id,"identity":copy.deepcopy(pipeline.identity),
        "box_candidates":_plain(candidates),"detections":_plain(detections),"instances":[],"raw_instances":[],
        "candidate_instances":[],"corner_observations":[],"quality_ready":False,"deployment_admitted":False,
        "diagnostic_candidate":diagnostic_candidate,"GT_used":False}
    if not detections:return result
    inputs=_prepare_corner_sam(pipeline,rgb,K,detections,owned_context)
    rois=inputs.rgb.to(pipeline.device)
    km=torch.as_tensor(np.stack(inputs.K112),device=pipeline.device,dtype=rois.dtype)
    canonical_tensor=torch.as_tensor(cad,device=pipeline.device,dtype=rois.dtype)[None].expand(len(detections),-1,-1)
    aux=device_mask_aux(inputs.mask_aux,pipeline.device,rois.dtype)
    packet=pipeline.a0(rois,km,canonical_tensor,mask_aux=aux,identities=aux.identities,return_corner=True)
    from model.action4_sam_joint import CornerCandidateOutput
    if not isinstance(packet,CornerCandidateOutput) or packet.corners.identities!=aux.identities:
        raise ValueError("Owned corner forward returned wrong instance order")
    output=packet.raw_pose
    for i,p in enumerate(detections):
        identity=aux.identities[i]
        if identity.image_id!=frame.image_id or identity.instance_id!=p["prediction_id"]:raise ValueError("Raw/head/ROI identity mismatch")
        obs=observations_from_head(packet.corners,i,identity=identity,pixel_space_id=frame.pixel_space_id,config=POLICY)
        rotation=output["rotation"][i].cpu().numpy();translation=output["translation"][i].cpu().numpy()
        camera=output["camera_corners"][i].cpu().numpy();uv112=output["corners"][i].cpu().numpy()
        raw_valid=bool(np.isfinite(camera).all() and np.isfinite(uv112).all() and (camera[:,2]>0).all())
        raw_corners=DataAdapter.transform_points(uv112,np.linalg.inv(inputs.A112[i])).tolist() if raw_valid else None
        gate=None if engineering_route_gates is None else engineering_route_gates.get(p["prediction_id"])
        if gate is not None:
            if (not isinstance(gate,EngineeringFullBoxGate) or type(gate.passed) is not bool or
                (gate.image_id,gate.prediction_id)!=(frame.image_id,p["prediction_id"]) or
                not np.array_equal(gate.bbox_full_native,p["bbox_full_xyxy"]) or not isinstance(gate.reason,str) or not gate.reason):
                raise ValueError("Engineering route gate identity/box mismatch")
        passed,gate_reason,gate_scope=_corner_full_box_quality(pipeline,p,quality_evidence)
        if gate is not None:
            passed=gate.passed;gate_reason=gate.reason;gate_scope="engineering_fixture_only"
        route={"kind":"full","identity":list(obs["identity"]),"pixel_space_id":frame.pixel_space_id,
            "full_box_native":list(p["bbox_full_xyxy"]),"full_box_quality_passed":bool(passed),
            "full_box_quality_reason":gate_reason,"quality_gate_scope":gate_scope}
        decoded=decode_single_frame_pose(canonical_m=cad,K_native=K,A224_from_native=inputs.A224[i],
            raw_rotation=rotation,raw_translation_m=translation,observations=obs,route=route,
            pixel_space_id=frame.pixel_space_id,raw_identity=identity,roi_identity=identity,
            quality_ready=False,config=POLICY,diagnostic_candidate=diagnostic_candidate)
        trace={"available":bool(inputs.mask_aux.available[i].item()),"reason":inputs.reasons[i],
            "dependencies":inputs.source_dependencies[i],"q_mask":float(packet.sam_aux.q_mask[i].item()),
            "q_mask_is_calibrated":False,"consumption_path":"native_SAM->sameROI->shared_enhancedC5->owned_corner_head"}
        raw=dict(_plain(p),instance_id=i,object_id=1,corner_ids=list(range(8)),source_detection_ids=[p["prediction_id"]],
            published=True,positive_depth_verified=raw_valid,corners_px=raw_corners,pose_valid=raw_valid,
            rotation=rotation.tolist(),translation_m=translation.tolist(),pose_diagnostic={"method":"raw_network","valid":raw_valid},
            sam_input=_plain(trace))
        candidate=copy.deepcopy(raw)
        candidate["pose_diagnostic"]={"method":"geometric_candidate_or_raw","reason":decoded["reason"],
            "geometric_candidate_accepted":decoded["geometric_candidate_accepted"],"deployed":False}
        if decoded["geometric_candidate_accepted"]:
            candidate.update(corners_px=decoded["candidate_corners_native"],rotation=decoded["candidate_rotation"],
                translation_m=decoded["candidate_translation_m"],pose_valid=True,positive_depth_verified=True)
        record={"image_id":frame.image_id,"prediction_id":p["prediction_id"],"roi_identity":list(obs["identity"]),
            "observations":_plain(obs),"A224_from_native":inputs.A224[i].tolist(),"K_native":K.tolist(),
            "canonical_m":cad.tolist(),"predicted_full_box":list(p["bbox_full_xyxy"]),
            "raw_rotation":rotation.tolist(),"raw_translation_m":translation.tolist(),"raw_corners_native":raw_corners,
            "route":_plain(route),"decoder_diagnostics":_plain(decoded),
            "quality_ready":False,"deployed":False}
        result["raw_instances"].append(raw);result["candidate_instances"].append(candidate)
        result["corner_observations"].append(record)
    result["instances"]=copy.deepcopy(result["raw_instances"])
    return result
