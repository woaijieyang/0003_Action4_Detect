"""Bind an already-calibrated development operating point to an exact quality sidecar."""
import copy
from data.action4_sam_native_export import _v2_checked,_v2_sha
from eval.action4_refined_quality_sidecar import RefinedQualitySidecar
from eval.action4_corner_prediction import validate_full_box_quality_gate


def verify_equivalent_capture(old_rows, sidecar, current_base):
    if len(old_rows)!=len(current_base) or len(sidecar["rows"])!=len(current_base):
        raise ValueError("Equivalent capture domain length differs")
    seen=set();count=0
    for old,base,row in zip(old_rows,current_base,sidecar["rows"]):
        sid=base["sample_id"]
        if sid in seen or old["sample_id"]!=sid or row["image"]["sample_id"]!=sid:
            raise ValueError("Equivalent capture ordered image domain differs")
        seen.add(sid)
        if len(old["instances"])!=len(base["instances"]) or len(row["qualities"])!=len(base["instances"]):
            raise ValueError("Equivalent candidate domain differs")
        for captured,original,q in zip(old["instances"],base["instances"],row["qualities"]):
            if ({k:v for k,v in captured.items() if k!="refined_quality_score"}!=original
                or q["refined_quality_score"]!=captured["refined_quality_score"]
                or q["grid_index"]!=original["grid_index"]):
                raise ValueError("Old calibrated capture and new sidecar are not exactly equivalent")
            count+=1
    return count


class CornerQualityBridge:
    """No calibration is learned here; reads no GT and never changes B0 fields."""
    def __init__(self, binding, *, b0_sha256, b0_config_id, b0_step, base_predictions, quality_gate):
        doc=_v2_checked(binding)
        if (doc.get("schema")!="action4.corner_quality_capture_bridge.v1"
            or doc.get("quality_gate")!=quality_gate or doc.get("base_predictions")!=base_predictions
            or doc.get("scope")!="phrase2_synthetic_development_full650"
            or doc.get("real_world_quality_calibrated") is not False):
            raise ValueError("Explicit development quality bridge binding required")
        gate=validate_full_box_quality_gate(quality_gate,b0_sha256=b0_sha256,b0_config_id=b0_config_id)
        report=_v2_checked(gate["calibration_report"])
        if doc.get("old_capture")!=report["predictions"]:
            raise ValueError("Bridge is not bound to the calibrated original capture")
        sidecar=_v2_checked(doc["sidecar"])
        if (sidecar["bindings"]["B0"]["sha256"]!=b0_sha256
            or sidecar["bindings"]["base_predictions"]!=base_predictions
            or sidecar["bindings"]["data_manifest"]!=report["data_manifest"]):
            raise ValueError("Sidecar model/data/base differ from calibration")
        count=verify_equivalent_capture(_v2_checked(doc["old_capture"]),sidecar,_v2_checked(base_predictions))
        if count!=1300 or sidecar["images"]!=650:raise ValueError("Full650 equivalence required")
        self.consumer=RefinedQualitySidecar(doc["sidecar"],bindings=sidecar["bindings"],
            model_config_id=b0_config_id,selected_step=b0_step)
        self._refs=copy.deepcopy([binding,quality_gate,gate["calibration_report"],doc["old_capture"]])
        self._gate=copy.deepcopy(gate)
        self.identity=copy.deepcopy(doc)

    def consume(self, *, sample_id, rgb, K, base_instances, b0_sha256):
        for ref in self._refs:
            if _v2_sha(ref["path"])!=ref["sha256"]:raise ValueError("Calibrated bridge source changed")
        payload=self.consumer.consume(sample_id=sample_id,rgb=rgb,K=K,
            base_instances=base_instances,b0_sha256=b0_sha256)
        return self._qualify(payload)

    def consume_previously_verified_original(self,**kwargs):
        for ref in self._refs:
            if _v2_sha(ref["path"])!=ref["sha256"]:raise ValueError("Calibrated bridge source changed")
        payload=self.consumer.consume_previously_verified_original(**kwargs)
        return None if payload is None else self._qualify(payload)

    def _qualify(self,payload):
        threshold=self._gate["threshold"]
        return {q["prediction_id"]:dict(score=q["refined_quality_score"],
            threshold=threshold,passed=q["refined_quality_score"]>=threshold,
            scope=self._gate["scope"],real_world_quality_calibrated=False)
            for q in payload["qualities"]}


def validate_eval_quality_dependencies(cfg,base_predictions):
    """Evaluator preflight uses the same exact-capture bridge as per-image inference.

    Verify actual B0 checkpoint metadata, not a sidecar's self-declared step.
    No model construction, predictions, threshold calibration or GT reads here.
    """
    configured_predictions=cfg.get("frozen_eval_b0_predictions")
    if configured_predictions is not None:
        from data.action4_prediction_roi_data import verify_prediction_binding_alias
        verify_prediction_binding_alias(base_predictions,configured_predictions)
        base_predictions=configured_predictions
    gate_ref=cfg.get("b0_full_box_quality_receipt")
    if gate_ref is None:raise ValueError("Explicit report-bound full-box quality receipt required")
    bridge_ref=cfg.get("corner_refined_quality_bridge")
    if bridge_ref is None:
        return validate_full_box_quality_gate(gate_ref,b0_sha256=cfg["evaluation_counterpart_sha256"],
            expected_reference_predictions=base_predictions)
    import torch
    from data.action4_sam_native_export import _v2_read,canonical
    b0_ref={"path":cfg["evaluation_counterpart_bundle"],"sha256":cfg["evaluation_counterpart_sha256"]}
    state=_v2_read(b0_ref,lambda stream:torch.load(stream,map_location="cpu",weights_only=True))
    if (state.get("role")!="B0" or state.get("architecture")!="online_b0"
            or state.get("config_id")!=canonical(state.get("config"))
            or type(state.get("selected_step")) is not int):
        raise ValueError("Actual B0 checkpoint metadata malformed")
    model_config=state["config_id"];step=state["selected_step"];del state
    bridge=CornerQualityBridge(bridge_ref,b0_sha256=b0_ref["sha256"],b0_config_id=model_config,
        b0_step=step,base_predictions=base_predictions,quality_gate=gate_ref)
    return copy.deepcopy(bridge._gate)
