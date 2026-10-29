"""Stage4 native prediction-ROI admission; no baseline or stage3 gate substitution."""
from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL,SAM_NATIVE_PROTOCOL,validate_config
from data.action4_sam_native_export import _v2_checked,canonical

INPUT_FIELDS=("stage4_parent_seal","parent_best_step","parent_bundle_sha256","evaluation_counterpart_sha256","prediction_roi_manifest",
 "sam_cache","sam_training_associations","sam_eval_cache","roi_expansion","roi_size",
 "expected_units","expected_eval_images","score_schema","corner_softmax_bias_frozen",
 "corner_refined_quality_bridge","b0_full_box_quality_receipt")


def validate_parent_seal(cfg):
    from data.action4_prediction_roi_data import roi_finetune
    if roi_finetune(cfg):
        from .action4_roi115_finetune import validate_parent
        return validate_parent(cfg)
    from data.action4_paths import PROJECT_ROOT as R,contained_path
    import json
    ref=cfg.get("stage4_parent_seal")
    if not isinstance(ref,dict):raise ValueError("Corner candidate scheduled training requires its actual sealed SAM80 parent")
    seal=_v2_checked(ref)
    if (seal.get("schema")!="action4.sealed_independent_role.v2" or seal.get("role")!="A0"
        or seal.get("strict_load") is not True or seal.get("frozen_backbone_verified") is not True
        or seal.get("sha256")!=cfg.get("parent_bundle_sha256") or seal.get("selected_step")!=cfg.get("parent_best_step")):
        raise ValueError("Corner parent is not the selected strictly sealed SAM model")
    spec=_v2_checked({"path":seal["run_spec"],"sha256":seal["run_spec_sha256"]})
    parent=json.loads(contained_path(R,spec["config"],exists=True).read_text())
    if (canonical(parent)!=seal["config_id"] or parent.get("training_protocol")!=SAM_NATIVE_PROTOCOL
        or parent.get("mode")!="full_training" or parent.get("epochs")!=80 or parent.get("a0_variant")!="a0_sam_image"):
        raise ValueError("Stage4 requires completed formal SAM80, not an engineering or baseline parent")
    consumed=contained_path(R,cfg["parent_bundle"],exists=True);original=contained_path(R,seal["bundle"],exists=True)
    if consumed!=original:
        if cfg.get("require_immutable_source") is not True:raise ValueError("Unbound corner parent copy")
        current=json.loads(contained_path(R,cfg["run_spec"],exists=True).read_text());code=contained_path(R,current["code_root"],exists=True)
        if not consumed.is_relative_to(code) or current["files"].get(str(consumed.relative_to(code)))!=seal["sha256"]:
            raise ValueError("Corner RunSpec does not bind selected parent copy")
    same=("dataset_root","prediction_roi_manifest","frozen_eval_b0_predictions","evaluation_counterpart_sha256",
          "validation_receipt_sha256","surface_source","roi_source","roi_expansion","roi_size","score_schema","b0_scoring_policy",
          "sam_cache","sam_eval_cache","sam_training_associations","sam_mask_augmentation")
    if any(parent.get(k)!=cfg.get(k) for k in same):raise ValueError("Corner stage changed parent data/ROI/SAM prediction contract")
    return seal


def validate_recipe(cfg):
    from data.action4_prediction_roi_data import roi_finetune
    if roi_finetune(cfg):
        from .action4_roi115_finetune import validate_recipe as validate_finetune
        return validate_finetune(cfg)
    if cfg.get("training_protocol")!=CORNER_NATIVE_PROTOCOL or not validate_config(cfg,"train"):
        raise ValueError("Registered native corner prediction-ROI protocol required")
    validate_parent_seal(cfg)
    from model.action4_sam_joint import validate_sam_training_config
    validate_sam_training_config(cfg)
    expected={"batch":48,"micro":2,"eval_every":25,"expected_eval_images":650,"eta_min":1e-7,
        "grad_clip_norm":1.,"corner_softmax_bias_frozen":True,"training_data_mode":"independent_images"}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError("Stage4 frozen recipe differs")
    if cfg.get("eval_schedule") is not None:raise ValueError("Stage4 must not inherit stage3 epoch-dependent cadence silently")
    world=cfg.get("world_size")
    if type(world) is not int or world not in (1,2,3,4) or 48%(2*world):raise ValueError("Exact corner global48 partition required")
    units=cfg.get("expected_units")
    if type(units) is not int or units<=0:raise ValueError("Actual positive prediction unit count required")
    if cfg.get("eval_indices") is not None:raise ValueError("Corner evaluation remains the complete650 domain")
    if cfg["mode"]=="full_training":
        if cfg.get("epochs")!=80 or cfg.get("train_indices") is not None or units!=cfg.get("formal_training_units"):
            raise ValueError("Stage4 formal training requires its complete80epoch domain")
    elif cfg["mode"]=="integration":
        if type(cfg.get("epochs")) is not int or not 1<=cfg["epochs"]<=6 or units>256:
            raise ValueError("Stage4 integration bounded to <=6 small epochs and <=256 real units")
        indices=cfg.get("train_indices")
        if (not isinstance(indices,list) or len(indices)!=units or len(set(indices))!=units
            or any(type(i) is not int or i<0 for i in indices)):
            raise ValueError("Stage4 integration requires an exact unique bounded unit list")
    else:raise ValueError("Explicit corner run mode required")
    for key in ("corner_refined_quality_bridge","b0_full_box_quality_receipt"):
        if not isinstance(cfg.get(key),dict) or set(cfg[key])!={"path","sha256"}:raise ValueError("Bound corner quality inputs required")


def validate_admission(cfg,gate):
    from data.action4_prediction_roi_data import roi_finetune
    if roi_finetune(cfg):
        from .action4_roi115_finetune import validate_admission as validate_finetune
        return validate_finetune(cfg,gate)
    validate_recipe(cfg)
    if (gate.get("schema")!="action4.stage4_prediction_roi_admission.v1" or gate.get("launch_allowed") is not True
        or gate.get("mode")!=cfg["mode"] or gate.get("role")!="A0" or gate.get("training_protocol")!=CORNER_NATIVE_PROTOCOL
        or any(gate.get(k)!=cfg.get(k) for k in INPUT_FIELDS)):
        raise ValueError("Corner-specific admission required; prior-stage receipts cannot authorize it")
    if cfg["mode"]=="full_training":
        proof=_v2_checked(cfg["stage4_execution_gate"])
        if (proof.get("status")!="CORNER_PREDICTION_ROI_STAGE4_EXECUTION_PASS" or proof.get("project_code_sha256")!=cfg.get("project_code_sha256")
            or any(proof.get(k)!=cfg.get(k) for k in INPUT_FIELDS if k!="expected_units")
            or proof.get("formal_training_units")!=cfg["expected_units"]):raise ValueError("Corner tested source/input proof differs")
        required=("real_prediction_ROI_consumption","corner_loss_to_SAM_adapter","actual_gradient_and_updates",
          "strict_bundle_reload","frozen_backbone_parameters_and_buffers","global_zero_and_independent_head_guards",
          "full_model_DDP_recovery","full650_eval","async_eval_coverage","whole_checkpoint_quality_selection",
          "unready_parent_fallback","sealed_decoder_contract","throughput_pass")
        if any(proof.get(k) is not True for k in required):raise ValueError("Incomplete corner execution checks")
