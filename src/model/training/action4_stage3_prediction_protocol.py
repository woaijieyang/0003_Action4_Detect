"""Independent SAM-enhancement admission, never a baseline continuation gate."""
from data.action4_prediction_roi_data import SAM_NATIVE_PROTOCOL,validate_config


def validate_parent_seal(cfg):
    from data.action4_sam_native_export import _v2_checked,canonical
    from data.action4_paths import PROJECT_ROOT,contained_path
    import json
    seal=_v2_checked(cfg["stage3_parent_seal"])
    if (seal.get("schema")!="action4.sealed_independent_role.v2" or seal.get("role")!="A0"
        or seal.get("strict_load") is not True or seal.get("frozen_backbone_verified") is not True
        or seal["sha256"]!=cfg.get("parent_bundle_sha256") or seal["selected_step"]!=cfg.get("parent_best_step")):
        raise ValueError("Stage3 parent must be the explicitly selected sealed baseline best")
    original=contained_path(PROJECT_ROOT,seal["bundle"],exists=True)
    consumed=contained_path(PROJECT_ROOT,cfg["parent_bundle"],exists=True)
    if consumed!=original:
        if cfg.get("require_immutable_source") is not True:raise ValueError("Unregistered parent copy")
        current=json.loads(contained_path(PROJECT_ROOT,cfg["run_spec"],exists=True).read_text())
        code_root=contained_path(PROJECT_ROOT,current["code_root"],exists=True)
        if not consumed.is_relative_to(code_root) or current["files"].get(str(consumed.relative_to(code_root)))!=seal["sha256"]:
            raise ValueError("Run snapshot does not bind its copied parent to the selected seal")
    spec=_v2_checked({"path":seal["run_spec"],"sha256":seal["run_spec_sha256"]})
    parent=json.loads(contained_path(PROJECT_ROOT,spec["config"],exists=True).read_text())
    if canonical(parent)!=seal["config_id"]:raise ValueError("Sealed parent configuration changed")
    same=("dataset_root","prediction_roi_manifest","frozen_eval_b0_predictions","evaluation_counterpart_sha256",
          "validation_receipt_sha256","surface_source","roi_source","roi_expansion","roi_size","score_schema","b0_scoring_policy")
    if any(parent.get(k)!=cfg.get(k) for k in same):raise ValueError("Stage3 changed the selected parent prediction/data/ROI contract")
    return seal


def validate_recipe(cfg):
    from model.training.action4_eval_schedule import validate_schedule
    validate_schedule(cfg)
    if cfg.get("training_protocol")!=SAM_NATIVE_PROTOCOL or not validate_config(cfg,"train"):
        raise ValueError("Registered native prediction-ROI SAM stage3 protocol required")
    validate_parent_seal(cfg)
    from model.action4_sam_joint import validate_sam_training_config
    validate_sam_training_config(cfg)
    if cfg.get("training_data_mode")!="independent_images":raise ValueError("Independent image training only")
    for key,value in {"batch":48,"micro":2,"eval_every":25,"expected_eval_images":650,"eta_min":1e-7,"grad_clip_norm":1.}.items():
        if cfg.get(key)!=value:raise ValueError("Stage3 recipe differs: "+key)
    world=cfg.get("world_size")
    if type(world) is not int or world not in (1,2,3,4) or 48%(2*world):raise ValueError("Exact global48 partition required")
    if cfg["mode"]=="full_training" and cfg.get("epochs")!=80:raise ValueError("Independent stage3 formal budget is80epochs")
    if cfg["mode"]=="integration" and (type(cfg.get("epochs")) is not int or not 1<=cfg["epochs"]<=2):raise ValueError("Bounded real-input integration required")
    if cfg.get("sam_mask_augmentation") is None:raise ValueError("Explicit registered mask dropout/perturbation policy required")
    if type(cfg.get("expected_units")) is not int or cfg["expected_units"]<=0:raise ValueError("Actual positive prediction-unit count required")
    if not isinstance(cfg.get("sam_eval_cache"),dict):raise ValueError("Explicit frozen original-image eval SAM cache required")


def validate_admission(cfg,gate):
    validate_recipe(cfg)
    if (gate.get("schema")!="action4.stage3_prediction_roi_admission.v1"
        or gate.get("launch_allowed") is not True or gate.get("mode")!=cfg["mode"]
        or gate.get("role")!="A0" or gate.get("training_protocol")!=SAM_NATIVE_PROTOCOL):
        raise ValueError("Independent stage3 admission required; stage1/baseline receipts are not interchangeable")
    fields=("parent_bundle_sha256","evaluation_counterpart_sha256","prediction_roi_manifest",
            "sam_cache","sam_training_associations","sam_eval_cache","roi_expansion","roi_size",
            "expected_units","expected_eval_images","score_schema")
    if any(gate.get(k)!=cfg.get(k) for k in fields):raise ValueError("Stage3 admission input/parent/model binding differs")
    if cfg["mode"]=="full_training":
        from data.action4_sam_native_export import _v2_checked
        proof=_v2_checked(cfg["stage3_execution_gate"])
        if (proof.get("status")!="SAM_PREDICTION_ROI_STAGE3_EXECUTION_PASS"
            or proof.get("project_code_sha256")!=cfg.get("project_code_sha256")
            or any(proof.get(k)!=cfg.get(k) for k in fields if k!="expected_units")
            or proof.get("formal_training_units")!=cfg["expected_units"]):
            raise ValueError("Stage3 requires its own tested immutable source/input proof")
        required=("real_prediction_ROI_consumption","actual_gradient_and_updates","strict_bundle_reload",
                  "frozen_backbone_parameters_and_buffers","full650_eval","async_eval_coverage","throughput_pass")
        if any(proof.get(k) is not True for k in required):raise ValueError("Incomplete stage3 execution checks")
