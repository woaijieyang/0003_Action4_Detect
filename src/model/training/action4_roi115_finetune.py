"""Explicit new75epoch ROI115 segment from a strictly sealed complete corner bundle."""
import json,math
from data.action4_prediction_roi_data import roi_finetune,validate_config,CORNER_NATIVE_PROTOCOL
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_checked,canonical

def validate_parent(cfg):
    if not roi_finetune(cfg):raise ValueError("Explicit ROI115 segment required")
    seal=_v2_checked(cfg["stage4_parent_seal"])
    if (seal.get("schema")!="action4.sealed_corner_all8_research.v1"
        or seal.get("status")!="SEALED_USER_STOPPED_ALL8_RESEARCH" or seal.get("selected_route")!="corner_candidate"
        or seal.get("strict_load") is not True or seal.get("frozen_backbone_verified") is not True
        or seal.get("sha256")!=cfg.get("parent_bundle_sha256")
        or seal.get("selected_step")!=cfg.get("parent_best_step") or seal.get("roi_expansion")!=1.5):
        raise ValueError("ROI115 requires the selected strictly sealed complete stage4 corner parent")
    spec=_v2_checked(seal["run_spec"])
    parent=json.loads(contained_path(R,spec["config"],exists=True).read_text())
    if canonical(parent)!=seal["config_id"] or parent.get("training_protocol")!=CORNER_NATIVE_PROTOCOL:
        raise ValueError("Sealed stage4 source config changed")
    if seal["model_B0"]["sha256"]!=cfg.get("evaluation_counterpart_sha256"):
        raise ValueError("Frozen parent B0 changed")
    same=("dataset_root","prediction_roi_manifest","frozen_eval_b0_predictions","evaluation_counterpart_sha256",
          "validation_receipt_sha256","surface_source","roi_source","roi_size","score_schema","b0_scoring_policy",
          "sam_cache","sam_eval_cache","sam_training_associations","sam_mask_augmentation",
          "corner_protocol","loss_weights","independent_head_dependencies","train_backbone")
    if any(parent.get(k)!=cfg.get(k) for k in same):
        raise ValueError("ROI115 changed frozen data/cache/model or loss contract")
    consumed=contained_path(R,cfg["parent_bundle"],exists=True)
    original=contained_path(R,seal["bundle"],exists=True)
    if consumed!=original:
        if cfg.get("require_immutable_source") is not True:raise ValueError("Unbound stage4 parent copy")
        current=json.loads(contained_path(R,cfg["run_spec"],exists=True).read_text())
        code=contained_path(R,current["code_root"],exists=True)
        if not consumed.is_relative_to(code) or current["files"].get(str(consumed.relative_to(code)))!=seal["sha256"]:
            raise ValueError("RunSpec does not bind stage4 parent copy")
    return seal

def validate_recipe(cfg):
    if not roi_finetune(cfg) or not validate_config(cfg,"train"):raise ValueError("ROI115 recipe required")
    validate_parent(cfg)
    from model.action4_sam_joint import validate_sam_training_config
    validate_sam_training_config(cfg)
    expected={"batch":48,"micro":2,"eval_every":10,"expected_eval_images":650,"eta_min":1e-7,
        "grad_clip_norm":1.,"corner_softmax_bias_frozen":True,"training_data_mode":"independent_images",
        "warmup_updates":200,"formal_epochs":75,"pose_lr":1e-5}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError("ROI115 frozen segment recipe differs")
    if cfg.get("eval_schedule") is not None or cfg.get("eval_indices") is not None:raise ValueError("Complete650 fixed10step evaluation required")
    world=cfg.get("world_size")
    if type(world) is not int or world not in (1,2,3,4) or 48%(2*world):raise ValueError("Exact global48 partition required")
    units=cfg.get("expected_units");formal=cfg.get("formal_training_units")
    if type(units) is not int or type(formal) is not int or not 0<units<=formal:raise ValueError("Actual positive training unit counts required")
    if cfg.get("schedule_total_updates")!=75*math.ceil(formal/48):raise ValueError("75epoch budget must use actual frozen manifest N")
    if cfg["mode"]=="full_training":
        if cfg.get("epochs")!=75 or cfg.get("train_indices") is not None or units!=formal:raise ValueError("Complete75epoch domain required")
    elif cfg["mode"]=="integration":
        indices=cfg.get("train_indices")
        if (type(cfg.get("epochs")) is not int or not 1<=cfg["epochs"]<=6 or units>256
            or not isinstance(indices,list) or len(indices)!=units or len(set(indices))!=units
            or any(type(i) is not int or not 0<=i<formal for i in indices)):
            raise ValueError("Bounded explicit real-unit integration required")
    else:raise ValueError("Explicit run mode required")

def validate_admission(cfg,gate):
    validate_recipe(cfg)
    from .action4_stage4_prediction_protocol import INPUT_FIELDS
    if (gate.get("schema")!="action4.stage4_prediction_roi_admission.v1" or gate.get("launch_allowed") is not True
        or gate.get("mode")!=cfg["mode"] or gate.get("role")!="A0"
        or gate.get("training_protocol")!=CORNER_NATIVE_PROTOCOL or gate.get("roi_finetune")!=cfg["roi_finetune"]
        or any(gate.get(k)!=cfg.get(k) for k in INPUT_FIELDS)):
        raise ValueError("New segment admission must bind actual ROI115 parent and inputs")
    if cfg["mode"]=="full_training":
        proof=_v2_checked(cfg["stage4_execution_gate"])
        if (proof.get("status")!="ROI115_FINETUNE_EXECUTION_PASS"
            or proof.get("project_code_sha256")!=cfg.get("project_code_sha256")
            or proof.get("parent_bundle_sha256")!=cfg.get("parent_bundle_sha256")
            or proof.get("roi_finetune")!=cfg.get("roi_finetune")
            or proof.get("formal_training_units")!=cfg.get("formal_training_units")):
            raise ValueError("ROI115 execution evidence differs")
        for key in ("strict_complete_parent_load","real_gradient_update","strict_reload","frozen_backbone",
                    "same_roi115_train_eval","actual_eval","zero_supervision_skip"):
            if proof.get(key) is not True:raise ValueError("Missing actual ROI115 validation: "+key)
