"""Fixed50epoch ROI recipe plus explicitly sealed-parent10/20epoch segments."""
import math
import torch
from data.action4_prediction_roi_data import enabled,validate_config,PROTOCOL

EXTENSION_SCHEMA="action4.roi15_best_extension.v1"
EXTENSION20_SCHEMA="action4.roi15_best_extension.v2"

def extension_segment(cfg):
    segment=cfg.get("continuation_segment")
    if segment is None:return False
    twenty=isinstance(segment,dict) and segment.get("schema")==EXTENSION20_SCHEMA
    epochs=20 if twenty else 10
    keys={"schema","epochs","batch","eval_every","parent_seal"}|({"global_selection"} if twenty else set())
    if (not isinstance(segment,dict) or set(segment)!=keys
            or segment["schema"] not in (EXTENSION_SCHEMA,EXTENSION20_SCHEMA) or segment["epochs"]!=epochs
            or segment["batch"]!=64 or segment["eval_every"]!=10
            or not isinstance(segment["parent_seal"],dict) or set(segment["parent_seal"])!={"path","sha256"}):
        raise ValueError("Only explicitly registered10/20epoch best extensions are allowed")
    for key,value in {"batch":64,"micro":2,"world_size":4,"eval_every":10,"formal_epochs":epochs,
                      "expected_eval_images":650}.items():
        if type(cfg.get(key)) is not int or cfg[key]!=value:raise ValueError("Extension budget differs: "+key)
    formal_units=cfg.get("formal_training_units")
    if type(formal_units) is not int or formal_units<=0 or cfg.get("schedule_total_updates")!=epochs*math.ceil(formal_units/64):
        raise ValueError("Extension scheduler must use complete parent formal unit count")
    if type(cfg.get("expected_units")) is not int or not 0<cfg["expected_units"]<=formal_units:
        raise ValueError("Explicit positive integration/formal unit count required")
    if cfg["mode"]=="full_training" and cfg["expected_units"]!=formal_units:
        raise ValueError("Formal extension must consume all parent training units")
    if cfg["mode"]=="full_training" and cfg["epochs"]!=epochs:
        raise ValueError("Formal extension must match its exact registered epoch budget")
    if cfg["mode"]=="integration" and (type(cfg.get("epochs")) is not int or not 1<=cfg["epochs"]<=epochs):
        raise ValueError("Integration must retain the registered formal segment budget")
    if cfg["mode"] not in ("integration","full_training"):raise ValueError("Explicit extension mode required")
    return True

def validate_extension_parent(cfg):
    """Authenticate selected50epoch seal and invariant inputs; loader still strict-loads actual weights."""
    if not extension_segment(cfg):return None
    import hashlib,json
    from data.action4_paths import PROJECT_ROOT,contained_path
    from data.action4_sam_native_export import _v2_checked
    segment=cfg["continuation_segment"]
    selection=None
    if segment["schema"]==EXTENSION20_SCHEMA:
        from data.action4_extension_selection import validate_global_selection
        selection=validate_global_selection(segment["global_selection"])
        if segment["parent_seal"]!=selection["receipt"]["winner_seal"]:
            raise ValueError("20epoch parent must be the authenticated cross-run winner")
    seal=_v2_checked(segment["parent_seal"])
    expected={"schema":"action4.sealed_independent_role.v2","role":"A0","epochs":selection["winner_config"]["epochs"] if selection else 50,
        "strict_load":True,"frozen_backbone_verified":True}
    if any(seal.get(k)!=v for k,v in expected.items()):
        raise ValueError("Extension requires the authenticated completed50epoch selected-best seal")
    if cfg.get("parent_bundle_sha256")!=seal["sha256"] or cfg.get("parent_best_step")!=seal["selected_step"]:
        raise ValueError("Extension parent best identity differs")
    # Actual weights are verified by the existing parent registry/strict loader.
    # Do not rehash a large bundle in every rank/recipe validation.
    for bundle in (seal["bundle"],cfg["parent_bundle"]):
        contained_path(PROJECT_ROOT,bundle,exists=True)
    spec=_v2_checked({"path":seal["run_spec"],"sha256":seal["run_spec_sha256"]})
    parent=json.loads(contained_path(PROJECT_ROOT,spec["config"],exists=True).read_text())
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest(parent)!=seal["config_id"] or (selection is None and parent.get("continuation_segment") is not None):
        raise ValueError("Parent sealed config identity/lineage differs")
    parent_units=parent.get("expected_units")
    parent_epochs=parent["epochs"];parent_batch=parent["batch"];parent_interval=parent["eval_every"]
    if ((selection is None and (parent_epochs,parent_batch,parent_interval)!=(50,48,25))
            or (selection is not None and parent!=selection["winner_config"])
            or cfg["formal_training_units"]!=parent_units):
        raise ValueError("Extension formal unit count differs from completed parent recipe")
    indices=cfg.get("train_indices")
    if cfg["mode"]=="full_training" and indices is not None:
        raise ValueError("Formal extension cannot subset/reorder training units")
    if cfg["mode"]=="integration":
        if indices is None:
            if cfg["expected_units"]!=parent_units:raise ValueError("Reduced integration requires explicit train_indices")
        elif (not isinstance(indices,list) or len(indices)!=cfg["expected_units"]
                or len(set(indices))!=len(indices)
                or any(type(i) is not int or not 0<=i<parent_units for i in indices)):
            raise ValueError("Integration indices must be unique in-range parent units")
    parent_total=parent_epochs*math.ceil(parent_units/parent_batch)
    parent_evals=len(set([0,parent_total]+list(range(parent_interval,parent_total+1,parent_interval))))
    if (seal.get("completed_optimizer_steps")!=parent_total
            or seal.get("completion",{}).get("evaluations")!=parent_evals
            or seal["completion"].get("best_step")!=seal["selected_step"]):
        raise ValueError("Parent completion/best coverage differs")
    keys=("training_protocol","dataset_root","seed","network","preprocess","roi_source","roi_expansion","roi_size",
          "prediction_roi_manifest","frozen_eval_b0_predictions","evaluation_counterpart_sha256",
          "validation_receipt","validation_receipt_sha256","surface_source","train_backbone",
          "a0_variant","sam_enabled","roi_jitter","photometric","geometry_batch_policy","loss_weights",
          "independent_head_dependencies","b0_scoring_policy","evaluation_scoring_policy","score_schema",
          "a0_selection_contract","surface_lr","lr","weight_decay","betas","eps","grad_clip_norm",
          "warmup_updates","eta_min")
    if any(cfg.get(k)!=parent.get(k) for k in keys):
        raise ValueError("Extension changed frozen parent recipe/data/B0/ROI inputs")
    contract=seal["completion"].get("inference_contract",{})
    if (contract.get("B0_sha256")!=cfg["evaluation_counterpart_sha256"]
            or contract.get("prediction_roi_manifest")!=cfg["prediction_roi_manifest"]
            or contract.get("frozen_eval_b0_predictions")!=cfg["frozen_eval_b0_predictions"]):
        raise ValueError("Extension parent inference input contract differs")
    return seal

def validate_recipe(cfg):
    validate_config(cfg,"train")
    extension=extension_segment(cfg)
    expected={"batch":64 if extension else 48,"micro":2,"surface_lr":2e-6,"lr":1e-5,"weight_decay":1e-4,
              "eval_every":10 if extension else 25,"warmup_updates":200,"eta_min":1e-7,"grad_clip_norm":1.0}
    for k,v in expected.items():
        if cfg.get(k)!=v:raise ValueError("Fixed ROI adaptation recipe differs: "+k)
    if tuple(cfg.get("betas",()))!=(.9,.999) or cfg.get("eps")!=1e-8:raise ValueError("AdamW recipe differs")
    if cfg["batch"]%(cfg["micro"]*cfg["world_size"]):raise ValueError("Global batch must equal micro x ranks x accumulation")
    if not extension and cfg["mode"]=="full_training" and cfg["epochs"]!=50:
        raise ValueError("This formal adaptation is exactly50epochs")
    if extension:validate_extension_parent(cfg)

def lr_factor(index,*,peak,total,warmup=200,minimum=1e-7):
    if total<=warmup or peak<=minimum or index<0:raise ValueError("Explicit feasible warmup/cosine budget required")
    if index<warmup:return .1+.9*index/(warmup-1)
    progress=min(1.,(index-(warmup-1))/(total-warmup))
    return (minimum+(peak-minimum)*.5*(1+math.cos(math.pi*progress)))/peak

def make_scheduler(optimizer,cfg,actual_total):
    from data.action4_prediction_roi_data import roi_finetune
    finetune=roi_finetune(cfg)
    if not enabled(cfg) and not finetune:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,actual_total,eta_min=min(g["lr"] for g in optimizer.param_groups)*.01)
    if finetune:
        from .action4_roi115_finetune import validate_recipe as validate_finetune
        validate_finetune(cfg)
    else:validate_recipe(cfg)
    total=cfg["schedule_total_updates"]
    if cfg["mode"]=="full_training" and total!=actual_total:raise ValueError("Scheduler budget differs from actual frozen unit manifest")
    return torch.optim.lr_scheduler.LambdaLR(optimizer,[
        (lambda step,peak=g["lr"]:lr_factor(step,peak=peak,total=total,warmup=cfg["warmup_updates"],minimum=cfg["eta_min"]))
        for g in optimizer.param_groups])

def validate_admission(cfg,gate):
    validate_recipe(cfg)
    if gate.get("launch_allowed") is not True or gate.get("mode")!=cfg["mode"] or gate.get("training_protocol")!=PROTOCOL:
        raise ValueError("Explicit adaptation admission required")
    if extension_segment(cfg) and gate.get("continuation_segment")!=cfg["continuation_segment"]:
        raise ValueError("Extension admission must bind the exact parent/segment")
    if cfg["mode"]=="full_training":
        from data.action4_sam_native_export import _v2_checked
        proof=_v2_checked(cfg["adaptation_execution_gate"])
        if (proof.get("status")!="ROI_ADAPTATION_EXECUTION_PASS" or proof["project_code_sha256"]!=cfg["project_code_sha256"]
                or proof["prediction_roi_manifest"]!=cfg["prediction_roi_manifest"] or proof["roi_expansion"]!=1.5
                or proof["parent_A0_sha256"]!=cfg["parent_bundle_sha256"]):
            raise ValueError("Untested adaptation source/input/parent")
        if extension_segment(cfg) and (proof.get("continuation_segment")!=cfg["continuation_segment"]
                or gate.get("continuation_segment")!=cfg["continuation_segment"]):
            raise ValueError("Extension admission did not bind the exact parent/segment")
        if gate["expected_units"]!=cfg["expected_units"] or gate["prediction_roi_manifest"]!=cfg["prediction_roi_manifest"]:
            raise ValueError("Actual prediction-unit admission differs")

def validate_parent_input_contract(parent_config,cfg):
    """A new stage cannot silently undo a selected prediction-ROI adaptation."""
    from data.action4_prediction_roi_data import SAM_NATIVE_PROTOCOL,CORNER_NATIVE_PROTOCOL
    protocol=parent_config.get("training_protocol")
    if protocol not in (PROTOCOL,SAM_NATIVE_PROTOCOL,CORNER_NATIVE_PROTOCOL):
        return
    if (parent_config.get("roi_source")!="frozen_b0_prediction" or parent_config.get("roi_expansion")!=1.5):
        raise ValueError("Malformed prediction-ROI parent contract")
    from data.action4_prediction_roi_data import roi_finetune
    expansion=1.15 if roi_finetune(cfg) else 1.5
    if (cfg.get("roi_source")!="frozen_b0_prediction" or cfg.get("roi_expansion")!=expansion
            or cfg.get("preprocess",{}).get("roi_expansion")!=expansion):
        raise ValueError("Prediction-ROI parent cannot silently return to GT/2.25 crops")
    # Future enhanced recipes must register and test their actual dataset path.
    # Declaring the source without a consumer is not sufficient.
    if not validate_config(cfg,"train"):
        raise ValueError("New stage has no registered prediction-ROI data consumer")
    if protocol in (SAM_NATIVE_PROTOCOL,CORNER_NATIVE_PROTOCOL):
        if cfg.get("training_protocol") not in (SAM_NATIVE_PROTOCOL,CORNER_NATIVE_PROTOCOL):
            raise ValueError("Native SAM parent requires a registered SAM prediction consumer")
        keys=("evaluation_counterpart_sha256","prediction_roi_manifest","frozen_eval_b0_predictions",
              "sam_cache","sam_eval_cache","sam_training_associations","b0_scoring_policy")
        if any(parent_config.get(k) is None or parent_config.get(k)!=cfg.get(k) for k in keys):
            raise ValueError("Native SAM parent frozen B0/data/cache identities changed")
