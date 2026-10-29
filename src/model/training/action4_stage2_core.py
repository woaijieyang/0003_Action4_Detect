"""Shared independent-role loss, best initialization and self-contained snapshot contract."""
import copy
import torch
from model.action4_joint_surface_rgb import JointSurfaceRGB
from model.action4_release_bundle import load_bundle,SCHEMA,SAM_SCHEMA,B0_GUIDANCE_SCHEMA,CORNER_SCHEMA
from model.action4_sparse_surface_losses import sparse_surface_losses
from model.action4_pose_maps import geometry_losses
from model.training.action4_online_b0_batch import batch_losses
from model.training.action4_runtime import canonical_hash

def construct(cfg,device):
    from data.action4_image_mode_contract import ensure_image_mode
    ensure_image_mode(cfg)
    from data.action4_sam_roi_routing import validate_config as validate_roi_routing_config
    validate_roi_routing_config(cfg,"train")
    from model.action4_sam_joint import validate_mask_augmentation_config
    validate_mask_augmentation_config(cfg)
    role=cfg["role"]
    pinned=None
    if cfg["mode"]=="full_training":
        from model.action4_best_registry import verify_reference,DEFAULT_REGISTRY
        pinned=verify_reference(role,cfg["parent_bundle"],cfg["parent_best_step"],cfg.get("parent_registry",DEFAULT_REGISTRY))
    if role=="A0":
        from model.action4_sam_joint import sam_variant, validate_sam_training_config,corner_variant
        validate_sam_training_config(cfg)
        if corner_variant(cfg):
            import hashlib
            from data.action4_paths import PROJECT_ROOT,contained_path
            refs=list(cfg["corner_counterpart"].values())
            for ref in refs:
                with contained_path(PROJECT_ROOT,ref["path"],exists=True).open("rb") as stream:
                    if hashlib.file_digest(stream,"sha256").hexdigest()!=ref["sha256"]:raise ValueError("Corner parent/counterpart source changed")
        model=JointSurfaceRGB(cfg["parent_bundle"],cfg["train_backbone"],device,enable_sam=sam_variant(cfg),enable_corner=corner_variant(cfg),parent_bundle_sha256=cfg.get("parent_bundle_sha256"))
        parent=model.parent
        from model.training.action4_roi_adaptation import validate_parent_input_contract
        validate_parent_input_contract(parent["config"],cfg)
        if corner_variant(cfg):
            model.corner_head.localization.heatmap.bias.requires_grad_(not cfg.get("corner_softmax_bias_frozen",False))
            model._corner_config_identity=canonical_hash(cfg)
    elif role=="B0":
        from model.action4_b0_fusion import enabled,GuidedB0
        model,_,parent=load_bundle(cfg["parent_bundle"],"B0",device)
        if enabled(cfg):
            if "guidance_lr" not in cfg or cfg["guidance_lr"]<=0:raise ValueError("Explicit guidance learning rate required")
            if not isinstance(model,GuidedB0):
                baseline=model;model=GuidedB0(cfg["network"])
                model.detector.load_state_dict(baseline.detector.state_dict(),strict=True)
                model.refiner.load_state_dict(baseline.refiner.state_dict(),strict=True)
        elif isinstance(model,GuidedB0):raise ValueError("Cannot silently discard guidance parent")
    else:raise ValueError("Independent A0/B0 role required")
    if parent["selected_step"]!=cfg["parent_best_step"]:raise ValueError("Selected best parent mismatch")
    if role=="B0" and parent["config"]["network"]!=cfg["network"]:raise ValueError("B0 architecture changed")
    if pinned is not None:
        from model.action4_best_registry import verify_loaded
        verify_loaded(pinned,parent)
    model=model.to(device)
    if role=="B0":
        from model.action4_b0_fusion import GuidedB0
        if isinstance(model,GuidedB0):
            import hashlib
            from data.action4_paths import PROJECT_ROOT as R,contained_path
            from model.action4_guidance_contract import SourceIdentity
            parent_sha=hashlib.sha256(contained_path(R,cfg["parent_bundle"],exists=True).read_bytes()).hexdigest()
            source_sha=cfg.get("project_code_sha256")
            if source_sha is None:
                from pathlib import Path
                source_sha=canonical_hash({name:hashlib.sha256((Path(__file__).resolve().parents[2]/name).read_bytes()).hexdigest() for name in
                    ('model/action4_b0_fusion.py','model/action4_b0_fusion_proposals.py','model/action4_b0_guidance_heads.py','model/action4_online_b0.py','model/action4_proposal_pool.py','model/action4_local_evidence.py','model/action4_guidance_contract.py')})
            model.bind_prediction_source(SourceIdentity('b0',parent_sha,source_sha,
                weight_identity_kind='initialization_sha256_plus_update',run_config_sha256=canonical_hash(cfg),optimizer_step=0))
    return model,parent

def optimizer_for(model,cfg):
    if cfg["role"]=="A0":
        from model.action4_sam_joint import sam_variant, validate_sam_training_config
        enabled=sam_variant(cfg)
        if enabled!=bool(getattr(model,"enable_sam",False)):raise ValueError("Optimizer/model SAM variant mismatch")
        validate_sam_training_config(cfg)
        from model.action4_sam_joint import corner_variant
        if corner_variant(cfg)!=bool(getattr(model,"enable_corner",False)):raise ValueError("Optimizer corner ownership mismatch")
        if corner_variant(cfg) and getattr(model,"_corner_config_identity",None)!=canonical_hash(cfg):raise ValueError("Constructed corner configuration changed")
        groups=[{"params":model.head.surface.parameters(),"lr":cfg["surface_lr"]},
                {"params":model.head.pose.parameters(),"lr":cfg["lr"]}]
        if model.train_backbone:groups.append({"params":model.base.backbone.parameters(),"lr":cfg["lr_backbone"]})
        if enabled:
            groups[0]["name"]="surface";groups[1]["name"]="pose"
            groups.extend([
                {"name":"sam_adapter","params":list(model.sam_adapter.encoder.parameters())+list(model.sam_adapter.projection.parameters()),"lr":cfg["sam_adapter_lr"]},
                {"name":"sam_quality","params":model.sam_adapter.quality_head.parameters(),"lr":cfg["sam_quality_lr"]}])
        if corner_variant(cfg):
            groups.extend([
                {"name":"corner_localization","params":[p for module in (model.corner_head.localization,model.corner_head.valid_head,model.corner_head.scale_head) for p in module.parameters() if p.requires_grad],"lr":cfg["corner_lr"]},
                {"name":"corner_quality","params":model.corner_head.quality_head.parameters(),"lr":cfg["corner_quality_lr"]}])

    else:
        groups=[{"params":model.detector.backbone.parameters(),"lr":cfg["lr_backbone"]},
                {"params":[p for n,p in model.detector.named_parameters() if not n.startswith("backbone.")],"lr":cfg["lr_detector"]},
                {"params":model.refiner.parameters(),"lr":cfg["lr_refiner"]}]
    if cfg["role"]=="B0":
        from model.action4_b0_fusion import enabled,GuidedB0
        if enabled(cfg)!=isinstance(model,GuidedB0):raise ValueError("B0 optimizer variant mismatch")
        if enabled(cfg):groups.append({"name":"guidance","params":model.guidance.parameters(),"lr":cfg["guidance_lr"]})
    from data.action4_prediction_roi_data import enabled as prediction_adaptation
    if prediction_adaptation(cfg):
        from model.training.action4_roi_adaptation import validate_recipe
        validate_recipe(cfg)
        groups[0]["name"]="surface";groups[1]["name"]="pose"
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg.get("weight_decay",.0001),
        betas=tuple(cfg.get("betas",(.9,.999))),eps=cfg.get("eps",1e-8))
    if prediction_adaptation(cfg):
        owned=[id(p) for group in optimizer.param_groups for p in group["params"]]
        expected={id(p) for p in model.parameters() if p.requires_grad}
        if len(owned)!=len(set(owned)) or set(owned)!=expected:raise ValueError("Adaptation optimizer ownership differs")
    if (cfg["role"]=="A0" and enabled) or (cfg["role"]=="B0" and isinstance(model,GuidedB0)):
        owned=[id(p) for group in optimizer.param_groups for p in group["params"]]
        expected={id(p) for p in model.parameters() if p.requires_grad}
        if len(owned)!=len(set(owned)) or set(owned)!=expected:
            raise ValueError("Enhanced optimizer has duplicated/missing trainable parameters")
    return optimizer

def losses_on_batch(model,items,cfg,device,corner_normalizers=None):
    if cfg["role"]=="B0":
        from model.action4_b0_fusion import enabled
        if enabled(cfg):raise ValueError("GuidedB0 needs candidate provider and global eligible guidance loss sums; baseline losses refused")
        return batch_losses(model,[x["item"] for x in items],cfg,device,
                            augmentation_seeds=[x["augmentation_seed"] for x in items] if cfg.get("photometric") else None)
    from model.action4_sam_joint import sam_variant
    if sam_variant(cfg) or bool(getattr(model.module if hasattr(model,"module") else model,"enable_sam",False)):
        raise ValueError("SAM requires loss_sums_on_batch and globally reduced eligible counts")
    rois=torch.stack([x["roi"] for x in items]).to(device)
    geometry={k:torch.stack([x["geometry"][k] for x in items]).to(device) for k in items[0]["geometry"]}
    targets={"masks":torch.stack([x["surface_masks"] for x in items]).to(device),
             "sparse":[{k:v.to(device) for k,v in x["surface_sparse"].items()} for x in items]}
    output=model(rois,geometry["K"],geometry["canonical"])
    loss=sparse_surface_losses(output["surface"],targets)
    loss.update(geometry_losses(output,geometry,corner_normalizers))
    return loss

def loss_sums_on_batch(model,items,cfg,device,*,epoch_one_based=None):
    from model.training.action4_supervision import a0_loss_sums
    rois=torch.stack([x['roi'] for x in items]).to(device)
    K=torch.stack([x['geometry']['K'] for x in items]).to(device)
    cad=torch.stack([x['geometry']['canonical'] for x in items]).to(device)
    from model.action4_sam_joint import sam_variant, stack_sam_items, QUALITY_KEY, sam_quality_eligible
    enabled=sam_variant(cfg)
    actual=model.module if hasattr(model,"module") else model
    if enabled!=bool(getattr(actual,"enable_sam",False)):raise ValueError("Loss/model SAM variant mismatch")
    if not enabled:
        if any(x.get("sam_enabled",False) or "mask_aux" in x for x in items):
            raise ValueError("SAM batch evidence requires explicit enhanced variant")
        return a0_loss_sums(model(rois,K,cad),items,device)
    aux,identities,gt,supervised=stack_sam_items(items,device,rois.dtype)
    from model.action4_sam_joint import corner_variant
    corner=corner_variant(cfg)
    if corner!=bool(getattr(actual,"enable_corner",False)):raise ValueError("Corner loss/model variant mismatch")
    corner_stats=None
    if corner:
        from model.training.action4_corner_targets import corner_targets_from_items,corner_target_counts
        from model.action4_corner_supervision import corner_loss_sums
        update_loss_weights(cfg,epoch_one_based)
        targets=corner_targets_from_items(items,device=device,dtype=rois.dtype)
        candidate=model(rois,K,cad,mask_aux=aux,identities=identities,return_corner=True)
        output,sam_output=candidate.raw_pose,candidate.sam_aux
        corner_stats=corner_loss_sums(candidate.corners,targets,epoch=epoch_one_based)
        if corner_stats.counts!=corner_target_counts(targets):raise ValueError("Corner loss/count eligibility mismatch")
    else:
        output,sam_output=model(rois,K,cad,mask_aux=aux,identities=identities,return_aux=True)
    sums=a0_loss_sums(output,items,device)
    from model.action4_sam_feature_adapter import mask_quality_loss
    quality=mask_quality_loss(sam_output,gt,gt_identities=tuple(x["gt_identity"] for x in items),supervised=supervised)
    if int(quality.count)!=sum(sam_quality_eligible(x) for x in items):
        raise ValueError("SAM quality sum/count eligibility mismatch")
    sums[QUALITY_KEY]=quality.loss_sum
    if corner_stats is not None:sums.update(corner_stats.sums)
    return sums

def update_loss_weights(cfg,epoch_one_based):
    """Effective weights for one whole optimizer update; no config mutation."""
    from model.action4_sam_joint import corner_variant,validate_corner_config
    weights=dict(cfg.get("loss_weights",{}))
    if corner_variant(cfg):
        from model.action4_corner_supervision import effective_weights
        validate_corner_config(cfg)
        weights.update(effective_weights(epoch_one_based))
    return weights


def objective(losses,cfg,active_count,global_count,world_size):
    # DDP averages gradients. Surface/box terms are per-object means; corner terms
    # already divide by global visible/hidden counts. Zero-weight padding is explicit.
    if global_count<=0 or active_count<0:raise ValueError("Invalid optimizer batch mass")
    return sum(cfg.get("loss_weights",{}).get(k,1.)*v*(
        (world_size if active_count else 0.) if cfg["role"]=="A0" and k in ("visible_corners","hidden_corners")
        else world_size*active_count/global_count) for k,v in losses.items())

def bundle_snapshot(model,parent,cfg,step):
    # CPU values are independent of subsequent optimizer updates.
    cpu=lambda state:{k:v.detach().cpu().clone() for k,v in state.items()}
    common={"schema":SCHEMA,"role":cfg["role"],"config":copy.deepcopy(cfg),"config_id":canonical_hash(cfg),
            "selected_step":int(step),"parent_selected_step":parent["selected_step"],"parent_bundle":cfg["parent_bundle"],
            "training_scope":cfg["mode"],"exact_optimizer_resume":False,
            "continuation":"best weights with fresh optimizer; exact recovery stored separately",
            "selection_scope":"snapshot awaiting development evaluation","resumable":False}
    if cfg["role"]=="A0":
        from model.action4_sam_joint import sam_variant, validate_sam_training_config, SAM_ARCHITECTURE, BUNDLE_PROTOCOL
        enabled=sam_variant(cfg)
        if enabled!=bool(getattr(model,"enable_sam",False)):raise ValueError("Bundle/model SAM variant mismatch; refusing parameter loss")
        validate_sam_training_config(cfg)
        common.update(architecture=SAM_ARCHITECTURE if enabled else "joint_pose",model=cpu(model.head.state_dict()),backbone=cpu(model.base.state_dict()),
            backbone_config=copy.deepcopy(parent["backbone_config"]),
            backbone_step=step if model.train_backbone else parent["backbone_step"],
            backbone_parent_step=parent["backbone_step"],backbone_frozen=not model.train_backbone,
            surface_frozen=False,pose_method="network")
        if enabled:
            common["schema"]=SAM_SCHEMA
            common["sam_adapter"]={k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                                   for k,v in model.sam_adapter.state_dict().items()}
            common["sam_protocol"]=copy.deepcopy(BUNDLE_PROTOCOL)
        from model.action4_sam_joint import corner_variant,CORNER_ARCHITECTURE,CORNER_PROTOCOL
        if corner_variant(cfg)!=bool(getattr(model,"enable_corner",False)):raise ValueError("Bundle corner ownership mismatch")
        if corner_variant(cfg) and getattr(model,"_corner_config_identity",None)!=canonical_hash(cfg):raise ValueError("Constructed corner configuration changed")
        if corner_variant(cfg):
            common.update(schema=CORNER_SCHEMA,architecture=CORNER_ARCHITECTURE,
                corner_head=cpu(model.corner_head.state_dict()),corner_protocol=copy.deepcopy(CORNER_PROTOCOL),
                corner_counterpart=copy.deepcopy(cfg["corner_counterpart"]),
                corner_parent={"path":cfg["parent_bundle"],"sha256":cfg["parent_bundle_sha256"],"selected_step":cfg["parent_best_step"]},
                quality_ready=False,deployment_admitted=False,pose_method="network_raw_corner_research_only",
                selection_scope="research candidate only; decoder/quality-ready/async admission pending")

    else:
        from model.action4_b0_fusion import enabled,GuidedB0,ARCHITECTURE,PROTOCOL
        if enabled(cfg)!=isinstance(model,GuidedB0):raise ValueError("Refuse B0 guidance parameter loss")
        common.update(architecture=ARCHITECTURE if enabled(cfg) else "online_b0",model=cpu(model.state_dict()),inference_scoring_policy=cfg["evaluation_scoring_policy"])
        if enabled(cfg):common.update(schema=B0_GUIDANCE_SCHEMA,guidance_protocol=copy.deepcopy(PROTOCOL))
    return common


def validate_run_gate(cfg,gate):
    """An integration receipt cannot silently authorize a full-data training run."""
    from data.action4_prediction_roi_data import enabled as prediction_adaptation,SAM_NATIVE_PROTOCOL
    from data.action4_prediction_roi_data import CORNER_NATIVE_PROTOCOL
    if cfg.get("training_protocol")==CORNER_NATIVE_PROTOCOL:
        from model.training.action4_stage4_prediction_protocol import validate_admission
        validate_admission(cfg,gate)
        return
    if cfg.get("training_protocol")==SAM_NATIVE_PROTOCOL:
        from model.training.action4_stage3_prediction_protocol import validate_admission
        validate_admission(cfg,gate)
        return
    if prediction_adaptation(cfg):
        from model.training.action4_roi_adaptation import validate_admission
        validate_admission(cfg,gate)
        return
    from data.action4_sam_roi_routing import validate_config as validate_roi_routing_config
    validate_roi_routing_config(cfg,"train")
    from model.action4_sam_joint import validate_mask_augmentation_config
    validate_mask_augmentation_config(cfg)
    if cfg.get('a0_selection_contract') is not None:
        if (cfg.get('a0_selection_contract')!='full_all8_training_candidate_v1' or cfg.get('role')!='A0'
                or cfg.get('a0_variant')=='a0_corner_image' or cfg.get('score_schema')!='action4.pose_projection_all8_002.v1'):
            raise ValueError('Explicit non-corner A0 All8 candidate selection required')
    if cfg.get("a0_variant")=="a0_corner_image" or cfg.get("corner_enabled",False):
        raise ValueError("Corner candidate: scheduled training refused until decoder/quality-ready/async admission")
    if cfg.get("b0_variant")=="b0_fusion_image":
        raise ValueError("Guidance component wired; full candidate provider/global loss/eval admission not complete")
    if not gate.get("launch_allowed") or gate.get("mode")!=cfg["mode"]:
        raise ValueError("Run mode differs from its explicit launch/admission gate")
    if cfg["mode"]=="full_training":
        if cfg.get('dataset_root')=='data/phrase2':
            import json,hashlib
            from data.action4_paths import PROJECT_ROOT as R,contained_path
            proof=json.loads(contained_path(R,cfg['execution_gate'],exists=True).read_text())
            if proof.get('status')!='STAGE1_EXECUTION_PASS' or proof.get('role')!=cfg['role']:
                raise ValueError('Missing stage1 engineering/integration/throughput gate')
            if proof['project_code_sha256']!=cfg['project_code_sha256']:
                raise ValueError('Formal code was not validated by the recorded integration')
            if hashlib.sha256(contained_path(R,cfg['validation_receipt'],exists=True).read_bytes()).hexdigest()!=cfg['validation_receipt_sha256']:
                raise ValueError('Data receipt changed after run freeze')
        if gate.get("formal_stage2_allowed") is not True:raise ValueError("Formal second stage not admitted")
        if gate.get("validation_receipt")!=cfg["validation_receipt"] or gate.get("role_units",{}).get(cfg["role"])!=cfg["expected_units"]:
            raise ValueError("Formal dataset/role coverage differs from admission")
        if gate.get("eval_images")!=cfg["expected_eval_images"]:raise ValueError("Evaluation dataset differs from admission")
        if cfg["role"]=="A0" and gate.get("surface_source")!=cfg["surface_source"]:
            raise ValueError("Surface sidecar version differs from admission")
        if cfg["role"]=="A0":
            import json
            from data.action4_paths import PROJECT_ROOT as R,contained_path
            receipt=json.loads(contained_path(R,cfg["validation_receipt"],exists=True).read_text())
            bindings=cfg["surface_source"].get("manifest_sha256",{})
            if any(bindings.get(split)!=receipt["splits"][split]["sha256"] for split in ("train","eval","test")):
                raise ValueError("Formal surface sidecars are not bound to the same three dataset manifests")
    elif cfg["mode"]!="integration":raise ValueError("Unknown run mode")
