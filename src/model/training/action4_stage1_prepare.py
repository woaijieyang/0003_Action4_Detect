"""Create explicit baseline integration/formal configurations; never starts a process."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path

def prepare(role,mode,run_id,counterpart=None,execution_gate=None,output_group="stage1_baseline"):
    if role not in ('A0','B0') or mode not in ('integration','full_training'):raise ValueError('Role/mode required')
    if output_group not in ('stage1_baseline','stage3_a0_sam') or (output_group=='stage3_a0_sam' and role!='A0'):
        raise ValueError('Explicit compatible output group required')
    root=R/'outputs/action4_phase2_stage1_v001'
    data_gate=json.loads((root/'data/FULL_TRAINING_ADMISSION.json').read_text())
    registry=json.loads((R/'configs/model/action4_best_registry_v003.json').read_text());parent=registry['entries'][role]
    bs=torch.load(R/'models/default/B0.pt',map_location='cpu',weights_only=True)
    output=f'outputs/action4_improvement/{output_group}/{run_id}'
    out=contained_path(R,output,exists=False);out.mkdir(parents=True,exist_ok=False)
    units=data_gate['role_units'][role]
    cfg={'role':role,'mode':mode,'output':output,'dataset_root':'data/phrase2','seed':20261024,
         'validation_receipt':data_gate['validation_receipt'],'expected_units':units,'expected_eval_images':650,
         'network':bs['config']['network'],'preprocess':bs['config']['preprocess'],
         'parent_bundle':parent['path'],'parent_best_step':parent['selected_step'],
         'parent_registry':'configs/model/action4_best_registry_v003.json','parent_bundle_sha256':parent['sha256'],
         'epochs':80 if mode=='full_training' else 1,'batch':32 if role=='B0' else 48,'micro':8 if role=='B0' else 2,
         'world_size':2,'threads':2,'loader_workers':4,'training_gpus':[4,5] if role=='B0' else [6,7],
         'eval_gpus':[6,7] if role=='B0' else [4,5], 'eval_loader_workers':4,'reserved_cpu_budget':25,'project_total_reserved_cpu':50,
         'min_free_mib':18000,'max_pending':8,'weight_decay':.0001,'eval_every':20 if role=='B0' else 25,
         'lr_backbone':1e-5,'lr_detector':5e-5,'lr_refiner':1e-4,'surface_lr':1e-5,'lr':5e-5,
         'train_backbone':False,'roi_jitter':{'center_fraction':.15,'scale_range':[.65,1.35]},
         'surface_source':data_gate['surface_source'],'evaluation_scoring_policy':'center','b0_scoring_policy':'center',
         'source_identity':{},'score_schema':'action4.pose_projection_all8_002.v1' if role=='A0' else 'action4.bbox.v1',
         'intrinsics_context':'outputs/action4_execution_source_release_v006/source_context.json',
         'scope':'stage1 existing architecture; independent best parent; fixed6500 data; no external new models or full-video execution',
         'independent_head_dependencies':[],'geometry_batch_policy':'instancewise_head_v1'}
    if mode=='integration':
        n=60*cfg['batch'];cfg['train_indices']=sorted(np.random.default_rng(1729).choice(units,n,replace=False).tolist());cfg['expected_units']=n
        cfg['integration_resume_step']=20 if role=='B0' else 25
        gate={'launch_allowed':True,'mode':'integration','scope':'60 actual updates with controlled same-run recovery; full650 image eval; not formal training'}
    else:
        if execution_gate is None:raise ValueError('Formal run requires verified code/integration/throughput receipt')
        proof=json.loads(contained_path(R,execution_gate,exists=True).read_text())
        if proof.get('status')!='STAGE1_EXECUTION_PASS' or proof.get('role')!=role:raise ValueError('Execution gate is not passed')
        cfg['execution_gate']=execution_gate;cfg['tested_project_code_sha256']=proof['project_code_sha256'];gate=data_gate
    gp=out/'ADMISSION.json';gp.write_text(json.dumps(gate,indent=2));cfg['integration_or_admission_receipt']=str(gp.relative_to(R))
    if role=='B0':cfg['photometric']=bs['config']['photometric']
    else:
        cfg['loss_weights']={'xyz':1.,'region':.1,'visible_mask':.5,'amodal_mask':.5,'rotation_point_matching':1.,
            'centroid':1.,'relative_depth':1.,'visible_corners':1.,'hidden_corners':1.,'positive_depth':1.}
        if mode=='full_training' and counterpart is None:raise ValueError('Formal1A must use completed1B counterpart')
        cp=contained_path(R,counterpart or 'models/default/B0.pt',exists=True)
        cfg['evaluation_counterpart_bundle']=str(cp.relative_to(R));cfg['evaluation_counterpart_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
        cfg['counterpart_scope']='completed1B' if mode=='full_training' else 'existingB0 integration-only'
    config=R/f'configs/model/action4_phrase2_{run_id}.json'
    with config.open('x') as f:json.dump(cfg,f,indent=2)
    return str(config.relative_to(R))

def prepare_corner(config_path,run_id):
    """Bind an explicit stage4 recipe to sealed parents; never launch or select defaults."""
    import copy,re
    from model.training.action4_stage4_prediction_protocol import (
        validate_recipe,validate_parent_seal,validate_admission,INPUT_FIELDS)
    from data.action4_sam_native_export import _v2_sha
    if not isinstance(run_id,str) or re.fullmatch(r"[A-Za-z0-9_-]+",run_id) is None:
        raise ValueError("Safe explicit stage4 run ID required")
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    # Verify before creating output or changing the supplied recipe.
    validate_recipe(cfg)
    seal=validate_parent_seal(cfg)
    if _v2_sha(cfg["parent_bundle"])!=seal["sha256"]:
        raise ValueError("Selected sealed SAM parent bytes changed")
    output=f"outputs/action4_improvement/stage4_corner/{run_id}"
    config=f"configs/model/action4_phrase2_{run_id}.json"
    out=contained_path(R,output,exists=False)
    dest=contained_path(R,config,exists=False)
    if out.exists() or dest.exists():
        raise FileExistsError("Preserve previous corner preparation/run")
    cfg=copy.deepcopy(cfg)
    cfg["output"]=output
    from model.action4_best_registry import selected_entry_from_bundle
    entry=selected_entry_from_bundle("A0",cfg["parent_bundle"],cfg["parent_bundle_sha256"],cfg["parent_best_step"])
    registry={"schema":"action4.selected_parent_registry.v1","entries":{"A0":entry},
        "scope":"actual sealed SAM parent only; not default promotion"}
    cfg["parent_registry"]=output+"/PARENT_REGISTRY.json"
    cfg["integration_or_admission_receipt"]=output+"/ADMISSION.json"
    gate={"schema":"action4.stage4_prediction_roi_admission.v1","launch_allowed":True,
        "mode":cfg["mode"],"role":"A0","training_protocol":cfg["training_protocol"],
        **{k:copy.deepcopy(cfg.get(k)) for k in INPUT_FIELDS}}
    if cfg.get("roi_finetune") is not None:gate["roi_finetune"]=copy.deepcopy(cfg["roi_finetune"])
    validate_admission(cfg,gate)
    out.mkdir(parents=True,exist_ok=False)
    for name,value in (("PARENT_REGISTRY.json",registry),("ADMISSION.json",gate)):
        with (out/name).open("x") as stream:json.dump(value,stream,indent=2)
    with dest.open("x") as stream:json.dump(cfg,stream,indent=2)
    return config

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--role',required=True);p.add_argument('--mode',choices=['integration','full_training'],required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--counterpart');p.add_argument('--execution-gate')
    p.add_argument('--output-group',choices=['stage1_baseline','stage3_a0_sam'],default='stage1_baseline')
    a=p.parse_args();print(prepare(a.role,a.mode,a.run_id,a.counterpart,a.execution_gate,a.output_group))
