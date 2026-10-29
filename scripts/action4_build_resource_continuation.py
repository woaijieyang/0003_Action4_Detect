"""Build a reviewed transport-only successor using the existing run freezer.

Does not signal or launch any process. Run from the authorized server root.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

R = Path('/data/zy/vln/a-test')
BASE = R/'runtime/action4_runs/p2_a0_parallel80_eval10_v001/RUN_SPEC.json'
WORK = R/'outputs/action4_a0_resource_continuation_v001'
ADDITIONS = ('src/model/training/action4_role_continuation.py',
             'src/eval/action4_role_continuation_eval.py',
             'src/model/training/action4_role_continuation_observer.py')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def ref(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(R): raise ValueError('Outside project')
    return {'path': str(path.relative_to(R)), 'sha256': sha(path)}

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def replace(text, old, new):
    if text.count(old) != 1: raise ValueError('Patch anchor differs: '+old[:90])
    return text.replace(old, new)

def prepare_code(variant):
    base = json.loads(BASE.read_text()); source = R/base['code_root']
    target = WORK/variant/'source'; target.mkdir(parents=True, exist_ok=False)
    for name, digest in base['files'].items():
        old = source/name
        if old.is_symlink() or not old.resolve().is_relative_to(source) or sha(old) != digest:
            raise ValueError('Predecessor source changed')
        new = target/name; new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old, new)
    for name in ADDITIONS:
        destination = target/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R/name, destination)
    trainer = target/'src/model/training/action4_stage2_train.py'
    subprocess.run(['patch', '--batch', str(trainer), str(WORK/'state_contract/TRAINER_CONTINUE_RECEIPT.patch')], check=True)
    text = trainer.read_text()
    text = replace(text, '        model._action4_continuation_lineage=lineage',
        '''        model._action4_continuation_lineage=lineage
        if lineage["purpose"]=="training_continuation":
            init=json.loads((out/"PREFIX_INITIALIZATION.json").read_text())
            if init["boundary_step"]!=step or init["prefix_config_id"]!=state["config_id"]:
                raise ValueError("Continuation boundary evaluation prefix was not initialized")''')
    text = replace(text, '        data_attempt=state.get("data_update_attempt",step);zero_updates=state.get("no_supervision_update_count",0)',
        '''        data_attempt=state.get("data_update_attempt",step);zero_updates=state.get("no_supervision_update_count",0)
        if state.get("continuation_lineage") is not None:
            model._action4_continuation_lineage=state["continuation_lineage"]''')
    text = replace(text, '                        "data_update_attempt":data_attempt,"no_supervision_update_count":zero_updates})',
        '''                        "data_update_attempt":data_attempt,"no_supervision_update_count":zero_updates,
                        "continuation_lineage":getattr(model.module,"_action4_continuation_lineage",None)})''')
    trainer.write_text(text)
    core = target/'src/model/training/action4_stage2_core.py'
    text = core.read_text()
    text = replace(text, '    return common\n',
        '''    if getattr(model,"_action4_continuation_lineage",None) is not None:
        common["continuation_lineage"]=copy.deepcopy(model._action4_continuation_lineage)
    return common
''')
    core.write_text(text)
    dispatch = target/'src/model/training/action4_stage2_dispatch.py'
    text = dispatch.read_text()
    text = replace(text, 'def run(config):',
        'def run(config,continue_receipt=None,continue_sha256=None,prefix_ledger=None,prefix_sha256=None):')
    text = replace(text, '    verify_running_snapshot(cfg)',
        '''    verify_running_snapshot(cfg)
    continuation=continue_receipt is not None
    if any(x is not None for x in (continue_sha256,prefix_ledger,prefix_sha256))!=continuation or (continuation and any(x is None for x in (continue_sha256,prefix_ledger,prefix_sha256))):
        raise ValueError("Explicit continuation needs all state and prefix references")
    if continuation and cfg["mode"]!="full_training":
        raise ValueError("Use direct bounded trainer entry for integration probe")''')
    text = replace(text, '    attempt=uuid.uuid4().hex',
        '''    prefix=None
    if continuation:
        from eval.action4_role_continuation_eval import load_prefix_reports,initialize_successor
        prefix=load_prefix_reports(prefix_ledger,prefix_sha256)
        initialize_successor(out,prefix)
    attempt=uuid.uuid4().hex''')
    text = replace(text, '    if cfg.get("integration_resume_step") is not None:',
        '''    if continuation:
        specs[0][1].extend(["--continue-receipt",continue_receipt,"--continue-receipt-sha256",continue_sha256])
    if cfg.get("integration_resume_step") is not None:''')
    text = replace(text, '    processes=[];workers=[]',
        '''    if continuation:
        for name,cmd,gpus in specs[1:]:
            cmd[cmd.index("eval.action4_stage2_eval")]="eval.action4_role_continuation_eval"
            cmd.extend(["--prefix-ledger",prefix_ledger,"--prefix-ledger-sha256",prefix_sha256])
    processes=[];workers=[]''')
    text = replace(text, '        requests=[p for p in (out/"eval").glob("step_*.json") if p.name.count(".")==1]',
        '''        if continuation:
            from eval.action4_role_continuation_eval import finalize_lineage
            joined=finalize_lineage(prefix,out,cfg)
            atomic_json(out/"EVAL_COMPLETE.json",{"attempt_id":attempt,"evaluations":len(joined["steps"]),"all_steps":joined["steps"],"best_step":joined["legacy_best_step"],"config_id":canonical_hash(cfg),"lineage":joined})
            atomic_json(out/"CONTROLLER_COMPLETE.json",{"attempt_id":attempt,"mode":cfg["mode"],"role":cfg["role"],"best_step":joined["legacy_best_step"],"config_id":canonical_hash(cfg),"accuracy_accepted":False,"prefix_ledger":{"path":prefix_ledger,"sha256":prefix_sha256},"parent_assets_pruned":False})
            return
        requests=[p for p in (out/"eval").glob("step_*.json") if p.name.count(".")==1]''')
    text = replace(text, '    p=argparse.ArgumentParser();p.add_argument("--config",required=True);run(p.parse_args().config)',
        '''    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--continue-receipt");p.add_argument("--continue-receipt-sha256")
    p.add_argument("--prefix-ledger");p.add_argument("--prefix-ledger-sha256");a=p.parse_args()
    run(a.config,a.continue_receipt,a.continue_receipt_sha256,a.prefix_ledger,a.prefix_ledger_sha256)''')
    dispatch.write_text(text)
    changed = []
    for path in sorted((target/'src').rglob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
        rel = str(path.relative_to(target)); after = sha(path)
        if base['files'].get(rel) != after:
            changed.append({'path':rel,'before_sha256':base['files'].get(rel),'after_sha256':after,'purpose':'transport_control'})
    code = {str(p.relative_to(target)):sha(p) for p in sorted((target/'src').rglob('*.py'))}
    code_id = hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    files = {str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    spec = target.parent/'SOURCE_SPEC.json'
    write(spec, {'code_root':str(target.relative_to(R)), 'files':files})
    write(target.parent/'REVIEWED_DELTAS.json', {'project_code_sha256':code_id,'approved_source_deltas':changed})
    for path in target.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    spec.chmod(0o444)
    return spec, code_id, changed

def build(run_id, purpose, checkpoint, source_variant, formal_gate=None):
    if Path.cwd().resolve()!=R: raise ValueError('Verified cwd required')
    os.environ['ACTION4_PROJECT_ROOT']=str(R);sys.dont_write_bytecode=True
    sys.path.insert(0,str(R/'src'))
    import torch
    from model.training.action4_run_snapshot import freeze
    from model.training.action4_runtime import canonical_hash
    spec=json.loads(BASE.read_text());old=json.loads((R/spec['config']).read_text())
    source_spec=WORK/source_variant/'SOURCE_SPEC.json'
    if source_spec.exists():
        code=json.loads((source_spec.parent/'REVIEWED_DELTAS.json').read_text())
        code_id=code['project_code_sha256'];deltas=code['approved_source_deltas']
    else:source_spec,code_id,deltas=prepare_code(source_variant)
    out=R/'outputs/action4_improvement/stage1_baseline'/run_id;out.mkdir(parents=True,exist_ok=False)
    cp=Path(checkpoint);cp=cp if cp.is_absolute() else R/cp
    if not cp.resolve().is_relative_to(R):raise ValueError('Checkpoint outside project')
    captured=out/'PREDECESSOR_STATE.pt';shutil.copyfile(cp,captured);captured.chmod(0o444)
    state=torch.load(captured,map_location='cpu',weights_only=True)
    assert state['config_id']==canonical_hash(old) and state['resumable']
    cfg=copy.deepcopy(old);cfg.update(mode='integration' if purpose=='validation_probe' else 'full_training',
        output=str(out.relative_to(R)),loader_workers=12,training_gpus=[5] if purpose=='validation_probe' else [7],
        eval_gpus=[4,4,4,4,6,6,6,6],reserved_cpu_budget=47,project_total_reserved_cpu=59)
    cfg['process_cpu_affinity']={'train':list(range(68,82)),**{f'eval_{i}':list(range(64-4*i,68-4*i)) for i in range(8)}}
    gate=json.loads((R/old['integration_or_admission_receipt']).read_text())
    if purpose=='validation_probe':
        gate={'launch_allowed':True,'mode':'integration','scope':'explicit two-update resource continuation probe, not formal training completion'}
        execution={'status':'RESOURCE_PROBE_ALLOWED_NOT_FORMAL_PASS','role':'A0','project_code_sha256':code_id,'stop_after':state['step']+2}
    else:
        if formal_gate is None:raise ValueError('Actual probe evidence required before formal preparation')
        execution=json.loads((R/formal_gate).read_text())
        if execution['status']!='STAGE1_EXECUTION_PASS' or execution['project_code_sha256']!=code_id:raise ValueError('Formal source/probe gate mismatch')
    write(out/'ADMISSION.json',gate);write(out/'EXECUTION_GATE.json',execution)
    cfg['integration_or_admission_receipt']=str((out/'ADMISSION.json').relative_to(R))
    cfg['execution_gate']=str((out/'EXECUTION_GATE.json').relative_to(R))
    cfg['tested_project_code_sha256']=code_id
    input_cfg=out/'PRE_FREEZE_CONFIG.json';write(input_cfg,cfg)
    result=freeze(str(input_cfg.relative_to(R)),run_id,source_spec=str(source_spec.relative_to(R)))
    new_spec=R/result['run_spec'];new_cfg_path=R/result['config'];new=json.loads(new_cfg_path.read_text())
    dispatch=R/old['output']/'DISPATCH.json';dispatch_data=json.loads(dispatch.read_text())
    receipt={'schema':'action4.role_resource_continuation.v1','purpose':purpose,
        'authorization':{'controlled_checkpoint_switch':True,'scope':'loader_resource_continuation'},
        'predecessor':{'run_spec':ref(BASE),'config':ref(R/spec['config']),'checkpoint':ref(captured),'dispatch':ref(dispatch),'config_id':canonical_hash(old),'attempt_id':dispatch_data['attempt_id']},
        'successor':{'run_spec':ref(new_spec),'config':ref(new_cfg_path),'config_id':canonical_hash(new)},
        'boundary':{k:state[k] for k in ('epoch','cursor','step','data_update_attempt','no_supervision_update_count')},
        'approved_source_deltas':deltas,
        'relocated_artifacts':{k:{'before':ref(R/old[k]),'after':ref(R/new[k])} for k in ('parent_bundle','parent_registry','evaluation_counterpart_bundle','execution_gate','integration_or_admission_receipt')},
        'observer_dependency':ref(R/'outputs/action4_training_candidate_selection_v001/source/observer.py')}
    path=out/'CONTINUATION_RECEIPT.json';write(path,receipt);path.chmod(0o444)
    result.update(receipt=ref(path),output=str(out.relative_to(R)),purpose=purpose,boundary=receipt['boundary'],project_code_sha256=code_id)
    write(out/'PREPARATION.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--purpose',required=True,choices=['validation_probe','training_continuation'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--source-variant',default='source_variant_v001');p.add_argument('--formal-gate')
    a=p.parse_args();build(a.run_id,a.purpose,a.checkpoint,a.source_variant,a.formal_gate)
