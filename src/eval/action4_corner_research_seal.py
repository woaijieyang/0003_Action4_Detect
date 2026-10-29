"""Terminal-only All8-first corner research seal; no deployment/default promotion.

Missing terminals return WAITING before source scans or imports. A fresh isolated
CPU worker verifies the immutable source before any model import or load.
"""
import argparse,hashlib,importlib.util,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test')
TERMINALS=('TRAIN_COMPLETE.json','EVAL_COMPLETE.json','CONTROLLER_COMPLETE.json')
SCORE='action4.pose_projection_all8_002.v1'
POLICY='action4.corner_all8_first_research_selection.v1'
_HELPER=None

def helpers():
    global _HELPER
    if _HELPER is None:
        p=ROOT/'scripts/action4_seal_completed_role.py';spec=importlib.util.spec_from_file_location('_existing_role_sealer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);_HELPER=m
    return _HELPER

def binding(p):
    h=helpers();p=h.path(p);return {'path':str(p.relative_to(ROOT)),'sha256':h.digest(p)}

def bound_json(ref,evidence):
    h=helpers();value=h.read_json(ref['path'],evidence)
    if evidence[str(h.path(ref['path']).relative_to(ROOT))]!=ref['sha256']:raise ValueError('Bound evidence changed: '+ref['path'])
    return value

def validate_user_stop(cfg,request,stopped,process):
    """Actual stopped traversal, without manufacturing formal completion records."""
    h=helpers();h.validate_formal_recipe(cfg)
    if cfg.get('training_protocol')!=h.CORNER_NATIVE_PROTOCOL or cfg.get('epochs')!=80:raise ValueError('Original planned80 corner config required')
    if request.get('status')!='USER_REQUESTED_STOP_KEEP_BEST_AND_DRAIN_EVAL' or request.get('formal80_completed') is not False:raise ValueError('Explicit user stop request required')
    if stopped.get('status')!='OWNED_TRAINER_AND_CONTROLLER_EXIT_CONFIRMED_EVAL_DRAINING' or stopped.get('formal_training_complete') is not False or stopped.get('no_new_training') is not True:raise ValueError('Actual owned trainer/controller stop proof required')
    a=stopped['actual_status'];before=request['train_status_before'];units=cfg['expected_units'];batch=cfg['batch']
    if a.get('role')!='A0' or a.get('mode')!='full_training' or before.get('role')!='A0' or before.get('mode')!='full_training':raise ValueError('Stop status role/mode differs')
    step=h.integer(a.get('step'),'stopped actual step',1);epoch=h.integer(a.get('epoch'),'actual epoch',1);cursor=h.integer(a.get('cursor'),'actual cursor');attempts=h.integer(a.get('data_update_attempt'),'actual data attempts',1);skipped=h.integer(a.get('no_supervision_update_count'),'actual zero supervision')
    if epoch>cfg['epochs'] or cursor>units or attempts!=(epoch-1)*math.ceil(units/batch)+math.ceil(cursor/batch) or step+skipped!=attempts or h.integer(before.get('step'),'prestop step')>step:raise ValueError('Actual stopped traversal counts differ')
    attempt=process.get('attempt_id')
    if process.get('role')!='A0' or process.get('units')!=units or not isinstance(attempt,str) or not attempt:raise ValueError('Owned training attempt identity missing')
    steps=stopped.get('submitted_eval_steps')
    if not isinstance(steps,list) or not steps or any(type(x) is not int or x<0 for x in steps) or steps!=sorted(set(steps)) or steps[0]!=0 or max(steps)>step:raise ValueError('Invalid submitted evaluation domain')
    if cfg.get('eval_schedule') is not None:raise ValueError('Stopped mode currently requires the fixed registered cadence')
    if steps!=sorted({0}|set(range(cfg['eval_every'],step+1,cfg['eval_every']))):raise ValueError('Stopped submitted domain omits registered fixed-cadence requests')
    return {'attempt_id':attempt,'submitted_eval_steps':steps,'actual_stop_step':step,'actual_epoch':epoch,'actual_cursor':cursor,'completed_epoch':epoch if cursor==units else epoch-1,'evaluated_boundary':max(steps),'discarded_unevaluated_updates':step-max(steps),'discarded_semantics':'excluded from checkpoint candidates; no weight deletion','formal_training_complete':False,'planned_epochs':cfg['epochs']}

def validate_drain_receipt(drain,proof,refs,config_id,done_steps):
    steps=proof['submitted_eval_steps']
    if (drain.get('schema')!='action4.user_stopped_eval_drained.v1' or drain.get('status')!='ALL_SUBMITTED_EVALUATIONS_DRAINED'
        or drain.get('config_id')!=config_id or drain.get('attempt_id')!=proof['attempt_id']
        or any(drain.get(k)!=v for k,v in refs.items()) or drain.get('submitted_eval_steps')!=steps or drain.get('done_steps')!=steps or done_steps!=steps
        or drain.get('request_count')!=len(steps) or drain.get('done_count')!=len(steps) or drain.get('failed_count')!=0
        or drain.get('eval_workers_stopped') is not True or drain.get('formal_training_complete') is not False or drain.get('no_new_training') is not True):raise ValueError('User-stop eval drain receipt identity/counts differ')

def stopped_precheck(run_spec,spec,cfg,evidence):
    h=helpers();out=h.path(cfg['output']);names=('USER_STOP_REQUEST.json','USER_STOPPED.json','TRAIN_PROCESS.json')
    missing=[n for n in names if not (out/n).is_file()]
    if missing:return None,{'status':'WAITING_EVAL_DRAIN','missing':missing,'sealed':False,'model_loaded':False,'termination_mode':'user_stopped'}
    records={n:h.read_json(out/n,evidence) for n in names};proof=validate_user_stop(cfg,records[names[0]],records[names[1]],records[names[2]])
    ledger=h.read_json(out/'CORNER_EVAL_IDENTITY.json',evidence);cid=h.canonical(cfg)
    if ledger.get('config_id')!=cid or ledger.get('corner_eval_source')!=cfg['corner_eval_source']:raise ValueError('Stopped run config/source identity differs')
    steps=proof['submitted_eval_steps'];queue=out/'eval';requests=sorted(p.name for p in queue.glob('step_*.json') if p.name.count('.')==1)
    if requests!=sorted(f'step_{x:08d}.json' for x in steps):raise ValueError('Submitted list differs from actual request set')
    for step in steps:
        req=h.read_json(queue/f'step_{step:08d}.json',evidence)
        if type(req.get('step')) is not int or req['step']!=step or req.get('config_id')!=cid or req.get('role')!='A0' or req.get('attempt_id',proof['attempt_id'])!=proof['attempt_id'] or h.path(req['snapshot'])!=out/f'eval_step_{step:08d}.pt':raise ValueError('Stopped evaluation request identity differs')
    if list(queue.glob('*.failed.json')):raise ValueError('Failed stopped-run evaluations remain')
    done_names=sorted(p.name for p in queue.glob('step_*.done.json'));expected_names={f'step_{x:08d}.done.json' for x in steps}
    if not set(done_names)<=expected_names:raise ValueError('Unexpected done outside submitted set')
    done_steps=sorted(int(x.split('_')[1].split('.')[0]) for x in done_names);pending=sorted(set(steps)-set(done_steps));drain_path=out/'USER_EVAL_DRAINED.json'
    if pending or not drain_path.is_file():return proof,{'status':'WAITING_EVAL_DRAIN','pending_steps':pending,'submitted_count':len(steps),'done_count':len(done_steps),'missing':[] if drain_path.is_file() else ['USER_EVAL_DRAINED.json'],'actual_stop_step':proof['actual_stop_step'],'evaluated_boundary':proof['evaluated_boundary'],'sealed':False,'model_loaded':False,'termination_mode':'user_stopped'}
    refs={'run_spec':binding(run_spec),'user_stop_request':binding(out/names[0]),'user_stopped':binding(out/names[1]),'train_process':binding(out/names[2])};drain=h.read_json(drain_path,evidence)
    validate_drain_receipt(drain,proof,refs,cid,done_steps);records['USER_EVAL_DRAINED.json']=drain;proof['records']=records;proof['input_bindings']=refs;proof['drain_receipt']=binding(drain_path)
    return proof,None

def precheck(run_spec,user_stopped=False):
    h=helpers();evidence={};spec=h.read_json(run_spec,evidence);cfg=h.read_json(spec['config'],evidence)
    if spec.get('schema')!='action4.immutable_run.v1' or spec.get('project_root')!=str(ROOT):raise ValueError('Invalid run spec')
    if not cfg.get('require_immutable_source') or h.path(cfg['run_spec'])!=h.path(run_spec) or not h.path(spec['config']).is_relative_to(h.path(spec['code_root'])):raise ValueError('Config/run/source binding differs')
    if user_stopped:
        _,waiting=stopped_precheck(run_spec,spec,cfg,evidence)
        return spec,cfg,evidence,waiting
    out=h.path(cfg['output']);missing=[n for n in TERMINALS if not (out/n).is_file()]
    return spec,cfg,evidence,({'status':'WAITING_FOR_COMPLETION','missing':missing,'sealed':False,'model_loaded':False,'selection_policy':POLICY} if missing else None)

def validate_terminal(cfg,receipts,expected):
    h=helpers()
    epochs=80
    if cfg.get('roi_finetune') is not None:
        from model.training.action4_roi115_finetune import validate_recipe
        validate_recipe(cfg);epochs=75
    else:h.validate_formal_recipe(cfg)
    if cfg.get('training_protocol')!=h.CORNER_NATIVE_PROTOCOL or cfg.get('epochs')!=epochs or cfg.get('formal_epochs')!=epochs or cfg.get('expected_eval_images')!=650 or cfg.get('score_schema')!=SCORE:raise ValueError('Complete formal80 corner recipe required')
    train,ev,ctrl=(receipts[n] for n in TERMINALS);cid=h.canonical(cfg)
    if any(x.get('config_id')!=cid for x in (train,ev,ctrl)):raise ValueError('Terminal config mismatch')
    attempt=train.get('attempt_id')
    if not isinstance(attempt,str) or not attempt or any(x.get('attempt_id')!=attempt for x in (ev,ctrl)):raise ValueError('Terminal attempts differ')
    if any(x.get('mode')!='full_training' or x.get('role')!='A0' for x in (train,ctrl)):raise ValueError('Nonformal role/mode')
    units=h.integer(cfg['expected_units'],'units',1);total=epochs*math.ceil(units/h.integer(cfg['batch'],'batch',1))
    step=h.integer(train.get('step'),'actual step',1);skipped=h.integer(train.get('no_supervision_update_count'),'zero-supervision attempts')
    if (train.get('epochs')!=epochs or train.get('units_per_epoch')!=units or train.get('formal_training_complete') is not True
        or step+skipped!=total or train.get('data_update_attempt')!=total or train.get('planned_update_attempts')!=total):raise ValueError('Incomplete80 traversal/actual update attempts')
    steps=ev.get('all_steps')
    if not isinstance(steps,list) or any(type(s) is not int for s in steps) or len(steps)!=len(set(steps)) or set(steps)!=set(expected) or ev.get('evaluations')!=len(expected):raise ValueError('Incomplete or duplicated evaluation domain')
    if ev.get('status') not in ('NO_DEPLOYABLE_CHECKPOINT','COMPLETE_WITH_DEPLOYABLE_CHECKPOINT') or ctrl.get('status')!=ev['status'] or ctrl.get('best_step')!=ev.get('best_step'):raise ValueError('Legacy terminal status differs')
    if ev['status']=='NO_DEPLOYABLE_CHECKPOINT' and ev.get('best_step') is not None:raise ValueError('False legacy selection')
    return attempt

def validate_report(report,image_ids,gt_units):
    h=helpers()
    if report.get('score_schema')!=SCORE or report.get('pose_gt_count')!=1300 or report.get('pose_output_count')!=1300:raise ValueError('Wrong route schema/full GT-output denominator')
    rows=report.get('per_gt_failure_reason',[]);keys=[(r['sample_id'],str(r['instance_id'])) for r in rows]
    if len(keys)!=1300 or len(set(keys))!=1300 or set(keys)!=set(gt_units) or {k[0] for k in keys}!=set(image_ids):raise ValueError('Actual route GT/image domain differs')
    if any(type(r.get('all8_at_002')) is not bool or type(r.get('all8_at_005')) is not bool for r in rows):raise ValueError('Unmeasured route outcomes')
    success=sum(r['all8_at_002'] for r in rows);success5=sum(r['all8_at_005'] for r in rows)
    if (report.get('pose_all8_at_002_success')!=success or report.get('pose_all8_at_002_rate')!=success/1300
        or report.get('pose_all8_at_005_rate')!=success5/1300 or report.get('pose_fp')!=1300-success or report.get('pose_fn')!=1300-success):raise ValueError('Route counts inconsistent with full outcome domain')
    p95=report.get('pose_corner_p95_valid_matched_only')
    if p95 is not None and (type(p95) not in (int,float) or not math.isfinite(p95) or p95<0):raise ValueError('Invalid P95')
    return report

def rank(row):
    r=row['metrics'];p=r['pose_corner_p95_valid_matched_only']
    return (r['pose_all8_at_002_rate'],-(r['pose_fp']+r['pose_fn']),-p if p is not None else -math.inf,-row['step'],row['route']=='raw')

def choose(rows):
    if not rows or len({(r['step'],r['route']) for r in rows})!=len(rows):raise ValueError('Missing/duplicate step-route')
    if any(r['route'] not in ('raw','corner_candidate') or type(r['quality_ready']) is not bool for r in rows):raise ValueError('Explicit research route/quality flag required')
    return max(rows,key=rank)

def verify_source(spec):
    h=helpers();code=h.path(spec['code_root']);h.verify_import_inventory(spec);verified={}
    for rel,sha in spec['files'].items():
        p=h.path(code/rel)
        if not p.is_relative_to(code):raise ValueError('Source manifest escape')
        stamp=h.stat_fingerprint(p)
        if h.digest(p)!=sha or h.stat_fingerprint(p)!=stamp:raise ValueError('Frozen source differs: '+rel)
        verified[str(p)]=(sha,stamp)
    if any(k=='model' or k.startswith(('model.','data.','eval.')) for k in sys.modules):raise ValueError('Worker already imported project implementation')
    os.environ['ACTION4_PROJECT_ROOT']=str(ROOT);os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path[:]=h.snapshot_pythonpath(code)
    return code,verified

def inspect_terminal(run_spec,spec,cfg,evidence,user_stopped=False):
    h=helpers();out=h.path(cfg['output']);stop_proof=None
    if user_stopped:
        stop_proof,waiting=stopped_precheck(run_spec,spec,cfg,evidence)
        if waiting:raise ValueError('User-stopped eval drain is not complete')
        receipts=stop_proof['records'];expected=set(stop_proof['submitted_eval_steps']);attempt=stop_proof['attempt_id']
    else:
        receipts={n:h.read_json(out/n,evidence) for n in TERMINALS};train=receipts['TRAIN_COMPLETE.json']
        from model.training.action4_eval_schedule import expected_eval_steps,read_schedule_ledger
        ledger=read_schedule_ledger(cfg,out,train)
        if ledger is not None:h.read_json(out/'EVAL_SCHEDULE_LEDGER.json',evidence)
        expected=expected_eval_steps(cfg,train,ledger);attempt=validate_terminal(cfg,receipts,expected)
    queue=out/'eval'
    if list(queue.glob('*.failed.json')):raise ValueError('Failed eval jobs remain')
    requests=sorted(p.name for p in queue.glob('step_*.json') if p.name.count('.')==1);dones=sorted(p.name for p in queue.glob('*.done.json'))
    if requests!=sorted(f'step_{s:08d}.json' for s in expected) or dones!=sorted(f'step_{s:08d}.done.json' for s in expected):raise ValueError('Request/done set not closed')
    validation=bound_json({'path':cfg['validation_receipt'],'sha256':cfg['validation_receipt_sha256']},evidence)
    annref={'path':validation['splits']['eval']['manifest'],'sha256':validation['splits']['eval']['sha256']};annpath=h.path(annref['path'])
    if h.digest(annpath)!=annref['sha256']:raise ValueError('Published eval annotations changed')
    evidence[str(annpath.relative_to(ROOT))]=annref['sha256'];records=[json.loads(line) for line in annpath.read_text().splitlines() if line.strip()]
    from data.action4_dataset import is_ignored
    image_ids={r['sample_id'] for r in records};gt_units={(r['sample_id'],str(g['instance_id'])) for r in records for g in r['instances'] if not is_ignored(g)}
    if len(records)!=650 or len(image_ids)!=650 or len(gt_units)!=1300 or any(r['split']!='eval' or r['status']!='training_accepted' for r in records):raise ValueError('Published full650/1300 required')
    canonical8=records[0]['corners_3d_mm']
    if any(r['corners_3d_mm']!=canonical8 for r in records):raise ValueError('Fixed canonical8 changed across eval')
    source=bound_json(cfg['corner_eval_source'],evidence)
    from model.action4_single_frame_pose import POLICY as decoder
    from eval.action4_corner_calibration import POLICY as calibration_policy, _record_sort
    decoder_hash=h.canonical({'policy':decoder,'source_sha256':source['files_sha256']['src/model/action4_single_frame_pose.py'],'scope':'verified_code_binding'})
    signature={'schema':'action4.corner_eval_run_identity.v1','config_id':h.canonical(cfg),'role':'A0','evaluation_variant':'a0_corner_image','corner_eval_source':cfg['corner_eval_source'],'parent_stage_fallback':{'path':cfg['parent_bundle'],'sha256':cfg['parent_bundle_sha256'],'selected_step':cfg['parent_best_step']}}
    if h.read_json(out/'CORNER_EVAL_IDENTITY.json',evidence)!=signature:raise ValueError('Corner source/parent identity ledger differs')
    rows=[];quality_count=deploy_count=0
    for step in sorted(expected):
        req=h.read_json(queue/f'step_{step:08d}.json',evidence);done=h.read_json(queue/f'step_{step:08d}.done.json',evidence)
        for item in (req,done):
            if type(item.get('step')) is not int or item['step']!=step or item.get('config_id')!=signature['config_id'] or item.get('role')!='A0' or item.get('attempt_id',attempt)!=attempt:raise ValueError('Per-eval step/config/attempt mismatch')
        if done.get('evaluation_variant')!='a0_corner_image' or done.get('training_mode')!='full_training' or done.get('corner_eval_source')!=cfg['corner_eval_source'] or done.get('parent_stage_fallback')!=signature['parent_stage_fallback']:raise ValueError('Per-eval mode/source/parent differs')
        snapshot=done['evaluated_snapshot'];expected_path=out/f'eval_step_{step:08d}.pt'
        if h.path(req['snapshot'])!=expected_path or h.path(snapshot['path'])!=expected_path:raise ValueError('Evaluated snapshot path differs')
        report=bound_json(done['report_artifact'],evidence)
        if report!=done['report'] or report.get('evaluation_images')!=650 or report.get('corner_eval_engineering_fixture') is not False or report.get('corner_eval_source')!=cfg['corner_eval_source']:raise ValueError('Actual report differs/incomplete')
        folder=h.path(done['report_artifact']['path']).parent
        if folder.parent!=out/'eval_runs' or not folder.name.startswith(f'step_{step:08d}_{attempt}_'):raise ValueError('Evaluation attempt directory differs')
        transaction=h.read_json(folder/'COMPLETE.json',evidence)
        if transaction.get('status')!='CORNER_EVAL_COMPLETE' or transaction.get('images')!=650 or transaction.get('engineering_fixture') is not False or transaction.get('prediction_artifact')!=report['prediction_artifact'] or transaction.get('report')!=done['report_artifact']:raise ValueError('Incomplete eval transaction')
        execution=bound_json(report['inference_execution_receipt'],evidence)
        if (execution.get('status')!='COMPLETE_PREDICTION_ONLY_FULL650' or execution.get('images')!=650 or execution.get('GT_passed_to_inference') is not False
            or h.path(execution['run_spec'])!=h.path(run_spec) or execution.get('code_root')!=spec['code_root'] or execution.get('snapshot')!=snapshot or execution.get('source_manifest')!=cfg['corner_eval_source']):raise ValueError('Actual inference execution binding differs')
        origins=execution.get('module_origins',{})
        if not origins or any(rel not in spec['files'] or not h.path(h.path(spec['code_root'])/rel).is_relative_to(h.path(spec['code_root'])) for rel in origins.values()):raise ValueError('Execution imports not in verified source inventory')
        cal=bound_json(report['calibration_receipt'],evidence);scoring=h.read_json(folder/'scoring/SCORING_INPUTS.json',evidence);domain=scoring['domain'];effective=scoring['effective_identity']
        if scoring['input_bindings']['predictions']!=report['prediction_artifact'] or scoring['input_bindings']['annotations']!=annref or scoring['input_bindings']['source_manifest']!=cfg['corner_eval_source']:raise ValueError('Scoring input bindings differ')
        domain_payload={k:v for k,v in domain.items() if k!='domain_sha256'}
        if domain['domain_sha256']!=h.canonical(domain_payload) or cal['domain_sha256']!=domain['domain_sha256'] or cal['scoring_records_sha256']!=h.canonical(sorted(scoring['records'],key=_record_sort)):raise ValueError('Actual scoring domain/record digest differs')
        if len(domain['image_ids'])!=650 or set(domain['image_ids'])!=image_ids or domain['data_sha256']!=annref['sha256'] or domain['predictions_sha256']!=report['prediction_artifact']['sha256']:raise ValueError('Actual full image domain differs')
        actualgt=[(u['image_id'],str(u['gt_instance_id'])) for u in domain['gt_units'] if u['pose_nonignore']]
        pids=[(u['image_id'],u['prediction_id']) for u in domain['prediction_units']]
        if len(actualgt)!=1300 or set(actualgt)!=gt_units or len(pids)!=1300 or len(set(pids))!=1300 or {i for i,_ in pids}!=image_ids:raise ValueError('Actual GT/prediction unit domain differs')
        if (effective['A0_step']!=step or effective['A0_config_id']!=signature['config_id'] or effective['models']['A0']!=snapshot or effective['models']['B0']['sha256']!=cfg['evaluation_counterpart_sha256'] or effective['decoder_policy']!=decoder or effective['input_contract']['roi_expansion']!=cfg['roi_expansion']):raise ValueError('Effective model/ROI/decoder differs')
        run={'checkpoint_sha256':snapshot['sha256'],'config_sha256':h.canonical(effective),'decoder_sha256':decoder_hash,'data_sha256':annref['sha256'],'source_sha256':cfg['corner_eval_source']['sha256'],'b0_model_sha256':cfg['evaluation_counterpart_sha256'],'sam_model_sha256':cfg['sam_eval_cache']['expected_sources']['sam_model_sha256'],'sam_cache_sha256':cfg['sam_eval_cache']['manifest']['sha256'],'predictions_sha256':report['prediction_artifact']['sha256']};run['identity']=h.canonical(run)
        ready=report.get('corner_quality_ready')
        if (type(ready) is not bool or cal.get('schema')!='action4.corner_checkpoint_calibration.v1' or cal.get('run_binding')!=run or cal.get('policy')!=calibration_policy or cal.get('quality_ready') is not ready or cal.get('quality_gate_passed_for_selection') is not ready or report.get('quality_gate_passed_for_selection') is not ready):raise ValueError('Original quality/source policy differs')
        if type(done.get('pose_protocol_eligible')) is not bool or done.get('deployment_eligible')!=(ready and done['pose_protocol_eligible']):raise ValueError('Original deployment flags inconsistent')
        quality_count+=ready;deploy_count+=done['deployment_eligible']
        # Hash the actual prediction bytes once, without loading all RGB/point payloads.
        pp=h.path(report['prediction_artifact']['path']);psha=h.digest(pp)
        if psha!=report['prediction_artifact']['sha256']:raise ValueError('Saved prediction artifact changed')
        evidence[str(pp.relative_to(ROOT))]=psha
        for route,key,reportkey in [('raw','raw_report','raw_pose_report_artifact'),('corner_candidate','candidate_report','candidate_pose_report_artifact')]:
            metric=bound_json(report[reportkey],evidence);wrapper=cal[key];validate_report(metric,image_ids,gt_units)
            if (wrapper.get('complete') is not True or wrapper.get('pose_report')!=metric or wrapper.get('image_ids')!=sorted(image_ids) or wrapper.get('pose_gt_count')!=1300 or wrapper.get('run_identity')!=run['identity'] or wrapper.get('data_sha256')!=annref['sha256'] or wrapper.get('score_schema')!=SCORE or wrapper.get('pose_all8_at_002_success')!=metric['pose_all8_at_002_success']):raise ValueError('Calibration raw/candidate report differs')
            outcomes={(x['image_id'],str(x['gt_instance_id'])):x['success'] for x in wrapper['outcomes']}
            if len(wrapper['outcomes'])!=1300 or outcomes!={(x['sample_id'],str(x['instance_id'])):x['all8_at_002'] for x in metric['per_gt_failure_reason']}:raise ValueError('Calibration route outcomes differ')
            rows.append({'step':step,'route':route,'metrics':{k:v for k,v in metric.items() if k not in ('per_gt_failure_reason','unmatched_pose_predictions')},'quality_ready':ready,'quality_reasons':cal['reasons'],'old_deployment_eligible':done['deployment_eligible'],'snapshot':snapshot,'report':report[reportkey],'calibration_receipt':report['calibration_receipt'],'prediction_artifact':report['prediction_artifact'],'effective_identity':effective})
    if not user_stopped:
        ev=receipts['EVAL_COMPLETE.json']
        if ev.get('quality_gate_passed_evaluations')!=quality_count or ev.get('deployment_eligible_evaluations')!=deploy_count or ev.get('rejected_evaluations')!=len(expected)-deploy_count:raise ValueError('Legacy aggregate eligibility differs')
        oldname='NO_DEPLOYABLE_CHECKPOINT.json' if ev['status']=='NO_DEPLOYABLE_CHECKPOINT' else 'BEST.json';old=h.read_json(out/oldname,evidence)
        if old.get('config_id')!=signature['config_id']:raise ValueError('Legacy selection audit config differs')
        if oldname.startswith('NO_') and (old.get('attempt_id')!=attempt or old.get('all_steps')!=ev['all_steps'] or old.get('evaluations')!=len(expected) or old.get('best_step') is not None or deploy_count):raise ValueError('Legacy no-deployment terminal differs')
        if oldname=='BEST.json' and (old.get('step')!=ev['best_step'] or old.get('deployment_eligible') is not True):raise ValueError('Legacy BEST audit mismatch')
    winner=choose(rows)
    policy=POLICY
    if cfg.get('roi_finetune') is not None:
        from model.training.action4_corner_research_selection import POLICY as policy
        winner=choose([row for row in rows if row['route']=='corner_candidate'])
        online=h.read_json(out/'PNP_RESEARCH_BEST.json',evidence)
        if (online.get('schema')!=policy or online.get('config_id')!=signature['config_id']
            or online.get('step')!=winner['step'] or online.get('metrics')!=winner['metrics']
            or online.get('evaluated_snapshot')!=winner['snapshot']):raise ValueError('Online PnP research best differs from terminal independently verified winner')
    return {'status':'USER_STOPPED_ALL8_RESEARCH_SELECTION_VERIFIED' if user_stopped else 'TERMINAL_ALL8_RESEARCH_SELECTION_VERIFIED','termination_mode':'user_stopped' if user_stopped else 'completed'+str(cfg['epochs']),'stop_summary':{k:v for k,v in stop_proof.items() if k!='records'} if stop_proof else None,'selection_policy':policy,'ranking':'All8@2% rate, fewer FP+FN, lower P95 (null worst), earlier step; candidate route only for ROI115, legacy exact route tie raw','winner':winner,'evaluations':len(expected),'routes_checked':len(rows),'expected_steps':sorted(expected),'config_id':signature['config_id'],'attempt_id':attempt,'completion':receipts,'canonical8_m':[[v/1000 for v in xyz] for xyz in canonical8],'decoder_policy':decoder,'data_manifest':annref,'evidence_sha256':evidence,'quality_upgraded':False,'deployment_admitted':False,'default_modified':False}

def worker(run_spec,destination=None,user_stopped=False):
    h=helpers();spec,cfg,evidence,waiting=precheck(run_spec,user_stopped)
    if waiting:return waiting
    sealer_ref=binding(Path(__file__));helper_ref=binding(ROOT/'scripts/action4_seal_completed_role.py')
    code,verified=verify_source(spec)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with h.completion_locks(h.path(cfg['output'])):
        result=inspect_terminal(run_spec,spec,cfg,evidence,user_stopped)
        if destination is None:return result
        dest=h.path(destination,False)
        if not dest.is_relative_to(ROOT/'models/candidates') or dest==ROOT/'models/candidates' or dest.exists():raise ValueError('Explicit new models/candidates destination required')
        deps=h.verify_dependencies(spec,cfg,verified)
        from model.action4_release_bundle import load_bundle
        from model.action4_best_registry import verify_reference,verify_loaded
        pinned=verify_reference('A0',cfg['parent_bundle'],cfg['parent_best_step'],cfg['parent_registry'])
        pm,pb,parent=load_bundle(cfg['parent_bundle'],'A0','cpu',allow_corner_candidate=cfg.get('roi_finetune') is not None,expected_sha256=cfg['parent_bundle_sha256']);verify_loaded(pinned,parent);del pm,pb
        bm,bb,bstate=load_bundle(cfg['evaluation_counterpart_bundle'],'B0','cpu',expected_sha256=cfg['evaluation_counterpart_sha256']);del bm,bb
        winner=result['winner'];source=h.path(winner['snapshot']['path']);sha=winner['snapshot']['sha256']
        if h.digest(source)!=sha:raise ValueError('Selected actual snapshot bytes changed')
        effective=winner['effective_identity']
        if bstate['config_id']!=effective['B0_config_id'] or bstate['selected_step']!=effective['B0_step']:raise ValueError('Strict B0 counterpart identity differs')
        dest.parent.mkdir(parents=True,exist_ok=True);pending=dest.with_name(dest.name+'.pending_'+str(os.getpid()));pending.mkdir(exist_ok=False);copy_path=pending/'A0.pt';shutil.copyfile(source,copy_path)
        if h.digest(copy_path)!=sha or h.digest(source)!=sha:raise ValueError('Winner changed during copy; pending retained')
        model,base,state=load_bundle(str(copy_path),'A0','cpu',allow_corner_candidate=True,expected_sha256=sha);h.verify_state(state,cfg,winner['step']);h.verify_frozen_backbone(state,parent)
        if state.get('schema')!='action4.stage_bundle.corner_candidate.v1' or state.get('quality_ready') is not False or state.get('deployment_admitted') is not False or state.get('corner_protocol')!=cfg['corner_protocol']:raise ValueError('Loaded research model contract differs')
        for name,module in list(sys.modules.items()):
            if name=='model' or name=='data' or name=='eval' or name.startswith(('model.','data.','eval.')):
                location=getattr(module,'__file__',None)
                if location and not Path(location).resolve().is_relative_to(code/'src'):raise ValueError('Non-snapshot project import: '+name)
        for filename,(expected,_) in verified.items():
            if h.verified_digest(h.path(filename),verified)!=expected:raise ValueError('Source changed during seal')
        h.unchanged(evidence)
        if h.verify_dependencies(spec,cfg,verified)!=deps or h.digest(copy_path)!=sha:raise ValueError('Dependencies/model changed during seal')
        if binding(Path(__file__))!=sealer_ref or binding(ROOT/'scripts/action4_seal_completed_role.py')!=helper_ref:raise ValueError('Sealer/helper implementation changed during verification')
        record={'schema':'action4.sealed_corner_all8_research.v1','status':'SEALED_USER_STOPPED_ALL8_RESEARCH' if user_stopped else 'SEALED_COMPLETED'+str(cfg['epochs'])+'_ALL8_RESEARCH_NOT_DEPLOYMENT','formal_training_complete':not user_stopped,'planned_epochs':cfg['epochs'],'stop_summary':result['stop_summary'],'role':'A0','bundle':str((dest/'A0.pt').relative_to(ROOT)),'sha256':sha,'selected_step':winner['step'],'selected_route':winner['route'],'selection_policy':result['selection_policy'],'quality_ready':winner['quality_ready'],'quality_reasons':winner['quality_reasons'],'model_internal_quality_ready':False,'deployment_admitted':False,'real_video_accuracy_accepted':False,'default_modified':False,'old_BEST_modified':False,'strict_load':True,'frozen_backbone_verified':True,'run_spec':binding(run_spec),'config_id':result['config_id'],'source_root':spec['code_root'],'source_manifest':cfg['corner_eval_source'],'model_B0':{'path':cfg['evaluation_counterpart_bundle'],'sha256':cfg['evaluation_counterpart_sha256'],'config_id':bstate['config_id'],'selected_step':bstate['selected_step']},'parent':{'path':cfg['parent_bundle'],'sha256':cfg['parent_bundle_sha256'],'selected_step':cfg['parent_best_step']},'roi_expansion':cfg['roi_expansion'],'roi_source':cfg['roi_source'],'canonical8_m':result['canonical8_m'],'decoder_policy':result['decoder_policy'],'selection':result,'dependencies':deps,'sealer_source':sealer_ref,'helpers_source':helper_ref,'time_unix':time.time()}
        (pending/'SEALED_RESEARCH.json').write_text(json.dumps(record,indent=2,allow_nan=False));copy_path.chmod(0o444);(pending/'SEALED_RESEARCH.json').chmod(0o444)
        if dest.exists():raise FileExistsError('Destination appeared')
        pending.rename(dest);return record

def launch(run_spec,destination=None,user_stopped=False):
    _,_,_,waiting=precheck(run_spec,user_stopped)
    if waiting:return waiting
    h=helpers();runtime=h.path('outputs/action4_corner_research_seal_preparation_v001/worker_runtime',False);runtime.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TMPDIR=str(runtime),XDG_CACHE_HOME=str(runtime),TORCH_HOME=str(runtime))
    cmd=[str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/'scripts/action4_seal_corner_all8.py'),'--worker','--run-spec',str(h.path(run_spec))]
    if destination is not None:cmd+=['--destination',str(destination)]
    if user_stopped:cmd+=['--user-stopped']
    completed=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
    if completed.returncode:raise RuntimeError('Isolated seal worker failed: '+completed.stderr)
    return json.loads(completed.stdout)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-spec',required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--user-stopped',action='store_true',help='Explicit user-stopped/drained research seal; never asserts80 completion');p.add_argument('--destination');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if Path.cwd().resolve()!=ROOT or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Verified project cwd and hidden GPU required')
    if args.check_only and args.destination:p.error('check-only cannot copy a model')
    if not args.check_only and not args.worker and not args.destination:p.error('Explicit destination or check-only required')
    result=worker(args.run_spec,args.destination,args.user_stopped) if args.worker else launch(args.run_spec,None if args.check_only else args.destination,args.user_stopped)
    print(json.dumps(result,allow_nan=False))
