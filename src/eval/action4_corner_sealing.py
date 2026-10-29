"""Immutable-worker corner selection evidence for role sealing; no model promotion."""
import copy,json
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_read,_v2_sha,canonical
from model.training.action4_async_recovery import corner_result_signature,checked_binding
from eval.action4_stage2_eval import _corner_identity,prepare_corner_selection
from eval.action4_pose_protocol import eligible
from model.action4_single_frame_pose import POLICY
from eval.action4_corner_calibration import POLICY as CALIBRATION_POLICY


def validate_corner_sealing_reports(cfg,reports,best_step):
    identity=_corner_identity(cfg)
    reference_path=contained_path(R,cfg['output']+'/REFERENCE_REPORT.json',exists=True)
    reference_binding={'path':str(reference_path.relative_to(R)),'sha256':_v2_sha(reference_path)}
    reference=_v2_checked(reference_binding)
    if reference.get('corner_reference_identity')!=identity:raise ValueError('Corner seal reference/source differs')
    receipt=_v2_checked({"path":cfg["validation_receipt"],"sha256":cfg["validation_receipt_sha256"]})
    eval_ref={"path":receipt["splits"]["eval"]["manifest"],"sha256":receipt["splits"]["eval"]["sha256"]}
    annotations=_v2_read(eval_ref,lambda stream:[json.loads(line) for line in stream if line.strip()])
    expected_ids={r["sample_id"] for r in annotations}
    source=_v2_checked(cfg["corner_eval_source"])
    decoder_hash=canonical({"policy":POLICY,"source_sha256":source["files_sha256"]["src/model/action4_single_frame_pose.py"],"scope":"verified_code_binding"})
    if len(annotations)!=650 or len(expected_ids)!=650:raise ValueError("Corner seal eval manifest must retain full650")
    by_step={r['step']:r for r in reports}
    if len(by_step)!=len(reports) or best_step not in by_step:raise ValueError('Corner seal completion domain differs')
    for row in reports:
        if corner_result_signature(row)!=identity:raise ValueError('Corner seal report/source differs')
        report=row['report'];cal=_v2_checked(report['calibration_receipt']);run=cal.get('run_binding',{})
        ids=cal["raw_report"]["image_ids"]
        if cal["candidate_report"]["image_ids"]!=ids:raise ValueError("Corner raw/candidate domains differ")
        if len(ids)!=650 or set(ids)!=expected_ids:raise ValueError("Corner seal evaluation image domain differs")
        ready=report['corner_quality_ready']
        if (cal.get('schema')!='action4.corner_checkpoint_calibration.v1' or cal.get('quality_ready') is not ready
            or cal.get('quality_gate_passed_for_selection') is not ready
            or run.get('checkpoint_sha256')!=row['evaluated_snapshot']['sha256']
            or run.get('source_sha256')!=cfg['corner_eval_source']['sha256']
            or run.get('predictions_sha256')!=report['prediction_artifact']['sha256']
            or run.get('b0_model_sha256')!=cfg['evaluation_counterpart_sha256']
            or run.get('sam_model_sha256')!=cfg['sam_eval_cache']['expected_sources']['sam_model_sha256']
            or run.get('sam_cache_sha256')!=cfg['sam_eval_cache']['manifest']['sha256']
            or run.get('data_sha256')!=eval_ref['sha256'] or run.get('decoder_sha256')!=decoder_hash
            or cal.get('policy')!=CALIBRATION_POLICY):
            raise ValueError('Corner seal calibration/snapshot/source differs')
        if row['pose_protocol_eligible'] is not bool(eligible(report,reference)):
            raise ValueError('Corner seal pose eligibility differs')
        if report.get('corner_eval_engineering_fixture') is not False or report.get('evaluation_images')!=650:
            raise ValueError('Corner seal requires actual full650 evaluation')
    selected=by_step[best_step]
    if selected['deployment_eligible'] is not True:raise ValueError('Cannot seal unready corner fallback as enhanced best')
    snapshot=checked_binding(selected['evaluated_snapshot'])
    fields,_=prepare_corner_selection(cfg,best_step,snapshot,selected['report'],reference)
    if any(selected.get(k)!=v for k,v in fields.items()):raise ValueError('Selected corner eligibility changed on independent recheck')
    report=selected['report'];cal=_v2_checked(report['calibration_receipt'])
    return dict(schema='action4.sealed_corner_inference_contract.v1',quality_ready=True,global_branch='candidate',
        decoder_policy=copy.deepcopy(POLICY),calibration_policy=copy.deepcopy(cal['policy']),stage_reference=reference_binding,
        calibration_receipt=copy.deepcopy(report['calibration_receipt']),report_artifact=copy.deepcopy(selected['report_artifact']),
        evaluated_snapshot=copy.deepcopy(selected['evaluated_snapshot']),parent_stage_fallback=copy.deepcopy(identity['parent_stage_fallback']),
        corner_eval_source=copy.deepcopy(cfg['corner_eval_source']),roi_expansion=cfg['roi_expansion'],roi_source=cfg['roi_source'],
        B0_sha256=cfg['evaluation_counterpart_sha256'],b0_full_box_quality_receipt=copy.deepcopy(cfg['b0_full_box_quality_receipt']),
        corner_refined_quality_bridge=copy.deepcopy(cfg['corner_refined_quality_bridge']),
        real_video_accuracy_accepted=False)
