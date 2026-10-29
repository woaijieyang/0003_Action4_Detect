"""Consume a calibrated corner seal through the existing image prediction path.

No new detector/backbone/decoder and no per-image GT branch choice. Video/test
providers require their own input admission; this entry uses the registered cache.
"""
import os,json,copy
from dataclasses import dataclass,field
from data.action4_paths import PROJECT_ROOT as R,CODE_ROOT,contained_path
from data.action4_sam_native_export import _v2_checked,_v2_sha,canonical
from eval.action4_corner_scoring_inputs import _source
from eval.action4_pose_protocol import eligible
from model.action4_single_frame_pose import POLICY
from eval.action4_corner_calibration import POLICY as CALIBRATION_POLICY

_TOKEN=object()

@dataclass(frozen=True)
class SealedCornerSelection:
    seal_binding_json:str
    seal_json:str
    config_json:str
    contract_json:str
    references:tuple
    _token:object=field(default=None,init=False,repr=False,compare=False)
    _pid:int=field(default=-1,init=False,repr=False,compare=False)
    def __reduce_ex__(self,protocol):raise TypeError('Verify corner selection inside its consuming process')
    @classmethod
    def load(cls,binding):
        seal=_v2_checked(binding);contract=seal.get('completion',{}).get('corner_inference_contract')
        if (seal.get('schema')!='action4.sealed_independent_role.v2' or seal.get('role')!='A0'
            or seal.get('strict_load') is not True or not isinstance(contract,dict)
            or contract.get('schema')!='action4.sealed_corner_inference_contract.v1'
            or contract.get('quality_ready') is not True or contract.get('global_branch')!='candidate'):
            raise ValueError('Explicit calibrated corner role seal required; baseline/unready fallback is not enhancement')
        spec_ref={'path':seal['run_spec'],'sha256':seal['run_spec_sha256']};spec=_v2_checked(spec_ref)
        code=contained_path(R,spec['code_root'],exists=True)
        if code!=CODE_ROOT or seal.get('actual_code_root')!=spec['code_root']:
            raise ValueError('Consume corner selection from its verified immutable code snapshot')
        config_path=contained_path(R,spec['config'],exists=True)
        if not config_path.is_relative_to(code):raise ValueError('Sealed config escaped immutable source')
        cfg_ref={'path':spec['config'],'sha256':spec['files'][str(config_path.relative_to(code))]};cfg=_v2_checked(cfg_ref)
        if canonical(cfg)!=seal['config_id'] or cfg.get('training_protocol')!='action4.corner_prediction_roi_enhancement.native_v1':
            raise ValueError('Sealed corner configuration differs')
        from model.training.action4_run_snapshot import verify_running_snapshot
        verify_running_snapshot(cfg)
        from model.training.action4_stage4_prediction_protocol import validate_recipe
        validate_recipe(cfg)
        if cfg["mode"]!="full_training" or seal.get("epochs")!=80:raise ValueError("Completed formal corner run required")
        terminals={}
        for name in ("TRAIN_COMPLETE.json","EVAL_COMPLETE.json","CONTROLLER_COMPLETE.json","BEST.json"):
            path=contained_path(R,cfg["output"]+"/"+name,exists=True);relative=str(path.relative_to(R))
            expected=seal["completion"]["evidence_sha256"].get(relative)
            if expected is None:raise ValueError("Seal omitted terminal receipt identity")
            terminals[name]=_v2_checked({"path":relative,"sha256":expected})
        train=terminals["TRAIN_COMPLETE.json"];evaluation=terminals["EVAL_COMPLETE.json"];controller=terminals["CONTROLLER_COMPLETE.json"]
        if (train.get("formal_training_complete") is not True or train.get("epochs")!=80
            or evaluation.get("best_step")!=seal["selected_step"] or controller.get("best_step")!=seal["selected_step"]
            or controller.get("status")!="COMPLETE_WITH_DEPLOYABLE_CHECKPOINT"
            or terminals["BEST.json"].get("deployment_eligible") is not True
            or any(t.get("config_id")!=seal["config_id"] for t in terminals.values())):
            raise ValueError("Corner terminal selection is incomplete or unadmitted")
        source=_source(contract['corner_eval_source'],False)
        if contained_path(R,source['code_root'],exists=True)!=code or contract['corner_eval_source']!=cfg['corner_eval_source']:
            raise ValueError('Corner calibrated source differs from inference source')
        bundle={'path':seal['bundle'],'sha256':seal['sha256']}
        if _v2_sha(bundle['path'])!=bundle['sha256'] or contract['evaluated_snapshot']['sha256']!=bundle['sha256']:
            raise ValueError('Selected corner weights changed')
        calibration=_v2_checked(contract['calibration_receipt']);report=_v2_checked(contract['report_artifact']);reference=_v2_checked(contract['stage_reference'])
        if (calibration.get('quality_ready') is not True or calibration.get('quality_gate_passed_for_selection') is not True
            or report.get('corner_quality_ready') is not True or report.get('corner_final_path')!='candidate'
            or calibration.get('final_path')!='candidate' or calibration['run_binding']['checkpoint_sha256']!=bundle['sha256']
            or not eligible(report,reference) or contract['decoder_policy']!=POLICY
            or contract['calibration_policy']!=calibration['policy']):
            raise ValueError('Sealed global quality/decoder/pose admission differs')
        if any(report.get(k)!=v for k,v in calibration['candidate_report']['pose_report'].items()):
            raise ValueError('Selected candidate scores differ from global calibration')
        if (contract["calibration_policy"]!=CALIBRATION_POLICY
            or contract["B0_sha256"]!=cfg["evaluation_counterpart_sha256"]
            or contract["b0_full_box_quality_receipt"]!=cfg["b0_full_box_quality_receipt"]
            or contract["corner_refined_quality_bridge"]!=cfg["corner_refined_quality_bridge"]
            or calibration["run_binding"]["source_sha256"]!=cfg["corner_eval_source"]["sha256"]
            or calibration["run_binding"]["sam_cache_sha256"]!=cfg["sam_eval_cache"]["manifest"]["sha256"]):
            raise ValueError("Corner calibrated input/quality policies differ")
        if contract['roi_expansion']!=cfg['roi_expansion'] or contract['roi_source']!=cfg['roi_source']:
            raise ValueError('Sealed ROI recipe changed')
        fallback=contract['parent_stage_fallback']
        if (fallback['path']!=cfg['parent_bundle'] or fallback['sha256']!=cfg['parent_bundle_sha256']
            or fallback['selected_step']!=cfg['parent_best_step'] or _v2_sha(fallback['path'])!=fallback['sha256']):
            raise ValueError('Independent parent fallback changed')
        refs=[binding,spec_ref,cfg_ref,bundle,contract['calibration_receipt'],contract['report_artifact'],contract['stage_reference'],
              contract['corner_eval_source'],{'path':fallback['path'],'sha256':fallback['sha256']}]
        refs.extend({"path":str(contained_path(R,cfg["output"]+"/"+name,exists=True).relative_to(R)),
                     "sha256":seal["completion"]["evidence_sha256"][str(contained_path(R,cfg["output"]+"/"+name,exists=True).relative_to(R))]}
                    for name in terminals)
        result=cls(json.dumps(binding,sort_keys=True),json.dumps(seal,sort_keys=True),json.dumps(cfg,sort_keys=True),json.dumps(contract,sort_keys=True),
            tuple((r['path'],r['sha256']) for r in refs))
        object.__setattr__(result,'_token',_TOKEN);object.__setattr__(result,'_pid',os.getpid());return result
    def consume(self,pipeline):
        if type(self) is not SealedCornerSelection or self._token is not _TOKEN or self._pid!=os.getpid():
            raise ValueError('Verified process-owned corner selection required')
        for path,sha in self.references:
            if _v2_sha(path)!=sha:raise ValueError('Corner selection dependency changed')
        seal=json.loads(self.seal_json);cfg=json.loads(self.config_json);contract=json.loads(self.contract_json);identity=pipeline.identity
        if (identity['models']['A0']['sha256']!=seal['sha256'] or identity['models']['B0']['sha256']!=cfg['evaluation_counterpart_sha256']
            or identity['A0_config_id']!=seal['config_id'] or identity['A0_step']!=seal['selected_step']
            or identity['decoder_policy']!=contract['decoder_policy'] or pipeline.roi_expansion!=contract['roi_expansion']
            or identity.get('engineering_fixture') is not False):
            raise ValueError('Actual corner pipeline differs from calibrated selection')
        return contract


def create_selected_pipeline(cls,binding,device):
    selection=SealedCornerSelection.load(binding);cfg=json.loads(selection.config_json);seal=json.loads(selection.seal_json)
    pipe=cls.from_corner_bundles(seal['bundle'],cfg['evaluation_counterpart_bundle'],cfg['sam_eval_cache'],device=device,
        full_box_quality_receipt=cfg['b0_full_box_quality_receipt'])
    selection.consume(pipe);pipe._sealed_corner_selection=selection
    return pipe


def selected_output(payload,contract,seal_binding):
    """Internal branch application; coordinates remain the saved decoder output."""
    if contract.get('quality_ready') is not True or contract.get('global_branch')!='candidate':raise ValueError('Unready selection cannot enable corner candidate')
    if payload.get('schema')!='action4.corner_prediction_bridge.v1':raise ValueError('Expected original corner producer payload')
    if payload.get('GT_used') is not False:raise ValueError('Prediction-only corner payload required')
    raw=payload['raw_instances'];candidate=payload['candidate_instances']
    raw_by_id={r['prediction_id']:r for r in raw}
    if len(raw_by_id)!=len(raw) or [r['prediction_id'] for r in candidate]!=[r['prediction_id'] for r in raw]:
        raise ValueError('Raw/candidate instance domain or order differs')
    result=copy.deepcopy(payload);result['instances']=copy.deepcopy(candidate)
    for instance in result['instances']:
        prior=raw_by_id[instance['prediction_id']]
        if instance['corner_ids']!=list(range(8)) or prior['corner_ids']!=list(range(8)):raise ValueError('Canonical corner ID order changed')
        instance['corners_raw']=copy.deepcopy(prior['corners_px']);instance['corners_final']=copy.deepcopy(instance['corners_px'])
        accepted=instance['pose_diagnostic'].get('geometric_candidate_accepted') is True
        if not accepted:
            for key in ("corners_px","rotation","translation_m","pose_valid","positive_depth_verified","published"):
                if key in prior and instance.get(key)!=prior[key]:raise ValueError("Local raw fallback was altered")
        else:
            import math
            points=instance["corners_px"]
            if (not isinstance(points,list) or len(points)!=8 or any(not isinstance(p,list) or len(p)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points)
                or instance.get("pose_valid") is not True):
                raise ValueError("Accepted geometric output is not a finite valid fixed8 prediction")
        instance['pose_diagnostic']['deployed']=accepted
        instance['selected_pose_source']='accepted_corner_geometry' if accepted else 'same_checkpoint_raw_pose_fallback'
    result.update(schema="action4.selected_corner_prediction.v1",producer_schema=payload["schema"],quality_ready=True,deployment_admitted=True,accuracy_accepted=False,diagnostic_candidate=False,
        producer_diagnostic_candidate=payload.get("diagnostic_candidate"),
        sealed_corner_selection=copy.deepcopy(seal_binding),global_selected_branch='candidate',
        parent_stage_fallback=copy.deepcopy(contract['parent_stage_fallback']),selection_scope='calibrated checkpoint; not real-video accuracy proof')
    return result


def predict_selected_corner(pipeline,rgb,K,canonical,*,sam_context):
    selection=getattr(pipeline,'_sealed_corner_selection',None)
    if not isinstance(selection,SealedCornerSelection):raise ValueError('Explicit sealed corner selection required')
    contract=selection.consume(pipeline)
    payload=pipeline.predict_corner(rgb,K,canonical,sam_context=sam_context,diagnostic_candidate=True)
    if payload['identity']!=pipeline.identity:raise ValueError('Prediction uses a different model/decoder identity')
    return selected_output(payload,contract,json.loads(selection.seal_binding_json))
