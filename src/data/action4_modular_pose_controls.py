"""Source-bound pose banks and explicit sliders; diagnostic-only, no renderer.
Import/propose/validate use CPU/cache metadata. Scenes use the existing sampler.
"""
from pathlib import Path
from copy import deepcopy
import json,math,re,hashlib
import numpy as np
from data import action4_modular_parameters as params
from data import action4_modular_layout_proposals as layouts
from data import action4_modular_assembly as assembly
from data import action4_global_environment as env
SCHEMA='action4.modular_pose_controls.v1'
STATUS='POSE_CONTROLS_ENGINEERING_CANDIDATE_PENDING_ACTUAL_SCENE'
SIDES=('left','right')
DERIVED_PRESET_RESOLVER_REF={'path': 'outputs/action4_derived_lift_preset_prepare_v001/derived_preset_resolver.py', 'sha256': '0914eaa90d003b5c741b429877fe6d7298444c538097e4f5046009b57da14e99'}
LIMITS={'common_distance_delta_m':[-.004,.004],'pair_spread_delta_m':[-.008,.008],'common_height_delta_m':[-.004,.004],'rotation_each_axis_deg':[-8.,8.],'area_light_gain':[.9,1.1],'exposure_offset_ev':[-.15,.15]}


def _bytes(reference,dependencies=None):
 """Fresh finite reference check; no mutable global cache or recursive crawl."""
 if not isinstance(reference,dict) or set(reference)!={'path','sha256'} or not isinstance(reference['sha256'],str) or re.fullmatch(r'[0-9a-f]{64}',reference['sha256']) is None:raise ValueError('Explicit path/SHA256 dependency required')
 path=params.bounded(reference['path']);name=str(path.relative_to(params.ROOT))
 if dependencies is not None:
  previous=dependencies.get(name)
  if previous is not None and previous!=reference['sha256']:raise ValueError('Conflicting dependency reference: '+name)
 data=path.read_bytes()
 if hashlib.sha256(data).hexdigest()!=reference['sha256']:raise ValueError('Frozen dependency changed: '+name)
 if dependencies is not None:dependencies[name]=reference['sha256']
 return data


def _read(reference,dependencies=None):return json.loads(_bytes(reference,dependencies))


def default_controls(bank_id='f0'):
 return {'bank_id':bank_id,'common_distance_delta_m':0.,'pair_spread_delta_m':0.,'common_height_delta_m':0.,'rotation_brace_local_xyz_deg':{s:[0.,0.,0.] for s in SIDES},'installation_angle_offset_deg':{s:[0.,0.,0.] for s in SIDES},'area_light_gain':1.,'exposure_offset_ev':0.,'occlusion_target_interval':None}


BANK_CACHE_TRANSFORM_REPLAY_ATOL=1e-9  # Existing named render-matrix replay policy.


def _coverage_banks(reference,source,pc,dependencies,registry,evidence_reader=None):
 _r=evidence_reader.read if evidence_reader is not None else _read;_b=evidence_reader.bytes if evidence_reader is not None else _bytes
 """Data-driven finite source-bound anchors, never inline matrix overrides.
 Frozen metadata is compared to serialized authority, not recomputed matrices.
 Actual scene/mask/range acceptance is deliberately outside this registry.
 """
 package=_r(registry['package_ref'],dependencies)
 if package['schema']!='action4.scene_coverage_candidate_handoff.v1' or package['status'] not in ('SEVEN_CACHE_FEASIBLE_NEW_BANK_CANDIDATES_PENDING_ACTUAL_SCENE','FINITE_CACHE_FEASIBLE_NEW_BANK_CANDIDATES_PENDING_ACTUAL_SCENE') or registry['source_context_sha256']!=package['source_context_sha256'] or package['source_context_sha256']!=env.digest(source):raise ValueError('Registry/package/source identity mismatch')
 if any(package[k] is not False for k in ('dataset_generation_allowed','training_accepted','sliders_actual_boundary_validated')) or package['provisional_sliders']!=LIMITS or package['large_bank_anchors_are_engineering_not_measured_motion_limits'] is not True:raise ValueError('Cache bank cannot promote actual evidence or control ranges')
 invariants=package['preserved_invariants']
 if invariants['W']!=source['W_from_capture'] or invariants['K']!=source['K_native'] or invariants['sides']!=source['sides'] or invariants['wearing_frames']!=source['wearing_frames'] or invariants['canonical8_m']!=pc['fixed']['canonical8_m'] or invariants['instance_ids']!={'left':1,'right':2}:raise ValueError('Registry changed fixed W/K/A/S/F/P/canonical8/physical IDs')
 declared=_r(package['declared_source_ref'],dependencies);checks=_r(package['checks_ref'],dependencies)
 if declared['schema']!='action4.finite_scene_coverage_candidates.v1' or declared['source_context_sha256']!=package['source_context_sha256'] or declared['frozen_invariants']!=invariants or checks['declared_candidates_ref']!=package['declared_source_ref'] or checks['control_expected493_observed']!=493:raise ValueError('Bound declared candidates and cache control required')
 for ref in declared['sources'].values():_b(ref,dependencies)
 _b(checks['checker_ref'],dependencies)
 rows=registry['rows'];candidates=package['candidates']
 if not isinstance(rows,list) or not rows or any(not isinstance(r,dict) or set(r)!={'bank_id','candidate_id'} or r['bank_id']!=r['candidate_id'] or not isinstance(r['bank_id'],str) or re.fullmatch(r'[a-z][a-z0-9_]{0,63}',r['bank_id']) is None or r['bank_id'] in ('f0','table_conjugate') for r in rows) or len({r['bank_id'] for r in rows})!=len(rows):raise ValueError('Unique finite registry IDs without aliases or legacy overrides required')
 if not isinstance(candidates,list) or not candidates or len({r['id'] for r in candidates})!=len(candidates):raise ValueError('Unique finite source candidate IDs required')
 by_id={r['id']:r for r in candidates}
 if any(r['candidate_id'] not in by_id for r in rows):raise ValueError('Registry ID missing from frozen package')
 result={}
 for row in rows:
  candidate=by_id[row['candidate_id']]
  name=candidate['id'];summary=candidate['cache_summary'];proof=_r(candidate['cache_evidence_ref'],dependencies)
  original=[r for r in declared['candidates'] if r['id']==name];checked=[r for r in checks['cases'] if r['id']==name]
  if len(original)!=1 or any(candidate[k]!=original[0][k] for k in ('id','family','B_capture_by_side','provenance','engineering_not_measured_pose','actual_scene_accepted')) or len(checked)!=1 or checked[0]!=summary or proof['summary']!=summary or len(proof['cross_pairs'])!=81:raise ValueError('Bank differs from declared candidate or bound cache evidence')
  if candidate['family'] not in ('coarse_orientation','near_double','partial_right_truncation') or candidate['engineering_not_measured_pose'] is not True or candidate['actual_scene_accepted'] is not False:raise ValueError('Engineering bank provenance required')
  if summary['cache_gate_pass'] is not True or summary['cross_surface_unresolved_pairs']!=0 or any(summary[k] is not False for k in ('actual_scene_checked','actual_masks_checked','containment_acceptance','candidate_accepted_for_production')) or summary['environment_table_collision_pending'] is not True:raise ValueError('Cache prior is not actual scene or family acceptance')
  if set(candidate['B_capture_by_side'])!=set(SIDES):raise ValueError('Both physical actor anchors required')
  W=assembly.rigid(source['W_from_capture'])
  for side in SIDES:
   B=assembly.rigid(candidate['B_capture_by_side'][side]);B0=assembly.rigid(source['sides'][side]['base_B_capture'])
   cache_delta=assembly.rigid(summary['world_delta_by_side'][side])
   if not np.allclose(W@B@np.linalg.inv(W@B0),cache_delta,atol=BANK_CACHE_TRANSFORM_REPLAY_ATOL,rtol=0):raise ValueError('Bank matrix differs from bound cache transform proof')
  # Frame numbers describe orientation-source provenance only. B includes the
  # separately declared engineering translation, not a measured video pose.
  anchor=candidate['provenance'].get('right_orientation_anchor',540) if isinstance(candidate['provenance'],dict) else 540
  result[name]={'status':'CANDIDATE_CACHE_SCENE_COVERAGE_ONLY','B_capture_by_side':deepcopy(candidate['B_capture_by_side']),'source_anchor_frames_by_side':{'left':0,'right':anchor},'engineering_not_pose_GT':True,'actual_scene_accepted':False,'registry_provenance':{'registry_ref':deepcopy(reference),'package_ref':deepcopy(registry['package_ref']),'candidate_id':name,'candidate_family':candidate['family'],'candidate_provenance':deepcopy(candidate['provenance']),'cache_evidence_ref':deepcopy(candidate['cache_evidence_ref']),'engineering_not_pose_GT':True,'actual_scene_accepted':False,'ranges_actual_boundary_validated':False}}
 return result


def _partial_forearm_banks(reference,source,pc,dependencies,group,evidence_reader=None):
 _r=evidence_reader.read if evidence_reader is not None else _read;_b=evidence_reader.bytes if evidence_reader is not None else _bytes
 """Consume this package's own finite prescription/result/proofs, not coverage-v1."""
 package=_r(group['package_ref'],dependencies)
 if package.get('schema')!='action4.partial_forearm_bank_candidates.v2' or package['status']!='TWO_PHYSICAL_KNOB_CACHE_CANDIDATES_PENDING_NATIVE_MASKS' or package['source_context_sha256']!=env.digest(source):raise ValueError('Explicit partial-forearm package/context required')
 if any(package[k] is not False for k in ('dataset_generation_allowed','training_accepted')):raise ValueError('Partial package cannot promote data')
 prescription=_r(package['prescription_ref'],dependencies);result=_r(package['result_ref'],dependencies);verification=_r(package['verification_ref'],dependencies)
 if prescription.get('schema')!='action4.partial_forearm_interpolation_prescription.v2' or prescription['status']!='PREDECLARED_BEFORE_ANY_V002_GEOMETRY_OR_RAY_CHECK' or prescription['source_context_sha256']!=package['source_context_sha256'] or prescription['initial_slider_limits']!=LIMITS or any(prescription[k] is not False for k in ('mask_acceptance','dataset_generation_allowed','training_accepted')):raise ValueError('Bound nonaccepted partial prescription required')
 invariants=package['fixed_invariants']
 if invariants!=prescription['fixed_invariants'] or invariants['W']!=source['W_from_capture'] or invariants['K']!=source['K_native'] or invariants['sides']!=source['sides'] or invariants['wearing_frames']!=source['wearing_frames'] or invariants['canonical8_m']!=pc['fixed']['canonical8_m'] or invariants['instance_ids']!={'left':1,'right':2}:raise ValueError('Partial package changed fixed source geometry')
 for ref in prescription['sources'].values():_b(ref,dependencies)
 original=_r(prescription['prior_declaration'],dependencies)
 # This is the explicit reused prior record, never a new partial493 control.
 _b(prescription['prior_summary'],dependencies);_b(prescription['prior_checker'],dependencies)
 _b(result['implementation_ref'],dependencies)
 source_config=_r(prescription['sources']['config'],dependencies)
 if prescription['visible_rotation_source_B']!=source_config['banks']['table_conjugate']['B_capture_by_side']['right']:raise ValueError('Partial visible orientation source differs')
 if result['status']!='FINITE_PHYSICAL_INTERPOLATION_CACHE_COMPLETE_ACTUAL_MASKS_PENDING' or result['prescription_ref']!=package['prescription_ref'] or result['calibration_493_reused']!=prescription['prior_summary'] or result['attempt_count']!=len(result['all_attempts']) or result['attempt_count']>prescription['maximum_all_checks']:raise ValueError('Partial result/prescription execution binding')
 if any(result[k] is not True for k in ('prior_package_and_sources_unchanged','geometry_algorithm_unchanged','no_full_alpha_domain_acceptance','no_linear_alpha_to_mask_area_claim')) or any(result[k] is not False for k in ('dataset_generation_allowed','training_accepted')) or verification['status']!='PASS_OUTPUT_RECOUNT_RIGID_COUPLING_AND_PRIOR_IDENTITY' or verification['actual_masks_checked'] is not False:raise ValueError('Partial cache result cannot imply native masks or ranges')
 knob=package['scalar_knob']
 if knob['mechanism']!=prescription['interpolation'] or knob['accepted_continuous_interval'] is not None or any(knob[k] is not True for k in ('only_two_discrete_candidate_alphas_have_partial_cache_evidence','not_linear_to_actual_area','initial_mild_pose_sliders_remain_separate_and_unvalidated_in_combination')):raise ValueError('Discrete alpha evidence only; no continuous mask control')
 rows=group['rows'];candidates=package['banks']
 _validate_registry_rows(rows,candidates)
 by_id={r['id']:r for r in candidates};out={};W=assembly.rigid(source['W_from_capture'])
 roles=('core','skin','band','quick_release','usb_connector','white_module','ball','link','cable');pair_keys={a+'__'+b for a in roles for b in roles}
 for row in rows:
  bank=by_id[row['candidate_id']];endpoint=bank['endpoint_id'];alpha=bank['alpha'];proof=_r(bank['evidence_ref'],dependencies);summary=proof['summary']
  if endpoint not in prescription['fully_occluded_endpoint_ids'] or type(alpha) not in (float,int) or not math.isfinite(alpha) or not 0<alpha<1:raise ValueError('Explicit evaluated partial alpha/endpoint required')
  endpoint_rows=[r for r in original['candidates'] if r['id']==endpoint]
  if len(endpoint_rows)!=1 or endpoint_rows[0]['B_capture_by_side']!=prescription['fully_occluded_endpoint_Bs'][endpoint]:raise ValueError('Partial endpoint differs from frozen declaration')
  selected=[r for r in result['selected'] if r['endpoint']==endpoint and r['alpha']==alpha];attempts=[r for r in result['all_attempts'] if r['id']==summary['id']];verified=[r for r in verification['checks'] if r['id']==summary['id']]
  if len(selected)!=1 or selected[0]!=summary or len(attempts)!=1 or attempts[0]!=summary or len(verified)!=1 or any(verified[0][k] is not True for k in ('cross81_recount_pass','semantic_ray_recount_pass','proper_rigid_pass','fixed_core_center_and_left_pass')):raise ValueError('Partial selected/per-case proof mismatch')
  if summary['endpoint']!=endpoint or summary['alpha']!=alpha or any(bank[k]!=summary[k] for k in ('B_capture_by_side','core_C_capture_by_side','canonical8_projection','visible_core_ray_fraction','own_forearm_ray_fraction','cache_geometry_gate_pass')) or bank['ray_counts']!=summary['rays']:raise ValueError('Partial bank differs from exact selected proof')
  if bank['actual_scene_accepted'] is not False or summary['actual_scene_mask_accepted'] is not False or summary['not_actual_pixel_area_fraction'] is not True or any(summary[k] is not True for k in ('cache_geometry_gate_pass','all_original8_inframe','partial_proxy_pass','selected_eligibility_pass')):raise ValueError('Partial cache candidate status cannot promote masks')
  if set(proof['cross81_pairs'])!=pair_keys or summary['cross_surface_unresolved_pairs']!=0 or sum(len(v['outside_or_unresolved_pairs']) for v in proof['cross81_pairs'].values())!=0:raise ValueError('Complete partial cross81 proof must remain clear')
  rays=proof['semantic_target_rays'];counts=summary['rays'];thresholds=prescription['thresholds'];categories=('target_core_first','own_forearm_first','own_other_skin_first','own_mount_first','other_actor_first')
  if len(rays)!=counts['target_core_hits'] or counts['grid_rays']!=math.prod(thresholds['ray_grid']) or any(sum(r['category']==k for r in rays)!=counts[k] for k in categories) or sum(counts[k] for k in categories)!=len(rays):raise ValueError('Partial semantic ray recount mismatch')
  if any(r['nearest_piece']!='right/skin' or r['source_region_id']!=thresholds['forearm_source_region_id'] for r in rays if r['category']=='own_forearm_first'):raise ValueError('Partial forearm source attribution mismatch')
  n=len(rays);vf=counts['target_core_first']/n if n else -1;ff=counts['own_forearm_first']/n if n else -1;mf=counts['own_mount_first']/n if n else -1
  if n<thresholds['minimum_target_core_hit_rays'] or not thresholds['visible_core_fraction'][0]<=vf<=thresholds['visible_core_fraction'][1] or ff<thresholds['minimum_own_forearm_fraction'] or mf>thresholds['maximum_own_mount_fraction'] or counts['other_actor_first']>thresholds['maximum_other_actor_first_hits'] or vf!=summary['visible_core_ray_fraction'] or ff!=summary['own_forearm_ray_fraction'] or mf!=summary['own_mount_ray_fraction']:raise ValueError('Partial ray eligibility or fraction mismatch')
  if set(bank['B_capture_by_side'])!=set(SIDES) or bank['B_capture_by_side']['left']!=prescription['fixed_left_B']:raise ValueError('Partial fixed-left actor changed')
  # Replay only the declared metadata law at the selected discrete alpha.
  R0=assembly.rigid(prescription['visible_rotation_source_B'])[:3,:3];R1=assembly.rigid(prescription['fully_occluded_endpoint_Bs'][endpoint]['right'])[:3,:3];relative=R0.T@R1
  theta=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
  if not 1e-8<theta<np.pi-1e-6:raise ValueError('Declared proper shortest SO3 interpolation domain')
  axis=(relative-relative.T)/(2*np.sin(theta));angle=alpha*theta;expected_R=R0@(np.eye(3)+np.sin(angle)*axis+(1-np.cos(angle))*(axis@axis))
  if not np.allclose(assembly.rigid(bank['B_capture_by_side']['right'])[:3,:3],expected_R,atol=BANK_CACHE_TRANSFORM_REPLAY_ATOL,rtol=0):raise ValueError('Partial alpha rotation differs from prescribed law')
  for side in SIDES:
   B=assembly.rigid(bank['B_capture_by_side'][side]);A=assembly.rigid(source['sides'][side]['A_brace_from_CAD']);C=assembly.rigid(bank['core_C_capture_by_side'][side]);B0=assembly.rigid(source['sides'][side]['base_B_capture'])
   # The proof stores exact B, not D. With frozen W/B0 its D is determined.
   assembly.rigid(W@B@np.linalg.inv(W@B0))
   if not np.allclose(B@A,C,atol=BANK_CACHE_TRANSFORM_REPLAY_ATOL,rtol=0):raise ValueError('Partial B/A/core coupling differs from proof')
   if side=='right' and not np.allclose(C[:3,3],prescription['right_core_center_capture_m'],atol=BANK_CACHE_TRANSFORM_REPLAY_ATOL,rtol=0):raise ValueError('Partial fixed core center changed')
  out[bank['id']]={'status':'CANDIDATE_PARTIAL_FOREARM_CACHE_ONLY','B_capture_by_side':deepcopy(bank['B_capture_by_side']),'source_anchor_frames_by_side':{'left':None,'right':None},'engineering_not_pose_GT':True,'actual_scene_accepted':False,'registry_provenance':{'registry_ref':deepcopy(reference),'package_ref':deepcopy(group['package_ref']),'package_kind':'partial_forearm_v2','candidate_id':bank['id'],'candidate_family':'self_forearm_occlusion','endpoint_id':endpoint,'alpha':alpha,'orientation_provenance':'finite engineered SO3 interpolation; no exact source frame or measured pose','prescription_ref':deepcopy(package['prescription_ref']),'evidence_ref':deepcopy(bank['evidence_ref']),'engineering_not_pose_GT':True,'actual_scene_accepted':False,'ranges_actual_boundary_validated':False,'semantic_rays_are_not_native_mask_fraction':True}}
 return out


def _validate_registry_rows(rows,candidates):
 if not isinstance(rows,list) or not rows or any(not isinstance(r,dict) or set(r)!={'bank_id','candidate_id'} or r['bank_id']!=r['candidate_id'] or not isinstance(r['bank_id'],str) or re.fullmatch(r'[a-z][a-z0-9_]{0,63}',r['bank_id']) is None or r['bank_id'] in ('f0','table_conjugate') for r in rows) or len({r['bank_id'] for r in rows})!=len(rows):raise ValueError('Unique finite registry IDs without aliases or legacy overrides required')
 if not isinstance(candidates,list) or not candidates or len({r['id'] for r in candidates})!=len(candidates):raise ValueError('Unique finite source candidate IDs required')
 if any(r['candidate_id'] not in {c['id'] for c in candidates} for r in rows):raise ValueError('Registry ID missing from frozen package')


def _registry_banks(reference,source,pc,dependencies,evidence_reader=None):
 _r=evidence_reader.read if evidence_reader is not None else _read;_b=evidence_reader.bytes if evidence_reader is not None else _bytes
 registry=_r(reference,dependencies);base={'schema','status','source_context_sha256','dataset_generation_allowed','training_accepted','ranges_actual_boundary_validated'}
 if registry.get('status')!='FINITE_CACHE_CANDIDATES_PENDING_ACTUAL_SCENE' or registry.get('source_context_sha256')!=env.digest(source) or any(registry.get(k) is not False for k in ('dataset_generation_allowed','training_accepted','ranges_actual_boundary_validated')):raise ValueError('Nonaccepted registry/context required')
 if registry.get('schema')=='action4.modular_pose_bank_registry.v2':
  if set(registry)!=base|{'package_ref','rows'}:raise ValueError('Explicit finite pose-bank registry v2')
  return _coverage_banks(reference,source,pc,dependencies,registry,evidence_reader)
 if registry.get('schema')!='action4.modular_pose_bank_registry.v3' or set(registry)!=base|{'packages'} or not isinstance(registry['packages'],list) or not registry['packages']:raise ValueError('Explicit finite typed registry v3 required')
 result={}
 for group in registry['packages']:
  if set(group)!={'kind','package_ref','rows'} or group['kind'] not in ('coverage_v1','partial_forearm_v2'):raise ValueError('Known typed package reference required')
  if group['kind']=='coverage_v1':banks=_coverage_banks(reference,source,pc,dependencies,{**group,'source_context_sha256':registry['source_context_sha256']},evidence_reader)
  else:banks=_partial_forearm_banks(reference,source,pc,dependencies,group,evidence_reader)
  if set(result)&set(banks):raise ValueError('Conflicting duplicate bank IDs across packages')
  result.update(banks)
 return result


def load_config(config,context):
 if isinstance(context,_ScopedPoseContext):
  p=context._loaded(config);return p['config'],p['bounds'],p['config_ref']
 return _load_config(config,context,{})


def dependency_refs(config,context):
 """Validated finite path->SHA closure consumed by this exact config loader.
 Includes config/module/context and typed validators' explicit edges only.
 Does not recurse into arbitrary historical manifests or unused proof records.
 A nondefault per-entry authority and generated parameter ref belong to callers.
 """
 if isinstance(context,_ScopedPoseContext):return context._loaded(config)['dependencies']
 dependencies={};_load_config(config,context,dependencies)
 return dict(sorted(dependencies.items()))


def _load_config(config,context,dependencies):
 from data import action4_modular_scene_sampling as sampling
 sampling.verify_sources(context);path=params.bounded(config);config_ref=layouts.ref(path);c=_read(config_ref,dependencies)
 if c.get('schema')=='action4.modular_pose_bank_config.v5':
  import importlib.util
  if c.get('resolver_ref')!=DERIVED_PRESET_RESOLVER_REF:raise ValueError('Exact reviewed derived preset resolver required')
  _bytes(DERIVED_PRESET_RESOLVER_REF,dependencies)
  spec=importlib.util.spec_from_file_location('action4_derived_preset_resolver',params.bounded(DERIVED_PRESET_RESOLVER_REF['path']))
  resolver=importlib.util.module_from_spec(spec);spec.loader.exec_module(resolver)
  expanded,pc=resolver.expand_v5(c,context,dependencies,candidate_consumer_ref=layouts.ref(__file__),resolver_ref=DERIVED_PRESET_RESOLVER_REF)
  return expanded,pc,config_ref
 if 'src/data/action4_mask_semantics.py' not in context['source_context']['frozen_refs']:raise ValueError('Shared mask semantics must be explicitly frozen in the source context')
 for name,digest in context['source_context']['frozen_refs'].items():_bytes({'path':name,'sha256':digest},dependencies)
 required={'schema','version','status','factory_ref','source_context_ref','source_context_sha256','parameter_bounds_ref','default_entry_source_ref','consumer_module_ref','banks','limits','occupancy_prior','environment_visibility','test_target_count','dataset_generation_allowed','training_accepted','ranges_actual_boundary_validated'}
 registry_version=c.get('schema') in ('action4.modular_pose_bank_config.v2','action4.modular_pose_bank_config.v3','action4.modular_pose_bank_config.v4')
 bridge_version=c.get('schema')=='action4.modular_pose_bank_config.v4'
 if bridge_version:required=required|{'bridge_ref'}
 if registry_version:required=required|{'registry_ref'}
 if set(c)!=required or c['schema'] not in ('action4.modular_pose_bank_config.v1','action4.modular_pose_bank_config.v2','action4.modular_pose_bank_config.v3','action4.modular_pose_bank_config.v4') or c['status']!='ENGINEERING_BANKS_NOT_PRODUCTION_ACCEPTED':raise ValueError('Explicit pose-bank config only')
 if c['dataset_generation_allowed'] is not False or c['training_accepted'] is not False or c['ranges_actual_boundary_validated'] is not False or c['limits']!=LIMITS:raise ValueError('Initial engineering controls only; bounds cannot self-expand')
 if c['consumer_module_ref']!=layouts.ref(__file__) or c['test_target_count']!=2:raise ValueError('Exact module and dual-only test authority')
 if c['environment_visibility']!={'wall_visible':True,'table_visible':True} or c['occupancy_prior']!={'minimum_bbox_area_fraction':.0001,'maximum_bbox_area_fraction':.25,'minimum_bbox_inside_image_fraction':.01,'kind':'engineering_bbox_prior_not_actual_mask_support'}:raise ValueError('Explicit frozen environment and engineering occupancy prior')
 _bytes(c['consumer_module_ref'],dependencies)
 source=_read(c['source_context_ref'],dependencies);_read(c['factory_ref'],dependencies)
 _bytes(c['parameter_bounds_ref'],dependencies);pc=params.load_config(c['parameter_bounds_ref']['path'])
 for ref in pc['source_refs'].values():_bytes(ref,dependencies)
 policy=pc['quota_contract'].get('presence_quota_policy')
 if policy is not None:_bytes(policy,dependencies)
 if pc['reference_sampling']['bounds'] is not None:
  for ref in pc['reference_sampling']['evidence_refs']:_bytes(ref,dependencies)
 _read(c['default_entry_source_ref'],dependencies)
 if source!=context['source_context'] or c['source_context_sha256']!=context['source_context_sha256'] or source['scene_config_ref']!=c['factory_ref'] or context.get('base_kind')!='palmar_wearing':raise ValueError('Exact palmar factory/context binding')
 params.validate_source_context(pc,source)
 if set(c['banks'])!={'f0','table_conjugate'}:raise ValueError('Explicit finite two-bank version only')
 for name,bank in c['banks'].items():
  if set(bank)!={'status','B_capture_by_side','source_refs','source_anchor_frames_by_side','engineering_not_pose_GT','actual_scene_accepted','cache_case_id'} or bank['actual_scene_accepted'] is not False or bank['engineering_not_pose_GT'] is not True:raise ValueError('Bank cannot promote a source candidate')
  refs={k:_read(v,dependencies) for k,v in bank['source_refs'].items()}
  for side in SIDES:assembly.rigid(bank['B_capture_by_side'][side])
  if name=='f0':
   if bank['status']!='VERIFIED_F0_SOURCE_COMPONENTS_VARIATION_PENDING' or any(bank['B_capture_by_side'][s]!=source['sides'][s]['base_B_capture'] for s in SIDES):raise ValueError('Exact original f0 bank')
   if {r['case'] for r in refs['presence_receipt']['cases']}!={'none','left','right','both'}:raise ValueError('Bound original four-presence source receipt')
  else:
   if bank['status']!='CANDIDATE_CACHE_CROSS_CLEARANCE_ONLY' or bank['B_capture_by_side']['left']!=refs['left_candidate']['B_capture'] or bank['B_capture_by_side']['right']!=refs['right_candidate']['sides']['right']['brace_B_capture']:raise ValueError('Exact candidate LEFT-B and fixed RIGHT-B provenance')
   selected=refs['selected_candidate']
   if selected.get('schema')!='action4.explicit_finite_wearing_pose_candidate.v1' or selected['training_accepted'] is not False or selected['dataset_generation_allowed'] is not False or selected['left']['brace_B_capture']!=bank['B_capture_by_side']['left'] or selected['right_fixed']['brace_B_capture']!=bank['B_capture_by_side']['right'] or selected['left']['A_brace_from_CAD']!=source['sides']['left']['A_brace_from_CAD'] or selected['W_from_capture']!=source['W_from_capture'] or selected['K_native']!=source['K_native']:raise ValueError('Selected bank source must preserve frozen wearing geometry and camera')
   report=refs['cache_report'];matches=[r for r in report['cases'] if r['candidate']==bank['cache_case_id']]
   if report['cross_pairs_each']!=81 or report['control_expected_actual_surface_count']!=493 or report['control_cache_surface_count']!=493 or len(matches)!=1 or matches[0]['outside_or_unresolved_surface_pairs']!=0:raise ValueError('Cache prior provenance required; it is not actual scene acceptance')
 bridge=None
 if bridge_version:
  from data import action4_execution_context_bridge as bridge_module
  _bytes(layouts.ref(bridge_module.__file__),dependencies)
  bridge=bridge_module.ExecutionBridge(c['bridge_ref'],source,dependencies)
  if c['source_context_ref']!=bridge.spec['current_source_ref'] or c['parameter_bounds_ref']!=bridge.spec['current_bounds_ref'] or c['registry_ref']!=bridge.historical_bank_config['registry_ref']:raise ValueError('Current v4 config differs from explicit bridge roots')
 if registry_version:c['banks'].update(_registry_banks(c['registry_ref'],bridge.historical_source if bridge else source,pc,dependencies,bridge))
 if bridge:bridge.close()
 return c,pc,config_ref


def _entry(entry,reference):
 source=_read(reference)
 if source.get('schema')=='action4.modular_entry_authority.v1':
  if source.get('status')!='FROZEN_ENTRY_IDENTITIES_ONLY_NOT_RENDER_PLAN' or source.get('training_accepted') is not False or source.get('dataset_generation_allowed') is not False:raise ValueError('Explicit nonaccepted immutable entry authority')
 elif source.get('schema_version')!='action4.synthesis.v2':raise ValueError('Immutable existing plan or explicit entry authority required')
 entries=source.get('entries',[])
 if not entries or len({e['index'] for e in entries})!=len(entries) or len({e['sample_id'] for e in entries})!=len(entries):raise ValueError('Unique source entries')
 if not isinstance(entry,dict) or set(entry)!=params.ENTRY_KEYS or type(entry['index']) is not int:raise ValueError('Full exact original entry identity')
 rows=[e for e in entries if e['index']==entry['index']]
 if len(rows)!=1 or rows[0]!=entry:raise ValueError('Entry differs from immutable source')
 from data import action4_synthesis as synthesis
 synthesis.target_indices(entry)
 if entry['scene_family'] not in params.FAMILIES or entry['split'] not in ('train','eval','test') or any(type(entry[k]) is not int or entry[k]<0 for k in ('seed','background_seed')):raise ValueError('Original family/split/seeds')
 if entry['split']=='test' and entry['target_count']!=2:raise ValueError('New pose controls require dual-target test entries')
 return deepcopy(entry)


def _controls(controls,config,presence):
 if not isinstance(controls,dict) or set(controls)!=set(default_controls()) or controls['bank_id'] not in config['banks']:raise ValueError('Exact supported slider fields and bank')
 json.dumps(controls,allow_nan=False)
 def bounded_number(value,bounds):return type(value) in (int,float) and math.isfinite(value) and bounds[0]<=value<=bounds[1]
 for key in ('common_distance_delta_m','pair_spread_delta_m','common_height_delta_m','area_light_gain','exposure_offset_ev'):
  if not bounded_number(controls[key],LIMITS[key]):raise ValueError('Slider exceeds initial engineering bound: '+key)
 if not isinstance(controls['rotation_brace_local_xyz_deg'],dict) or not isinstance(controls['installation_angle_offset_deg'],dict):raise ValueError('Explicit per-side controls required')
 for side in SIDES:
  angles=controls['rotation_brace_local_xyz_deg'].get(side);installation=controls['installation_angle_offset_deg'].get(side)
  if not isinstance(angles,list) or len(angles)!=3 or any(not bounded_number(a,LIMITS['rotation_each_axis_deg']) for a in angles):raise ValueError('Per-actor bounded rotation only')
  if not isinstance(installation,list) or len(installation)!=3 or any(type(a) not in (int,float) or a!=0 for a in installation):raise ValueError('Installation offset is locked to0')
 if set(controls['rotation_brace_local_xyz_deg'])!=set(SIDES) or set(controls['installation_angle_offset_deg'])!=set(SIDES):raise ValueError('Both actors explicit')
 interval=controls['occlusion_target_interval']
 if config['banks'][controls['bank_id']].get('registry_provenance',{}).get('package_kind')=='partial_forearm_v2' and interval is not None:raise ValueError('Partial forearm bank requires unset global occlusion interval; per-instance scope pending')
 if interval is not None and (not any(presence) or not isinstance(interval,list) or len(interval)!=2 or any(not bounded_number(a,[0.,1.]) for a in interval) or interval[0]>interval[1]):raise ValueError('Occlusion is a target interval for present targets only')
 if 'derived_preset' in config['banks'][controls['bank_id']]:
  if any(controls[k]!=0 for k in ('common_distance_delta_m','common_height_delta_m','pair_spread_delta_m')) or any(a!=0 for s in SIDES for a in controls['rotation_brace_local_xyz_deg'][s]):raise ValueError('Discrete derived lift anchor: local pose sliders disabled')
 return deepcopy(controls)


def _cache_prior(actor_poses,pc,presence):
 # Reuse exact18-piece normalization/transforms from the existing geometry helper.
 full=layouts._pair_prior(actor_poses,pc);pieces={};actors={}
 for side,on in zip(SIDES,presence):
  pieces[side]={k:v for k,v in full['piece_AABB_by_side_capture_m'][side].items() if on or k in ('skin','band','white_module','ball')}
  actors[side]={'min_m':np.min([v['min_m'] for v in pieces[side].values()],axis=0).tolist(),'max_m':np.max([v['max_m'] for v in pieces[side].values()],axis=0).tolist()}
 unresolved=[]
 for a,A in pieces['left'].items():
  for b,B in pieces['right'].items():
   if not np.any(np.maximum(np.asarray(A['min_m'])-B['max_m'],np.asarray(B['min_m'])-A['max_m'])>0):unresolved.append({'left_piece':a,'right_piece':b,'status':'AABB_overlap_requires_actual_surface_check'})
 return {'status':'CACHE_GEOMETRY_PRIOR_ONLY','actor_AABB_by_side_capture_m':actors,'piece_AABB_by_side_capture_m':pieces,'retained_piece_counts':{s:len(pieces[s]) for s in SIDES},'cross_pair_count':len(pieces['left'])*len(pieces['right']),'unresolved_AABB_pairs':unresolved,'near_plane_pass':all(v['min_m'][2]>pc['near_plane_m'] for v in actors.values()),'actual_surface_pass':False,'containment_accepted':False,'environment_collision_accepted':False,'actual_masks_pending':True}


def propose_pose_controls(entry,config,controls,context,*,entry_source_ref=None):
 c,pc,config_ref=load_config(config,context)
 consumer_ref=context._loaded(config)['consumer_ref'] if isinstance(context,_ScopedPoseContext) else layouts.ref(__file__)
 return _propose_from_loaded(entry,c,pc,config_ref,controls,context,consumer_ref,entry_source_ref=entry_source_ref)


def _propose_from_loaded(entry,c,pc,config_ref,controls,context,consumer_ref,*,entry_source_ref=None):
 """Loaded-data kernel; caller supplies the validated static payload explicitly.
 Entry authority, controls and every numeric/derived prior are replayed per entry.
 This private kernel does not itself establish source verification.
 """
 entry_ref=entry_source_ref or c['default_entry_source_ref'];entry=_entry(entry,entry_ref);presence=entry['target_presence'];controls=_controls(controls,c,presence);bank=c['banks'][controls['bank_id']];source=context['source_context'];actor={};sides={};derived={}
 for i,side in enumerate(SIDES):
  baseline=assembly.rigid(bank['B_capture_by_side'][side]);B=baseline.copy();B[:3,:3]=baseline[:3,:3]@params.rotation_xyz_degrees(controls['rotation_brace_local_xyz_deg'][side]);B[:3,3]+=np.array([(-.5 if side=='left' else .5)*controls['pair_spread_delta_m'],controls['common_height_delta_m'],controls['common_distance_delta_m']]);A=assembly.rigid(source['sides'][side]['A_brace_from_CAD']);C=B@A;B0=assembly.rigid(source['sides'][side]['base_B_capture']);delta=B@np.linalg.inv(B0)
  actor[side]={'brace_B_capture':B.tolist(),'core_C_capture':C.tolist(),'whole_actor_delta_capture':delta.tolist()}
  sides[side]={k:deepcopy(source['sides'][side][k]) for k in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace')};sides[side].update(present=presence[i],core_C_capture=C.tolist() if presence[i] else None,brace_B_capture=B.tolist() if presence[i] else None,whole_actor_delta_capture=delta.tolist() if presence[i] else None,translation_capture_m=(B[:3,3]-B0[:3,3]).tolist() if presence[i] else None,rotation_brace_local_xyz_deg=deepcopy(controls['rotation_brace_local_xyz_deg'][side]) if presence[i] else None)
  metric=params.project_canonical(C,source['K_native'],pc['fixed']['canonical8_m'],pc['near_plane_m']) if presence[i] else None;derived[side]=metric
  if metric is not None:
   occupancy=c['occupancy_prior']
   if not metric['valid_positive_depth'] or not occupancy['minimum_bbox_area_fraction']<=metric['bbox_area_over_image']<=occupancy['maximum_bbox_area_fraction'] or metric['bbox_inside_image_fraction']<occupancy['minimum_bbox_inside_image_fraction']:raise ValueError('Engineering native occupancy/positive-depth prior failed')
 prior=_cache_prior(actor,pc,presence)
 if not prior['near_plane_pass']:raise ValueError('Retained full-actor geometry near-plane prior failed')
 family=pc['family_profiles'][entry['scene_family']]
 out={'schema':SCHEMA,'status':STATUS,'estimated':True,'entry':entry,'target_presence':presence,'sides':sides,'actor_pose_capture_by_side':actor,'W_from_capture':deepcopy(source['W_from_capture']),'K_native':deepcopy(source['K_native']),'source_anchor_frames_by_side':{s:bank['source_anchor_frames_by_side'][s] if on else None for s,on in zip(SIDES,presence)},'controls':controls,'environment_controls':{**c['environment_visibility'],'area_light_gain':controls['area_light_gain'],'exposure_offset_ev':controls['exposure_offset_ev']},'scene_family_controls':{'family':entry['scene_family'],'required_capabilities':['whole_actor_rigid','target_presence','wall_table_visibility','area_light_gain','exposure_offset'],'pending_reasons':deepcopy(family.get('pending_reasons',[])),'coarse_occluders':[],'effect_pending':entry['scene_family'] not in ('wall_rear','hard_negative'),'effect_claim':'pose_controls_only_not_verified_family_effect'},'augmentation':{'counts_as_formal_coverage':False,'reference_or_augmentation':'generic_pose_bank_engineering_sliders','ranges_actual_boundary_validated':False},'derived':{'by_side':derived,'original8_unclamped':True,'exact_video_frame_match_required':False,'actual_mask_occlusion_by_instance_id':None},'occlusion_condition':{'target_interval':deepcopy(controls['occlusion_target_interval']),'actual_mask_verification_pending':True,'requested_interval_is_not_measured_occlusion':True},'safety_prior':prior,'provenance':{'source_context_sha256':context['source_context_sha256'],'bank_config_ref':config_ref,'entry_source_ref':deepcopy(entry_ref),'pose_controls_module_ref':deepcopy(consumer_ref),'bank_id':controls['bank_id'],'bank_status':bank['status']},'formal_entry_adapter':'PENDING_NOT_IMPLEMENTED_BY_DIAGNOSTIC_INTERFACE','dataset_generation_allowed':False,'training_accepted':False}
 if 'registry_provenance' in bank:out['provenance']['registry_bank']=deepcopy(bank['registry_provenance'])
 if c.get('schema') in ('action4.modular_pose_bank_config.v4','action4.modular_pose_bank_config.v5'):out['provenance']['execution_context_bridge_ref']=deepcopy(c['bridge_ref'])
 json.dumps(out,allow_nan=False);return out


def _same_derivatives(a,b,path=()):
 if type(a) is not type(b):return False
 if isinstance(b,dict):return set(a)==set(b) and all(_same_derivatives(a[k],v,path+(str(k),)) for k,v in b.items())
 if isinstance(b,list):return len(a)==len(b) and all(_same_derivatives(x,y,path+(str(i),)) for i,(x,y) in enumerate(zip(a,b)))
 if type(b) is float:
  if not math.isfinite(a) or not math.isfinite(b):return False
  if a==b:return True
  name='/'.join(path);tol=layouts._derived_replay_tolerance(name)
  if re.fullmatch(r'(?:actor_pose_capture_by_side|sides)/(?:left|right)/(?:brace_B_capture|core_C_capture|whole_actor_delta_capture)/[0-2]/[0-3]',name):tol=1e-9
  return tol is not None and abs(a-b)<=tol
 return a==b


def validate_pose_controls(parameters,context):
 if parameters.get('schema')!=SCHEMA or parameters.get('status')!=STATUS:raise ValueError('Explicit diagnostic pose controls schema/status')
 provenance=parameters['provenance'];expected=propose_pose_controls(parameters['entry'],provenance['bank_config_ref']['path'],parameters['controls'],context,entry_source_ref=provenance['entry_source_ref'])
 if not _same_derivatives(parameters,expected):raise ValueError('Pose controls differ from source-bound bank/entry/slider reconstruction')
 return deepcopy(parameters)


def verify_actual_pose_geometry(session):
 from data import action4_modular_pixels as pixels
 session.verifier(session.scene);capture=np.linalg.inv(assembly.rigid(session.parameters['W_from_capture']));near=session.scene.camera.data.clip_start;minimum={}
 for side in SIDES:
  depths=[]
  for name in session.bindings[side]['names']:
   obj=session.scene.objects[name]
   if obj.type=='MESH' and not obj.hide_render:
    V,_=assembly.world_geometry(obj);depths.append(float(assembly.transform(capture,V)[:,2].min()))
  if not depths or min(depths)<=near:raise ValueError('Actual retained actor near-plane failure')
  minimum[side]=min(depths)
 for instance,target in session.targets().items():
  C=capture@np.asarray(pixels._evaluated_world_matrix(target['core']));side='left' if instance==1 else 'right'
  if not np.allclose(C,session.parameters['actor_pose_capture_by_side'][side]['core_C_capture'],atol=2e-6,rtol=0):raise ValueError('Actual core differs from complete actor controls')
 return {'status':'ACTUAL_POSE_GEOMETRY_DIAGNOSTIC_ONLY','retained_actor_min_depth_m':minimum,'collision_report_status':session.contract['actor_collision_diagnostics']['status'],'actual_masks_pending':True,'training_accepted':False,'dataset_generation_allowed':False}


def verify_actual_masks(parameters,record):
 if parameters.get('schema')!=SCHEMA or any(record.get(k)!=v for k,v in parameters['entry'].items()):raise ValueError('Actual immutable pose-control entry required')
 ids={i for i,on in enumerate(parameters['target_presence'],1) if on};rows=record.get('instances',[])
 if record.get('width')!=3248 or record.get('height')!=2464 or record.get('excluded_instances') or {r['instance_id'] for r in rows}!=ids or len(rows)!=len(ids):raise ValueError('Original native physical IDs and retained mask records required')
 from data import action4_mask_semantics as mask_semantics
 verified=mask_semantics.verify_record_masks(record)['measurements_by_instance_id']
 values={};passes={};interval=parameters['controls']['occlusion_target_interval']
 for row in rows:
  total,inside,visible=(row[k] for k in ('area_full_amodal_px','area_inframe_amodal_px','area_visible_px'))
  if any(type(v) is not int for v in (total,inside,visible)) or not total>=inside>0 or visible<=0:raise ValueError('Actual positive mask counts required')
  measured=verified[str(row['instance_id'])];occlusion=measured['occlusion_inframe']
  values[str(row['instance_id'])]={'occlusion_inframe':occlusion,'occlusion_inframe_raw':measured['occlusion_inframe_raw'],'truncation_fraction':measured['truncation'],'mask_nonnested_boundary_pixels':measured['mask_nonnested_boundary_pixels'],'counts_full_inframe_visible':[total,inside,visible]}
  passes[str(row['instance_id'])]=interval is None or interval[0]<=occlusion<=interval[1]
 return {'status':'ACTUAL_MASK_NUMBERS_CHECK_ONLY_HOST_PNG_AND_FORMAL_GATES_PENDING','target_interval':deepcopy(interval),'actual_mask_metrics_by_instance_id':values,'target_interval_pass':all(passes.values()),'per_instance_interval_pass':passes,'host_PNG_validation_still_required':True,'dataset_generation_allowed':False,'training_accepted':False}


def rebind_historical_pose_parameters(historical_ref,current_config,context,*,scene_contract=False):
 if isinstance(context,_ScopedPoseContext):
  identity={'historical_ref':historical_ref,'current_config_ref':context._loaded(current_config)['config_ref'],'scene_contract':scene_contract}
  result=context._memo('historical_rebind',identity,lambda:_rebind_historical_pose_parameters(historical_ref,current_config,context,scene_contract=scene_contract))
  return tuple(result)
 return _rebind_historical_pose_parameters(historical_ref,current_config,context,scene_contract=scene_contract)


def _rebind_historical_pose_parameters(historical_ref,current_config,context,*,scene_contract=False):
 """Replay historical serialization with current typed validation and identities.
 Metadata changes are explicit and source-derived; no historical code executes.
 """
 from data import action4_execution_context_bridge as bridge_module
 c,_,cref=load_config(current_config,context)
 if c.get('schema') not in ('action4.modular_pose_bank_config.v4','action4.modular_pose_bank_config.v5'):raise ValueError('Explicit v4/v5 bridge config required')
 bridge=bridge_module.ExecutionBridge(c['bridge_ref'],context['source_context'])
 old=bridge.read(historical_ref)
 if scene_contract:
  if old.get('schema')!='action4.modular_dynamic_scene.v1' or old.get('source_context_sha256')!=bridge_module.digest(bridge.historical_source) or old.get('parameters_sha256')!=bridge_module.digest(old.get('parameters')):raise ValueError('Bound historical scene contract required')
  old=old['parameters']
 expected=propose_pose_controls(old['entry'],current_config,old['controls'],context,entry_source_ref=old['provenance']['entry_source_ref'])
 # Retain the exact historical numeric serialization. Current typed replay
 # remains authoritative within its existing named tolerances; the bridge then
 # proves exact historical preservation outside these four identity fields.
 current=deepcopy(old)
 for key in ('source_context_sha256','bank_config_ref','pose_controls_module_ref','execution_context_bridge_ref'):
  current['provenance'][key]=deepcopy(expected['provenance'][key])
 current=validate_pose_controls(current,context)
 receipt=bridge.assert_parameter_rebinding(historical_ref,current,cref,scene_contract=scene_contract);bridge.close()
 return current,receipt


class _LoadedPoseSnapshot:
 """Owner-bound immutable payload, only for one opening OR closing index.
 Production construction is through _TypedPoseScope after the full typed loader.
 This lower layer intentionally accepts explicit already-loaded data for CPU
 kernel checks; it is not a public source-validation or trust bypass API.
 """
 def __init__(self,c,pc,reference,dependencies,context,consumer_ref,token):
  import os,threading
  if token is None:raise ValueError('Explicit typed-scope token required')
  self.__token=token;self.__pid=os.getpid();self.__thread=threading.get_ident();self.__sealed=False
  source={'base_kind':context.get('base_kind'),'source_context':context['source_context'],'source_context_sha256':context['source_context_sha256']}
  if env.digest(source['source_context'])!=source['source_context_sha256']:raise ValueError('Typed source identity mismatch')
  payload={'config':c,'bounds':pc,'config_ref':reference,'dependencies':dependencies,'source':source,'consumer_ref':consumer_ref}
  self.__raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  self.__sha=hashlib.sha256(self.__raw).hexdigest()
 def __getstate__(self):raise TypeError('Typed scope cannot cross process/run boundary')
 def _payload(self,reference,context,token):
  import os,threading
  if self.__sealed or token is not self.__token or os.getpid()!=self.__pid or threading.get_ident()!=self.__thread:raise ValueError('Typed scope sealed or owner/token mismatch')
  if hashlib.sha256(self.__raw).hexdigest()!=self.__sha:raise ValueError('Typed immutable snapshot changed')
  payload=json.loads(self.__raw)
  source={'base_kind':context.get('base_kind'),'source_context':context['source_context'],'source_context_sha256':context['source_context_sha256']}
  if source!=payload['source'] or env.digest(source['source_context'])!=source['source_context_sha256'] or reference!=payload['config_ref']:raise ValueError('Typed config/source identity differs from owned snapshot')
  return payload
 def validate(self,parameters,context,token):
  if parameters.get('schema')!=SCHEMA or parameters.get('status')!=STATUS:raise ValueError('Explicit diagnostic pose controls schema/status')
  provenance=parameters['provenance'];p=self._payload(provenance['bank_config_ref'],context,token)
  expected=_propose_from_loaded(parameters['entry'],p['config'],p['bounds'],p['config_ref'],parameters['controls'],p['source'],p['consumer_ref'],entry_source_ref=provenance['entry_source_ref'])
  if not _same_derivatives(parameters,expected):raise ValueError('Pose controls differ from source-bound bank/entry/slider reconstruction')
  return deepcopy(parameters)
 def dependencies(self,reference,context,token):
  return self._payload(reference,context,token)['dependencies']
 def seal(self,reference,context,token):
  self._payload(reference,context,token);self.__sealed=True
 def _seal_owned(self,token):
  # Recheck owner and serialized integrity even if there was no final row.
  p=json.loads(self.__raw)
  self.seal(p['config_ref'],p['source'],token)


class _TypedPoseScope(_LoadedPoseSnapshot):
 """Fresh full typed load once; never share this instance with closing scope."""
 def __init__(self,reference,context,token):
  _bytes(reference)
  dependencies={};c,pc,actual=_load_config(reference['path'],dict(context),dependencies)
  if actual!=reference:raise ValueError('Typed config changed during full load')
  super().__init__(c,pc,actual,dict(sorted(dependencies.items())),context,layouts.ref(__file__),token)


class _ScopedPoseContext(dict):
 """Explicit context capability for already owned static validation.
 Sealed historical readers need no patch: their existing controls APIs receive
 this context and reuse only its owner's exact validated config payloads.
 Arbitrary dicts keep the standalone full-loader behavior.
 """
 def __init__(self,context,owner,token):
  super().__init__(context)
  owner._guard()
  if token is not owner._typed_token:raise ValueError('Wrong scoped context token')
  self.__owner=owner;self.__token=token
 def __getstate__(self):raise TypeError('Scoped context cannot cross owner/run')
 def _owner(self):
  self.__owner._guard()
  if self.__token is not self.__owner._typed_token:raise ValueError('Scoped context owner token changed')
  self.__owner._validation_source(self)
  return self.__owner
 def _loaded(self,config):
  owner=self._owner();name=str(params.bounded(config).relative_to(params.ROOT))
  reference={'path':name,'sha256':owner.hash_path(name)}
  scope=owner._typed_scope(reference,self)
  return scope._payload(reference,self,self.__token)
 def _memo(self,kind,identity,producer):
  return self._owner().validation_once(kind,identity,self,producer)
