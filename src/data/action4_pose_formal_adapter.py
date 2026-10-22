"""Candidate pose-controls adapter for finite formal-plan releases.
Candidate records remain unaccepted. Only explicit base-wearing wall_rear is
implemented; production all-family gates remain the existing backend's duty.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from data import action4_synthesis as synth
from data import action4_modular_pose_controls as controls
from data import action4_modular_scene_sampling as sampling
from data import action4_modular_pixels as pixels
from data import action4_modular_render as camera
SCHEMA='action4.pose_formal_binding.v1'

def backend():
 from data import action4_modular_synthesis_backend as module
 return module

def validate_binding(row,context=None,plan=None,*,_index=None):
 if _index is not None and context is not None:context=_index.pose_context(context)
 b=backend();read_ref=_index.read_ref if _index is not None else lambda ref:synth._read(b._ref(ref))
 parameters=read_ref(row['parameters_ref'])
 if parameters.get('schema')!=controls.SCHEMA:raise ValueError('Pose binding requires exact pose controls parameters')
 binding=read_ref(row['pose_execution_ref'])
 if set(binding)!={'schema','parameters_ref','entry_authority_ref','bank_config_ref','environment_composition','training_accepted','dataset_generation_allowed'} or binding['schema'] not in (SCHEMA,'action4.pose_formal_binding.v2') or binding['parameters_ref']!=row['parameters_ref'] or binding['training_accepted'] is not False or binding['dataset_generation_allowed'] is not False:raise ValueError('Explicit nonpromoting pose formal binding required')
 authority=read_ref(binding['entry_authority_ref'])
 if authority.get('schema')!='action4.modular_entry_authority.v1' or authority.get('status')!='FROZEN_ENTRY_IDENTITIES_ONLY_NOT_RENDER_PLAN' or authority.get('training_accepted') is not False or authority.get('dataset_generation_allowed') is not False:raise ValueError('Explicit current entry authority only; historical render plans forbidden')
 entry=b.entry_identity(row['entry'])
 matches=([_index.authority_entry(binding['entry_authority_ref'],entry['sample_id'])] if _index is not None else [e for e in authority['entries'] if e['sample_id']==entry['sample_id']])
 if matches!=[entry] or parameters['entry']!=entry or parameters['provenance']['entry_source_ref']!=binding['entry_authority_ref']:raise ValueError('Full pose entry must originate from bound authority before rendering')
 if entry['split']=='test' and entry['target_count']!=2:raise ValueError('Test split remains dual only')
 if binding['schema']==SCHEMA and entry['scene_family']!='wall_rear':raise ValueError('Pose formal adapter currently implements wall_rear only; explicit family composition missing')
 if binding['bank_config_ref']!=parameters['provenance']['bank_config_ref']:raise ValueError('Exact pose-bank configuration')
 bank=read_ref(binding['bank_config_ref'])
 composition=binding['environment_composition']
 if binding['schema']==SCHEMA and composition!={'kind':'base_wearing','layers':[],'source_refs':{'factory':bank['factory_ref'],'source_context':bank['source_context_ref']}}:raise ValueError('Only finite source-bound base environment composition implemented')
 if binding['schema']=='action4.pose_formal_binding.v2':
  from data import action4_formal_composition as composer
  composer.validate_descriptor(row,parameters,binding,bank,context,_index)
 for ref in composition['source_refs'].values():(_index.check_ref(ref) if _index is not None else b._ref(ref))
 if plan is not None:
  if _index is not None:_index.check_plan_authority(binding['entry_authority_ref'],plan)
  elif plan.get('entry_authority_ref')!=binding['entry_authority_ref'] or authority!=synth.build_entry_authority(plan['recipe'],plan['count'],plan['seed']) or authority['entries']!=plan['entries']:raise ValueError('Pose authority differs from actual formal plan allocation')
 if context is not None:
  if _index is not None:_index.validate_pose_once(row['parameters_ref'],parameters,context)
  else:controls.validate_pose_controls(parameters,context)
  if context['source_context']['scene_config_ref']!=composition['source_refs']['factory']:raise ValueError('Pose scene factory differs from formal composition')
 return parameters,binding

def dependency_refs(row,context,*,_index=None):
 b=backend();p,binding=validate_binding(row,context,_index=_index)
 refs=(_index.bank_dependencies(binding['bank_config_ref'],context) if _index is not None else controls.dependency_refs(binding['bank_config_ref']['path'],context))
 for reference in [row['parameters_ref'],row['pose_execution_ref'],binding['entry_authority_ref'],*binding['environment_composition']['source_refs'].values()]:
  (_index.check_ref(reference) if _index is not None else b._ref(reference))
  if reference['path'] in refs and refs[reference['path']]!=reference['sha256']:raise ValueError('Conflicting pose formal dependency')
  refs[reference['path']]=reference['sha256']
 if binding['schema']=='action4.pose_formal_binding.v2':
  from data import action4_formal_composition as composer
  for path,digest in composer.dependency_refs(binding['environment_composition'],context,_index=_index).items():
   if path in refs and refs[path]!=digest:raise ValueError('Composition dependency collision')
   reference={'path':path,'sha256':digest};(_index.check_ref(reference) if _index is not None else b._ref(reference));refs[path]=digest
 return refs

def _projection(geometry,canonical,ids):
 if set(map(int,geometry))!=set(ids):raise ValueError('Original physical IDs missing from zero-render geometry')
 errors={}
 for key,row in geometry.items():
  if row['corner_ids']!=list(range(8)) or row['points_object_mm']!=canonical:raise ValueError('Original ordered canonical8 changed')
  xyz=(np.linalg.inv(np.asarray(row['camera_to_world_blender_m']))@np.asarray(row['object_to_world_m'])@np.c_[np.asarray(canonical)/1000.,np.ones(8)].T)[:3].T*np.array([1.,-1.,-1.])
  q=(np.asarray(row['K_native'])@xyz.T).T;uv=q[:,:2]/q[:,2,None];actual=np.asarray(row['points_native_px'])
  if not np.isfinite(xyz).all() or not (xyz[:,2]>row['near_plane_m']).all() or actual.shape!=(8,2) or not np.isfinite(actual).all() or not np.allclose(uv,actual,atol=1e-6,rtol=0):raise ValueError('Independent original8 projection failed')
  errors[str(key)]=float(np.max(abs(uv-actual)))
 return errors

def verify_plan_source(plan):
 b=backend();reference=plan.get('_runtime_plan_ref')
 frozen=synth._read(b._ref(reference))
 if reference['sha256']!=plan.get('_runtime_plan_sha256') or frozen!={k:v for k,v in plan.items() if k not in ('_runtime_plan_sha256','_runtime_plan_ref')}:raise ValueError('Actual formal plan file differs from runtime plan')
 return deepcopy(reference)


# Diagnostic-statistic agreement only; never a point/transform acceptance gate.
# Use binary64 ULPs at pixel/intrinsic scale, not relative error of a near-zero
# residual. The small fixed affine/pinhole replay chain can round differently
# across NumPy/BLAS runtimes. 32 ULP is a conservative engineering allowance,
# capped four orders below the unchanged 1e-6 pixel physical projection gate.
PROJECTION_STATISTIC_ULPS=32
PROJECTION_STATISTIC_CAP_PX=1e-10
PROJECTION_STATISTIC_MAX_RESIDUAL_PX=1e-6


def _projection_statistic_budget_px(geometry_row):
 points=np.asarray(geometry_row['points_native_px'],dtype=np.float64)
 K=np.asarray(geometry_row['K_native'],dtype=np.float64)
 if points.shape!=(8,2) or K.shape!=(3,3) or not np.isfinite(points).all() or not np.isfinite(K).all():raise ValueError('Finite original8 pixel/intrinsic scale required for statistic comparison')
 # Include K so cancellation near image origin does not imply a zero scale.
 scale=max(1.,float(np.max(np.abs(points))),float(np.max(np.abs(K))))
 ulp=float(np.spacing(np.float64(scale)))
 if not np.isfinite(ulp) or ulp<=0:raise ValueError('Invalid float64 projection statistic scale')
 return PROJECTION_STATISTIC_CAP_PX if ulp>=PROJECTION_STATISTIC_CAP_PX/PROJECTION_STATISTIC_ULPS else PROJECTION_STATISTIC_ULPS*ulp


def _verify_projection_residual_statistics(recomputed,stored,geometry):
 if not isinstance(recomputed,dict) or not isinstance(stored,dict) or not isinstance(geometry,dict):raise ValueError('Explicit projection statistic dictionaries required')
 by_id={str(k):v for k,v in geometry.items()}
 if len(by_id)!=len(geometry) or set(recomputed)!=set(stored) or set(recomputed)!=set(by_id):raise ValueError('Projection statistic instance keys changed')
 for key,current in recomputed.items():
  saved=stored[key]
  if any(type(v) not in (int,float) or not np.isfinite(v) or not 0<=v<=PROJECTION_STATISTIC_MAX_RESIDUAL_PX for v in (current,saved)):raise ValueError('Projection statistics must remain finite nonnegative and within original physical budget')
  budget=_projection_statistic_budget_px(by_id[key])
  if abs(current-saved)>budget:raise ValueError('Pose proof projection statistic differs beyond float64 roundoff budget')
 return True


ACTUAL_TRANSFORM_ATOL=2e-6  # Existing pose.verify_actual_pose_geometry replay budget.

def verify_geometry_contract(parameters,contract,geometry,depth):
 b=backend();state=contract['expected_state'];canonical=contract['canonical_definition'];objects=state['objects']
 if contract['parameters']!=parameters or contract['parameters_sha256']!=b._digest(parameters) or contract['source_context_sha256']!=parameters['provenance']['source_context_sha256'] or contract['expected_state_sha256']!=b._digest(state):raise ValueError('Actual scene contract parameter/state identity mismatch')
 bank=synth._read(b._ref(parameters['provenance']['bank_config_ref']));bounds=synth._read(b._ref(bank['parameter_bounds_ref']));source=synth._read(b._ref(bank['source_context_ref']))
 b._ref(canonical['source_manifest'])
 if canonical['source_manifest']!=bounds['source_refs']['canonical_source'] or canonical['corner_order']!=bounds['fixed']['corner_order'] or canonical['corners_3d_m']!=bounds['fixed']['canonical8_m'] or not np.array_equal(np.asarray(canonical['corners_3d_mm'])/1000.,bounds['fixed']['canonical8_m']):raise ValueError('Actual canonical metadata differs from frozen original8 source')
 if parameters['K_native']!=source['K_native'] or parameters['W_from_capture']!=source['W_from_capture'] or state['K']!=parameters['K_native']:raise ValueError('Actual fixed K/W source mismatch')
 cameras=[v for v in objects.values() if v['type']=='CAMERA']
 if len(cameras)!=1:raise ValueError('One actual observer required')
 camera_world=np.asarray(cameras[0]['world'],float);W=np.asarray(parameters['W_from_capture'],float)
 if not np.allclose(camera_world,W@np.diag([1.,-1.,-1.,1.]),atol=ACTUAL_TRANSFORM_ATOL,rtol=0):raise ValueError('Actual observer differs from source capture frame')
 near=state['observer_data']['clip_start']
 if type(near) not in (int,float) or not np.isfinite(near) or near!=float(np.float32(bounds['near_plane_m'])):raise ValueError('Actual camera near-plane differs from original source bound')
 if depth!=contract['pose_controls_geometry'] or depth.get('status')!='ACTUAL_POSE_GEOMETRY_DIAGNOSTIC_ONLY' or depth.get('collision_report_status')!=contract['actor_collision_diagnostics']['status']:raise ValueError('Actual retained depth evidence differs from scene contract')
 minimum=depth['retained_actor_min_depth_m']
 if set(minimum)!={'left','right'} or any(type(v) not in (int,float) or not np.isfinite(v) or v<=near for v in minimum.values()):raise ValueError('Retained actor depth must exceed actual near-plane')
 ids=[i for i,on in enumerate(parameters['target_presence'],1) if on]
 errors=_projection(geometry,canonical['corners_3d_mm'],ids)
 for key,g in geometry.items():
  instance=int(key);side='left' if instance==1 else 'right';name=side+'_Action4_core';want=W@np.asarray(parameters['sides'][side]['core_C_capture'],float)
  if g['instance_id']!=instance or name not in contract['actor_bindings'][side]['names'] or g['canonical_source_manifest']!=canonical['source_manifest'] or g['corner_order']!=canonical['corner_order'] or g['K_native']!=parameters['K_native'] or g['near_plane_m']!=near:raise ValueError('Original canonical/camera metadata differs from actual contract')
  if not np.allclose(objects[name]['world'],want,atol=ACTUAL_TRANSFORM_ATOL,rtol=0) or not np.allclose(g['object_to_world_m'],want,atol=ACTUAL_TRANSFORM_ATOL,rtol=0) or not np.allclose(g['object_to_world_m'],objects[name]['world'],atol=ACTUAL_TRANSFORM_ATOL,rtol=0):raise ValueError('Original8 geometry differs from requested W@C or actual core')
  if not np.allclose(g['camera_to_world_blender_m'],camera_world,atol=ACTUAL_TRANSFORM_ATOL,rtol=0):raise ValueError('Original8 geometry camera differs from actual observer')
 return errors

def resolve_record_plan(record):
 # Resolve complete disk authority without calling validate_record again.
 b=backend();reference=record['pose_controls_proof']['plan_ref'];plan=synth._read(b._ref(reference))
 required={'schema_version','status','recipe','recipe_path','recipe_sha256','entries','count','seed','split_counts','asset_identities','entry_authority_ref','synthesis_code_sha256','render_base_sha256'}
 if not required<=set(plan) or plan['schema_version']!=synth.SCHEMA or plan['status']!='render_candidates_pending_acceptance' or plan['recipe'].get('asset_backend')!=b.BACKEND:raise ValueError('Complete frozen formal plan required for host record readback')
 recipe_ref={'path':plan['recipe_path'],'sha256':plan['recipe_sha256']}
 if synth._read(b._ref(recipe_ref))!=plan['recipe']:raise ValueError('Formal disk recipe differs from frozen plan')
 if plan['synthesis_code_sha256']!=synth._hash(synth.__file__) or plan['render_base_sha256']!=synth._hash('src/data/action4_render.py'):raise ValueError('Formal plan runtime source identity changed')
 synth.validate_recipe(plan['recipe']);b.validate_plan(plan)
 if plan['asset_identities']!=b.asset_identities(plan['recipe']):raise ValueError('Formal readback source closure differs from frozen plan')
 context,rows=b.prepare_context(plan['recipe']);entry=b.entry_identity(record);row=rows.get(entry['sample_id'])
 if row is None or row['entry']!=entry or row.get('pose_execution_ref')!=record['pose_controls_proof']['binding_ref'] or row['parameters_ref']!=record['modular_backend']['parameters_ref'] or reference['sha256']!=record['modular_backend']['plan_sha256']:raise ValueError('Formal readback plan/release/record binding differs')
 validate_binding(row,context,plan)
 return plan,rows

def pre_render(session,entry,parameters,row,plan,runtime,*,render_scope=None):
 b=backend()
 if render_scope is None:
  plan_ref=verify_plan_source(plan);validate_binding(row,session.context,plan)
 else:
  if type(render_scope) is not b._RenderEntryScope:raise ValueError('Explicit active worker render scope required')
  plan_ref=render_scope.pre_render_binding(session,entry,parameters,row,plan)
 session.verifier(session.scene)
 g=session.contract['actor_collision_diagnostics'];counts={side:9 if on else 4 for side,on in zip(('left','right'),parameters['target_presence'])}
 if g['cross_pair_count']!=counts['left']*counts['right'] or g['internal_pair_counts']!={side:n*(n-1)//2 for side,n in counts.items()} or g['outside_or_unresolved_surface_pairs']!=0:raise ValueError('All retained actual actor collision pairs required before render')
 depth=controls.verify_actual_pose_geometry(session);ids=[i for i,on in enumerate(parameters['target_presence'],1) if on]
 data=runtime/'pose_preflight_data';out=runtime/'pose_preflight_runtime'
 data.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
 probe=pixels.render_frame(scene=session.scene,entry=deepcopy(entry),targets=session.targets(),canonical=session.contract['canonical_definition'],K=camera.actual_K(session.scene),data_out=data,runtime_out=out,output_role='diagnostic',source_identities=plan['asset_identities'],render_settings=plan['recipe']['render'],verify_scene=session.verifier,probe_only=True)
 if probe['render_calls']!=0 or probe['render_attempts']!=0 or probe['rendered'] is not False or probe['restoration_verified'] is not True or probe['source_identities_verified'] is not True:raise ValueError('Pose preflight must not render')
 errors=verify_geometry_contract(parameters,session.contract,probe['geometry'],depth)
 return {'schema':'action4.pose_formal_actual_proof.v1','binding_ref':deepcopy(row['pose_execution_ref']),'plan_ref':plan_ref,'parameters_sha256':b._digest(parameters),'entry_sha256':b._digest(b.entry_identity(entry)),'geometry':deepcopy(probe['geometry']),'actual_collision':deepcopy(g),'actual_depth':depth,'independent_projection_error_px':errors,'render_calls':0,'training_accepted':False}

def finish_record(parameters,record,proof):
 # Uses current authoritative pose mask semantics unchanged. The reported
 # renderer edge-tolerance discrepancy remains a root-owned separate blocker.
 masks=controls.verify_actual_masks(parameters,record)
 if masks['target_interval_pass'] is not True:raise ValueError('Pose requested actual mask interval failed')
 for row in record['instances']:
  before=proof['geometry'].get(row['instance_id'],proof['geometry'].get(str(row['instance_id'])))
  if before is None or not np.allclose(row['canonical8_geometry']['points_native_px'],before['points_native_px'],atol=1e-6,rtol=0):raise ValueError('Pose original8 changed after actual zero-render gate')
 return {**proof,'actual_masks':masks}

def validate_record(row,record,plan=None,*,validated_plan=None,run_token=None):
 b=backend()
 if validated_plan is not None:
  parameters,binding=validated_plan.record_binding(row,record,run_token,plan)
 elif plan is None:
  plan,rows=resolve_record_plan(record)
  if rows[row['entry']['sample_id']]!=row:raise ValueError('Supplied pose row differs from complete disk release')
 if validated_plan is None:parameters,binding=validate_binding(row,plan=plan)
 proof=record['pose_controls_proof']
 if proof.get('schema')!='action4.pose_formal_actual_proof.v1' or proof.get('binding_ref')!=row['pose_execution_ref'] or proof.get('parameters_sha256')!=b._digest(parameters) or proof.get('entry_sha256')!=b._digest(b.entry_identity(row['entry'])) or proof.get('render_calls')!=0 or proof.get('training_accepted') is not False:raise ValueError('Pose actual proof provenance mismatch')
 frozen_plan=(validated_plan.bound_plan(proof['plan_ref'],run_token) if validated_plan is not None else synth._read(b._ref(proof['plan_ref'])))
 if proof['plan_ref']['sha256']!=record['modular_backend']['plan_sha256'] or frozen_plan.get('entry_authority_ref')!=binding['entry_authority_ref']:raise ValueError('Pose proof formal plan source differs')
 if plan is not None and frozen_plan!={k:v for k,v in plan.items() if k not in ('_runtime_plan_sha256','_runtime_plan_ref')}:raise ValueError('Pose record belongs to a different formal plan')
 contract=record['modular_backend']['scene_contract']
 if contract['parameters']!=parameters or contract['actor_collision_diagnostics']!=proof['actual_collision']:raise ValueError('Pose record actual scene differs from proof')
 g=proof['actual_collision'];counts={side:9 if on else 4 for side,on in zip(('left','right'),parameters['target_presence'])}
 if g['cross_pair_count']!=counts['left']*counts['right'] or g['internal_pair_counts']!={side:n*(n-1)//2 for side,n in counts.items()} or g['outside_or_unresolved_surface_pairs']!=0:raise ValueError('Recorded actual collision gate no longer passes')
 depths=proof['actual_depth']['retained_actor_min_depth_m']
 if set(depths)!={'left','right'} or any(not np.isfinite(v) or v<=0 for v in depths.values()):raise ValueError('Recorded actual retained depth invalid')
 ids=[i for i,on in enumerate(parameters['target_presence'],1) if on]
 recomputed_errors=verify_geometry_contract(parameters,contract,proof['geometry'],proof['actual_depth'])
 _verify_projection_residual_statistics(recomputed_errors,proof['independent_projection_error_px'],proof['geometry'])
 rendered={i['instance_id']:i['canonical8_geometry'] for i in record['instances']}
 verify_geometry_contract(parameters,contract,rendered,proof['actual_depth'])
 camera=record['camera'];actual_camera=next(v['world'] for v in contract['expected_state']['objects'].values() if v['type']=='CAMERA')
 if camera['K']!=parameters['K_native'] or not np.allclose(camera['camera_to_world_m'],actual_camera,atol=ACTUAL_TRANSFORM_ATOL,rtol=0):raise ValueError('Rendered record camera differs from actual scene camera')
 if finish_record(parameters,record,{k:v for k,v in proof.items() if k!='actual_masks'})!=proof:raise ValueError('Pose mask proof differs from original record')
 if binding['schema']=='action4.pose_formal_binding.v2':
  from data import action4_formal_composition as composer
  composer.verify_evidence(parameters,binding,proof,record,contract)
 elif 'composition' in proof:raise ValueError('Legacy v1 record cannot claim composition execution')
 return record
