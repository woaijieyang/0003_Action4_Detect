"""Finite formal composition routing; candidates never authorize training.
The reviewed geometry builders own their objects. No scene/renderer on import.
"""
from copy import deepcopy
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
from data import action4_synthesis as synth
from data import action4_modular_pose_controls as controls
from data import action4_modular_pixels as pixels
from data import action4_modular_geometric_effects as effects
from data import action4_modular_geometric_effects_v002 as near_effects
SCHEMA='action4.pose_formal_binding.v2'
REVIEWED={'interface': {'path': 'outputs/action4_derived_presence_binding_prepare_v001/derived_presence_interface.py', 'sha256': '8ca2f00ab650280a2c6012f817114eeb96a920562473a4823b0e1dbdd351b5f5'}, 'fixed_interface': {'path': 'outputs/action4_forearm_continuation_interface_v002/continuation_candidate_interface.py', 'sha256': '5cf05c92fb32d3d958a2ba15e8f7e2f987a90995160e237e99b9faa67b5c3794'}, 'selection_handoff': {'path': 'outputs/action4_forearm_continuation_prepare_v001/HANDOFF.json', 'sha256': '71f46c867aefc5212491b857fd6d11c167b60e829ec927fa773733df17a617da'}, 'spec_review': {'path': 'outputs/action4_forearm_continuation_spec_v001/SPEC_REVIEW.json', 'sha256': 'f180f03c58d96970e9e0b85999dbe101fe42da0629ff47cfd8c80708f25871c2'}, 'geometry_checks': {'path': 'outputs/action4_current_continuation_runner_prepare_v001/geometry_checks.py', 'sha256': 'ca6600b23c27c68e9acb3c82b064cbf9d9b8321078c2e4b6e01039c6778f9a45'}, 'actual_presence_ref': {'path': 'outputs/action4_parametric_presence_actual_prepare_v001/actual_presence.py', 'sha256': '90be2dab4bf066083fb177938ed96e6f69cf2b6767a50fffebf4f9575ac9e1eb'}, 'render_runtime': {'path': 'outputs/action4_continuation_cleanup_fix_v001/render_runtime_cleanup.py', 'sha256': '1328e919335c4b1a00d8df797ba6a1cc1828060694a9fe6b71b0ce8f39a28c8d'}, 'environment_provider_module_ref': {'path': 'outputs/action4_parametric_environment_prepare_v002/environment_provider.py', 'sha256': '697c7ffb52397da8669b92d4c62bf926b9f736b718ec7c836cf83c43d0838d81'}, 'garment_module_ref': {'path': 'outputs/action4_current_composition_prepare_v001/action4_garment_background.py', 'sha256': '6c9a50ae449b84a730508028179f99f79b3f2eb2d99894b558b7e136d49e36d9'}, 'dual_base': {'path': 'outputs/action4_parametric_continuation_actual_prepare_v001/continuation_layer.py', 'sha256': '88b1a03dc4efe272f600e0446a4ce28c6ee2d376b2d2fa6d93bf68d8d6c13ba4'}, 'registry': {'path': 'outputs/action4_ownership_snapshot_reuse_prepare_v001/ownership_registry.py', 'sha256': '55575bbc0cfda16698893de1b0873377a9493211bfa719548fad286ba2b244c7'}}

REVIEWED.update({'presence_rules': {'path': 'outputs/action4_parametric_presence_prepare_v001/presence_guards.py', 'sha256': '4b6a2fbe6186749bee2956a9345f1eb9d3959bfd8484c21276229f14fd724ce5'}, 'presence_host': {'path': 'outputs/action4_parametric_presence_prepare_v001/presence_host_receipt_guard.py', 'sha256': 'ed5e932a8be8d92dacaf03e5f14fadbf5c4e2cba1ad233e603104d1f07101be7'}, 'environment_effects': {'path': 'outputs/action4_scene_effects_prepare_v001/action4_modular_scene_effects.py', 'sha256': '9732efcb41541eaaba5584e7a85390155f3de491e06259801afefa80ef67aa95'}})
PROFILES={'cloth_lift':'cloth_lift10_full_garment','self_forearm_occlusion':'bare_continuation','wall_rear':'bare_continuation','hard_negative':'bare_continuation','near_double':'bare_continuation','wall_transition_truncation':'bare_continuation','table_fastening':'table_full_garment','table_turn':'table_full_garment'}
BLOCKED={}
def module_ref(name):
 module=importlib.import_module('data.'+name);path=Path(module.__file__).resolve()
 return {'path':str(path.relative_to(synth.ROOT)),'sha256':synth._hash(path)}
FORMAL_MODULES=('action4_formal_composition','action4_formal_continuation','action4_formal_table_composition','action4_formal_forearm_attribution','action4_formal_environment','action4_formal_lift','action4_formal_lift_composition')
def require_runtime_sources(context):
 frozen=context['source_context']['frozen_refs']
 for name in FORMAL_MODULES:
  r=module_ref(name)
  if r['path']!='src/data/'+name+'.py' or frozen.get(r['path'])!=r['sha256']:raise ValueError('Formal composition consumers require coordinated current source publication: '+name)
 return True
def backend():
 from data import action4_modular_synthesis_backend as b
 return b
def read(r):return synth._read(backend()._ref(r))
def load_fixed(key,name):
 r=REVIEWED[key];p=backend()._ref(r);sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def _refs(value):
 # Finite in-memory JSON tree only; never follows arbitrary historical JSON.
 if isinstance(value,dict):
  if set(value)=={'path','sha256'}:yield value
  else:
   for child in value.values():yield from _refs(child)
 elif isinstance(value,list):
  for child in value:yield from _refs(child)

def normalized_config(c):
 if c.get('environment_profile')!='cloth_lift10_full_garment':return c
 q=read(c['request_ref']);out=deepcopy(c)
 if any(k not in c for k in ('entry_authority_ref','baseline_scene_contract_ref','baseline_source_contract_ref')) or c['entry_authority_ref']!=q['entry_authority_ref'] or c['baseline_scene_contract_ref']!=q['baseline_scene_contract_ref'] or c['baseline_source_contract_ref']!=q['baseline_scene_contract_ref']:raise ValueError('Explicit exact lift authority and both baseline keys required; no repair')
 out.update(case=c['bank_id'],current_factory_ref=q['factory_ref'],current_config_ref=q['config_ref'],current_source_context_ref=q['source_context_ref'],actual_presence_ref=REVIEWED['actual_presence_ref'])
 return out

def validate_layer_config(c):
 c=normalized_config(c)
 required={'case','request_ref','entry_authority_ref','parameters_override_ref','baseline_source_contract_ref','current_factory_ref','current_config_ref','current_source_context_ref','fixed_interface','selection_handoff','spec_review','interface','actual_presence_ref','geometry_checks','candidates','dependencies'}
 allowed=required|{'baseline_scene_contract_ref','environment_provider_module_ref','environment_source_ref','environment_binding_ref','environment_module','garment_module_ref','garment_config_ref','environment_profile','bank_id','geometry_bank_id','composition_module_ref','lift_module_ref','angle_deg'}
 if not required<=set(c) or not set(c)<=allowed:raise ValueError('Exact explicit continuation config, no unknown transform/matrix override fields')
 for k in ('fixed_interface','selection_handoff','spec_review','interface','actual_presence_ref','geometry_checks'):
  if c[k]!=REVIEWED[k]:raise ValueError('Only reviewed continuation executable/helper: '+k)
 q=read(c['request_ref']);p=read(c['parameters_override_ref'])
 if any(key in c and c[key]!=q[key] for key in ('bank_id','geometry_bank_id')):raise ValueError('Explicit config bank mapping differs from request')
 api=load_fixed('interface','formal_typed_presence_metadata')
 if set(q)!=api.REQUEST_KEYS or q['schema']!=api.SCHEMA or q['status']!=api.STATUS or any(q[k] is not False for k in ('training_accepted','dataset_generation_allowed')):raise ValueError('Exact typed derived-presence request required')
 expected={'bank_id':c['case'],'parameters_ref':c['parameters_override_ref'],'config_ref':c['current_config_ref'],'factory_ref':c['current_factory_ref'],'source_context_ref':c['current_source_context_ref'],'entry_authority_ref':c['entry_authority_ref'],'baseline_scene_contract_ref':c['baseline_source_contract_ref'],'candidate_refs':c['candidates']}
 if any(q[k]!=v for k,v in expected.items()):raise ValueError('Layer request identity mismatch')
 assets=api.bank_assets(q['geometry_bank_id'])
 if any(q[k]!=assets[k] for k in ('baseline_scene_contract_ref','candidate_refs','geometry_refs')):raise ValueError('Exact immutable same geometry-bank selected pair')
 if p['provenance']['bank_config_ref']!=q['config_ref'] or p['provenance']['entry_source_ref']!=q['entry_authority_ref'] or p['controls']['bank_id']!=q['bank_id']:raise ValueError('Current parameters/bank/authority mismatch')
 for r in _refs(c):backend()._ref(r)
 return True

def validate_descriptor(row,p,binding,bank,context=None,index=None):
 if index is not None:
  scoped=index.pose_context(context)
  identity={'row':row,'parameters':p,'binding':binding,'bank':bank}
  return index.validation_once('composition_descriptor',identity,scoped,lambda:_validate_descriptor(row,p,binding,bank,scoped,index))
 return _validate_descriptor(row,p,binding,bank,context,index)


def _validate_descriptor(row,p,binding,bank,context=None,index=None):
 if context is not None:require_runtime_sources(context)
 e=p['entry'];family=e['scene_family']
 if family in BLOCKED:raise ValueError(BLOCKED[family])
 if family not in PROFILES:raise ValueError('Unimplemented composition family')
 n=e['target_count']
 if (family=='hard_negative' and n!=0) or (family=='wall_rear' and n not in (1,2)) or (family not in ('wall_rear','hard_negative') and n!=2):raise ValueError('Exact formal family/presence allocation')
 comp=binding['environment_composition']
 if set(comp)!={'kind','profile','layer_config_ref','source_refs'} or comp['kind']!='actual_composition' or comp['profile']!=PROFILES[family] or comp['source_refs']!={'factory':bank['factory_ref'],'source_context':bank['source_context_ref']}:raise ValueError('Exact composition source/profile descriptor')
 c=(index.read_ref(comp['layer_config_ref']) if index is not None else read(comp['layer_config_ref']))
 validate_layer_config(c);c=normalized_config(c)
 if c['parameters_override_ref']!=row['parameters_ref'] or c['entry_authority_ref']!=binding['entry_authority_ref'] or read(c['parameters_override_ref'])!=p:raise ValueError('Composition current entry/parameters differ')
 if c['current_factory_ref']!=bank['factory_ref'] or c['current_source_context_ref']!=bank['source_context_ref'] or c['current_config_ref']!=binding['bank_config_ref']:raise ValueError('Composition current factory/context/config differs')
 if comp['profile'] in ('table_full_garment','cloth_lift10_full_garment'):
  for key in ('environment_provider_module_ref','garment_module_ref'):
   if c.get(key)!=REVIEWED[key]:raise ValueError('Reviewed immutable environment helper required')
  if c.get('environment_module')!=module_ref('action4_formal_environment') or c.get('environment_profile')!=comp['profile']:raise ValueError('Exact current formal environment and complete garment profile')
  provider=load_fixed('environment_provider_module_ref','formal_environment_descriptor');sources,_=provider.descriptor(c['environment_source_ref'])
  if any(sources[k]!=bank[k] for k in ('factory_ref','source_context_ref','source_context_sha256','bridge_ref')):raise ValueError('Environment is bound to a different current source')
  envbinding=read(c['environment_binding_ref'])
  if envbinding!={'schema':'action4.parametric_environment_execution_binding.v1','provider_ref':c['environment_source_ref'],'environment_module_ref':c['environment_module'],'garment_module_ref':c['garment_module_ref'],'garment_config_ref':c['garment_config_ref']}:raise ValueError('Exact explicit environment binding')
 if family=='cloth_lift':
  if c.get('composition_module_ref')!=module_ref('action4_formal_lift_composition') or c.get('lift_module_ref')!=module_ref('action4_formal_lift') or c.get('angle_deg')!=10:raise ValueError('Exact derived10deg actual lift implementation required')
  if context is not None:
   from data import action4_formal_lift_composition as lift_composition
   lift_composition.validate_selection(c,context)
 # Different complete frozen actor anchors with the existing typed local range.
 # The active validator reconstructs B/C/W; actual8 proves that requested result.
 if family in ('table_fastening','table_turn'):
  anchor={'table_fastening':'source540_side_oblique','table_turn':'source960_turn'}[family]
  q=read(c['request_ref'])
  if p['controls']['bank_id']!=anchor or q['bank_id']!=anchor or q['geometry_bank_id']!=anchor:raise ValueError('Exact distinct table pose anchor required; no action-GT claim')
 if context is not None and context['source_context']['scene_config_ref']!=bank['factory_ref']:raise ValueError('Actual current parent factory mismatch')
 return c

def dependency_refs(comp,context=None,*,_index=None):
 if _index is not None:
  scoped=_index.pose_context(context)
  return _index.validation_once('composition_dependencies',comp,scoped,lambda:_dependency_refs(comp,scoped))
 return _dependency_refs(comp,context)


def _dependency_refs(comp,context=None):
 refs={};b=backend()
 def add(r):
  b._ref(r)
  if r['path'] in refs and refs[r['path']]!=r['sha256']:raise ValueError('Conflicting composition dependency')
  refs[r['path']]=r['sha256']
 c=read(comp['layer_config_ref']);add(comp['layer_config_ref'])
 for r in _refs(c):add(r)
 for r in REVIEWED.values():add(r)
 from data import action4_formal_forearm_attribution as attribution
 add(attribution.SEMANTIC_REF)
 # Explicit current data module consumers; source migration must rebind them.
 for name in ('action4_formal_composition','action4_formal_continuation','action4_formal_table_composition','action4_formal_forearm_attribution','action4_formal_environment','action4_formal_lift','action4_formal_lift_composition'):
  mod=importlib.import_module('data.'+name);path=Path(mod.__file__).resolve();add({'path':str(path.relative_to(synth.ROOT)),'sha256':synth._hash(path)})
 # All current typed request reader dependencies, no mesh/scene construction.
 q=read(c['request_ref'])
 for r in _refs(q):add(r)
 if context is not None:
  api=load_fixed('interface','formal_dependency_typed_reader');reader=api._ReadSet()
  api._validate(reader,c['request_ref'],read(c['parameters_override_ref']),context)
  for path,h in reader.close().items():add({'path':path,'sha256':h})
 if comp['profile'] in ('table_full_garment','cloth_lift10_full_garment'):
  from data import action4_formal_environment as environment
  view=environment.prepare(c)
  for path,h in view['bridge_receipt']['actual_dependency_refs'].items():add({'path':path,'sha256':h})
  for r in _refs(read(c['garment_config_ref'])):add(r)
  # This original table builder directly imports the reviewed effects module.
  path=Path(environment.effects.__file__).resolve();add({'path':str(path.relative_to(synth.ROOT)),'sha256':synth._hash(path)})
 if comp['profile']=='cloth_lift10_full_garment':
  from data import action4_formal_lift as lift
  for r in lift.REFS.values():add(r)
  add(lift.prepare(c)['prescription']['sources']['v002_garment_10'])
 for name in ('action4_modular_cloth_effects','action4_dataset','action4_modular_geometric_effects','action4_modular_geometric_effects_v002'):
  module=importlib.import_module('data.'+name);path=Path(module.__file__).resolve();add({'path':str(path.relative_to(synth.ROOT)),'sha256':synth._hash(path)})
 return refs

def create(session,comp):
 c=read(comp['layer_config_ref'])
 if session.parameters!=read(c['parameters_override_ref']):raise ValueError('Actual parent parameters mismatch')
 if comp['profile']=='bare_continuation':
  from data import action4_formal_continuation as implementation
 elif comp['profile']=='table_full_garment':
  from data import action4_formal_table_composition as implementation
 elif comp['profile']=='cloth_lift10_full_garment':
  from data import action4_formal_lift_composition as implementation
 else:raise ValueError('No runtime implementation for profile')
 return implementation.create(session,c)

def projection_equal(before,after):
 if set(map(int,before))!=set(map(int,after)):raise ValueError('Original physical ID set changed')
 for key,g in before.items():
  other=after.get(key,after.get(str(key),after.get(int(key))))
  for field in ('points_object_mm','object_to_world_m','camera_to_world_blender_m','K_native','corner_ids'):
   if g[field]!=other[field]:raise ValueError('Original8 fixed field changed: '+field)
  if not np.allclose(g['points_native_px'],other['points_native_px'],rtol=0,atol=1e-6):raise ValueError('Original8 composition projection changed')
 return True

def geometry_check(p,contract,geometry,source_contract=None):
 verify_named_inventory(p,contract,geometry,source_contract)
 n=p['entry']['target_count'];full=p['entry']['scene_family'] in ('table_fastening','table_turn','cloth_lift')
 expected=45 if full else {0:21,1:31,2:41}[n]
 if geometry.get('status')!='ACTUAL_CONTINUATION_GEOMETRY_ONLY' or geometry.get('pair_count')!=expected or len(geometry.get('pairs',{}))!=expected or geometry.get('unresolved_count')!=0 or geometry.get('new_whitelist_count')!=0:raise ValueError('Complete actual extension geometry required')
 if geometry.get('original_actor_piece_count')!=8+5*n or len(geometry.get('self_pairs',{}))!=2:raise ValueError('All retained actors and two self-checks required')
 for r in geometry['pairs'].values():
  if r['outside_or_unresolved_pairs'] or r['environment_inside_vertices']:raise ValueError('Actual extension intersection/containment')
 for r in geometry['self_pairs'].values():
  if r['outside_or_unresolved_pairs']:raise ValueError('Actual extension self intersection')
 if full:
  if contract.get('schema')!='action4.complete_table_garment_continuation.v1' or set(contract.get('owner_contracts',{}))!={'table','garment','continuation'}:raise ValueError('Complete real table/garment owners required')
  h=geometry['composition_halfspace_and_containment'];g=h['all20_actor_and_continuation_top_gaps_m']
  if len(g)!=20 or min(g.values())<.032-2e-6 or h['minimum_top_gap_m']!=min(g.values()) or h['garment_bottom_above_table_m']<=0 or h['actor_garment_plane_margin_m']<.002 or h.get('table_and_garment_wall_depth_gap_m',h.get('all20_table_garment_wall_depth_gap_m',0))<=0:raise ValueError('Full20 table/garment separation evidence required')
  if p['entry']['scene_family']=='cloth_lift':
   lift=contract.get('rigid_lift',{});garment=contract['owner_contracts']['garment']
   if lift.get('angle_deg')!=10 or lift.get('preset_bank_id')!=p['controls']['bank_id'] or h.get('full_actor_piece_count')!=20 or h.get('required_nominal_margin_m')!=.020 or h.get('actual_float32_numeric_budget_m')!=2e-6 or h['actor_garment_plane_margin_m']<.020-2e-6 or garment.get('rigid_lift_kind')!='current_derived_10deg' or garment.get('no_panel_seam_geometry_deformation') is not True:raise ValueError('Actual full20 10deg/20mm lift and unchanged full garment required')
 else:
  if contract.get('schema')!='action4.parametric_presence_continuation_layer.v1' or len(contract.get('objects',{}))!=2 or 'presence_pair_inventory' not in geometry:raise ValueError('Actual presence-aware continuation contract required')
 # Reject NaNs before min/comparison can silently accept them.
 def finite(v):
  if isinstance(v,float) and not np.isfinite(v):raise ValueError('Nonfinite actual composition evidence')
  if isinstance(v,dict):
   for x in v.values():finite(x)
  elif isinstance(v,list):
   for x in v:finite(x)
 finite(geometry)
 return True

def effect_check(p,record,attribution=None):
 family=p['entry']['scene_family']
 if family in BLOCKED:raise ValueError(BLOCKED[family])
 if family=='self_forearm_occlusion':
  result=effects.mask_effect_check('self_forearm',record,2,attribution)
  if result['effect_condition_pass'] is not True:raise ValueError('Actual same-actor forearm attribution plus real mask occlusion required')
  return result
 if family=='near_double':
  result=near_effects.actual_near_double_check(record,{'outside_or_unresolved_surface_pairs':0})
  if result['effect_condition_pass'] is not True:raise ValueError('Actual near-double projection and two visible masks required')
  return result
 if family=='wall_transition_truncation':
  passed=[]
  for r in record['instances']:
   flags=r['canonical8_geometry']['point_inframe']
   if any(flags) and not all(flags) and effects.mask_effect_check('truncation',record,r['instance_id'])['effect_condition_pass']:passed.append(r['instance_id'])
  if not passed:raise ValueError('Real partially inframe original8 plus actual amodal truncation required')
  return {'effect':'truncation','original_ids':passed,'effect_condition_pass':True}
 if family=='hard_negative':
  if record['instances'] or record.get('excluded_instances') or record['target_count']!=0 or record['planned_target_count']!=0:raise ValueError('Actual target-absent frame only; no pseudo points/masks')
 return {'effect':family,'effect_condition_pass':True,'construction_verified_separately':True,'fine_finger_action_claimed':False}

def verify_evidence(p,binding,proof,record,source_contract):
 c=proof['composition'];descriptor=binding['environment_composition'];b=backend()
 if set(c)!={'descriptor','contract','contract_sha256','geometry','preflight_geometry','cleanup','runtime_cleanup','source_contract_sha256','effect','forearm_attribution'} or c['descriptor']!=descriptor or c['source_contract_sha256']!=b._digest(source_contract) or c['contract_sha256']!=b._digest(c['contract']):raise ValueError('Full composition contract/source identity mismatch')
 construction=c['contract'];full=descriptor['profile']!='bare_continuation'
 continuation=construction['owner_contracts']['continuation'] if full else construction
 pair=continuation['parametric_binding_proof'];config=read(descriptor['layer_config_ref']);request=read(config['request_ref'])
 if pair.get('schema')!='action4.derived_presence_binding_proof.v1' or not isinstance(pair.get('dependency_identities'),dict) or not pair['dependency_identities'] or pair.get('request_ref')!=config['request_ref'] or pair.get('parameters_ref')!=binding['parameters_ref'] or pair.get('config_ref')!=binding['bank_config_ref'] or pair.get('current_entry')!=p['entry'] or pair.get('bank_id')!=p['controls']['bank_id'] or pair.get('geometry_bank_id')!=request['geometry_bank_id'] or pair.get('entry_authority',{}).get('authority_ref')!=binding['entry_authority_ref'] or pair['entry_authority'].get('full_allocation_replayed') is not True:raise ValueError('Actual full typed pair construction evidence required')
 if pair.get('consumer_ref')!=REVIEWED['interface'] or continuation.get('parameters_override_ref')!=binding['parameters_ref'] or continuation.get('request_ref')!=config['request_ref']:raise ValueError('Actual continuation consumed different current interface/request')
 verify_pair_contract(p,pair,continuation)
 verify_consumed_sources(c['contract'],record['pixel_source_identities'])
 geometry_check(p,c['contract'],c['geometry'],source_contract);projection_equal(proof['geometry'],c['preflight_geometry'])
 projection_equal(proof['geometry'],{str(r['instance_id']):r['canonical8_geometry'] for r in record['instances']})
 clean=c['cleanup'];status='REMOVED_CONTINUATION_GARMENT_TABLE_BASE_VERIFIED' if descriptor['profile'] in ('table_full_garment','cloth_lift10_full_garment') else 'REMOVED_CONTINUATION_ORIGINAL_PARENT_VERIFIED'
 if clean.get('status')!=status or (descriptor['profile'] in ('table_full_garment','cloth_lift10_full_garment') and clean.get('reverse_order')!=['continuation','garment','table']):raise ValueError('Strict layer reverse cleanup required')
 runtime=c['runtime_cleanup']
 if runtime.get('status')!='RESTORED_EXACT_INVENTORY' or runtime.get('original_ids_preserved') is not True or runtime.get('unknown_changes_accepted') is not False:raise ValueError('Exact native-render runtime inventory cleanup required')
 if p['entry']['scene_family']=='self_forearm_occlusion':
  from data import action4_formal_forearm_attribution as attribution
  attribution.validate(c['forearm_attribution'],p,c['contract'],c['preflight_geometry'])
 elif c['forearm_attribution'] is not None:raise ValueError('Unrequested semantic attribution')
 if c['effect']!=effect_check(p,record,c['forearm_attribution']):raise ValueError('Actual family effect evidence changed')
 if c['contract'].get('training_accepted') is not False or c['contract'].get('dataset_generation_allowed') is not False:raise ValueError('Composition cannot self-promote')
 return True


def render_composition(session,entry,p,row,plan,runtime,proof,render_settings,output_dir):
 from data import action4_pose_formal_adapter as adapter
 from data import action4_modular_render as camera
 binding=read(row['pose_execution_ref']);desc=binding['environment_composition']
 refs=dependency_refs(desc,session.context)
 if any(plan['asset_identities'].get(k)!=v for k,v in refs.items()):raise ValueError('Actual composition dependencies absent from formal frozen plan')
 layer=None;primary=None;cleanup=None;result=None;contract=None;geometry=None;probe=None;transaction=None;forearm=None
 try:
  layer=create(session,desc);layer.verifier(session.scene)
  geometry=layer.geometry_diagnostics();contract=deepcopy(layer.contract);geometry_check(p,contract,geometry,session.contract)
  verify_consumed_sources(contract,plan['asset_identities'])
  data=runtime/('composition_preflight_data_'+entry['sample_id']);out=runtime/('composition_preflight_runtime_'+entry['sample_id']);data.mkdir();out.mkdir()
  probe=pixels.render_frame(scene=session.scene,entry=deepcopy(entry),targets=layer.targets(),canonical=session.contract['canonical_definition'],K=camera.actual_K(session.scene),data_out=data,runtime_out=out,output_role='diagnostic',source_identities=deepcopy(plan['asset_identities']),render_settings=deepcopy(render_settings),verify_scene=layer.verifier,probe_only=True)
  if probe['render_calls']!=0 or probe['render_attempts']!=0 or probe['rendered'] is not False or probe['restoration_verified'] is not True or probe['source_identities_verified'] is not True:raise ValueError('Composition original8 probe must not render')
  adapter.verify_geometry_contract(p,session.contract,probe['geometry'],proof['actual_depth']);projection_equal(proof['geometry'],probe['geometry'])
  if entry['scene_family']=='self_forearm_occlusion':
   from data import action4_formal_forearm_attribution as attribution
   forearm=attribution.observe(session,layer,probe['geometry'])
  transaction=load_fixed('render_runtime','formal_render_inventory').BlenderRenderTransaction()
  result=transaction.run(lambda:pixels.render_frame(scene=session.scene,entry=deepcopy(entry),targets=layer.targets(),canonical=session.contract['canonical_definition'],K=camera.actual_K(session.scene),data_out=output_dir,runtime_out=runtime,output_role='dataset_candidate',source_identities=deepcopy(plan['asset_identities']),render_settings=deepcopy(render_settings),verify_scene=layer.verifier,probe_only=False))
  calls=1+(1 if entry['target_count'] else 0)+entry['target_count']
  if result['render_calls']!=calls or result['render_attempts']!=calls or result['rendered'] is not True or result['restoration_verified'] is not True or result['source_identities_verified'] is not True:raise ValueError('Native presence render budget/source/restoration mismatch')
  projection_equal(proof['geometry'],{str(r['instance_id']):r['canonical8_geometry'] for r in result['record']['instances']})
 except BaseException as exc:primary=exc
 finally:
  if layer is not None:
   try:cleanup=layer.remove()
   except BaseException as exc:
    raise RuntimeError('Composition cleanup failed; primary='+repr(primary)+'; cleanup='+repr(exc)) from (primary or exc)
 if primary is not None:raise primary
 # RenderResult transaction and composition removal have already completed.
 c={'descriptor':deepcopy(desc),'contract':contract,'contract_sha256':backend()._digest(contract),'geometry':geometry,'preflight_geometry':deepcopy(probe['geometry']),'cleanup':cleanup,'runtime_cleanup':deepcopy(transaction.report),'source_contract_sha256':backend()._digest(session.contract),'forearm_attribution':forearm,'effect':effect_check(p,result['record'],forearm)}
 proof={**proof,'composition':c}
 verify_evidence(p,binding,proof,result['record'],session.contract)
 if dependency_refs(desc,session.context)!=refs:raise ValueError('Composition sources changed across transaction')
 return result,proof


def verify_consumed_sources(contract,identities):
 """Actual builder readsets must be a subset of the pre-render frozen closure."""
 def walk(v):
  if isinstance(v,dict):
   for key,value in v.items():
    if key in ('dependency_identities','actual_dependency_refs'):
     if not isinstance(value,dict) or any(identities.get(p)!=h for p,h in value.items()):raise ValueError('Actual builder consumed an unbound source')
    else:walk(value)
  elif isinstance(v,list):
   for x in v:walk(x)
 walk(contract)
 return True


def prepare_binding(parameters_ref,context,output_dir,*,environment_refs=None):
 """Write one exclusive v2 row after real typed/source validation.
 Used by the existing finite release planner AFTER coordinated source migration.
 Does not create a render plan, scene, image, acceptance or training artifact.
 """
 b=backend();p=read(parameters_ref);controls.validate_pose_controls(p,context)
 bank_ref=p['provenance']['bank_config_ref'];loaded,_,_=controls.load_config(bank_ref['path'],context)
 api=load_fixed('interface','formal_binding_writer');reader=api._ReadSet();authority_ref=p['provenance']['entry_source_ref'];api._authority(reader,authority_ref,p['entry'])
 bank_id=p['controls']['bank_id'];geometry_bank=loaded['banks'][bank_id].get('geometry_bank_id',bank_id)
 assets,_,_=api._geometry_mapping(reader,loaded,bank_id,geometry_bank);reader.close()
 target=synth._path(output_dir)
 if not target.is_relative_to(synth.ROOT/'outputs') or target.exists():raise ValueError('Fresh exclusive project outputs directory required')
 target.mkdir(parents=True,exist_ok=False)
 def write(name,value):
  path=target/name;raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  with path.open('xb') as stream:stream.write(raw)
  return {'path':str(path.relative_to(synth.ROOT)),'sha256':hashlib.sha256(raw).hexdigest()}
 q={'schema':api.SCHEMA,'status':api.STATUS,'bank_id':bank_id,'geometry_bank_id':geometry_bank,'parameters_ref':parameters_ref,'config_ref':bank_ref,'factory_ref':loaded['factory_ref'],'source_context_ref':loaded['source_context_ref'],'source_context_sha256':p['provenance']['source_context_sha256'],'bridge_ref':loaded['bridge_ref'],'entry_authority_ref':authority_ref,**{k:assets[k] for k in ('baseline_scene_contract_ref','candidate_refs','geometry_refs')},'training_accepted':False,'dataset_generation_allowed':False}
 request_ref=write('request.json',q)
 config={'case':bank_id,'bank_id':bank_id,'geometry_bank_id':geometry_bank,'request_ref':request_ref,'entry_authority_ref':authority_ref,'parameters_override_ref':parameters_ref,'baseline_source_contract_ref':assets['baseline_scene_contract_ref'],'baseline_scene_contract_ref':assets['baseline_scene_contract_ref'],'current_factory_ref':loaded['factory_ref'],'current_config_ref':bank_ref,'current_source_context_ref':loaded['source_context_ref'],'candidates':assets['candidate_refs'],'dependencies':[],**{k:REVIEWED[k] for k in ('fixed_interface','selection_handoff','spec_review','interface','actual_presence_ref','geometry_checks')}}
 family=p['entry']['scene_family']
 if family not in PROFILES:raise ValueError('No executable profile for family')
 profile=PROFILES[family]
 if profile!='bare_continuation':
  if not isinstance(environment_refs,dict) or set(environment_refs)!={'environment_source_ref','environment_binding_ref','garment_config_ref'}:raise ValueError('Explicit current environment metadata required; no historical default')
  config.update(deepcopy(environment_refs));config.update({k:REVIEWED[k] for k in ('environment_provider_module_ref','garment_module_ref')});config.update(environment_module=module_ref('action4_formal_environment'),environment_profile=profile)
 if family=='cloth_lift':config.update(composition_module_ref=module_ref('action4_formal_lift_composition'),lift_module_ref=module_ref('action4_formal_lift'),angle_deg=10)
 config_ref=write('layer_config.json',config)
 composition={'kind':'actual_composition','profile':profile,'layer_config_ref':config_ref,'source_refs':{'factory':loaded['factory_ref'],'source_context':loaded['source_context_ref']}}
 binding={'schema':SCHEMA,'parameters_ref':parameters_ref,'entry_authority_ref':authority_ref,'bank_config_ref':bank_ref,'environment_composition':composition,'training_accepted':False,'dataset_generation_allowed':False}
 row={'entry':deepcopy(p['entry']),'parameters_ref':parameters_ref}
 validate_descriptor(row,p,binding,loaded,context)
 # A live, fully closed consumer is required before final row is written. This
 # deliberately rejects old-release/new-source mixtures instead of relabeling.
 deps=dependency_refs(composition,context)
 row['pose_execution_ref']=write('pose_execution.json',binding)
 write('row.json',row);write('dependencies.json',{'identities':deps,'training_accepted':False,'dataset_generation_allowed':False})
 return row


def verify_pair_contract(p,pair,continuation):
 assets,metadata,baseline=host_geometry_assets(p,pair)
 W=np.asarray(p['W_from_capture'],float)
 if set(pair['sides'])!={'left','right'} or set(continuation['objects'])!={'left_forearm_continuation','right_forearm_continuation'}:raise ValueError('Both unchanged physical forearms required')
 for side in ('left','right'):
  proof=pair['sides'][side];obj=continuation['objects'][side+'_forearm_continuation']
  for key in ('request_ref','parameters_ref','config_ref','bridge_ref','source_context_sha256'):
   if proof[key]!=pair[key]:raise ValueError('Side construction provenance differs from bound pair')
  meta=metadata[side];skin=meta['source_skin_object'];frozen=baseline['expected_state']['objects'][skin]
  if proof['candidate_ref']!=assets['candidate_refs'][side] or proof['geometry_ref']!=assets['geometry_refs'][side] or proof['baseline_scene_contract_ref']!=assets['baseline_scene_contract_ref'] or proof['geometry_bank_id']!=meta['bank_id'] or proof['bank_id']!=p['controls']['bank_id'] or obj['parent']!=skin:raise ValueError('Host continuation references/parent differ from immutable selected geometry')
  if frozen['world']!=meta['source_skin_world_matrix'] or frozen['mesh']['geometry_uv_sha256']!=meta['source_mesh_geometry_uv_sha256'] or proof['baseline_skin_world']!=meta['source_skin_world_matrix']:raise ValueError('Host source-skin baseline differs from immutable selected asset')
  B0_expected=baseline['parameters']['actor_pose_capture_by_side'][side]['brace_B_capture']
  if not np.allclose(proof['Bbaseline_capture'],B0_expected,atol=1e-12,rtol=0):raise ValueError('Host baseline B differs from immutable geometry-bank parameters')
  parent=baseline['expected_state']['objects'][frozen['parent']]
  if not np.allclose(np.asarray(parent['world'])@np.asarray(frozen['parent_inverse'])@np.asarray(frozen['basis']),meta['source_skin_world_matrix'],atol=2e-6,rtol=0):raise ValueError('Immutable source skin parent hierarchy differs')
  with np.load(backend()._ref(assets['geometry_refs'][side]),allow_pickle=False) as arrays:
   rim=arrays['source_boundary_vertex_ids'];triangles=arrays['triangles'];vertices=arrays['vertices_skin_object_local_m']
   edges={tuple(sorted((int(a),int(b)))) for a,b in zip(rim,np.roll(rim,-1))}
   if obj['vertex_count']!=len(vertices) or not np.array_equal(obj['triangles'],triangles) or obj['source_rim_ids']!=rim.tolist() or len(obj['source_boundary_edges'])!=len(edges) or {tuple(sorted(x)) for x in obj['source_boundary_edges']}!=edges:raise ValueError('Host continuation topology/source44 rim differs from immutable NPZ')
  B=np.asarray(p['actor_pose_capture_by_side'][side]['brace_B_capture'],float);B0=np.asarray(proof['Bbaseline_capture'],float)
  D=W@B@np.linalg.inv(W@B0);expected=D@np.asarray(proof['baseline_skin_world'],float)
  comparisons=[(proof['Bnew_capture'],B,1e-12),(proof['Dworld_from_bank'],D,1e-9),(proof['expected_skin_world'],expected,1e-9),(proof['observed_skin_world'],expected,2e-6),(obj['world'],expected,2e-6)]
  for actual,want,tolerance in comparisons:
   a=np.asarray(actual,float)
   if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a,want,atol=tolerance,rtol=0):raise ValueError('Actual continuation transform differs from current W/B and immutable geometry baseline')
  if proof['side']!=side or proof['current_entry_sha256']!=backend()._digest(p['entry']) or proof['frozen_A_S_F']!={k:p['sides'][side][k] for k in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace')} or obj['typed_pose_proof']!=proof or obj['candidate_ref']!=proof['candidate_ref']:raise ValueError('Actual continuation fixed physical/current entry binding mismatch')
  if not 0<=obj['seam_world_error_m']<=1e-7 or not 0<=obj['seam_normal_max_error']<=2e-4 or len(obj['source_rim_ids'])!=44 or len(set(obj['source_rim_ids']))!=44 or obj['material_shared_original_unchanged'] is not True or obj['generated_coordinate_texspace_copied_from_source'] is not True:raise ValueError('Exact source44 seam/normals/original material binding required')
 return True


def host_geometry_assets(p,pair):
 """Read only the exact selection and geometry references. Typed current parameter
 validation remains the upstream adapter's responsibility; no old code executes.
 """
 api=load_fixed('interface','formal_host_frozen_geometry_selection')
 q=read(pair['request_ref'])
 if q['parameters_ref']!=pair['parameters_ref'] or q['config_ref']!=p['provenance']['bank_config_ref'] or q['bank_id']!=p['controls']['bank_id'] or q['geometry_bank_id']!=pair['geometry_bank_id']:raise ValueError('Host geometry request differs from current typed parameters')
 bank_id=q['bank_id'];geometry_id=q['geometry_bank_id'];assets=api.bank_assets(geometry_id)
 if bank_id!=geometry_id:
  # Reuse the reviewed resolver's finite mapping and immutable package assets.
  # The parent typed validator already validates expand_v5/resolve_rows in full.
  cfg=read(q['config_ref'])
  if cfg.get('schema')!='action4.modular_pose_bank_config.v5' or cfg.get('resolver_ref')!=api.DERIVED_RESOLVER:raise ValueError('Only registered derived preset geometry mapping')
  reader=api._ReadSet();resolver=api._load(reader,api.DERIVED_RESOLVER,'formal_host_derived_mapping');package=reader.read(cfg['presets_ref']);reader.close()
  if resolver.NAMES.get(bank_id)!=geometry_id or package['base_config_ref']!=cfg['base_config_ref'] or package['presets'][bank_id]['geometry_bank_id']!=geometry_id or package['presets'][bank_id]['geometry_assets']!=assets:raise ValueError('Host derived geometry mapping differs from registered immutable assets')
 for key in ('baseline_scene_contract_ref','candidate_refs','geometry_refs'):
  if q[key]!=assets[key]:raise ValueError('Host request differs from exact frozen geometry-bank selection')
 baseline=read(assets['baseline_scene_contract_ref']);metadata={side:read(assets['candidate_refs'][side]) for side in ('left','right')}
 if baseline['parameters']['controls']['bank_id']!=geometry_id or baseline['parameters']['target_presence']!=[True,True]:raise ValueError('Immutable baseline must be the original dual geometry bank')
 if pair['baseline_entry_identity']!=baseline['parameters']['entry'] or p['W_from_capture']!=baseline['parameters']['W_from_capture']:raise ValueError('Immutable geometry baseline entry/W changed')
 for side,meta in metadata.items():
  if meta['bank_id']!=geometry_id or meta['side']!=side or meta['geometry_ref']!=assets['geometry_refs'][side] or meta['scene_contract_ref']!=assets['baseline_scene_contract_ref']:raise ValueError('Frozen selected candidate metadata mismatch')
 return assets,metadata,baseline

def verify_named_inventory(p,contract,geometry,source):
 """Exact role bijection and pair names, using immutable baseline source names.
 Cardinalities remain useful diagnostics, never substitute for set coverage.
 """
 if source is None or source['parameters']!=p:raise ValueError('Actual original source contract required for named geometry coverage')
 full=contract.get('schema')=='action4.complete_table_garment_continuation.v1'
 continuation=contract['owner_contracts']['continuation'] if full else contract
 _,metadata,baseline=host_geometry_assets(p,continuation['parametric_binding_proof'])
 presence=load_fixed('presence_rules','formal_host_presence_inventory')
 from data import action4_modular_render as camera
 expected=presence.expected_pair_inventory(p['entry']);requirements=presence.presence_contract(p['entry'],p);mapping={};retained={}
 for side,i,on in zip(('left','right'),(1,2),p['target_presence']):
  names={'core':pixels.KNOWN_TARGET_PARTS[i]['core'],'decal':pixels.KNOWN_TARGET_PARTS[i]['decal'],'skin':metadata[side]['source_skin_object'],'band':'left_brace_band' if side=='left' else 'brace_band',**{r:'mount_'+side+'_'+r for r in ('quick_release','usb_connector','white_module','ball','link')}}
  original=baseline['actor_collision_diagnostics']['retained_actor_surface_names'][side]
  remaining=set(original)-set(names.values())
  if len(original)!=9 or len(remaining)!=1 or set(names.values())-set(original)!={names['decal']}:raise ValueError('Immutable actor source-role bijection is not the known fixed geometry')
  names['cable']=remaining.pop()
  if not set(names.values())<=set(source['actor_bindings'][side]['names']) or any(source['expected_state']['objects'][name]['type']!='MESH' for name in names.values()):raise ValueError('Current original source actor names/types differ from immutable source roles')
  skin=names['skin'];actual_skin=source['expected_state']['objects'][skin];sideproof=continuation['parametric_binding_proof']['sides'][side]
  if actual_skin['parent']!=baseline['expected_state']['objects'][skin]['parent'] or not np.allclose(actual_skin['world'],sideproof['observed_skin_world'],atol=2e-6,rtol=0):raise ValueError('Actual source skin parent/world differs from bound immutable hierarchy and observed pair')
  mapping.update({side+'/'+role:name for role,name in names.items()});mapping[side+'/continuation']=side+'_forearm_continuation'
  roles=presence.PRESENT_ROLES if on else presence.RETAINED_ROLES;retained[side]=[names[role] for role in roles]
 mapping.update({'environment/wall':camera.ENV_NAMES[0],'environment/table':camera.ENV_NAMES[1]})
 if len(set(mapping.values()))!=len(mapping):raise ValueError('Source and owned object role names overlap')
 collision=source['actor_collision_diagnostics'];actual_original=[]
 for key,row in collision['cross_pairs'].items():
  a,b=key.split('__');actual_original.append(('left/'+a,'right/'+b))
  if row['outside_or_unresolved_pairs']:raise ValueError('Actual original cross intersection')
 for side,rows in collision['internal_pairs'].items():
  for key,row in rows.items():
   a,b=key.split('__');actual_original.append((side+'/'+a,side+'/'+b))
   if row['outside_or_unresolved_pairs']:raise ValueError('Actual original internal intersection')
 presence.validate_pair_inventory(p['entry'],actual_original,expected['bare_continuation'])
 for side in ('left','right'):
  if len(collision['retained_actor_surface_names'][side])!=len(retained[side]) or set(collision['retained_actor_surface_names'][side])!=set(retained[side]):raise ValueError('Exact actual retained actor names required')
 actors=retained['left']+retained['right'];continuations=[mapping[s+'/continuation'] for s in ('left','right')]
 def pairset(rows):
  if any(len(row)!=2 or row[0]==row[1] for row in rows):raise ValueError('Exact two distinct object names per pair')
  keys=[tuple(sorted(row)) for row in rows]
  if len(set(keys))!=len(keys):raise ValueError('Duplicate named geometry pair')
  return set(keys)
 if full:
  registry=load_fixed('registry','formal_host_registered_owners')
  for role in ('table','garment','continuation'):
   if set(contract['owner_contracts'][role]['objects'])!=set(registry.EXPECTED[role]):raise ValueError('Exact registered full composition owner names required')
  table=contract['owner_contracts']['table']['table_name'];garment=contract['owner_contracts']['garment'];panel=garment['panel_name'];seam=garment['seam_name']
  if table!='modular_environment_v002_table' or panel!='wearing_garment_table_garment' or seam!='wearing_garment_table_garment_boundary_stitch':raise ValueError('Registered table/panel/seam roles differ')
  others=actors+[camera.ENV_NAMES[0],table,panel,seam];want=[(a,b) for a in continuations for b in others]+[tuple(continuations)]
  if set(geometry['composition_halfspace_and_containment']['all20_actor_and_continuation_top_gaps_m'])!=set(actors+continuations):raise ValueError('Exact full20 halfspace actor and continuation names required')
 else:
  want=[(mapping[a],mapping[b]) for a,b in expected['bare_continuation']];inv=geometry['presence_pair_inventory']
  if not isinstance(inv,dict) or set(inv)!={'requirements','object_by_semantic_role','original_actual_pairs','retained_actor_surface_names','observed_visibility','continuation_objects_observed','bare_actual_pairs'}:raise ValueError('Complete exact presence inventory fields required')
  if inv['requirements']!=requirements or inv['object_by_semantic_role']!=mapping or inv['continuation_objects_observed'] is not True:raise ValueError('Complete observed presence role inventory required')
  presence.validate_pair_inventory(p['entry'],inv['original_actual_pairs'],inv['bare_actual_pairs']);presence.validate_role_visibility(p['entry'],inv['observed_visibility'])
  if pairset(inv['original_actual_pairs'])!=pairset(actual_original):raise ValueError('Presence inventory differs from actual source pair rows')
  for side in ('left','right'):
   if len(inv['retained_actor_surface_names'][side])!=len(retained[side]) or set(inv['retained_actor_surface_names'][side])!=set(retained[side]):raise ValueError('Presence inventory retained objects differ')
 actual=[key.split('__') for key in geometry['pairs']]
 if pairset(actual)!=pairset(want) or set(geometry['self_pairs'])!=set(continuations):raise ValueError('Exact bare/full pair and two self-pair object coverage required')
 return True
