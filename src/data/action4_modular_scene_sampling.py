"""Consume bounded modular parameters on a separately verified bilateral base.

No scene/sampling/render on import or prepare. Dynamic contracts replace the
f0 verifier only while the allowed whole-actor/environment changes are active.
Unimplemented family effects fail closed; no training or anatomical acceptance.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from data import action4_modular_bilateral_scene as bilateral
from data import action4_modular_pixels as pixels
from data import action4_modular_render as neutral
from data import action4_modular_assembly as assembly
from data import action4_global_environment as env
from data import action4_mount_modules as mounts
ROOT=Path('/data/zy/vln/a-test')
CAPABILITIES=('whole_actor_rigid','target_presence','wall_table_visibility','area_light_gain','exposure_offset','layout_conditioned_preview')
PENDING_FAMILY_EFFECTS={'table_fastening':'cloth_patch','table_turn':'cloth_patch','cloth_lift':'cloth_patch','near_double':'near_double_effect','self_forearm_occlusion':'self_forearm_occlusion_effect','wall_transition_truncation':'truncation_effect'}
FAMILIES=('wall_rear','table_fastening','table_turn','near_double','self_forearm_occlusion','cloth_lift','wall_transition_truncation','hard_negative')

class PendingFamilyError(ValueError):
 def __init__(self,reasons):
  self.reasons=list(reasons);super().__init__('Pending scene family capability/effect: '+str(self.reasons))

class ActorCollisionError(ValueError):
 def __init__(self,report):
  self.report=report;super().__init__('Dynamic actor collision outside/unresolved pairs: '+str(report['outside_or_unresolved_surface_pairs']))

def _same(a,b):return env.digest(a)==env.digest(b)

def _metadata_rigid_inverse(matrix):
 """Identity metadata only: fixed Python-double transpose/translation inverse.
 Validation uses the existing rigid guard; actual bilateral geometry keeps its
 original NumPy operations. Products and sums below are explicit scalar steps.
 """
 assembly.rigid(matrix)
 out=[[float(matrix[j][i]) for j in range(3)]+[0.0] for i in range(3)]
 for i in range(3):
  total=0.0
  for k in range(3):
   product=out[i][k]*float(matrix[k][3])
   total=total+product
  out[i][3]=-total
 return out+[[0.0,0.0,0.0,1.0]]

def _metadata_rigid_product(left,right):
 """Identity metadata only: fixed k=0..3 separate multiply then add."""
 assembly.rigid(left);assembly.rigid(right);out=[]
 for i in range(4):
  row=[]
  for j in range(4):
   total=0.0
   for k in range(4):
    product=float(left[i][k])*float(right[k][j])
    total=total+product
   row.append(total)
  out.append(row)
 return out

def prepare_base(scene_config):
 """CPU/cache-only explicit base factory; legacy bilateral schema is retained."""
 path=bilateral.bounded(scene_config)
 packet=bilateral.read(path)
 if packet.get('schema')=='action4.wearing_sampling_base.v1':return _prepare_wearing_base(path,packet)
 c,ctx=bilateral.prepare(path)
 right=ctx['right_context'];refs={str(path.relative_to(ROOT)):bilateral.sha(path)}
 for name in pixels.REQUIRED_SOURCE_PATHS:refs[name]=bilateral.sha(name)
 refs[str(Path(__file__).resolve().relative_to(ROOT))]=bilateral.sha(__file__)
 for config in (c,ctx['right_config'],right['neutral_config']):
  for row in config['sources'].values():
   if row['path'] in refs and refs[row['path']]!=row['sha256']:raise ValueError('Conflicting frozen source identities')
   refs[row['path']]=row['sha256']
 registry=bilateral.read(c['sources']['registry_v003']['path'])
 for component in registry['components']:
  for name,digest in component['files'].items():
   if name in refs and refs[name]!=digest:raise ValueError('Frozen registry reference conflict')
   refs[name]=digest
 source={'W_from_capture':ctx['left']['W'].tolist(),'K_native':ctx['left']['K'].tolist(),'baseline_environment':deepcopy(ctx['right_config']['environment']),'capabilities':list(CAPABILITIES),'frozen_refs':refs,'scene_config_ref':{'path':str(path.relative_to(ROOT)),'sha256':bilateral.sha(path)},'sides':{}}
 for side,g,anchors in [('left',ctx['left'],{'0':ctx['left']['C'].tolist()}),('right',right['geometry'],right['rc']['C_per_frame'])]:
  original=g['j'] if side=='left' else right['rc']
  metadata_B=_metadata_rigid_product(original['C_per_frame']['0'],_metadata_rigid_inverse(original['A_shared']))
  source['sides'][side]={'A_brace_from_CAD':g['A'].tolist(),'S_skin_relative_to_brace':g['S'].tolist(),'F_body_from_brace':deepcopy(g['wc']['rigid_F_new_body']),'anchor_C_capture_by_frame':deepcopy(anchors),'base_C_capture':g['C'].tolist(),'base_B_capture':metadata_B}
 return {'config':c,'bilateral_context':ctx,'source_context':source,'source_context_sha256':env.digest(source)}

def _prepare_wearing_base(path,packet):
 """Lazy factory avoids the wearing wrapper's legacy initializer import cycle."""
 from data import action4_modular_wearing_base as wearing
 if set(packet)!={'schema','base_kind','wearing_base_config','sources','dataset_generation_allowed','training_accepted'} or packet['base_kind']!='palmar_wearing' or packet['dataset_generation_allowed'] is not False or packet['training_accepted'] is not False:raise ValueError('Explicit nonaccepted palmar wearing sampling factory')
 for row in [packet['wearing_base_config'],*packet['sources'].values()]:wearing.check_ref(row)
 if packet['sources'].get('sampling')!={'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':bilateral.sha(__file__)} or packet['sources'].get('wrapper')!={'path':str(Path(wearing.__file__).resolve().relative_to(ROOT)),'sha256':bilateral.sha(wearing.__file__)}:raise ValueError('Exact sampling/wearing factory source bindings')
 base=wearing.prepare(packet['wearing_base_config']['path']);source=deepcopy(base['source_context'])
 source['base_kind']='palmar_wearing';source['absence_policy']={'hidden_mount_roles':['quick_release','usb_connector','link','cable'],'retained_background_roles':['skin','band','white_module','ball'],'always_hidden':['mount_right_cable'],'claim':'Omit core/decal and camera-specific connector/link/cable; retain wrist skin/band, white module and black ball base, all pass0.'};source['scene_config_ref']={'path':str(path.relative_to(ROOT)),'sha256':bilateral.sha(path)}
 for row in [source['scene_config_ref'],packet['wearing_base_config'],*packet['sources'].values()]:
  previous=source['frozen_refs'].get(row['path'])
  if previous is not None and previous!=row['sha256']:raise ValueError('Wearing sampling identity conflict')
  source['frozen_refs'][row['path']]=row['sha256']
 collision=deepcopy(base['ancestor']['bilateral_context']['right_context']['mount_contract'])
 collision['objects']['cable']['name']=base['config']['new_cable_object']
 right={**base['ancestor']['bilateral_context']['right_context'],'mount_contract':collision}
 context={'base_kind':'palmar_wearing','factory_config':deepcopy(packet),'config':base['ancestor']['config'],'bilateral_context':{**base['ancestor']['bilateral_context'],'right_context':right},'wearing_context':base,'source_context':source,'source_context_sha256':env.digest(source)}
 verify_sources(context);return context

def verify_sources(context):
 if context.get('base_kind')!=context['source_context'].get('base_kind'):raise ValueError('Explicit base factory/source context mismatch')
 for name,digest in context['source_context']['frozen_refs'].items():
  if bilateral.sha(name)!=digest:raise ValueError('Frozen scene sampling source changed: '+name)
 if env.digest(context['source_context'])!=context['source_context_sha256']:raise ValueError('Prepared source context changed')

def validate_parameters(parameters,context):
 """Pure metadata guards; no target/occlusion effect is inferred from a family."""
 verify_sources(context);s=context['source_context'];p=parameters
 layout_mode=isinstance(p,dict) and p.get('schema') in ('action4.modular_layout_proposal.v1','action4.modular_layout_proposal.v2')
 if layout_mode:
  from data import action4_modular_layout_proposals as layout
  layout.validate_layout_parameters(p,context)
 if not isinstance(p,dict) or p.get('schema') not in ('action4.modular_parameters.v1','action4.modular_layout_proposal.v1','action4.modular_layout_proposal.v2') or p.get('estimated') is not True:raise ValueError('Estimated modular parameter JSON required')
 if layout_mode!=(p.get('status') in ('LAYOUT_CONDITIONED_PREVIEW_CANDIDATE','LAYOUT_ORIENTATION_CONDITIONED_REQUIRES_ACTUAL_GEOMETRY')):raise ValueError('Explicit layout schema/status pair required; no old-delta disguise')
 if p.get('status') not in ('BASELINE_REGRESSION_ENGINEERING_ONLY','BOUNDED_AUGMENTATION_CANDIDATE','BOUNDED_REFERENCE_CANDIDATE','LAYOUT_CONDITIONED_PREVIEW_CANDIDATE','LAYOUT_ORIENTATION_CONDITIONED_REQUIRES_ACTUAL_GEOMETRY'):raise PendingFamilyError(['Global parameter status '+str(p.get('status'))])
 if p['augmentation']['counts_as_formal_coverage'] is not False:raise ValueError('Engineering candidate cannot count as formal coverage')
 if p['status']=='BOUNDED_REFERENCE_CANDIDATE':
  info=p['augmentation']
  if info['reference_bounds_known'] is not True or info['reference_bounds_status']!='evidence_checked_candidate' or not isinstance(info['reference_bounds'],dict) or not info['reference_bounds'] or not isinstance(info['reference_evidence_refs'],list) or not info['reference_evidence_refs']:raise PendingFamilyError(['Reference bounds lack explicit evidence binding'])
 if p.get('training_accepted') is not False or p.get('dataset_generation_allowed') is not False:raise ValueError('Explicit nonaccepted parameter status required')
 if p.get('dataset_generation_allowed',False) is not False or p.get('training_accepted',False) is not False:raise ValueError('Parameters cannot authorize production')
 for name in ('W_from_capture','K_native'):
  if not _same(p[name],s[name]):raise ValueError('Frozen common '+name)
 presence=p['target_presence']
 if not isinstance(presence,list) or len(presence)!=2 or any(type(v) is not bool for v in presence):raise ValueError('Explicit LEFT/RIGHT boolean presence')
 if set(p['sides'])!={'left','right'} or set(p['source_anchor_frames_by_side'])!={'left','right'}:raise ValueError('Complete explicit side metadata')
 entry=p['entry']
 if entry['target_presence']!=presence or entry['target_count']!=sum(presence) or entry['scene_family']!=p['scene_family_controls']['family']:raise ValueError('Frozen entry/presence/family mismatch')
 if p['provenance']['source_context_sha256']!=context['source_context_sha256']:raise ValueError('Parameter source context binding')
 for ref in [*p['provenance'].get('refs',{}).values(),*p['augmentation'].get('reference_evidence_refs',[])]:
  if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or Path(ref['path']).is_absolute() or '..' in Path(ref['path']).parts or bilateral.sha(ref['path'])!=ref['sha256']:raise ValueError('Parameter provenance source changed')
 if not layout_mode and presence==[True,True] and p['source_anchor_frames_by_side']!={'left':0,'right':0}:raise ValueError('Paired actors require the actual same f0 anchor')
 for i,side in enumerate(('left','right')):
  row=p['sides'][side];original=s['sides'][side]
  if row['present'] is not presence[i]:raise ValueError('Side/presence mismatch')
  for name in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace'):
   if not _same(row[name],original[name]):raise ValueError('Frozen side '+name)
  frame=p['source_anchor_frames_by_side'][side]
  if not presence[i]:
   if row['core_C_capture'] is not None or row['brace_B_capture'] is not None or row['whole_actor_delta_capture'] is not None or frame is not None:raise ValueError('Absent side must have null C/B/anchor')
   continue
  if type(frame) is not int or str(frame) not in original['anchor_C_capture_by_frame']:raise ValueError('Unbound source anchor')
  C=assembly.rigid(row['core_C_capture']);B=assembly.rigid(row['brace_B_capture']);A=assembly.rigid(original['A_brace_from_CAD'])
  if np.max(abs(C-B@A))>1e-9:raise ValueError('C = B A must preserve fixed assembly')
  if layout_mode:continue # Larger translation/R-template closure was independently checked above.
  anchor=assembly.rigid(original['anchor_C_capture_by_frame'][str(frame)])@np.linalg.inv(A)
  translation=np.asarray(row['translation_capture_m'],float);angles=np.asarray(row['rotation_brace_local_xyz_deg'],float)
  if translation.shape!=(3,) or angles.shape!=(3,) or not np.isfinite(translation).all() or not np.isfinite(angles).all() or np.any(abs(translation)>.004+1e-12) or np.any(abs(angles)>8+1e-12):raise ValueError('Whole actor bounded4mm/8degree controls required')
  ax,ay,az=np.deg2rad(angles);cx,sx=np.cos(ax),np.sin(ax);cy,sy=np.cos(ay),np.sin(ay);cz,sz=np.cos(az),np.sin(az)
  R=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
  if np.max(abs(B[:3,:3]-anchor[:3,:3]@R))>1e-9 or np.max(abs(B[:3,3]-anchor[:3,3]-translation))>1e-9 or np.max(abs(assembly.rigid(row['whole_actor_delta_capture'])-B@np.linalg.inv(anchor)))>1e-9:raise ValueError('Bounded actor metadata differs from actual B/delta')
 e=p['environment_controls']
 if set(e)!={'area_light_gain','exposure_offset_ev','wall_visible','table_visible'} or any(type(e[k]) is not bool for k in ('wall_visible','table_visible')):raise ValueError('Explicit supported environment controls')
 for key,lo,hi in [('area_light_gain',.9,1.1),('exposure_offset_ev',-.15,.15)]:
  if isinstance(e[key],bool) or not np.isfinite(e[key]) or not lo<=e[key]<=hi:raise ValueError('Bounded photometry control '+key)
 family=p['scene_family_controls']
 if family['family'] not in FAMILIES:raise ValueError('Unknown scene family')
 reasons=list(family['pending_reasons'])
 if family['family'] in PENDING_FAMILY_EFFECTS:reasons.append('Consumer pending '+PENDING_FAMILY_EFFECTS[family['family']])
 missing=set(family['required_capabilities'])-set(CAPABILITIES)
 if missing:reasons.append('Unsupported capabilities '+str(sorted(missing)))
 if family['coarse_occluders']:reasons.append('Cloth/occluder geometry not implemented')
 if family.get('effect_pending'):reasons.append('Family effect has no actual geometry/visibility evidence')
 if reasons:raise PendingFamilyError(reasons)
 if family['family']=='hard_negative' and any(presence):raise ValueError('Hard negative cannot contain targets')
 return deepcopy(p)

def _snapshot(scene):
 for o in scene.objects:bilateral._check_static(o)
 state=neutral.new_scene_state(scene,None)
 state['camera_ray_visibility']={o.name:bool(o.visible_camera) for o in scene.objects}
 state['collection_visibility']=env._collection_visibility(scene)
 state['skin_groups']={o.name:bilateral._group_state(o) for o in scene.objects if o.type=='MESH' and len(o.vertex_groups)}
 return state

def _hidden_when_absent(mc,instance_id,wearing=False):
 names=set(pixels.KNOWN_TARGET_PARTS[instance_id].values())
 if wearing:return names|{mc['objects'][role]['name'] for role in ('quick_release','usb_connector','link','cable')}
 return names|set(mc['root_objects'].values())|{row['name'] for row in mc['objects'].values()}

def actor_bindings(scene,context):
 ctx=context['bilateral_context'];out={}
 for side,g,mc in [('left',ctx['left'],ctx['left_mount_contract']),('right',ctx['right_context']['geometry'],ctx['right_context']['mount_contract'])]:
  roots=[('left_brace_module_root' if side=='left' else 'brace_module_root'),g['wc']['root_name'],mc['root_objects']['newbrace']]
  names={o.name for root in roots for o in [scene.objects[root],*scene.objects[root].children_recursive]}
  expected={g['wc']['root_name'],g['wc']['object_name'],('left_brace_module_root' if side=='left' else 'brace_module_root'),('left_brace_band' if side=='left' else 'brace_band'),('left_wrist_reference' if side=='left' else 'wrist_reference'),*pixels.KNOWN_TARGET_PARTS[1 if side=='left' else 2].values(),*mc['root_objects'].values(),*(row['name'] for row in mc['objects'].values())}
  if context.get('base_kind')=='palmar_wearing' and side=='right':
   old=context['wearing_context']['cable_contract']['replace_only_old_cable_object'];expected.add(old)
   if not scene.objects[old].hide_render:raise ValueError('Superseded cable must remain hidden')
  if names!=expected or any(scene.objects[n].parent is not None for n in roots):raise ValueError('Exact independent source actor roots/hierarchy')
  hide_names=_hidden_when_absent(mc,1 if side=='left' else 2,context.get('base_kind')=='palmar_wearing')
  out[side]={'roots':roots,'names':sorted(names),'hidden_when_absent':sorted(hide_names)}
  if context.get('base_kind')=='palmar_wearing' and side=='right':out[side]['always_hidden']=[old]
 if set(out['left']['names'])&set(out['right']['names']):raise ValueError('Actor ownership overlaps')
 return out

def expected_state(baseline,bindings,parameters,source):
 """Analytic allowed-state delta; never learn an expected contract after mutation."""
 expected=deepcopy(baseline);W=assembly.rigid(source['W_from_capture'])
 for i,side in enumerate(('left','right')):
  present=parameters['target_presence'][i]
  if parameters.get('schema')=='action4.modular_pose_controls.v1':
   actor_B=parameters['actor_pose_capture_by_side'][side]['brace_B_capture']
   delta=W@assembly.rigid(actor_B)@np.linalg.inv(W@assembly.rigid(source['sides'][side]['base_B_capture']))
  else:
   delta=W@assembly.rigid(parameters['sides'][side]['brace_B_capture'])@np.linalg.inv(W@assembly.rigid(source['sides'][side]['base_B_capture'])) if present else np.eye(4)
  for name in bindings[side]['names']:
   row=expected['objects'][name];row['world']=(delta@np.array(row['world'])).tolist()
   if name in bindings[side]['roots']:row['basis']=(delta@np.array(row['basis'])).tolist()
   if not present:
    if name in bindings[side]['hidden_when_absent']:row['hide_render']=True
    row['pass_index']=0
   if name in bindings[side].get('always_hidden',[]):
    if not baseline['objects'][name]['hide_render']:raise ValueError('Superseded cable baseline must be hidden')
    row['hide_render']=True
   if row['mesh'] is not None:row['mesh']['pass_index']=row['pass_index']
 e=parameters['environment_controls'];base=source['baseline_environment']
 expected['objects'][neutral.ENV_NAMES[0]]['hide_render']=not e['wall_visible']
 expected['objects'][neutral.ENV_NAMES[1]]['hide_render']=not e['table_visible']
 expected['objects'][neutral.ENV_NAMES[2]]['light']['properties']['energy']=base['area_energy_w']*e['area_light_gain']
 expected['view']['exposure']=base['exposure']+e['exposure_offset_ev']
 return expected

def verify_state(actual,expected):
 """Only matrix roundoff and Blender float photometry rounding are tolerated."""
 value=deepcopy(actual);target=deepcopy(expected)
 if set(value['objects'])!=set(target['objects']):raise ValueError('Dynamic scene inventory changed')
 for name,row in target['objects'].items():
  for key in ('world','basis'):
   if np.max(abs(np.asarray(value['objects'][name][key])-np.asarray(row[key])))>2e-6:raise ValueError('Dynamic allowed matrix differs '+name+'/'+key)
   value['objects'][name][key]=row[key]
 for name,row in target['objects'].items():
  if row['light'] is not None:
   energy=value['objects'][name]['light']['properties']['energy'];want=row['light']['properties']['energy']
   if not np.isclose(energy,want,atol=1e-6,rtol=0):raise ValueError('Dynamic light energy differs')
   value['objects'][name]['light']['properties']['energy']=want
 if not np.isclose(value['view']['exposure'],target['view']['exposure'],atol=1e-6,rtol=0):raise ValueError('Dynamic exposure differs')
 value['view']['exposure']=target['view']['exposure']
 if not _same(value,target):raise ValueError('Source-local geometry/material/scale/camera/presence or nonallowed scene state changed')
 return True

class AppliedScene:
 """One dynamic contract with explicit restoration; use as a context manager."""
 def __init__(self,scene,context,base_contract,parameters,bindings,baseline,expected,base_session=None):
  self.scene,self.context,self.base_contract=scene,context,base_contract;self.base_session=base_session
  self.parameters,self.bindings,self.baseline,self.expected=parameters,bindings,baseline,expected
  self.root_channels={name:{'location':list(scene.objects[name].location),'rotation_mode':scene.objects[name].rotation_mode,'rotation_quaternion':list(scene.objects[name].rotation_quaternion),'rotation_euler':list(scene.objects[name].rotation_euler),'rotation_axis_angle':list(scene.objects[name].rotation_axis_angle),'scale':list(scene.objects[name].scale)} for side in ('left','right') for name in bindings[side]['roots']}
  self.restored=False;self.contract={'schema':'action4.modular_dynamic_scene.v1','parameters':parameters,'parameters_sha256':env.digest(parameters),'source_context_sha256':context['source_context_sha256'],'actor_bindings':bindings,'base_state_sha256':env.digest(baseline),'expected_state':expected,'expected_state_sha256':env.digest(expected),'canonical_definition':base_contract['canonical_definition'],'estimated':True,'dataset_generation_allowed':False,'training_accepted':False}
  if base_session is not None:self.contract['base_kind']='palmar_wearing'
 def targets(self):
  targets={}
  for i,side in enumerate(('left','right'),1):
   if self.parameters['target_presence'][i-1]:
    names=pixels.KNOWN_TARGET_PARTS[i];targets[i]={'core':self.scene.objects[names['core']],'parts':[self.scene.objects[names['core']],self.scene.objects[names['decal']]]}
  return targets
 def verifier(self,scene):
  if scene!=self.scene or self.restored:raise ValueError('Active dynamic scene identity required')
  verify_sources(self.context)
  if env.digest(self.parameters)!=self.contract['parameters_sha256']:raise ValueError('Bound dynamic parameters changed')
  if env.digest(self.contract['expected_state'])!=self.contract['expected_state_sha256'] or not _same(self.contract['expected_state'],self.expected):raise ValueError('Dynamic contract changed')
  verify_state(_snapshot(scene),self.expected)
  entry={'target_count':sum(self.parameters['target_presence']),'target_presence':self.parameters['target_presence'],'scene_family':self.parameters['scene_family_controls']['family']}
  pixels._validate_targets(scene,entry,self.targets())
  return {'status':'PASS_DYNAMIC_SOURCE_TRANSFORM_CONTRACT_ONLY','dataset_generation_allowed':False,'training_accepted':False}
 def _restore_to_base(self):
  if self.restored:raise ValueError('Dynamic restoration is one-shot')
  for side in ('left','right'):
   for name in self.bindings[side]['roots']:
    o=self.scene.objects[name];row=self.root_channels[name]
    o.location=row['location'];o.rotation_mode=row['rotation_mode'];o.rotation_quaternion=row['rotation_quaternion'];o.rotation_euler=row['rotation_euler'];o.rotation_axis_angle=row['rotation_axis_angle'];o.scale=row['scale']
   for name in self.bindings[side]['names']:
    o=self.scene.objects[name];row=self.baseline['objects'][name];o.hide_render=row['hide_render'];o.pass_index=row['pass_index']
  for name in neutral.ENV_NAMES[:2]:self.scene.objects[name].hide_render=self.baseline['objects'][name]['hide_render']
  lamp=neutral.ENV_NAMES[2];self.scene.objects[lamp].data.energy=self.baseline['objects'][lamp]['light']['properties']['energy']
  self.scene.view_settings.exposure=self.baseline['view']['exposure'];_update()
  verify_state(_snapshot(self.scene),self.baseline)
  if self.base_session is None:bilateral.verify_bilateral_scene(self.scene,self.context['config'],self.context['bilateral_context'],self.base_contract)
  else:self.base_session.verifier(self.scene)
  verify_sources(self.context)
 def restore(self):
  if self.restored:raise ValueError('Dynamic restoration is one-shot')
  if self.base_session is None:
   self._restore_to_base();self.restored=True
   return {'status':'RESTORED_VERIFIED_BILATERAL_BASE','whole_source_scene_restoration_claim':False}
  primary=None;receipt=None;restoration=None
  try:self._restore_to_base()
  except Exception as exc:primary=exc
  if self.base_session is not None:
   try:receipt=self.base_session.restore()
   except Exception as exc:restoration=exc
  if primary is not None or restoration is not None:raise RuntimeError('Dynamic restoration failed; base='+repr(primary)+'; original='+repr(restoration)) from (primary or restoration)
  self.restored=True
  return {'status':'RESTORED_WEARING_THEN_ORIGINAL_BASE' if self.base_session is not None else 'RESTORED_VERIFIED_BILATERAL_BASE','wearing_restoration':receipt,'whole_source_scene_restoration_claim':False}
 def __enter__(self):return self
 def __exit__(self,kind,error,tb):
  try:self.restore()
  except Exception as restoration:raise RuntimeError('Dynamic restoration failed; primary='+repr(error)+'; restoration='+str(restoration)) from error
  return False

def _update():
 import bpy
 bpy.context.view_layer.update()

def create_scene_for_entry(context,parameters):
 """Explicit Blender-only assembly; caller owns approval and failure evidence."""
 if isinstance(parameters,dict) and parameters.get('schema')=='action4.modular_pose_controls.v1':
  from data import action4_modular_pose_controls as pose
  p=pose.validate_pose_controls(parameters,context)
  session=_create_validated_scene(context,p)
  try:
   session.contract['pose_controls_execution']={'provenance':deepcopy(p['provenance']),'controls':deepcopy(p['controls']),'formal_entry_adapter':p['formal_entry_adapter'],'training_accepted':False}
   session.contract['pose_controls_geometry']=pose.verify_actual_pose_geometry(session)
   return session
  except Exception:
   session.restore()
   raise
 if isinstance(parameters,dict) and parameters.get('schema')=='action4.modular_geometric_execution.v1':
  from data import action4_modular_geometric_consumer as geometric
  p=geometric.validate_execution(parameters,context)
  session=_create_validated_scene(context,p)
  try:
   session.contract['geometric_effect_execution']=deepcopy(p['geometric_effect_execution'])
   session.contract['geometric_effect_geometry']=geometric.verify_actual_geometry(session)
   return session
  except Exception:
   session.restore()
   raise
 p=validate_parameters(parameters,context)
 return _create_validated_scene(context,p)

def create_scene_for_geometric_request(context,request_ref,**entry_binding):
 """Explicit diagnostic/formal-entry effect path; bind identity before render."""
 from data import action4_modular_geometric_consumer as geometric
 return create_scene_for_entry(context,geometric.prepare(request_ref,context,**entry_binding))

def _create_validated_scene(context,p):
 """Shared existing mutation/restoration path after independent validation."""
 base_session=None;session=None
 if context.get('base_kind')=='palmar_wearing':
  from data import action4_modular_wearing_base as wearing
  base_session=wearing.create(context['wearing_context']);scene,base_contract=base_session.scene,base_session.contract
 else:
  scene,base_contract=bilateral.create_scene(context['config'],context['bilateral_context'])
  bilateral.verify_bilateral_scene(scene,context['config'],context['bilateral_context'],base_contract)
 try:
  bindings=actor_bindings(scene,context);baseline=_snapshot(scene);expected=expected_state(baseline,bindings,p,context['source_context'])
  session=AppliedScene(scene,context,base_contract,p,bindings,baseline,expected,base_session=base_session)
  for side in ('left','right'):
   for name in bindings[side]['roots']:assembly.set_rigid_basis(scene.objects[name],expected['objects'][name]['basis'])
   for name in bindings[side]['names']:
    row=expected['objects'][name];scene.objects[name].hide_render=row['hide_render'];scene.objects[name].pass_index=row['pass_index']
  for name in neutral.ENV_NAMES[:2]:scene.objects[name].hide_render=expected['objects'][name]['hide_render']
  lamp=neutral.ENV_NAMES[2];scene.objects[lamp].data.energy=expected['objects'][lamp]['light']['properties']['energy']
  scene.view_settings.exposure=expected['view']['exposure'];_update();session.verifier(scene)
  session.contract['actor_collision_diagnostics']=geometry_diagnostics(session)
  session.verifier(scene)
  return session
 except Exception as exc:
  try:
   if session is not None:session.restore()
   elif base_session is not None:base_session.restore()
  except Exception as restoration:raise RuntimeError('Dynamic application failed: '+str(exc)+'; restoration failed: '+str(restoration)) from exc
  raise

def geometry_diagnostics(session):
 """Original actor surface/zone classifiers, selected by actual presence and B."""
 from mathutils.bvhtree import BVHTree
 session.verifier(session.scene);ctx=session.context['bilateral_context'];p=session.parameters;geometries={};sides={};trees={}
 for side in ('left','right'):
  g=ctx['left'] if side=='left' else ctx['right_context']['geometry'];mc=ctx['left_mount_contract'] if side=='left' else ctx['right_context']['mount_contract']
  names={'core':side+'_Action4_core','skin':g['wc']['object_name'],'band':'left_brace_band' if side=='left' else 'brace_band',**{key:row['name'] for key,row in mc['objects'].items()}}
  names={key:name for key,name in names.items() if not session.scene.objects[name].hide_render}
  sides[side]=(names,mc)
  for key,name in names.items():geometries[side+'/'+key]=assembly.world_geometry(session.scene.objects[name])
 def pair(a,b,zone=None):
  V,F=geometries[a];U,G=geometries[b];sep=bilateral.aabb_separation(V,U)
  if sep['strictly_disjoint']:return {**sep,'outside_or_unresolved_pairs':[],'BVH_evaluated':False}
  for name,vertices,faces in [(a,V,F),(b,U,G)]:
   if name not in trees:trees[name]=BVHTree.FromPolygons(vertices.tolist(),faces.tolist(),all_triangles=True,epsilon=0)
  raw=trees[a].overlap(trees[b]);retained,separated=bilateral.filter_separated_bvh_candidates(V,F,U,G,raw,zone)
  return {**sep,**mounts.classify_pairs(V,F,U,G,retained,zone),'BVH_evaluated':True,'BVH_candidate_pair_count':len(raw),'separated_BVH_candidates':separated}
 cross={};internal={}
 if set(sides)=={'left','right'}:cross={a+'__'+b:pair('left/'+a,'right/'+b) for a in sides['left'][0] for b in sides['right'][0]}
 for side,(names,mc) in sides.items():
  keys=sorted(names);rows={};inv=np.linalg.inv(np.asarray(p['W_from_capture'])@np.asarray(p['sides'][side]['brace_B_capture'] if p['sides'][side]['present'] else session.context['source_context']['sides'][side]['base_B_capture']))
  if session.context.get('base_kind')=='palmar_wearing':
   # P installation belongs to the actual mount root, including source float residue.
   inv=np.linalg.inv(np.asarray(session.expected['objects'][session.bindings[side]['roots'][2]]['world']))
  for i,a in enumerate(keys):
   for b in keys[i+1:]:
    zone=mc['joint_candidate_zones'].get(a+'__'+b) or mc['joint_candidate_zones'].get(b+'__'+a)
    if zone is not None:zone={**zone,'module_to_zone':(np.asarray(zone['module_to_zone'])@inv).tolist()}
    rows[a+'__'+b]=pair(side+'/'+a,side+'/'+b,zone)
  internal[side]=rows
 count=sum(len(row['outside_or_unresolved_pairs']) for row in [*cross.values(),*(row for rows in internal.values() for row in rows.values())])
 report={'status':'DYNAMIC_ACTOR_SURFACE_DIAGNOSTICS_ONLY','cross_pair_count':len(cross),'internal_pair_counts':{side:len(rows) for side,rows in internal.items()},'cross_pairs':cross,'internal_pairs':internal,'outside_or_unresolved_surface_pairs':count,'candidate_joint_zones_unchanged':True,'retained_actor_surface_names':{side:list(names.values()) for side,(names,_) in sides.items()},'absent_target_band_skin_context_retained':True,'self_intersection_baselines':{'left':{'polygon_pairs':ctx['left']['row']['self_intersections']['pair_count'],'triangle_pairs':ctx['left']['row']['self_triangle_pair_count']},'right':{'polygon_pairs':ctx['right_context']['Y25_row']['self_intersections']['pair_count'],'triangle_pairs':ctx['right_context']['Y25_row']['self_triangle_pair_count']}},'containment_accepted':False,'environment_collision_accepted':False,'mechanical_acceptance':False,'training_accepted':False}
 if session.context.get('base_kind')=='palmar_wearing':report['absence_policy']=deepcopy(session.context['source_context']['absence_policy'])
 if count:raise ActorCollisionError(report)
 return report
