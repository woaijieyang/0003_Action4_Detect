"""Finite source-referenced cloth patches; no Blender/proposal/render on import.

The caller owns scene authority. Existing cloth/material helpers create only
new candidate cloth; original actor meshes, transforms, K and materials stay
unchanged. A candidate effect is not a rendered family/physical-contact pass.
"""
from copy import deepcopy
from pathlib import Path
import json,re,numpy as np
from data import action4_render as base
from data import action4_modular_scene_sampling as sampling
from data import action4_modular_assembly as assembly
from data import action4_modular_bilateral_scene as bilateral
from data import action4_modular_render as neutral
from data import action4_global_environment as env
ROOT=Path('/data/zy/vln/a-test')
CONFIG=ROOT/'configs/data/action4_modular_cloth_candidate_v001.json'
PREFIX='sampling_cloth_'
CONTROL_KEYS={'control_id','kind','parent_side','pose_brace','dimensions_m','fold_lift_m','source'}

class ClothCollisionError(ValueError):
 def __init__(self,report):
  self.report=report;super().__init__('Cloth actor intersections: '+str(report['outside_or_unresolved_surface_pairs']))

def prepare_candidate(config=CONFIG):
 """CPU-only source checks. No download, scene, sampler or mesh build."""
 path=bilateral.bounded(config);c=bilateral.read(path)
 if c['schema']!='action4.modular_cloth_candidate.v1' or c['estimated'] is not True or c['training_accepted'] is not False or c['dataset_generation_allowed'] is not False:raise ValueError('Explicit estimated nonaccepted cloth candidate')
 if set(c['profiles'])!={'table_fastening','table_turn','cloth_lift'}:raise ValueError('Exactly three finite cloth family profiles')
 for row in c['sources'].values():
  if set(row)!={'path','sha256'} or Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or bilateral.sha(row['path'])!=row['sha256']:raise ValueError('Cloth candidate source changed')
 if c['sources']['cloth_module']['path']!=str(Path(__file__).resolve().relative_to(ROOT)):raise ValueError('Canonical cloth module source identity')
 for role,module in [('render_helper',base),('sampling_helper',sampling),('bilateral_helper',bilateral),('assembly_helper',assembly),('environment_helper',env),('neutral_helper',neutral)]:
  if bilateral.bounded(c['sources'][role]['path'])!=Path(module.__file__).resolve():raise ValueError('Loaded cloth dependency identity '+role)
 for name,f in c['profiles'].items():
  dims=np.asarray(f['dimensions_m'],float)
  if dims.shape!=(3,) or not np.isfinite(dims).all() or not .08<=dims[0]<=.30 or not .06<=dims[1]<=.20 or dims[2]!=.0012 or not 0<=f['fold_lift_m']<=.025 or f['parent_side'] not in ('left','right') or f['anchor_mode'] not in ('parent_brace','bilateral_midpoint') or not abs(f['capture_roll_deg'])<=15:raise ValueError('Finite patch dimensions/anchor/lift prior')
 if c['depth_policy']!={'clearance_m':.012,'original_cloth_wave_bound_m':.008,'claim':'estimated behind-actor initial candidate; no physical contact or measured depth'}:raise ValueError('Fixed explicit starting-depth candidate')
 return c,{'path':str(path.relative_to(ROOT)),'sha256':bilateral.sha(path)}

def _B(context,parameters,side):
 row=parameters['sides'][side]
 B=row['brace_B_capture'] if row['present'] else context['source_context']['sides'][side]['base_B_capture']
 return assembly.rigid(B)

def actor_capture_bounds(context,parameters):
 """Read exact cached actors; RIGHT neutral collision skin is replaced by Y25."""
 sampling.verify_sources(context);ctx=context['bilateral_context'];rows={}
 for side,g,path in [('left',ctx['left'],context['config']['sources']['left_mount_geometry']['path']),('right',ctx['right_context']['geometry'],ctx['right_config']['sources']['mount_geometry']['path'])]:
  archive=np.load(bilateral.bounded(path),allow_pickle=False);B=_B(context,parameters,side)
  keys=['core','skin','band',*bilateral.MOUNT_KEYS] if parameters['sides'][side]['present'] else ['skin','band']
  for key in keys:
   if key=='skin':
    V=g['cache']['skin_capture_vertices'] if side=='left' else g['y']['skin_capture_vertices']
    V=assembly.transform(B@np.linalg.inv(g['B']),V)
   else:V=assembly.transform(B,archive[key+'_vertices_m'])
   if not np.isfinite(V).all():raise ValueError('Finite cached actor bounds')
   rows[side+'/'+key]=[V.min(0).tolist(),V.max(0).tolist()]
 return rows

def finite_lift(vertices,dimensions,lift):
 """Limited new-cloth-only near-edge lift; no fingertip pin/skin deformation."""
 V=np.asarray(vertices,float);dims=np.asarray(dimensions,float)
 if V.ndim!=2 or V.shape[1]!=3 or not np.isfinite(V).all() or dims.shape!=(3,) or dims[1]<=0 or not np.isfinite(lift) or not 0<=lift<=.025:raise ValueError('Finite cloth vertices/dimensions/lift')
 out=V.copy();t=np.clip((dims[1]/2-V[:,1])/dims[1],0,1);out[:,2]-=lift*t*t
 return out

def propose(family,context,parameters,config=CONFIG):
 """One deterministic estimated candidate, not a pose search or sampling task."""
 c,ref=prepare_candidate(config);sampling.verify_sources(context)
 if c['sources']['base_scene_config']!=context['source_context']['scene_config_ref']:raise ValueError('Selected base scene identity differs')
 if family not in c['profiles']:raise ValueError('Only three cloth families supported')
 if parameters['W_from_capture']!=context['source_context']['W_from_capture'] or parameters['K_native']!=context['source_context']['K_native']:raise ValueError('Cloth cannot alter W/K')
 f=c['profiles'][family];side=f['parent_side'];bounds=actor_capture_bounds(context,parameters)
 centers={s:_B(context,parameters,s)[:3,3] for s in ('left','right')}
 center=centers[side].copy() if f['anchor_mode']=='parent_brace' else (centers['left']+centers['right'])/2
 center[:2]+=f['anchor_offset_capture_xy_m']
 center[2]=max(row[1][2] for row in bounds.values())+.012+.008+f['fold_lift_m']+f['dimensions_m'][2]/2
 angle=np.deg2rad(f['capture_roll_deg']);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
 pose_capture=np.eye(4);pose_capture[:3,:3]=R;pose_capture[:3,3]=center
 control={'control_id':family+'_patch_v001','kind':'cloth_patch','parent_side':side,'pose_brace':(np.linalg.inv(_B(context,parameters,side))@pose_capture).tolist(),'dimensions_m':f['dimensions_m'],'fold_lift_m':f['fold_lift_m'],'source':'engineering_candidate'}
 return {'schema':'action4.modular_cloth_proposal.v1','family':family,'coarse_occluders':[control],'candidate_config':ref,'source_context_sha256':context['source_context_sha256'],'actor_states_by_side':{name:{'present':parameters['sides'][name]['present'],'brace_B_capture':_B(context,parameters,name).tolist()} for name in ('left','right')},'parent_B_capture':_B(context,parameters,side).tolist(),'W_from_capture':parameters['W_from_capture'],'K_native':parameters['K_native'],'initial_pose_capture':pose_capture.tolist(),'cached_actor_bounds_capture':bounds,'depth_is_estimated':True,'placement_layer':'background_context_only','occlusion_layer_covered':False,'future_foreground_fold_pending':True,'grasp_contact_verified':False,'family_effect_render_verified':False,'dataset_generation_allowed':False,'training_accepted':False}

def validate_proposal(proposal,context,config=CONFIG):
 c,ref=prepare_candidate(config);sampling.verify_sources(context);p=deepcopy(proposal)
 if p['schema']!='action4.modular_cloth_proposal.v1' or p['candidate_config']!=ref or p['source_context_sha256']!=context['source_context_sha256'] or p['family'] not in c['profiles'] or len(p['coarse_occluders'])!=1:raise ValueError('Exact candidate/source-bound cloth proposal')
 row=p['coarse_occluders'][0];f=c['profiles'][p['family']]
 if set(row)!=CONTROL_KEYS or row['kind']!='cloth_patch' or row['source']!='engineering_candidate' or not re.fullmatch('[A-Za-z0-9_]{1,70}',row['control_id']):raise ValueError('Finite cloth control schema/identity')
 for name in ('dimensions_m','fold_lift_m','parent_side'):
  if row[name]!=f[name]:raise ValueError('Frozen finite cloth profile differs '+name)
 if p['W_from_capture']!=context['source_context']['W_from_capture'] or p['K_native']!=context['source_context']['K_native']:raise ValueError('Frozen capture metadata')
 B=assembly.rigid(p['parent_B_capture']);pose=assembly.rigid(row['pose_brace']);target=assembly.rigid(p['initial_pose_capture'])
 if np.max(abs(B@pose-target))>1e-9:raise ValueError('Cloth brace/capture closure')
 expected=propose(p['family'],context,{'sides':p['actor_states_by_side'],'W_from_capture':p['W_from_capture'],'K_native':p['K_native']},config)
 if env.digest(p)!=env.digest(expected):raise ValueError('Cloth proposal differs from frozen finite source-derived candidate')
 if p['family_effect_render_verified'] is not False or p['training_accepted'] is not False or p['dataset_generation_allowed'] is not False:raise ValueError('Cloth candidate cannot self-accept effect')
 return p

def _source_snapshot(scene,excluded=()):
 excluded=set(excluded)
 for o in scene.objects:
  if o.name not in excluded:bilateral._check_static(o)
 state=neutral.new_scene_state(scene,None);state['objects']={name:row for name,row in state['objects'].items() if name not in excluded}
 state['ray_visibility']={o.name:bool(o.visible_camera) for o in scene.objects if o.name not in excluded}
 state['skin_groups']={o.name:bilateral._group_state(o) for o in scene.objects if o.name not in excluded and o.type=='MESH' and len(o.vertex_groups)}
 visibility=env._collection_visibility(scene)
 def prune(value):
  if isinstance(value,dict):return {key:[name for name in item if name not in excluded] if key=='objects' else prune(item) for key,item in value.items()}
  if isinstance(value,list):return [prune(v) for v in value]
  return value
 state['collection_visibility']=prune(visibility)
 return state

def table_context(scene,context):
 """Read original table location; bbox coverage is an upper bound, not a mask."""
 table=scene.objects[neutral.ENV_NAMES[1]];V,_=assembly.world_geometry(table)
 capture=assembly.transform(np.linalg.inv(context['source_context']['W_from_capture']),V)
 K=neutral.actual_K(scene);uv=assembly.project(capture,K);lo,hi=uv.min(0),uv.max(0)
 width,height=scene.render.resolution_x,scene.render.resolution_y
 extent=np.maximum(0,np.minimum(hi,[width-.5,height-.5])-np.maximum(lo,[-.5,-.5]))
 return {'object_name':table.name,'camera_render_enabled':sampling.pixels._camera_render_enabled(scene,table),'actual_world_matrix':np.asarray(table.matrix_world).tolist(),'capture_bounds_m':[capture.min(0).tolist(),capture.max(0).tolist()],'projected_native_bbox_xyxy':[*lo.tolist(),*hi.tolist()],'bbox_inframe_area_fraction_upper_bound':float(np.prod(extent)/(width*height)),'source_manual_back_edge_y_px':context['source_context']['baseline_environment']['table_back_edge_native_y_px'],'visible_pixel_coverage_measured':False,'family_table_semantics_verified':False,'limitation':'Original f0 narrow bottom strip; visible=true does not prove540/960 table background. Finite table pose/back-edge candidate and actual coverage remain pending.'}

def _ids():
 import bpy
 return {name:set(getattr(bpy.data,name)) for name in ('objects','meshes','materials')}

def _remove_created(before,owned=None):
 import bpy
 for name in ('objects','meshes','materials'):
  collection=getattr(bpy.data,name)
  for item in (set(collection)-before[name] if owned is None else owned[name]):
   if name=='objects':collection.remove(item,do_unlink=True)
   elif item.users==0:collection.remove(item)
   else:raise RuntimeError('New cloth datablock still has users: '+item.name)

class ClothEffect:
 """Context manager: remove cloth before the outer dynamic actor restoration."""
 def __init__(self,scene,context,proposal,before,state,verify_existing,config):
  self.scene,self.context,self.proposal,self.before,self.source_state=scene,context,proposal,before,state
  self.verify_existing,self.config,self.contract=verify_existing,config,None;self.removed=False;self.owned=None
 def verifier(self,scene):
  if scene!=self.scene or self.removed:raise ValueError('Active cloth scene identity')
  validate_proposal(self.proposal,self.context,self.config);o=scene.objects[self.contract['object_name']]
  if env.digest(_source_snapshot(scene,[o.name]))!=env.digest(self.source_state):raise ValueError('Existing source actors/camera/environment changed during cloth effect')
  expected=self.contract
  if o.pass_index!=0 or not sampling.pixels._camera_render_enabled(scene,o) or o.parent.name!=expected['parent_name'] or tuple(o.scale)!=(1.,1.,1.) or not np.array_equal(o.matrix_parent_inverse,np.eye(4)) or np.max(abs(np.asarray(o.matrix_basis)-expected['pose_brace']))>1e-6 or np.max(abs(np.asarray(o.matrix_world)-expected['expected_world']))>2e-6:raise ValueError('Actual added cloth pass/visibility/rigid brace anchor differs')
  if len(o.constraints) or o.animation_data is not None or len(o.modifiers)!=1 or o.modifiers[0].type!='SOLIDIFY' or abs(o.modifiers[0].thickness-expected['dimensions_m'][2])>1e-9 or o.modifiers[0].offset!=0 or env.digest(assembly.mesh_state(o))!=expected['mesh_state_sha256']:raise ValueError('Added cloth geometry/material/thickness changed')
  return {'status':'PASS_ADDED_CLOTH_SOURCE_CONTRACT_ONLY','family_effect_render_verified':False,'training_accepted':False}
 def geometry_diagnostics(self):
  from mathutils.bvhtree import BVHTree
  self.verifier(self.scene);cloth=self.scene.objects[self.contract['object_name']];V,F=assembly.world_geometry(cloth);tree=None;rows={}
  ctx=self.context['bilateral_context']
  for side,g,mc in [('left',ctx['left'],ctx['left_mount_contract']),('right',ctx['right_context']['geometry'],ctx['right_context']['mount_contract'])]:
   names={'core':side+'_Action4_core','skin':g['wc']['object_name'],'band':'left_brace_band' if side=='left' else 'brace_band',**{k:r['name'] for k,r in mc['objects'].items()}}
   for role,name in names.items():
    o=self.scene.objects[name]
    if o.hide_render:continue
    U,G=assembly.world_geometry(o);sep=bilateral.aabb_separation(V,U)
    if sep['strictly_disjoint']:rows[side+'/'+role]={**sep,'outside_or_unresolved_pairs':[],'BVH_evaluated':False};continue
    if tree is None:tree=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True,epsilon=0)
    other=BVHTree.FromPolygons(U.tolist(),G.tolist(),all_triangles=True,epsilon=0);raw=tree.overlap(other)
    retained,separated=bilateral.filter_separated_bvh_candidates(V,F,U,G,raw)
    rows[side+'/'+role]={**sep,'BVH_evaluated':True,'BVH_candidate_pair_count':len(raw),'separated_BVH_candidates':separated,'outside_or_unresolved_pairs':[list(p) for p in retained]}
  count=sum(len(row['outside_or_unresolved_pairs']) for row in rows.values())
  report={'cloth_object':cloth.name,'pairs':rows,'outside_or_unresolved_surface_pairs':count,'joint_zone_exemptions':False,'containment_accepted':False,'grasp_contact_verified':False,'family_effect_render_verified':False,'training_accepted':False}
  if count:raise ClothCollisionError(report)
  return report
 def remove(self):
  if self.removed:raise ValueError('Cloth removal is one-shot')
  _remove_created(self.before,self.owned);_update()
  if env.digest(_source_snapshot(self.scene))!=env.digest(self.source_state):raise ValueError('Source scene changed after cloth removal')
  self.verify_existing(self.scene);sampling.verify_sources(self.context);self.removed=True
  return {'status':'REMOVED_CLOTH_SOURCE_VERIFIED','whole_scene_inventory_restoration_claim':False}
 def __enter__(self):return self
 def __exit__(self,kind,error,tb):
  try:self.remove()
  except Exception as restoration:raise RuntimeError('Cloth removal failed; primary='+repr(error)+'; restoration='+str(restoration)) from error
  return False

def _update():
 import bpy
 bpy.context.view_layer.update()

def materialize(scene,context,proposal,*,verify_existing,config=CONFIG):
 """Explicit Blender only. Upper layer must authorize and record the attempt."""
 import bpy
 from mathutils import Matrix
 p=validate_proposal(proposal,context,config)
 if bpy.context.scene!=scene or not callable(verify_existing):raise ValueError('Active supplied scene and actual existing-scene verifier required')
 verify_existing(scene)
 for side,actor in p['actor_states_by_side'].items():
  root=scene.objects['left_brace_module_root' if side=='left' else 'brace_module_root']
  if np.max(abs(np.asarray(root.matrix_world)-np.asarray(p['W_from_capture'])@np.asarray(actor['brace_B_capture'])))>2e-6 or sampling.pixels._camera_render_enabled(scene,scene.objects[side+'_Action4_core']) is not actor['present']:raise ValueError('Actual source actor frame/presence differs from cloth proposal')
 row=p['coarse_occluders'][0];name=PREFIX+row['control_id']
 if any(name in getattr(bpy.data,key) or name+'_mesh' in getattr(bpy.data,key) for key in ('objects','meshes')):raise FileExistsError('Fresh cloth names; no overwrite/retry')
 before=_ids();state=_source_snapshot(scene);effect=ClothEffect(scene,context,p,before,state,verify_existing,config)
 try:
  material=base._denim_material();obj=base._cloth(name,[0,0,0],row['dimensions_m'][:2],material,phase=0)
  vertices=np.array([v.co[:] for v in obj.data.vertices]);deformed=finite_lift(vertices,row['dimensions_m'],row['fold_lift_m'])
  for vertex,position in zip(obj.data.vertices,deformed):vertex.co=position
  obj.data.update();solid=obj.modifiers.new('candidate_cloth_thickness','SOLIDIFY');solid.thickness=row['dimensions_m'][2];solid.offset=0
  parent=scene.objects['left_brace_module_root' if row['parent_side']=='left' else 'brace_module_root']
  if np.max(abs(np.asarray(parent.matrix_world)-np.asarray(p['W_from_capture'])@np.asarray(p['parent_B_capture'])))>2e-6:raise ValueError('Actual selected brace root differs from proposal')
  obj.parent=parent;obj.matrix_parent_inverse=Matrix.Identity(4);assembly.set_rigid_basis(obj,row['pose_brace']);obj.pass_index=0;obj['geometry_role']='estimated_finite_cloth_candidate';obj['dataset_generation_allowed']=False;obj.hide_render=False;obj.visible_camera=True;_update()
  effect.owned={key:set(getattr(bpy.data,key))-before[key] for key in before}
  if {key:len(ids) for key,ids in effect.owned.items()}!={'objects':1,'meshes':1,'materials':1}:raise ValueError('Only one new cloth object/mesh/material may be created')
  effect.contract={'schema':'action4.modular_added_cloth.contract.v1','object_name':obj.name,'parent_name':parent.name,'pose_brace':row['pose_brace'],'expected_world':(np.asarray(parent.matrix_world)@np.asarray(row['pose_brace'])).tolist(),'dimensions_m':row['dimensions_m'],'fold_lift_m':row['fold_lift_m'],'mesh_state_sha256':env.digest(assembly.mesh_state(obj)),'original_source_state_sha256':env.digest(state),'proposal':p,'helper':'action4_render._cloth20x24 + existing procedural denim; bounded new-cloth-only near-edge lift','placement_layer':'background_context_only','occlusion_layer_covered':False,'future_foreground_fold_pending':True,'actual_table_context':table_context(scene,context),'grasp_contact_verified':False,'family_effect_render_verified':False,'dataset_generation_allowed':False,'training_accepted':False}
  effect.verifier(scene);effect.contract['actor_collision_diagnostics']=effect.geometry_diagnostics()
  return effect
 except Exception as exc:
  try:effect.remove()
  except Exception as restoration:raise RuntimeError('Cloth materialization failed: '+str(exc)+'; cleanup failed: '+str(restoration)) from exc
  raise
