"""Palmar wearing base: old verified initializer, explicit P, independent cable.

No scene/render on import/prepare. Identity metadata copies sealed JSON;
NumPy is used only for guards, predicted scene transforms and actual checks.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import hashlib,json,re,numpy as np
from data import action4_modular_scene_sampling as sampling
from data import action4_modular_bilateral_scene as bilateral
from data import action4_modular_assembly as assembly
from data import action4_modular_pixels as pixels
from data import action4_modular_render as camera
from data.action4_historical_contract import load_historical_contract
ROOT=Path('/data/zy/vln/a-test')
CONFIG=ROOT/'outputs/action4_modular_wearing_base_prepare_v001/config.json'
env=sampling.env
def check_ref(row):
 if not isinstance(row,dict) or set(row)!={'path','sha256'} or Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or not re.fullmatch('[0-9a-f]{64}',row['sha256']) or bilateral.sha(row['path'])!=row['sha256']:raise ValueError('Wearing source SHA/reference differs')
 return bilateral.bounded(row['path'])

def verify_sources(ctx):
 for name,digest in ctx['source_refs'].items():check_ref({'path':name,'sha256':digest})
 sampling.verify_sources(ctx['ancestor'])
 if env.digest(ctx['source_context'])!=ctx['source_context_sha256']:raise ValueError('Wearing metadata changed')

def frame_plan(phase,source,candidates):
 out={}
 for side in ('left','right'):
  row=phase[side];candidate=next(v for v in candidates['candidates'] if v['candidate']==side+'_palmar')
  A=assembly.rigid(row['Anew_fixed_candidate']);B=assembly.rigid(row['Bnew_fixed_candidate']);P=assembly.rigid(row['P_added_installation_subframe']);old=source['sides'][side]
  if row['Aold']!=old['A_brace_from_CAD'] or candidate['A_candidate']!=row['Anew_fixed_candidate'] or candidate['B_capture_candidate']!=row['Bnew_fixed_candidate'] or candidate['fixed_C']!=old['base_C_capture'] or candidate['fixed_S']!=old['S_skin_relative_to_brace']:raise ValueError('Selected sealed palmar/source frames disagree')
  if max(np.max(abs(A-P@np.asarray(row['Aold']))),np.max(abs(B@P-np.asarray(row['Bold_derived']))),np.max(abs(B@A-np.asarray(candidate['fixed_C']))))>1e-12:raise ValueError('P/A/B/C closure')
  out[side]={'P':deepcopy(row['P_added_installation_subframe']),'Aold':deepcopy(row['Aold']),'Anew':deepcopy(row['Anew_fixed_candidate']),'Bold':deepcopy(row['Bold_derived']),'Bnew':deepcopy(row['Bnew_fixed_candidate']),'C':deepcopy(candidate['fixed_C']),'S':deepcopy(candidate['fixed_S'])}
 return out

def prepare(config=CONFIG):
 c=bilateral.read(config)
 if c['schema']!='action4.palmar_wearing_base.v1' or c['frame_id']!=0 or c['dataset_generation_allowed'] is not False or c['training_accepted'] is not False or c['selected_candidates']!=['left_palmar','right_palmar']:raise ValueError('One diagnostic selected palmar base')
 for row in c['sources'].values():check_ref(row)
 if check_ref(c['sources']['wrapper'])!=Path(__file__).resolve():raise ValueError('Loaded wearing wrapper identity')
 ancestor=sampling.prepare_base(c['sources']['ancestor_scene']['path']);a=ancestor['bilateral_context'];phase=bilateral.read(c['sources']['phase']['path']);candidates=bilateral.read(c['sources']['candidates']['path'])
 frames=frame_plan(phase,ancestor['source_context'],candidates);cc=bilateral.read(c['sources']['cable_contract']['path']);hand=bilateral.read(c['sources']['cable_handoff']['path'])
 if hand['status']!='ONE_CPU_BUILD_EXACT_RELOAD_17PAIR_ZERO_CANDIDATE' or not hand['clean_reload_exact'] or hand['other_component_pairs']!=17 or hand['raw_surface_candidates'] or hand['unresolved_pairs'] or hand['new_whitelist_count'] or not hand['old_sources_assets_unchanged']:raise ValueError('Actual clean-reloaded17-pair zero cable required')
 for row in hand['refs'].values():check_ref(row)
 if cc['saved_asset']!=c['sources']['cable_asset'] or cc['geometry']!=c['sources']['cable_geometry'] or cc['object_name']!=c['new_cable_object'] or cc['parent_in_asset'] is not None or cc['basis_local']!=np.eye(4).tolist() or cc['pass_index']!=0 or env.digest(cc['mesh_state'])!=cc['mesh_state_sha256'] or cc['installation_phase_hypothesis']!=phase['right']:raise ValueError('Exact standalone cable contract')
 if env.digest(cc['mesh_state']['material_states'])!=env.digest(a['right_context']['mount_contract']['objects']['cable']['state']['material_states']):raise ValueError('Original cable material graph required')
 history,normalization=load_historical_contract(c['sources']['historical_contract'],return_receipt=True)
 right_root=a['right_context']['mount_contract']['root_objects']['newbrace']
 if cc['W_from_capture']!=ancestor['source_context']['W_from_capture'] or cc['required_parent_world_from_old_mount_local']!=history['state']['objects'][right_root]['world'] or cc['replace_only_old_cable_object']!=a['right_context']['mount_contract']['objects']['cable']['name']:raise ValueError('Cable original actual parent frame/source replacement identity')
 refs=dict(ancestor['source_context']['frozen_refs'])
 for row in [*c['sources'].values(),{'path':str(bilateral.bounded(config).relative_to(ROOT)),'sha256':bilateral.sha(config)},*hand['refs'].values()]:
  if row['path'] in refs and refs[row['path']]!=row['sha256']:raise ValueError('Wearing source identity conflict')
  refs[row['path']]=row['sha256']
 metadata=deepcopy(ancestor['source_context']);metadata['frozen_refs']=refs;metadata['wearing_frames']=deepcopy(frames)
 for side in ('left','right'):
  metadata['sides'][side]['A_brace_from_CAD']=deepcopy(frames[side]['Anew']);metadata['sides'][side]['base_B_capture']=deepcopy(frames[side]['Bnew']);metadata['sides'][side]['mount_installation_from_new_brace']=deepcopy(frames[side]['P'])
 ctx={'config':c,'ancestor':ancestor,'frames':frames,'source_refs':refs,'source_context':metadata,'source_context_sha256':env.digest(metadata),'K':deepcopy(ancestor['source_context']['K_native']),'canonical_definition':deepcopy(a['right_context']['oldcontract']['canonical_definition']),'render':deepcopy(a['right_config']['render']),'blender_ref':deepcopy(a['right_context']['neutral_config']['sources']['blender']),'historical_contract':history,'historical_normalization':normalization,'cable_contract':cc,'cable_geometry':np.load(check_ref(c['sources']['cable_geometry']),allow_pickle=False),'candidate_geometry':np.load(check_ref(c['sources']['candidate_geometry']),allow_pickle=False)}
 verify_sources(ctx);return ctx

def predict_state(before,bindings,ctx):
 """Only declared basis changes and the one cable addition, before mutation."""
 expected=deepcopy(before);W=np.asarray(ctx['ancestor']['source_context']['W_from_capture']);changed=set()
 for side in ('left','right'):
  f=ctx['frames'][side];root=bindings[side]['roots'][0];wrist=bindings[side]['roots'][1];core=pixels.KNOWN_TARGET_PARTS[1 if side=='left' else 2]['core']
  for name,basis in [(root,W@np.asarray(f['Bnew'])),(wrist,W@np.asarray(f['Bnew'])@np.asarray(f['S'])),(core,np.asarray(f['Anew']))]:expected['objects'][name]['basis']=basis.tolist();changed.add(name)
 memo={}
 def affected(name):
  parent=expected['objects'][name]['parent'];return name in changed or (parent is not None and affected(parent))
 def world(name):
  if name in memo:return memo[name]
  row=expected['objects'][name]
  if not affected(name):value=np.asarray(before['objects'][name]['world'])
  elif row['parent'] is None:value=np.asarray(row['basis'])
  else:value=world(row['parent'])@np.asarray(row['parent_inverse'])@np.asarray(row['basis'])
  memo[name]=value;return value
 for name in list(expected['objects']):expected['objects'][name]['world']=world(name).tolist()
 old_cable=ctx['cable_contract']['replace_only_old_cable_object'];expected['objects'][old_cable]['hide_render']=True
 name=ctx['config']['new_cable_object'];parent=ctx['ancestor']['bilateral_context']['right_context']['mount_contract']['root_objects']['newbrace'];cc=ctx['cable_contract']
 expected['objects'][name]={'type':'MESH','parent':parent,'parent_inverse':np.eye(4).tolist(),'basis':deepcopy(cc['basis_local']),'world':deepcopy(before['objects'][parent]['world']),'scale':[1.,1.,1.],'pass_index':0,'hide_render':False,'hide_viewport':False,'mesh':deepcopy(cc['mesh_state']),'light':None}
 expected['camera_ray_visibility'][name]=True
 for layer in expected['collection_visibility'].values():layer['collections']['objects']=sorted([*layer['collections']['objects'],name])
 return expected

def _channels(o):return {key:list(getattr(o,key)) if key!='rotation_mode' else o.rotation_mode for key in ('location','rotation_mode','rotation_quaternion','rotation_euler','rotation_axis_angle','scale')}
def _put_channels(o,row):
 for key,value in row.items():setattr(o,key,value)
def _owned():
 import bpy
 return {key:set(getattr(bpy.data,key)) for key in ('objects','meshes','materials','images')}
def _cleanup(owned):
 import bpy
 for key in ('objects','meshes','materials','images'):
  for item in owned.get(key,set()):
   if key=='objects':getattr(bpy.data,key).remove(item,do_unlink=True)
   elif item.users==0:getattr(bpy.data,key).remove(item)
   else:raise RuntimeError('Owned cable datablock still used: '+item.name)

class WearingSession:
 def __init__(self,scene,ctx,original,bindings,before,expected):
  self.scene,self.ctx,self.original=scene,ctx,original;self.bindings=bindings;self.restored=False;self.owned={};self.cores={pixels.KNOWN_TARGET_PARTS[i]['core']:_channels(scene.objects[pixels.KNOWN_TARGET_PARTS[i]['core']]) for i in (1,2)}
  parameters={'schema':'action4.wearing_base_operation.v1','target_presence':[True,True],'scene_family_controls':{'family':'wall_rear'},'frames':deepcopy(ctx['frames'])}
  transaction_context={**ctx['ancestor'],'source_context':ctx['source_context'],'source_context_sha256':ctx['source_context_sha256']}
  self.transaction=sampling.AppliedScene(scene,transaction_context,original,parameters,bindings,before,expected)
  self.contract={'schema':'action4.palmar_wearing_scene.contract.v1','canonical_definition':ctx['canonical_definition'],'frames':ctx['frames'],'source_refs':ctx['source_refs'],'source_context_sha256':ctx['source_context_sha256'],'expected_state':expected,'expected_state_sha256':env.digest(expected),'old_base_state_sha256':env.digest(before),'new_cable':{'name':ctx['config']['new_cable_object'],'contract_ref':ctx['config']['sources']['cable_contract']},'mount_actual_world_before':{side:deepcopy(before['objects'][bindings[side]['roots'][2]]['world']) for side in ('left','right')},'dataset_generation_allowed':False,'training_accepted':False,'wearing_accepted':False}
 def targets(self):return self.transaction.targets()
 def verifier(self,scene):
  if self.restored or scene!=self.scene:raise ValueError('Active wearing session required')
  verify_sources(self.ctx);self.transaction.verifier(scene);W=np.asarray(self.ctx['ancestor']['source_context']['W_from_capture']);cache=self.ctx['candidate_geometry'];errors={}
  for side in ('left','right'):
   g=self.ctx['ancestor']['bilateral_context']['left'] if side=='left' else self.ctx['ancestor']['bilateral_context']['right_context']['geometry'];names={'skin':g['wc']['object_name'],'band':'left_brace_band' if side=='left' else 'brace_band','core':side+'_Action4_core'}
   for role,name in names.items():
    V,F=assembly.world_geometry(scene.objects[name]);target=assembly.transform(W,cache[side+'_palmar_'+role+'_capture_vertices'])
    if not np.array_equal(F,cache[side+'_palmar_'+role+'_triangles']) or np.max(abs(V-target))>1e-6:raise ValueError('Actual palmar geometry/cache mismatch '+side+'/'+role)
    errors[side+'/'+role]=float(np.max(abs(V-target)))
   mount=scene.objects[self.bindings[side]['roots'][2]]
   if not np.array_equal(np.asarray(mount.matrix_world),self.contract['mount_actual_world_before'][side]):raise ValueError('Mount actual world must remain exactly premutation')
  name=self.ctx['config']['new_cable_object'];V,F=assembly.world_geometry(scene.objects[name]);c=self.ctx['cable_geometry']
  if not np.array_equal(F,c['cable_triangles']) or np.max(abs(V-c['cable_vertices_world_m']))>1e-6:raise ValueError('New actual cable differs from verified17-pair world cache')
  if env.digest(assembly.mesh_state(scene.objects[name]))!=self.ctx['cable_contract']['mesh_state_sha256']:raise ValueError('New cable mesh/material local source state differs')
  exports={}
  for j,target in self.targets().items():
   value=pixels._canonical_geometry(canonical=self.ctx['canonical_definition'],instance_id=j,object_to_world_m=pixels._evaluated_world_matrix(target['core']),camera_to_world_blender_m=np.asarray(scene.camera.matrix_world).tolist(),K=camera.actual_K(scene),image_size_wh=[3248,2464],near_plane_m=scene.camera.data.clip_start)
   a=self.ctx['ancestor']['bilateral_context'];baseline=a['left']['row']['canonical8'] if j==1 else a['right_context']['Y25_row']['canonical8']
   if value['corner_ids']!=list(range(8)) or value['corner_order']!=baseline['corner_order'] or np.max(abs(np.asarray(value['points_native_px'])-np.asarray(baseline['points_native_px'])))>.003:raise ValueError('Original fixedC/canonical8 reprojection changed')
   exports[str(j)]=value
  return {'status':'PASS_NEW_PALMAR_SOURCE_TRANSFORM_CACHE_CONTRACT_ONLY','geometry_max_component_errors_m':errors,'new_cable_world_cache_max_m':float(np.max(abs(V-c['cable_vertices_world_m']))),'canonical8_by_instance':exports,'dataset_generation_allowed':False}
 def geometry_diagnostics(self):
  a=self.ctx['ancestor']['bilateral_context'];right={**a['right_context'],'mount_contract':deepcopy(a['right_context']['mount_contract'])};right['mount_contract']['objects']['cable']['name']=self.ctx['config']['new_cable_object'];collision_context={**a,'right_context':right};W=np.asarray(self.ctx['ancestor']['source_context']['W_from_capture'])
  p={'W_from_capture':W.tolist(),'sides':{side:{'present':True,'brace_B_capture':(np.linalg.inv(W)@np.asarray(self.contract['mount_actual_world_before'][side])).tolist()} for side in ('left','right')}}
  proxy=SimpleNamespace(scene=self.scene,context={'bilateral_context':collision_context,'source_context':self.ctx['source_context']},parameters=p,verifier=self.verifier)
  report=sampling.geometry_diagnostics(proxy)
  if report['cross_pair_count']!=81 or report['internal_pair_counts']!={'left':36,'right':36} or self.ctx['config']['new_cable_object'] not in report['retained_actor_surface_names']['right'] or self.ctx['cable_contract']['replace_only_old_cable_object'] in report['retained_actor_surface_names']['right']:raise ValueError('Complete153 pairs must include new cable, exclude hidden old cable')
  report['mount_zone_frame']='preserved actual mount world; explicit P installation, no new zones';return report
 def restore(self):
  if self.restored:raise ValueError('One wearing restoration only')
  errors=[];receipt={}
  try:_cleanup(self.owned)
  except Exception as exc:errors.append('new cable cleanup: '+str(exc))
  for name,row in self.cores.items():
   try:_put_channels(self.scene.objects[name],row)
   except Exception as exc:errors.append('core local restore: '+str(exc))
  try:
   receipt['source_transaction']=self.transaction.restore()
   receipt['original_memory_base']=bilateral.verify_bilateral_scene(self.scene,self.ctx['ancestor']['config'],self.ctx['ancestor']['bilateral_context'],self.original)
   receipt['typed_historical_base']=bilateral.verify_bilateral_scene(self.scene,self.ctx['ancestor']['config'],self.ctx['ancestor']['bilateral_context'],self.ctx['historical_contract'])
   receipt['historical_normalization']=self.ctx['historical_normalization'];verify_sources(self.ctx)
  except Exception as exc:errors.append(str(exc))
  if errors:raise RuntimeError('Wearing restoration failed: '+str(errors))
  self.restored=True;return {'status':'RESTORED_ORIGINAL_MEMORY_AND_TYPED_HISTORICAL_BASE','checks':receipt,'dataset_generation_allowed':False}
 def __enter__(self):return self
 def __exit__(self,kind,error,tb):
  try:self.restore()
  except Exception as exc:raise RuntimeError('Wearing restoration failed; primary='+repr(error)+'; restore='+str(exc)) from error
  return False

def create(ctx):
 import bpy
 from mathutils import Matrix
 verify_sources(ctx);scene,original=bilateral.create_scene(ctx['ancestor']['config'],ctx['ancestor']['bilateral_context']);bilateral.verify_bilateral_scene(scene,ctx['ancestor']['config'],ctx['ancestor']['bilateral_context'],original)
 bindings=sampling.actor_bindings(scene,ctx['ancestor']);before=sampling._snapshot(scene);expected=predict_state(before,bindings,ctx);session=WearingSession(scene,ctx,original,bindings,before,expected);ids=_owned()
 try:
  W=np.asarray(ctx['ancestor']['source_context']['W_from_capture'])
  phase_residual={}
  for side in ('left','right'):
   f=ctx['frames'][side];root,wrist,mount=bindings[side]['roots'];core=pixels.KNOWN_TARGET_PARTS[1 if side=='left' else 2]['core']
   phase_residual[side]=float(np.max(abs(W@np.asarray(f['Bnew'])@np.asarray(f['P'])-np.asarray(before['objects'][mount]['world']))))
   if phase_residual[side]>2e-6:raise ValueError('Analytic BnewP vs preserved actual mount world')
   assembly.set_rigid_basis(scene.objects[root],W@np.asarray(f['Bnew']));assembly.set_rigid_basis(scene.objects[wrist],W@np.asarray(f['Bnew'])@np.asarray(f['S']));assembly.set_rigid_basis(scene.objects[core],f['Anew'])
  cc=ctx['cable_contract'];name=cc['object_name']
  if name in bpy.data.objects:raise FileExistsError('Fresh independent cable name required')
  with bpy.data.libraries.load(str(check_ref(ctx['config']['sources']['cable_asset'])),link=False) as (available,loaded):
   if name not in available.objects:raise ValueError('Exact new cable object missing')
   loaded.objects=[name]
  cable=loaded.objects[0];scene.collection.objects.link(cable);assembly.share_exact_materials(cable,cc['mesh_state']['material_states'],[])
  if cable.parent is not None or len(cable.modifiers) or len(cable.constraints) or cable.animation_data is not None or not np.array_equal(cable.matrix_basis,np.eye(4)) or env.digest(assembly.mesh_state(cable))!=cc['mesh_state_sha256']:raise ValueError('Original standalone cable source-local asset state')
  cable.parent=scene.objects[bindings['right']['roots'][2]];cable.matrix_parent_inverse=Matrix.Identity(4);cable.matrix_basis=Matrix.Identity(4);cable.pass_index=0;cable.hide_render=False;cable.hide_viewport=False;cable.visible_camera=True
  scene.objects[cc['replace_only_old_cable_object']].hide_render=True;sampling._update()
  session.owned={key:set(getattr(bpy.data,key))-ids[key] for key in ids};session.contract['analytic_mount_vs_preserved_actual_max_m']=phase_residual
  if len(session.owned['objects'])!=1 or len(session.owned['meshes'])!=1:raise ValueError('Exactly one new independent cable object/mesh')
  session.contract['verification']=session.verifier(scene);session.contract['collision_diagnostics']=session.geometry_diagnostics();return session
 except Exception as exc:
  session.owned={key:set(getattr(bpy.data,key))-ids[key] for key in ids}
  try:session.restore()
  except Exception as restoration:raise RuntimeError('Wearing create failed: '+str(exc)+'; restoration failed: '+str(restoration)) from exc
  raise
