"""Single f0 diagnostic: cached Y25 skin, fixed six-piece mount, v003 light.

Import and prepare use only CPU/cache. Scene creation and rendering are explicit,
review-bound future operations. Frozen libraries and the old renderer stay intact.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,traceback
import numpy as np
from data import action4_modular_render as neutral
from data import action4_mount_modules as mount
from data import action4_global_render as render_helpers
from data import action4_global_environment as env
from data import action4_modular_assembly as assembly
from data import action4_render as base
from data import action4_synthesis as synth
from data import action4_amodal as amodal
from data import action4_canonical_export as canonical
from data.action4_devices import ALLOWED_GPU_UUIDS,worker_environment
ROOT=Path('/data/zy/vln/a-test')
OWN=ROOT/'outputs/action4_modular_composed_prepare_v002'
CONFIG=ROOT/'configs/data/action4_modular_composed_preview_v002.json'
APPROVAL=OWN/'root_run_approval.json'
STATUS='COMPOSED_RIGHT_F0_Y25_MOUNT_ESTIMATED_NOT_TRAINING'
TARGET_NAMES=neutral.TARGET_NAMES
MOUNT_KEYS=('quick_release','usb_connector','white_module','ball','link','cable')
bounded,sha,read=neutral.bounded,neutral.sha,neutral.read
actual_K,targets=neutral.actual_K,neutral.targets
transform=assembly.transform

def write(p,value):
 p=bounded(p)
 if not p.is_relative_to(OWN):raise ValueError('Assigned composed output only')
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def frozen_sources(c):
 refs=dict(c['sources'])
 # Check the consumed old neutral inputs too, before and after rendering.
 refs.update({'neutral/'+k:v for k,v in read(c['sources']['neutral_config']['path'])['sources'].items()})
 for name,r in refs.items():
  if set(r)!={'path','sha256'} or sha(r['path'])!=r['sha256']:raise ValueError('Frozen composed source changed: '+name)
 registry=read(c['sources']['registry_v003']['path'])
 for component in registry['components']:
  for p,h in component['files'].items():
   if sha(p)!=h:raise ValueError('Frozen registry input changed '+p)
def geometry_inputs(c,rc):
 wc=read(c['sources']['wrist_contract']['path']);mod=np.load(bounded(c['sources']['wrist_geometry']['path']),allow_pickle=False)
 y=np.load(bounded(c['sources']['y25_geometry']['path']),allow_pickle=False)
 full=np.load(bounded(c['sources']['source_neutral_geometry']['path']),allow_pickle=False)
 ids,tids=mod['source_vertex_ids'],mod['source_triangle_ids'];faces=mod['faces']
 if len(full['vertices_world_m'])!=60270 or len(full['triangles'])!=120536 or len(ids)!=8663 or faces.shape!=(17280,3) or len(tids)!=17280:raise ValueError('Original full/module topology counts')
 if not np.array_equal(full['vertices_world_m'][ids],mod['source_vertices_m']) or not np.array_equal(full['triangles'][tids],ids[faces]) or not np.array_equal(full['triangle_polygon_ids'][tids],mod['source_polygon_ids']) or not np.array_equal(full['triangle_source_regions'][tids],mod['source_regions']):raise ValueError('Frozen source vertex/triangle/region mapping')
 if not np.array_equal(y['skin_faces'],faces) or y['source_body_vertices'].shape!=(8663,3) or y['source_loop_normals'].shape!=(51840,3) or mod['source_loop_ids'].shape!=(51840,):raise ValueError('Derived TRI mesh and51840 actual source-loop normals required')
 for name in wc['uv_layers']:
  if not isinstance(name,str):raise ValueError('Frozen named UV')
 for i in range(len(wc['uv_layers'])):
  if mod['uv_'+str(i)].shape!=(51840,2) or not np.isfinite(mod['uv_'+str(i)]).all():raise ValueError('51840 TRI-loop UV correspondence')
 if not all(np.isfinite(y[k]).all() for k in y.files):raise ValueError('Nonfinite actual Y25 cache')
 if np.max(abs(np.linalg.norm(y['source_loop_normals'],axis=1)-1))>1e-5:raise ValueError('Actual source loop unit normals')
 A=np.asarray(rc['A_shared']);C=np.asarray(rc['C_per_frame']['0']);B=C@np.linalg.inv(A);S=np.asarray(rc['S_skin_relative_to_brace']);W=np.asarray(rc['W_from_capture_engineering_embedding']);Fi=np.linalg.inv(wc['rigid_F_new_body'])
 expected=transform(B@S@Fi,y['source_body_vertices']);error=float(abs(expected-y['skin_capture_vertices']).max())
 if error>1e-12:raise ValueError('Y25 source/capture exact coordinate chain')
 return {'wc':wc,'mod':mod,'y':y,'A':A,'B':B,'C':C,'S':S,'W':W,'Fi':Fi,'capture_error_m':error}
def prepare(config=CONFIG,require_contact=True):
 if Path.cwd().resolve()!=ROOT:raise ValueError('Physical project root')
 c=read(config)
 if set(c)!={'schema','status','dataset_generation_allowed','frame_id','instance_id','threads','resolution_wh','sources','render','environment','review_crop_xyxy','pending','output_scope'} or c['schema']!='action4.modular_composed_right_f0.v1' or c['status']!='estimated_diagnostic_pending_RGB' or c['dataset_generation_allowed'] is not False:raise ValueError('Explicit composed diagnostic schema')
 if (c['frame_id'],c['instance_id'],c['threads'],c['resolution_wh'])!=(0,2,2,[3248,2464]):raise ValueError('One native rightf0 CPU2')
 if c['output_scope']!='outputs/action4_modular_composed_prepare_v002/runs':raise ValueError('Composed output scope')
 frozen_sources(c)
 if bounded(c['sources']['composed_module']['path'])!=Path(__file__).resolve():raise ValueError('Loaded composed controller binding')
 nc,rel,rc,oldcfg,oldcontract,oldregistry=neutral.prepare(c['sources']['neutral_config']['path'])
 for k in ['render','environment','review_crop_xyxy']:
  if c[k]!=nc[k]:raise ValueError('Exact photometry_v003 setting reuse: '+k)
 reg=read(c['sources']['registry_v003']['path']);registry_files={p:h for component in reg['components'] for p,h in component['files'].items()}
 if len(registry_files)!=18:raise ValueError('Frozenv003 exact18')
 for p,h in registry_files.items():
  if sha(p)!=h:raise ValueError('Frozen18 changed '+p)
 yresult=read(c['sources']['y25_result']['path']);row=read(c['sources']['y25_row']['path']);review=read(c['sources']['y25_review']['path'])
 if yresult['status']!='ACTUAL_ONE_Y25_DIAGNOSTIC_COMPLETE_NOT_WEARING_ACCEPTED' or yresult['actual_mesh_extractions']!=1 or yresult['rows']!=[row] or row['pose']!={'right_hand':[0,25,0]} or yresult['source_files_unchanged'] is not True or yresult['restoration']['status']!='RESTORED_ORIGINAL_BONES_FOREARM_CORE' or yresult['restoration']['errors']:raise ValueError('Verified original ONE Y25 only')
 if row['baseline_stable_polygon_pair_count']!=95 or row['baseline_self_triangle_pair_count']!=163 or row['new_stable_polygon_pairs'] or row['new_self_triangle_pairs'] or row['core_skin_pairs'] or row['band_skin_pairs']:raise ValueError('Retain95/163 baseline; no newY25 core/band/self hits')
 if review['status']!='PASS' or review['actual_result_sha256']!=c['sources']['y25_result']['sha256'] or review['actual_npz_sha256']!=c['sources']['y25_geometry']['sha256']:raise ValueError('Independent Y25 review binding')
 mc=read(c['sources']['mount_contract']['path']);receipt=read(c['sources']['mount_receipt']['path']);diagnostic=read(c['sources']['mount_collision']['path'])
 if receipt['status']!='CPU_MODULE_BUILD_RELOAD_DIAGNOSTIC_ONLY' or receipt['reloaded_geometry_exact'] is not True or receipt['all_fixed_source_bytes_unchanged'] is not True:raise ValueError('Actual mount build/reload provenance')
 for key,sourcekey in [('asset','mount_asset'),('contract','mount_contract'),('collisions','mount_collision'),('geometry','mount_geometry')]:
  if receipt[key]!=c['sources'][sourcekey]:raise ValueError('Mount actual receipt binding')
 if set(mc['objects'])!=set(MOUNT_KEYS) or len(diagnostic['pairs'])!=36 or any(v['outside_or_unresolved_pairs'] for v in diagnostic['pairs'].values()):raise ValueError('Complete unresolved-free old neutral mount context')
 if not np.array_equal(mc['routes']['A_new_brace_from_CAD'],rc['A_shared']) or not np.array_equal(mc['collision_skin_context']['S_skin_relative_to_brace'],rc['S_skin_relative_to_brace']):raise ValueError('Original A/S mount context differs')
 if mc['objects']['ball']['scale']!=[0.006000000052154064]*3:raise ValueError('Original fixed ball .006 childscale')
 g=geometry_inputs(c,rc)
 if require_contact:
  contact=read(c['sources']['mount_y25_contact']['path'])
  if contact['status']!='PASS_CACHE_MOUNT_VS_Y25_NO_SURFACE_HITS' or contact['checked_pairs']!=6 or contact['unresolved_surface_pairs']!=0 or contact['scene_loaded'] or contact['pose_evaluations']:raise ValueError('New Y25 contact cache gate')
  for key in ['y25_geometry','mount_geometry','mount_contract','neutral_config']:
   if contact['sources'][key]!=c['sources'][key]:raise ValueError('Y25 contact source drift')
 return c,{'neutral_config':nc,'rel':rel,'rc':rc,'oldcfg':oldcfg,'oldcontract':oldcontract,'registry':{'path':c['sources']['registry_v003']['path'],'verified_files':18},'mount_contract':mc,'geometry':g,'Y25_row':row}
def cache_contacts(config=CONFIG):
 """Only cached arrays + existing mathutils; never import bpy/load scene/pose."""
 from mathutils.bvhtree import BVHTree
 c,ctx=prepare(config,require_contact=False);g=ctx['geometry'];d=np.load(bounded(c['sources']['mount_geometry']['path']),allow_pickle=False)
 skin=transform(np.linalg.inv(g['B']),g['y']['skin_capture_vertices']);faces=g['y']['skin_faces'];tree=BVHTree.FromPolygons(skin.tolist(),faces.tolist(),all_triangles=True,epsilon=0)
 rows={}
 for key in MOUNT_KEYS:
  V,F=d[key+'_vertices_m'],d[key+'_triangles'];part=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True,epsilon=0);pairs=part.overlap(tree)
  zone=ctx['mount_contract']['joint_candidate_zones'].get(key+'__skin') or ctx['mount_contract']['joint_candidate_zones'].get('skin__'+key)
  rows[key+'__Y25skin']=mount.classify_pairs(V,F,skin,faces,pairs,zone)
 unresolved=sum(len(x['outside_or_unresolved_pairs']) for x in rows.values());hits=sum(x['surface_pairs'] for x in rows.values())
 result={'status':'PASS_CACHE_MOUNT_VS_Y25_NO_SURFACE_HITS' if hits==0 else 'BLOCKED_NEW_MOUNT_Y25_CONTACT_REQUIRES_REVIEW','sources':{k:c['sources'][k] for k in ['y25_geometry','mount_geometry','mount_contract','neutral_config']},'checked_pairs':6,'pairs':rows,'surface_pairs':hits,'unresolved_surface_pairs':unresolved,'design_zones_unchanged':True,'coordinate_frame':'new34brace; inv(B_f0) @ actual Y25 capture vertices','scene_loaded':False,'pose_evaluations':0,'mesh_extractions':0,'builds':0,'GPU':False,'wearing_accepted':False,'containment_proven':False}
 write(OWN/'mount_Y25_cache_contacts.json',result)
 return result

def create_scene(c,ctx):
 """Verify frozen neutral scene before replacing only its in-memory skin data."""
 import bpy
 from mathutils import Matrix
 nc,rel,rc,oldcfg,oldcontract=(ctx[k] for k in ['neutral_config','rel','rc','oldcfg','oldcontract'])
 scene,neutral_contract=neutral.create_scene(nc,rel,rc,oldcfg,oldcontract)
 # The old neutral verifier was used above. Its assumptions cease at this line.
 g=ctx['geometry'];wc,mod,y=g['wc'],g['mod'],g['y'];skin=scene.objects[wc['object_name']];original_mesh=skin.data
 original_mesh_state=assembly.mesh_state(skin)
 mesh=bpy.data.meshes.new('composed_Y25_source_triangles')
 mesh.from_pydata(y['source_body_vertices'].tolist(),[],mod['faces'].tolist());mesh.update()
 if len(mesh.loops)!=51840 or any(len(poly.vertices)!=3 for poly in mesh.polygons):raise ValueError('New mesh must have17280tri/51840loops; no quad-loop injection')
 mesh.use_auto_texspace=False;mesh.texspace_location=wc['source_texture_space']['location'];mesh.texspace_size=wc['source_texture_space']['size']
 for material in original_mesh.materials:mesh.materials.append(material)
 for poly in mesh.polygons:poly.use_smooth=True
 mesh.use_auto_smooth=True;mesh.auto_smooth_angle=np.pi
 for i,name in enumerate(wc['uv_layers']):mesh.uv_layers.new(name=name).data.foreach_set('uv',mod['uv_'+str(i)].ravel())
 mesh.normals_split_custom_set(y['source_loop_normals'].tolist());skin.data=mesh;assembly.set_rigid_basis(skin,g['Fi']);skin.pass_index=0
 skin['geometry_role']='derived_Y25_source_positions_normals; frozen REST library unchanged';skin['dataset_generation_allowed']=False
 bpy.context.view_layer.update()
 # Preserve the original mesh datablock too; only the instance pointer is new.
 if original_mesh==skin.data:raise ValueError('Derived skin cannot overwrite original REST mesh')
 with bpy.data.libraries.load(str(bounded(c['sources']['mount_asset']['path'])),link=False) as (available,loaded):
  if available.collections!=['right_mount_components']:raise ValueError('Exact independent mount collection')
  loaded.collections=available.collections;loaded.texts=available.texts
 collection=loaded.collections[0];scene.collection.children.link(collection);bpy.context.view_layer.update();mc=ctx['mount_contract']
 expected_contract=dict(mc);expected_contract.pop('saved_asset')
 if env.digest(json.loads(bpy.data.texts[mount.TEXT].as_string()))!=env.digest(expected_contract):raise ValueError('Mount embedded/source contract differs')
 expected_names=set(mc['root_objects'].values())|{v['name'] for v in mc['objects'].values()}
 if set(o.name for o in collection.all_objects)!=expected_names:raise ValueError('Mount collection includes unexpected dependency')
 for key,row in mc['objects'].items():
  o=scene.objects[row['name']]
  if env.digest(assembly.mesh_state(o))!=env.digest(row['state']) or not np.array_equal(o.matrix_world,row['world_matrix']) or not np.array_equal(o.matrix_basis,row['basis']) or not np.array_equal(o.matrix_parent_inverse,row['parent_inverse']) or (o.parent.name if o.parent else None)!=row['parent']:raise ValueError('Imported source-local mount state differs: '+key)
 root=scene.objects[mc['root_objects']['newbrace']]
 if root.parent or not np.array_equal(root.matrix_basis,np.eye(4)):raise ValueError('Original newbrace root must be identity')
 for key,M in [('oldbrace_subframe',mc['routes']['old_brace_to_new_brace']),('CAD_subframe',mc['routes']['A_new_brace_from_CAD'])]:
  child=scene.objects[mc['root_objects'][key]]
  if child.parent!=root or not np.allclose(child.matrix_basis,M,rtol=0,atol=2e-6) or not np.array_equal(child.matrix_parent_inverse,np.eye(4)):raise ValueError('Original mount subroot A/sourcebrace differs')
 assembly.set_rigid_basis(root,g['W']@g['B']);bpy.context.view_layer.update()
 scene['modular_preview_status']=STATUS;scene['dataset_generation_allowed']=False
 state=neutral.new_scene_state(scene,None)
 contract={'schema':'action4.modular_composed_scene.v1','status':STATUS,'dataset_generation_allowed':False,'source_identities':c['sources'],'state':state,'state_sha256':env.digest(state),'neutral_source_verified_before_derived_changes':True,'neutral_scene_state_sha256':neutral_contract['state_sha256'],'derived_skin':{'object':skin.name,'actual_pose':{'right_hand':[0,25,0]},'positions_frame':'source_body; world=W B S inv(F) positions','triangles':17280,'loops':51840,'frozen_UV_mapping':True,'material_and_source_texture_space_preserved':True,'REST_library_changed':False,'old_neutral_mesh_state_sha256':env.digest(original_mesh_state),'old_neutral_mesh_datablock_preserved_in_memory':True,'source_capture_error_m':g['capture_error_m']},'mount':{'components':list(MOUNT_KEYS),'newbrace_world':(g['W']@g['B']).tolist(),'A_child_root_preserved':True,'ball_source_childscale_preserved':True,'cached_Y25_contact':c['sources']['mount_y25_contact'],'mechanical_acceptance':False},'estimated_environment':c['environment'],'target_parts':TARGET_NAMES,'pending':c['pending'],'wearing_alignment_accepted':False,'training_accepted':False,'whole_source_scene_restoration_claim':False}
 verify_composed_scene(scene,c,ctx,contract)
 return scene,contract

def _verify_source_normal_roundtrip(actual,target):
 """Inherited wrist0.002rad custom-normal storage agreement, not bitwise RGB/fold acceptance."""
 actual=np.asarray(actual,dtype=float);target=np.asarray(target,dtype=float)
 if actual.shape!=(51840,3) or target.shape!=actual.shape:raise ValueError('Exact TRI-loop normal correspondence shape required')
 if not np.isfinite(actual).all() or not np.isfinite(target).all():raise ValueError('Nonfinite source normal storage roundtrip')
 length_actual=np.linalg.norm(actual,axis=1);length_target=np.linalg.norm(target,axis=1)
 if np.any(length_actual<=0) or np.any(length_target<=0):raise ValueError('Zero source normal storage roundtrip')
 if max(float(abs(length_actual-1).max()),float(abs(length_target-1).max()))>1e-5:raise ValueError('Nonunit source normal storage roundtrip')
 a=actual/length_actual[:,None];b=target/length_target[:,None]
 angles=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b));maximum=float(angles.max())
 # Same0.002rad direction tolerance as frozen action4_wrist_modules._verify_mesh.
 if maximum>.002:raise ValueError('Source normal storage roundtrip exceeds original0.002rad angular tolerance')
 return {'maximum_source_normal_roundtrip_angle_rad':maximum,'storage_angular_tolerance_rad':.002,'bitwise_normal_or_RGB_equality_claim':False,'physiological_or_fold_acceptance':False}

def verify_composed_scene(scene,c,ctx,contract):
 """Own composed contract; never reuse the now-inapplicable neutral validator."""
 g=ctx['geometry'];mc=ctx['mount_contract'];oldcontract=ctx['oldcontract']
 expected=set(oldcontract['object_names'])|set(neutral.ENV_NAMES)|set(mc['root_objects'].values())|{v['name'] for v in mc['objects'].values()}
 if set(scene.objects.keys())!=expected:raise ValueError('Exact single-right composed inventory')
 _,world,_=ctx['rel'].expected_world(ctx['rc'],oldcontract,0)
 for name,row in oldcontract['objects'].items():
  o=scene.objects[name]
  if (o.parent.name if o.parent else None)!=row['parent'] or not np.array_equal(o.matrix_parent_inverse,np.eye(4)) or tuple(o.scale)!=(1.,1.,1.) or len(o.modifiers) or len(o.constraints) or o.animation_data is not None:raise ValueError('Static module hierarchy changed: '+name)
  expected_world=g['W']@g['B']@g['S']@g['Fi'] if name==g['wc']['object_name'] else world[name]
  if np.max(abs(np.array(o.matrix_world)-expected_world))>2e-6:raise ValueError('Original hardware/derived skin frame differs: '+name)
  if o.type=='MESH' and name!=g['wc']['object_name'] and env.digest(assembly.mesh_state(o))!=env.digest(row['mesh_state']):raise ValueError('Original frozen hardware mesh changed: '+name)
 skin=scene.objects[g['wc']['object_name']];mesh=skin.data
 if len(mesh.vertices)!=8663 or len(mesh.polygons)!=17280 or len(mesh.loops)!=51840 or not np.array_equal(np.array([tuple(poly.vertices) for poly in mesh.polygons]),g['mod']['faces']):raise ValueError('Actual derived TRI topology mismatch')
 V=np.array([v.co[:] for v in mesh.vertices]);actual_world=transform(skin.matrix_world,V);expected_world=transform(g['W'],g['y']['skin_capture_vertices'])
 if np.max(abs(actual_world-expected_world))>1e-6:raise ValueError('Actual composed skin vs saved Y25 closed coordinate chain')
 if not np.array_equal(V,g['y']['source_body_vertices']):raise ValueError('Actual source Y25 float32 vertex equality')
 for i,layer in enumerate(mesh.uv_layers):
  if layer.name!=g['wc']['uv_layers'][i] or not np.array_equal(np.array([d.uv[:] for d in layer.data]),g['mod']['uv_'+str(i)]):raise ValueError('Actual TRI-loop frozen UV differs')
 if len(mesh.uv_layers)!=len(g['wc']['uv_layers']) or mesh.use_auto_texspace or not np.array_equal(mesh.texspace_location,g['wc']['source_texture_space']['location']) or not np.array_equal(mesh.texspace_size,g['wc']['source_texture_space']['size']) or env.digest(env._material_state(mesh.materials[0]))!=env.digest(g['wc']['material_state']):raise ValueError('Source texture space/material altered')
 mesh.calc_normals_split();N=np.array([l.normal[:] for l in mesh.loops])
 _verify_source_normal_roundtrip(N,g['y']['source_loop_normals'])
 d=np.load(bounded(c['sources']['mount_geometry']['path']),allow_pickle=False)
 for key,row in mc['objects'].items():
  o=scene.objects[row['name']]
  if len(o.modifiers) or len(o.constraints) or o.animation_data is not None or (o.parent.name if o.parent else None)!=row['parent'] or not np.array_equal(o.matrix_basis,row['basis']) or not np.array_equal(o.matrix_parent_inverse,row['parent_inverse']) or not np.array_equal(o.scale,row['scale']):raise ValueError('Original mount local state/ball scale differs')
  if env.digest(assembly.mesh_state(o))!=env.digest(row['state']) or np.max(abs(np.array(o.matrix_world)-g['W']@g['B']@row['world_matrix']))>2e-6:raise ValueError('Imported mount state/world frame differs')
  V=np.array([v.co[:] for v in o.data.vertices]);o.data.calc_loop_triangles();F=np.array([t.vertices[:] for t in o.data.loop_triangles])
  if not np.array_equal(F,d[key+'_triangles']) or np.max(abs(transform(o.matrix_world,V)-transform(g['W']@g['B'],d[key+'_vertices_m'])))>1e-6:raise ValueError('Mount actual/cache closed coordinate chain: '+key)
 if not np.allclose(actual_K(scene),ctx['rc']['K_native'],rtol=0,atol=1e-6):raise ValueError('Original native K')
 targets(scene)
 if env.digest(neutral.new_scene_state(scene,None))!=contract['state_sha256']:raise ValueError('Composed skin/mount/environment scene state changed')
 return True

def output_path(value,fresh=True):
 p=bounded(value)
 if not p.is_relative_to(OWN/'runs') or p==OWN/'runs':raise ValueError('Unique composed runs output only')
 if fresh and p.exists():raise FileExistsError('Fresh output; no automatic retry')
 return p

def approval(c,config,out,probe_only):
 a=read(APPROVAL);expected='APPROVED_ONE_RIGHT_F0_COMPOSED_CPU_PREFLIGHT' if probe_only else 'APPROVED_ONE_RIGHT_F0_COMPOSED_RGB_MASK_DIAGNOSTIC'
 if a.get('status')!=expected or a.get('preparation_handoff_sha256')!=sha(OWN/'preparation_handoff.json') or a.get('config_sha256')!=sha(config) or bounded(a.get('output_dir',''))!=out or a.get('samples')!=1:raise ValueError('New exact composed one-view approval')
 for prefix in ['spec_review','root_review']:
  if a.get(prefix+'_pass') is not True or sha(a[prefix+'_path'])!=a[prefix+'_sha256']:raise ValueError('New independent/root review binding')
 for p,h in read(OWN/'preparation_handoff.json')['files'].items():
  if sha(p)!=h:raise ValueError('Prepared composed implementation changed')
 return a

def _render_target_pixels(scene,contract,c,ctx,out,gpu_uuid,probe_only,instance_id,target):
 import bpy
 oldcontract=ctx['oldcontract'];corners=oldcontract['canonical_definition']['corners_3d_m'];sid=f'f0_i{instance_id}'
 before=render_helpers.state_snapshot(scene);calls=0;primary=None;checks=[]
 try:
  with render_helpers.temporary_render_state(scene):
   scene.render.threads_mode='FIXED';scene.render.threads=2;scene.frame_set(1)
   scene.render.resolution_x,scene.render.resolution_y=c['resolution_wh'];scene.render.resolution_percentage=100
   scene.render.use_border=False;scene.render.use_crop_to_border=False;scene.render.film_transparent=False
   scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
   scene.cycles.samples=64;scene.cycles.seed=c['render']['seed'];scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.transparent_max_bounces=4;scene.cycles.use_adaptive_sampling=False
   inspection=render_helpers.inspection_visibility_guard(scene,{instance_id:target},corners,[instance_id])
   pixels,visible,rays=render_helpers.project_canonical(scene,target['core'],corners,return_ray_clear=True)
   geometry=canonical.build_export(canonical=oldcontract['canonical_definition'],instance_id=instance_id,object_to_world_m=np.array(target['core'].evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world).tolist(),camera_to_world_blender_m=np.array(scene.camera.matrix_world).tolist(),K=actual_K(scene),image_size_wh=c['resolution_wh'],near_plane_m=scene.camera.data.clip_start,ray_clear_to_virtual_point=rays,legacy_corners_px=pixels,legacy_corner_visible=visible)
   canonical.verify_export(geometry,canonical=oldcontract['canonical_definition'],instance_id=instance_id,object_to_world_m=np.array(target['core'].matrix_world).tolist(),camera_to_world_blender_m=np.array(scene.camera.matrix_world).tolist(),K=actual_K(scene),image_size_wh=c['resolution_wh'],near_plane_m=scene.camera.data.clip_start,ray_clear_to_virtual_point=rays,legacy_corners_px=pixels,legacy_corner_visible=visible)
   if np.max(abs(np.asarray(geometry['points_native_px'])-ctx['Y25_row']['canonical8']['points_native_px']))>.003:raise ValueError('Original fixed8 projection exceeds existing camera tolerance')
   with render_helpers.isolated_expanded(scene,target['parts']):coverage=render_helpers.crop_geometry_preflight(scene,target['core'],target['parts'],corners)
   record={'schema_version':'action4.v1','sample_id':sid,'frame_reference':0,'status':'diagnostic_render_not_training_accepted','dataset_generation_allowed':False,'width':3248,'height':2464,'instance_id':instance_id,'canonical8_geometry':geometry,'target_parts':TARGET_NAMES,'inspection_visibility':inspection,'amodal_geometry_coverage':coverage,'camera':{'K':actual_K(scene),'camera_to_world_m':np.array(scene.camera.matrix_world).tolist(),'calibration_status':'estimated_not_calibrated'},'pending':c['pending'],'scene_contract_sha256':sha(out/'scene_contract.json'),'RGB_rendered':False,'instances':[]}
   if not probe_only:
    device=synth._render_device(scene,{**c['render'],'worker_gpu_uuid':gpu_uuid});denoiser=synth._set_denoiser(scene,c['render'])
    for folder in ['rgb','masks','scratch']:(out/folder).mkdir()
    nodes=base._mask_nodes(out/'masks')
    for node in nodes:node.mute=True
    rgb=out/'rgb'/(sid+'.png');scene.cycles.samples=64;scene.cycles.use_denoising=True;scene.render.filepath=str(rgb)
    bpy.ops.render.render(write_still=True);calls+=1
    for j,node in enumerate(nodes,1):node.mute=False;node.file_slots[0].path=f'{sid}_i{j}_visible_'
    scene.cycles.samples=1;scene.cycles.use_denoising=False;bpy.ops.render.render(write_still=False);calls+=1
    vp=out/'masks'/f'{sid}_i{instance_id}_visible_0001.png';ap=out/'masks'/f'{sid}_i{instance_id}_amodal_for_{instance_id}_0001.png'
    for j,node in enumerate(nodes,1):node.file_slots[0].path=f'{sid}_i{j}_amodal_for_{instance_id}_'
    with render_helpers.isolated_expanded(scene,target['parts']):
     full,info=amodal.render_crop(scene,scene.camera,target['core'],target['parts'],corners,nodes,instance_id,ap,out/'scratch'/sid);calls+=1
    measurements=render_helpers.mask_measurements(base._mask_array(vp),full,3248,2464)
    if measurements.get('excluded'):raise ValueError('Required rightf0 target outside image')
    a=np.asarray(pixels);instance={'instance_id':instance_id,'object_id':1,'corners_px':pixels,'corner_visible':visible,'canonical8_geometry':geometry,'bbox_full_xyxy':[float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())],'mask_visible':str(vp.relative_to(ROOT)),'mask_amodal_expanded':str(ap.relative_to(ROOT)),'mask_visible_sha256':sha(vp),'mask_amodal_expanded_sha256':sha(ap),'amodal_expanded_origin_px':[-3248,-2464],'amodal_expanded_shape_hw':[7392,9744],'amodal_render':info,**measurements}
    record.update(image=str(rgb.relative_to(ROOT)),rgb_sha256=sha(rgb),RGB_rendered=True,instances=[instance],unused_indices_are_not_instances=True,render_settings={**c['render'],'actual_device':device,'actual_denoiser':denoiser})
 except Exception as e:
  primary=(e,traceback.format_exc())
 finally:
  try:
   if render_helpers.state_snapshot(scene)!=before:checks.append('Temporary RGB/mask state restoration failed')
  except Exception as e:checks.append('Temporary restoration check '+str(e))
  try:verify_composed_scene(scene,c,ctx,contract)
  except Exception as e:checks.append('Composed scene restore check '+str(e))
  try:frozen_sources(c)
  except Exception as e:checks.append('Frozen source restore check '+str(e))
 if primary or checks:
  write(out/'render_failure.json',{'status':'FAILED_RETAINED_NO_RETRY','primary_error':str(primary[0]) if primary else None,'primary_traceback':primary[1] if primary else None,'restoration_errors':checks,'render_calls_completed':calls,'GPU_render_attempted':not probe_only,'training_accepted':False})
  if primary:raise primary[0]
  raise RuntimeError(checks)
 return {'status':'PASS_COMPOSED_CPU_PREFLIGHT_NO_RENDER' if probe_only else 'COMPOSED_RGB_MASK_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED','rendered':not probe_only,'dataset_generation_allowed':False,'render_calls':calls,'single_sample':True,'registry':ctx['registry'],'record':record,'source_assets_unchanged':True,'restoration_verified':True,'wearing_alignment_accepted':False,'scene_or_training_accepted':False}

def execute_blender(config,out,threads,gpu_uuid=None,probe_only=False):
 import bpy
 c,ctx=prepare(config);out=output_path(out,False);approval(c,config,out,probe_only)
 if threads!=2 or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('Pinned CPU2/GPU UUID resource boundary')
 write(out/'started.json',{'status':'STARTED_ONE_COMPOSED_ATTEMPT','config_sha256':sha(config),'approval_sha256':sha(APPROVAL),'probe_only':probe_only,'automatic_retry':False})
 stage='compose_scene'
 try:
  scene,contract=create_scene(c,ctx);write(out/'scene_contract.json',contract);stage='pixels'
  target_id=c['instance_id'];report=_render_target_pixels(scene,contract,c,ctx,out,gpu_uuid,probe_only,target_id,targets(scene)[target_id]);report['source_config_sha256']=sha(config)
  write(out/('preflight.json' if probe_only else 'diagnostic_report.json'),report)
  if not probe_only:
   with bounded(out/'manifest.jsonl').open('x') as f:f.write(json.dumps(report['record'],allow_nan=False)+chr(10))
  return report
 except Exception as e:
  write(out/'failure.json',{'status':'FAILED_RETAINED_NO_AUTOMATIC_RETRY','stage':stage,'error':str(e),'traceback':traceback.format_exc(),'probe_only':probe_only,'scene_saved':False,'training_accepted':False});raise

def review_panel(out,report,c):
 from PIL import Image,ImageDraw,ImageFont
 record=report['record'];row=record['instances'][0];crop=c['review_crop_xyxy'];rgb=Image.open(bounded(record['image'])).convert('RGB');visible=Image.open(bounded(row['mask_visible'])).convert('RGB');full=Image.open(bounded(row['mask_amodal_expanded'])).convert('RGB');am=full.crop((3248,2464,6496,4928));draw=ImageDraw.Draw(rgb);points=row['corners_px'];font=ImageFont.load_default(size=20)
 for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:draw.line([tuple(points[a]),tuple(points[b])],fill='red',width=3)
 for i,point in enumerate(points):draw.text(tuple(point),str(i),fill='yellow',stroke_width=1,stroke_fill='black',font=font)
 parts=[im.crop(crop) for im in [rgb,visible,am]];panel=Image.new('RGB',(parts[0].width*3,parts[0].height+70),'white')
 for i,part in enumerate(parts):panel.paste(part,(i*part.width,70))
 draw=ImageDraw.Draw(panel);draw.text((8,8),'ACTUAL Y25 skin + mount RGB / visible ID2 / inframe isolated amodal ID2',fill='black',font=font);draw.text((8,35),'Fixed original8; v003 light estimated; no wearing, mechanical or training acceptance.',fill='black',font=font)
 panel.save(out/'RGB_mask_review_native_crop.png');panel.thumbnail((1500,1100));panel.save(out/'RGB_mask_review_view.jpg',quality=90)
 for p,h in [(record['image'],record['rgb_sha256']),(row['mask_visible'],row['mask_visible_sha256']),(row['mask_amodal_expanded'],row['mask_amodal_expanded_sha256'])]:
  if sha(p)!=h:raise ValueError('Review changed raw RGB/mask artifact')

def launch(config,out,threads,gpu_uuid=None,probe_only=False):
 c,ctx=prepare(config);out=output_path(out);approval(c,config,out,probe_only)
 if threads!=2 or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('Only CPU2 and authorized GPU UUID')
 child=worker_environment(None if probe_only else gpu_uuid)
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:child[key]='2'
 child['PYTHONDONTWRITEBYTECODE']='1';child['PYTHONNOUSERSITE']='1'
 for key,subdir in [('TMPDIR','tmp'),('TEMP','tmp'),('TMP','tmp'),('XDG_CACHE_HOME','runtime_cache'),('CUDA_CACHE_PATH','runtime_cache/cuda'),('BLENDER_USER_CONFIG','blender_user/config'),('BLENDER_USER_SCRIPTS','blender_user/scripts'),('BLENDER_USER_DATAFILES','blender_user/datafiles')]:child[key]=str(out/subdir)
 blender=ctx['neutral_config']['sources']['blender']['path']
 cmd=[str(bounded(blender)),'--factory-startup','--disable-autoexec','-noaudio','--background','--threads','2','--python-exit-code','1','--python',str(bounded(c['sources']['entrypoint']['path'])),'--','--inside-blender','--config',str(bounded(config)),'--output',str(out),'--threads','2']
 cmd+=['--probe-only'] if probe_only else ['--gpu-uuid',gpu_uuid]
 out.mkdir(parents=True)
 for subdir in ['tmp','runtime_cache/cuda','blender_user/config','blender_user/scripts','blender_user/datafiles']:(out/subdir).mkdir(parents=True,exist_ok=True)
 write(out/'launch.json',{'command':cmd,'probe_only':probe_only,'gpu_uuid':gpu_uuid,'config_sha256':sha(config),'approval_sha256':sha(APPROVAL),'source_sha256':sha(__file__),'max_samples':1,'automatic_retry':False})
 try:
  with (out/'blender.log').open('xb') as log:process=subprocess.run(cmd,cwd=ROOT,env=child,stdout=log,stderr=subprocess.STDOUT)
  if process.returncode:raise RuntimeError('Preserved failed composed render/preflight: '+str(out/'blender.log'))
  report=read(out/('preflight.json' if probe_only else 'diagnostic_report.json'))
  if not probe_only:review_panel(out,report,c)
  return report
 except Exception as e:
  write(out/'launch_failure.json',{'status':'FAILED_RETAINED_NO_RETRY','error':str(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise

def main():
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--threads',type=int);p.add_argument('--gpu-uuid');p.add_argument('--probe-only',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--inside-blender',action='store_true',help=argparse.SUPPRESS);a=p.parse_args(args)
 if a.prepare_only:
  if a.output or a.gpu_uuid or a.inside_blender or a.probe_only:raise ValueError('Prepare-only cannot mix with execution flags')
  prepare(a.config);print('PREPARED_COMPOSED_ONLY_NO_SCENE_OR_GPU');return
 if not a.output or a.threads!=2:raise ValueError('Explicit output and CPU2')
 if a.inside_blender and 'bpy' not in sys.modules:raise ValueError('Internal Blender entry only')
 report=(execute_blender if a.inside_blender else launch)(a.config,a.output,a.threads,a.gpu_uuid,a.probe_only)
 print(json.dumps({'status':report['status'],'rendered':report['rendered'],'dataset_generation_allowed':False}))
if __name__=='__main__':main()
