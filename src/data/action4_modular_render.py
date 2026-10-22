"""One right f0 diagnostic RGB/mask preview using frozen modular components.

No renderer is initialized on import/prepare. Existing global-render helpers own
canonical projection, isolation, crop amodal and mask measurements. This module
only supplies the separate modular scene contract and a one-sample controller.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys
import numpy as np
from data import action4_global_render as render_helpers
from data import action4_global_environment as env
from data import action4_modular_assembly as assembly
from data import action4_render as base
from data import action4_synthesis as synth
from data import action4_amodal as amodal
from data import action4_canonical_export as canonical
from data.action4_devices import ALLOWED_GPU_UUIDS,worker_environment

ROOT=Path('/data/zy/vln/a-test')
OWN=ROOT/'outputs/action4_modular_render_prepare_v001'
STATUS='MODULAR_RIGHT_F0_ESTIMATED_PREVIEW_NOT_TRAINING'
CONFIG=ROOT/'configs/data/action4_modular_preview_v001.json'
APPROVAL=OWN/'root_run_approval.json'
TARGET_NAMES=['right_Action4_core','right_Action4_core_top_id_decal']
ENV_NAMES=['modular_preview_wall','modular_preview_table','modular_preview_area']
CONFIG_KEYS={'schema','status','dataset_generation_allowed','frame_id','instance_id','threads','resolution_wh','sources','render','environment','review_crop_xyxy','pending','output_scope'}
SOURCE_KEYS={'baseline_config','baseline_asset','baseline_contract','relation_adapter','relation_config','relation_result','joint_result','relation_spec','relation_actual_frame','relation_actual_geometry','raw_frame','environment_observation','global_render_helper','render_helper','environment_helper','synthesis_helper','amodal_helper','canonical_helper','assembly_helper','devices_helper','blender','modular_render_module','entrypoint'}
ENV_KEYS={'claim','wall_front_capture_z_m','wall_size_xy_m','wall_thickness_m','wall_linear_rgb','roughness','table_front_capture_z_m','table_back_edge_native_y_px','table_width_m','table_thickness_m','grid_pitch_m','grid_line_fraction','grid_cell_linear_rgb','grid_line_linear_rgb','world_linear_rgb','world_strength','area_location_capture_m','area_target_capture_m','area_energy_w','area_size_m','area_color','view_transform','look','exposure','gamma'}
def bounded(p):
 p=Path(p);p=(p if p.is_absolute() else ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p
def sha(p):return hashlib.sha256(bounded(p).read_bytes()).hexdigest()
def read(p):return json.loads(bounded(p).read_text())
def write(p,value):
 p=bounded(p)
 if not p.is_relative_to(OWN):raise ValueError('Assigned diagnostic output only')
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def load_module(p):
 spec=importlib.util.spec_from_file_location('frozen_modular_relation_render_input',bounded(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate_config(c):
 if set(c)!=CONFIG_KEYS or c['schema']!='action4.modular_right_f0_preview.v1' or c['status']!='estimated_diagnostic_pending_RGB' or c['dataset_generation_allowed'] is not False:raise ValueError('Fixed diagnostic-only schema')
 if c['frame_id']!=0 or c['instance_id']!=2 or c['resolution_wh']!=[3248,2464] or c['threads']!=2:raise ValueError('One native rightf0 only')
 if set(c['sources'])!=SOURCE_KEYS:raise ValueError('Complete explicit source bindings')
 for r in c['sources'].values():
  if set(r)!={'path','sha256'} or sha(r['path'])!=r['sha256']:raise ValueError('Frozen source changed')
 if bounded(c['sources']['modular_render_module']['path'])!=Path(__file__).resolve():raise ValueError('Wrong loaded controller identity')
 if c['render']!={'device':'CUDA','denoiser':'OPTIX','samples':64,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':20261020}:raise ValueError('Fixed existing64RGB/1mask pipeline')
 synth._denoiser_name(c['render']);amodal.validate_settings(c['render'])
 if set(c['environment'])!=ENV_KEYS:raise ValueError('Explicit estimated environment only')
 e=c['environment']
 for key,value in e.items():
  if key in ['claim','view_transform','look']:continue
  a=np.asarray(value,float)
  if not np.isfinite(a).all():raise ValueError('Finite environment prior')
 if e['wall_front_capture_z_m']<=e['table_front_capture_z_m'] or min(e['wall_thickness_m'],e['table_thickness_m'],e['grid_pitch_m'],e['area_size_m'],e['area_energy_w'])<=0:raise ValueError('Positive environment geometry/light')
 for key in ['wall_linear_rgb','grid_cell_linear_rgb','grid_line_linear_rgb','world_linear_rgb','area_color']:
  if np.asarray(e[key]).shape!=(3,) or min(e[key])<0 or max(e[key])>1:raise ValueError('Linear RGB triple prior')
 if len(e['wall_size_xy_m'])!=2 or min(e['wall_size_xy_m'])<=0 or e['table_width_m']<=0 or not 0<e['grid_line_fraction']<1 or not 0<=e['roughness']<=1 or e['world_strength']<0:raise ValueError('Environment dimensions/material range')
 for key in ['area_location_capture_m','area_target_capture_m']:
  if np.asarray(e[key]).shape!=(3,):raise ValueError('Light capture coordinates')
 if np.linalg.norm(np.array(e['area_location_capture_m'])-e['area_target_capture_m'])<1e-6:raise ValueError('Degenerate area-light direction')
 if e['view_transform']!='Standard' or e['look']!='Medium High Contrast' or e['gamma']!=1:raise ValueError('Explicit source-style color management')
 if c['output_scope']!=str((OWN/'runs').relative_to(ROOT)):raise ValueError('Diagnostic output scope')
 if c['review_crop_xyxy']!=[1800,450,3100,2464]:raise ValueError('Fixed native review crop')
 return c
def prepare(config=CONFIG):
 if Path.cwd().resolve()!=ROOT:raise ValueError('Physical project root required')
 c=validate_config(read(config));rel=load_module(c['sources']['relation_adapter']['path']);rc,oldcfg,oldcontract,result,registry=rel.prepare()
 if rc!=read(c['sources']['relation_config']['path']) or result!=read(c['sources']['joint_result']['path']):raise ValueError('Relation input mismatch')
 if oldcontract!=read(c['sources']['baseline_contract']['path']) or rc['baseline_scene']!=c['sources']['baseline_asset']['path'] or rc['baseline_config']!=c['sources']['baseline_config']['path']:raise ValueError('Baseline source mismatch')
 actualresult=read(c['sources']['relation_result']['path']);spec=read(c['sources']['relation_spec']['path'])
 if actualresult['joint_result_sha256']!=c['sources']['joint_result']['sha256'] or spec['result']['sha256']!=c['sources']['relation_result']['sha256']:raise ValueError('Actual result/review linkage mismatch')
 frame=read(c['sources']['relation_actual_frame']['path'])
 actualrow=dict(next(x for x in actualresult['frames'] if x['frame']==0));actualrow.pop('geometry_preview_status',None)
 if actualrow!=frame:raise ValueError('Actualframe result disagreement')
 if frame['frame']!=0 or frame['same_shared_A_S'] is not True:raise ValueError('Verified actualrightf0 required')
 observation=read(c['sources']['environment_observation']['path'])
 if observation['source_frame']!=c['sources']['raw_frame'] | {'resolution_wh':[3248,2464]}:raise ValueError('Native environment observation source mismatch')
 if c['environment']['table_back_edge_native_y_px']!=observation['manual_table_back_edge_initial_y_px']:raise ValueError('Table visible strip prior changed without evidence')
 return c,rel,rc,oldcfg,oldcontract,registry

def environment_geometry(c,K):
 """Explicit engineering geometry in capture coordinates, not measured world."""
 e=c['environment'];K=np.asarray(K);z=e['wall_front_capture_z_m'];y=(e['table_back_edge_native_y_px']-K[1,2])*z/K[1,1]
 wall=np.eye(4);wall[:3,3]=[0,0,z+e['wall_thickness_m']/2]
 table=np.eye(4);table[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
 table[:3,3]=[0,y+e['table_thickness_m']/2,(z+e['table_front_capture_z_m'])/2]
 return {'wall_capture_matrix':wall.tolist(),'wall_dimensions_m':[*e['wall_size_xy_m'],e['wall_thickness_m']],
         'table_capture_matrix':table.tolist(),'table_dimensions_m':[e['table_width_m'],z-e['table_front_capture_z_m'],e['table_thickness_m']],
         'table_back_top_capture_points_m':[[-e['table_width_m']/2,y,z],[e['table_width_m']/2,y,z]],
         'table_top_capture_y_m':y,'grid_pitch_at_back_px':float(K[0,0]*e['grid_pitch_m']/z),'depth_and_photometry_calibrated':False}
def actual_K(scene):
 camera=scene.camera
 if camera is None or camera.name!='assembly_observer':raise ValueError('Only actual assembly observer')
 d=camera.data;w=scene.render.resolution_x;h=scene.render.resolution_y
 if d.type!='PERSP' or d.sensor_fit!='HORIZONTAL' or d.shift_x or d.shift_y or scene.render.resolution_percentage!=100 or scene.render.pixel_aspect_x!=1 or scene.render.pixel_aspect_y!=1:raise ValueError('Native centered square-pixel pinhole')
 f=d.lens/d.sensor_width*w
 return [[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]]
def targets(scene):
 core=scene.objects['right_Action4_core'];parts=[core]+list(core.children_recursive)
 if set(o.name for o in parts)!=set(TARGET_NAMES) or any(o.type!='MESH' or o.pass_index!=2 for o in parts):raise ValueError('Full originalrightcore+decal ownership required')
 for o in scene.objects:
  if o not in parts and o.pass_index!=0:raise ValueError('Nontarget carries target index')
 return {2:{'core':core,'parts':parts}}
def add_environment(scene,c,W,K):
 import bpy
 e=c['environment'];g=environment_geometry(c,K)
 if any(n in bpy.data.objects for n in ENV_NAMES):raise ValueError('Reserved newenvironment name collision')
 mat=base._material('modular_preview_wall_material',e['wall_linear_rgb'],e['roughness'])
 wall=base._cube(ENV_NAMES[0],[0,0,0],g['wall_dimensions_m'],mat)
 assembly.set_rigid_basis(wall,np.asarray(W)@g['wall_capture_matrix'])
 grid=base._grid_material();grid.name='modular_preview_grid_material'
 for node in grid.node_tree.nodes:
  if node.type=='MATH' and node.operation=='MULTIPLY':node.inputs[1].default_value=1/e['grid_pitch_m']
  if node.type=='MATH' and node.operation=='LESS_THAN':node.inputs[1].default_value=e['grid_line_fraction']
  if node.type=='MIX_RGB':node.inputs[1].default_value=(*e['grid_cell_linear_rgb'],1);node.inputs[2].default_value=(*e['grid_line_linear_rgb'],1)
 grid.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=e['roughness']
 table=base._cube(ENV_NAMES[1],[0,0,0],g['table_dimensions_m'],grid)
 assembly.set_rigid_basis(table,np.asarray(W)@g['table_capture_matrix'])
 location=assembly.transform(W,[e['area_location_capture_m']])[0];target=assembly.transform(W,[e['area_target_capture_m']])[0]
 lamp=base._light(location,e['area_energy_w'],e['area_size_m']);lamp.name=ENV_NAMES[2];base._look_at(lamp,target);lamp.data.color=e['area_color']
 for o in [wall,table,lamp]:o.pass_index=0;o['target_geometry']=False;o['dataset_generation_allowed']=False
 world=bpy.data.worlds.new('modular_preview_world');world.use_nodes=True;scene.world=world
 world.node_tree.nodes['Background'].inputs['Color'].default_value=(*e['world_linear_rgb'],1)
 world.node_tree.nodes['Background'].inputs['Strength'].default_value=e['world_strength']
 scene.render.engine='CYCLES'
 for k in ['view_transform','look','exposure','gamma']:setattr(scene.view_settings,k,e[k])
 bpy.context.view_layer.update()
 for o,key,dimkey in [(wall,'wall_capture_matrix','wall_dimensions_m'),(table,'table_capture_matrix','table_dimensions_m')]:
  if not np.allclose(o.matrix_world,np.asarray(W)@g[key],rtol=0,atol=2e-6) or tuple(o.scale)!=(1.,1.,1.):raise ValueError('Environment effective transform/scale')
  dims=np.ptp(np.array([v.co[:] for v in o.data.vertices]),axis=0)
  if not np.allclose(dims,g[dimkey],rtol=0,atol=1e-7):raise ValueError('Environment mesh dimensions')
 if not np.allclose(mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value,[*e['wall_linear_rgb'],1],rtol=0,atol=1e-7):raise ValueError('Wall color setter mismatch')
 if not np.allclose([lamp.data.energy,lamp.data.size],[e['area_energy_w'],e['area_size_m']],rtol=0,atol=1e-6):raise ValueError('Light setter mismatch')
 return g

def new_scene_state(scene,contract):
 """Finite dedicated modularscene contract, not old integrated/fullbody identity."""
 objects={}
 for o in scene.objects:
  objects[o.name]={'type':o.type,'parent':o.parent.name if o.parent else None,'parent_inverse':np.array(o.matrix_parent_inverse).tolist(),
    'basis':np.array(o.matrix_basis).tolist(),'world':np.array(o.matrix_world).tolist(),'scale':list(o.scale),'pass_index':o.pass_index,
    'hide_render':o.hide_render,'hide_viewport':o.hide_viewport,
    'mesh':assembly.mesh_state(o) if o.type=='MESH' else None,'light':env._light_state(o.data) if o.type=='LIGHT' else None}
 return {'objects':objects,'observer_data':{k:getattr(scene.camera.data,k) for k in ['type','lens','sensor_width','sensor_height','sensor_fit','shift_x','shift_y','clip_start','clip_end']},'observer_dof_enabled':scene.camera.data.dof.use_dof,'world':env._world_state(scene.world),'view':{k:getattr(scene.view_settings,k) for k in ['view_transform','look','exposure','gamma']},
         'K':actual_K(scene),'render_engine':scene.render.engine,'units':[scene.unit_settings.system,scene.unit_settings.scale_length],
         'modular_preview_status':scene.get('modular_preview_status'),'dataset_generation_allowed':scene.get('dataset_generation_allowed')}
def verify_modular_scene(scene,c,oldcontract,newcontract,rc):
 import bpy
 basis,world,B=load_module(c['sources']['relation_adapter']['path']).expected_world(rc,oldcontract,0)
 if set(scene.objects.keys())!=set(oldcontract['object_names'])|set(ENV_NAMES):raise ValueError('Only frozenmodule and smallenvironment inventory')
 for name,row in oldcontract['objects'].items():
  o=scene.objects[name]
  if (o.parent.name if o.parent else None)!=row['parent'] or not np.array_equal(o.matrix_parent_inverse,np.eye(4)):raise ValueError('Module parenting changed')
  if tuple(o.scale)!=(1.,1.,1.) or len(o.modifiers) or len(o.constraints) or o.animation_data is not None:raise ValueError('Frozen static module changed')
  if np.max(abs(np.array(o.matrix_world)-world[name]))>2e-6:raise ValueError('WB/WBS/WC changed')
  if o.type=='MESH' and env.digest(assembly.mesh_state(o))!=env.digest(row['mesh_state']):raise ValueError('Frozen mesh/material changed')
 if not np.allclose(actual_K(scene),rc['K_native'],rtol=0,atol=1e-6):raise ValueError('Actual K changed')
 targets(scene)
 if env.digest(new_scene_state(scene,newcontract))!=newcontract['state_sha256']:raise ValueError('Modular scene/environment/light contract changed')
 return True
def create_scene(c,rel,rc,oldcfg,oldcontract):
 import bpy
 bpy.ops.wm.open_mainfile(filepath=str(bounded(c['sources']['baseline_asset']['path'])),load_ui=False,use_scripts=False)
 assembly.verify_scene(oldcfg,oldcontract)
 basis,_,_=rel.expected_world(rc,oldcontract,0)
 for name in ['brace_module_root','right_wrist_module_root','right_Action4_core','assembly_observer']:assembly.set_rigid_basis(bpy.data.objects[name],basis[name])
 scene=bpy.context.scene;scene['assembly_status']=rel.NEW_STATUS;bpy.context.view_layer.update()
 actual,geometries,local=rel.verify_actual_frame(rc,oldcontract,0)
 geometry=add_environment(scene,c,np.array(rc['W_from_capture_engineering_embedding']),rc['K_native'])
 scene['modular_preview_status']=STATUS;scene['dataset_generation_allowed']=False
 state=new_scene_state(scene,None)
 newcontract={'schema':'action4.modular_preview_scene.v1','status':STATUS,'dataset_generation_allowed':False,'source_identities':c['sources'],
   'state':state,'state_sha256':env.digest(state),'original_baseline_verified_before_modular_changes':True,'new_modular_relation_actual':actual,
   'estimated_environment':c['environment'],'environment_geometry':geometry,'target_parts':TARGET_NAMES,'pending':c['pending'],
   'old_integrated_scene_acceptance_claim':False,'wearing_alignment_accepted':False,'training_accepted':False}
 verify_modular_scene(scene,c,oldcontract,newcontract,rc)
 return scene,newcontract

def output_path(value,fresh=True):
 p=bounded(value)
 if not p.is_relative_to(OWN/'runs') or p==OWN/'runs':raise ValueError('Unique outputs-only diagnostic run')
 if fresh and p.exists():raise FileExistsError('Fresh output; no automatic retry')
 return p
def approval(c,config,out,probe_only):
 a=read(APPROVAL)
 expected='APPROVED_ONE_RIGHT_F0_MODULAR_CPU_PREFLIGHT' if probe_only else 'APPROVED_ONE_RIGHT_F0_MODULAR_RGB_MASK_DIAGNOSTIC'
 if a.get('status')!=expected or a.get('preparation_handoff_sha256')!=sha(OWN/'preparation_handoff.json') or a.get('config_sha256')!=sha(config) or bounded(a.get('output_dir',''))!=out or a.get('samples')!=1:raise ValueError('Exact new single-view approval required')
 for prefix in ['spec_review','root_review']:
  if a.get(prefix+'_pass') is not True or sha(a[prefix+'_path'])!=a[prefix+'_sha256']:raise ValueError('Bound independent/root review required')
 for p,h in read(OWN/'preparation_handoff.json')['files'].items():
  if sha(p)!=h:raise ValueError('Prepared implementation changed')
 return a

def execute_blender(config,out,threads,gpu_uuid=None,probe_only=False):
 import bpy
 c,rel,rc,oldcfg,oldcontract,registry=prepare(config);out=output_path(out,False);approval(c,config,out,probe_only)
 if threads!=2 or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('Pinned resources')
 scene,contract=create_scene(c,rel,rc,oldcfg,oldcontract);target=targets(scene)[2];corners=oldcontract['canonical_definition']['corners_3d_m'];sid='f0_R'
 write(out/'scene_contract.json',contract);before=render_helpers.state_snapshot(scene);calls=0
 try:
  with render_helpers.temporary_render_state(scene):
   scene.render.threads_mode='FIXED';scene.render.threads=2;scene.frame_set(1)
   scene.render.resolution_x,scene.render.resolution_y=c['resolution_wh'];scene.render.resolution_percentage=100
   scene.render.use_border=False;scene.render.use_crop_to_border=False;scene.render.film_transparent=False
   scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
   scene.cycles.samples=64;scene.cycles.seed=c['render']['seed'];scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.transparent_max_bounces=4;scene.cycles.use_adaptive_sampling=False
   inspection=render_helpers.inspection_visibility_guard(scene,{2:target},corners,[2])
   pixels,visible,rays=render_helpers.project_canonical(scene,target['core'],corners,return_ray_clear=True)
   geometry=canonical.build_export(canonical=oldcontract['canonical_definition'],instance_id=2,object_to_world_m=np.array(target['core'].evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world).tolist(),camera_to_world_blender_m=np.array(scene.camera.matrix_world).tolist(),K=actual_K(scene),image_size_wh=c['resolution_wh'],near_plane_m=scene.camera.data.clip_start,ray_clear_to_virtual_point=rays,legacy_corners_px=pixels,legacy_corner_visible=visible)
   canonical.verify_export(geometry,canonical=oldcontract['canonical_definition'],instance_id=2,object_to_world_m=np.array(target['core'].matrix_world).tolist(),camera_to_world_blender_m=np.array(scene.camera.matrix_world).tolist(),K=actual_K(scene),image_size_wh=c['resolution_wh'],near_plane_m=scene.camera.data.clip_start,ray_clear_to_virtual_point=rays,legacy_corners_px=pixels,legacy_corner_visible=visible)
   with render_helpers.isolated_expanded(scene,target['parts']):coverage=render_helpers.crop_geometry_preflight(scene,target['core'],target['parts'],corners)
   record={'schema_version':'action4.v1','sample_id':sid,'frame_reference':0,'status':'diagnostic_render_not_training_accepted','dataset_generation_allowed':False,'width':3248,'height':2464,'instance_id':2,'canonical8_geometry':geometry,'target_parts':TARGET_NAMES,'inspection_visibility':inspection,'amodal_geometry_coverage':coverage,'camera':{'K':actual_K(scene),'camera_to_world_m':np.array(scene.camera.matrix_world).tolist(),'calibration_status':'estimated_not_calibrated'},'pending':c['pending'],'scene_contract_sha256':sha(out/'scene_contract.json'),'RGB_rendered':False,'instances':[]}
   if not probe_only:
    device=synth._render_device(scene,{**c['render'],'worker_gpu_uuid':gpu_uuid});denoiser=synth._set_denoiser(scene,c['render'])
    for folder in ['rgb','masks','scratch']:(out/folder).mkdir()
    nodes=base._mask_nodes(out/'masks')
    for node in nodes:node.mute=True
    rgb=out/'rgb'/(sid+'.png');scene.cycles.samples=64;scene.cycles.use_denoising=True;scene.render.filepath=str(rgb)
    bpy.ops.render.render(write_still=True);calls+=1
    for j,node in enumerate(nodes,1):node.mute=False;node.file_slots[0].path=f'{sid}_i{j}_visible_'
    scene.cycles.samples=1;scene.cycles.use_denoising=False;bpy.ops.render.render(write_still=False);calls+=1
    vp=out/'masks'/f'{sid}_i2_visible_0001.png';ap=out/'masks'/f'{sid}_i2_amodal_for_2_0001.png'
    for j,node in enumerate(nodes,1):node.file_slots[0].path=f'{sid}_i{j}_amodal_for_2_'
    with render_helpers.isolated_expanded(scene,target['parts']):
     full,info=amodal.render_crop(scene,scene.camera,target['core'],target['parts'],corners,nodes,2,ap,out/'scratch'/sid);calls+=1
    measurements=render_helpers.mask_measurements(base._mask_array(vp),full,3248,2464)
    if measurements.get('excluded'):raise ValueError('Required rightf0 target outside image')
    a=np.asarray(pixels);instance={'instance_id':2,'object_id':1,'corners_px':pixels,'corner_visible':visible,'canonical8_geometry':geometry,'bbox_full_xyxy':[float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())],'mask_visible':str(vp.relative_to(ROOT)),'mask_amodal_expanded':str(ap.relative_to(ROOT)),'mask_visible_sha256':sha(vp),'mask_amodal_expanded_sha256':sha(ap),'amodal_expanded_origin_px':[-3248,-2464],'amodal_expanded_shape_hw':[7392,9744],'amodal_render':info,**measurements}
    record.update(image=str(rgb.relative_to(ROOT)),rgb_sha256=sha(rgb),RGB_rendered=True,instances=[instance],unused_target_index1_is_not_an_instance=True,render_settings={**c['render'],'actual_device':device,'actual_denoiser':denoiser})
 finally:
  if render_helpers.state_snapshot(scene)!=before:raise RuntimeError('Temporary RGB/mask state restoration failed')
  verify_modular_scene(scene,c,oldcontract,contract,rc)
  for ref in c['sources'].values():
   if sha(ref['path'])!=ref['sha256']:raise ValueError('Source changed during diagnostic')
 report={'status':'PASS_MODULAR_CPU_PREFLIGHT_NO_RENDER' if probe_only else 'MODULAR_RGB_MASK_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED','rendered':not probe_only,'dataset_generation_allowed':False,'render_calls':calls,'single_sample':True,'source_config_sha256':sha(config),'registry':registry,'record':record,'source_assets_unchanged':True,'restoration_verified':True,'wearing_alignment_accepted':False,'scene_or_training_accepted':False}
 write(out/('preflight.json' if probe_only else 'diagnostic_report.json'),report)
 if not probe_only:
  with (out/'manifest.jsonl').open('x') as f:f.write(json.dumps(record,allow_nan=False)+chr(10))
 return report

def review_panel(out,report,c):
 """Project-Python display only; original RGB/masks stay unmodified."""
 from PIL import Image,ImageDraw,ImageFont
 record=report['record'];row=record['instances'][0];crop=c['review_crop_xyxy'];rgb=Image.open(bounded(record['image'])).convert('RGB')
 visible=Image.open(bounded(row['mask_visible'])).convert('RGB');full=Image.open(bounded(row['mask_amodal_expanded'])).convert('RGB');am=full.crop((3248,2464,6496,4928))
 draw=ImageDraw.Draw(rgb);points=row['corners_px'];font=ImageFont.load_default(size=20)
 for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:draw.line([tuple(points[a]),tuple(points[b])],fill='red',width=3)
 for i,p in enumerate(points):draw.text(tuple(p),str(i),fill='yellow',stroke_width=1,stroke_fill='black',font=font)
 parts=[x.crop(crop) for x in [rgb,visible,am]];panel=Image.new('RGB',(parts[0].width*3,parts[0].height+70),'white')
 for i,part in enumerate(parts):panel.paste(part,(i*part.width,70))
 d=ImageDraw.Draw(panel);d.text((8,8),'ACTUAL RGB + fixed cuboid0..7 | visible ID2 mask | isolated inframe amodal ID2',fill='black',font=font);d.text((8,35),'Diagnostic only. Missing mount, coarse wrist pose and estimated environment; no calibration/training acceptance.',fill='black',font=font)
 panel.save(out/'RGB_mask_review_native_crop.png');panel.thumbnail((1500,1100));panel.save(out/'RGB_mask_review_view.jpg',quality=90)
 for p,k in [(bounded(record['image']),'rgb_sha256'),(bounded(row['mask_visible']),'mask_visible_sha256'),(bounded(row['mask_amodal_expanded']),'mask_amodal_expanded_sha256')]:
  if sha(p)!=(record[k] if k=='rgb_sha256' else row[k]):raise ValueError('Review panel changed rawartifact')
def launch(config,out,threads,gpu_uuid=None,probe_only=False):
 c,*_=prepare(config);out=output_path(out);approval(c,config,out,probe_only)
 if threads!=2 or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('Only CPU2 and authorizedGPU UUID')
 child=worker_environment(None if probe_only else gpu_uuid)
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:child[k]='2'
 child['PYTHONDONTWRITEBYTECODE']='1';child['PYTHONNOUSERSITE']='1';child['TMPDIR']=str(out/'tmp')
 child['XDG_CACHE_HOME']=str(out/'runtime_cache');child['CUDA_CACHE_PATH']=str(out/'runtime_cache'/'cuda')
 for key,subdir in [('BLENDER_USER_CONFIG','config'),('BLENDER_USER_SCRIPTS','scripts'),('BLENDER_USER_DATAFILES','datafiles')]:child[key]=str(ROOT/'runtime'/'blender_user'/subdir)
 cmd=[str(bounded(c['sources']['blender']['path'])),'--factory-startup','--disable-autoexec','-noaudio','--background','--threads','2','--python-exit-code','1','--python',str(bounded(c['sources']['entrypoint']['path'])),'--','--inside-blender','--config',str(bounded(config)),'--output',str(out),'--threads','2']
 cmd+=['--probe-only'] if probe_only else ['--gpu-uuid',gpu_uuid]
 out.mkdir(parents=True);(out/'tmp').mkdir();write(out/'launch.json',{'command':cmd,'probe_only':probe_only,'gpu_uuid':gpu_uuid,'config_sha256':sha(config),'approval_sha256':sha(APPROVAL),'source_sha256':sha(__file__),'max_samples':1,'automatic_retry':False})
 with (out/'blender.log').open('xb') as log:p=subprocess.run(cmd,cwd=ROOT,env=child,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('Preserved failed modular render/preflight: '+str(out/'blender.log'))
 report=read(out/('preflight.json' if probe_only else 'diagnostic_report.json'))
 if not probe_only:review_panel(out,report,c)
 return report
def main():
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--threads',type=int);p.add_argument('--gpu-uuid');p.add_argument('--probe-only',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--inside-blender',action='store_true',help=argparse.SUPPRESS);a=p.parse_args(args)
 if a.prepare_only:prepare(a.config);print('PREPARED_ONLY_NO_BLENDER_OR_GPU');return
 if not a.output or a.threads!=2:raise ValueError('Explicit output andCPU2')
 if a.inside_blender and 'bpy' not in sys.modules:raise ValueError('Internal Blender entry only')
 report=(execute_blender if a.inside_blender else launch)(a.config,a.output,a.threads,a.gpu_uuid,a.probe_only)
 print(json.dumps({'status':report['status'],'rendered':report['rendered'],'dataset_generation_allowed':False}))
if __name__=='__main__':main()
