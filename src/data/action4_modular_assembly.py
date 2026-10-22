"""Frozen component assembly diagnostic; never grants dataset generation."""
from pathlib import Path
import json, hashlib, importlib.util, sys
import numpy as np
from data import action4_wrist_modules as wrist
from data import action4_brace_module as brace
from data import action4_global_environment as env
ROOT=Path('/data/zy/vln/a-test')
STATUS='FAILED_CALIBRATION_INITIALIZATION_ONLY'
OUTPUT=ROOT/'outputs/action4_realism_reassessment_v001/modular_assembly_probe_v001'
TEXT='Action4_Modular_Assembly_Diagnostic.json'
def bounded(p):
 p=Path(p);p=(p if p.is_absolute() else ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p
def sha(p):return hashlib.sha256(bounded(p).read_bytes()).hexdigest()
def read(p):return json.loads(bounded(p).read_text())
def write(p,d):
 with bounded(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def ref(p):return {'path':str(bounded(p).relative_to(ROOT)),'sha256':sha(p)}
def transform(M,V):return np.asarray(V)@np.asarray(M)[:3,:3].T+np.asarray(M)[:3,3]
def rigid(M):
 M=np.asarray(M,float)
 if M.shape!=(4,4) or not np.isfinite(M).all() or not np.array_equal(M[3],[0,0,0,1]) or np.max(abs(M[:3,:3].T@M[:3,:3]-np.eye(3)))>1e-6 or abs(np.linalg.det(M[:3,:3])-1)>1e-6:raise ValueError('Proper metric SE3 required; no scale/mirror')
 return M
def project(V,K):
 V=np.asarray(V);K=np.asarray(K)
 if not np.isfinite(V).all() or np.any(V[:,2]<=0):raise ValueError('Projection requires finite positive depth')
 return (V@K.T)[:,:2]/V[:,2,None]
def load_config(p):
 c=read(p)
 if c['schema']!='action4.modular_assembly_probe.v1' or c['status']!=STATUS or c['dataset_generation_allowed'] is not False or c['frame_id']!=960 or c['K_ratio']!=.8125 or c['source_anatomical_side']!='right':raise ValueError('Fixed diagnostic-only scope')
 if bounded(c['output_dir'])!=OUTPUT:raise ValueError('Assigned output scope')
 rigid(c['W_from_capture'])
 for r in c['sources'].values():
  if set(r)!={'path','sha256'} or sha(r['path'])!=r['sha256']:raise ValueError('Source identity mismatch')
 spec=importlib.util.spec_from_file_location('frozen_module_verifier',bounded(c['sources']['registry_validator']['path']));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 registry=mod.verify(c['sources']['registry']['path'])
 return c,registry
def frames(c):
 p=read(c['sources']['pose_candidate']['path'])
 if p['id']!='A3_K0p8125_c11_front_patch_inner' or p['production_accepted'] is not False:raise ValueError('Expected retained failed shared-A candidate')
 A=rigid(p['A_shared_newreference_from_camera']);C=rigid(p['C_all_fixed']['960']);B=C@np.linalg.inv(A);W=rigid(c['W_from_capture'])
 width,height=c['native_resolution_wh'];f=c['K_ratio']*width;K=np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
 return A,C,rigid(B),W,K,p
def set_rigid_basis(o,M):
 from mathutils import Matrix
 M=rigid(M);o.rotation_mode='QUATERNION';o.rotation_quaternion=Matrix(M).to_quaternion();o.location=M[:3,3];o.scale=(1,1,1)
 if np.max(abs(np.array(o.matrix_basis)-M))>1e-6:raise ValueError('Rigid Blender basis assignment error')
def mesh_state(o):
 from data.action4_global_scene import _geometry
 o.data.calc_normals_split()
 return {'geometry_uv_sha256':_geometry(o),'material_indices':[p.material_index for p in o.data.polygons],'smooth':[p.use_smooth for p in o.data.polygons],'loop_normals_sha256':hashlib.sha256(np.array([tuple(l.normal) for l in o.data.loops],np.float32).tobytes()).hexdigest(),'use_auto_smooth':o.data.use_auto_smooth,'auto_smooth_angle':o.data.auto_smooth_angle,'texture_space':[list(o.data.texspace_location),list(o.data.texspace_size),o.data.use_auto_texspace],'material_states':[env._material_state(m) for m in o.data.materials],'pass_index':o.pass_index}
def load_objects(asset,names):
 import bpy
 with bpy.data.libraries.load(str(bounded(asset)),link=False) as (a,b):
  if not set(names)<=set(a.objects):raise ValueError('Missing exact library objects')
  b.objects=names
 for o in b.objects:bpy.context.scene.collection.objects.link(o)
 return b.objects
def share_exact_materials(o,expected,actions):
 import bpy
 if len(o.data.materials)!=len(expected):raise ValueError('Material slot count')
 for i,state in enumerate(expected):
  if env.digest(env._material_state(o.data.materials[i]))==env.digest(state):continue
  matches=[m for m in bpy.data.materials if m.use_nodes and env.digest(env._material_state(m))==env.digest(state)]
  if not matches:raise ValueError('No exact frozen material available for namespace collision')
  old=o.data.materials[i].name;o.data.materials[i]=matches[0]
  actions.append({'object':o.name,'slot':i,'from':old,'to':matches[0].name,'proof':'Full frozen supported graph digest equality'})
def world_geometry(o):
 import bpy
 e=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh()
 try:
  m.calc_loop_triangles();V=np.array([v.co[:] for v in m.vertices]);F=np.array([t.vertices[:] for t in m.loop_triangles],int)
  return transform(e.matrix_world,V),F
 finally:e.to_mesh_clear()
def verify_scene(c,contract):
 import bpy
 from bpy_extras.object_utils import world_to_camera_view
 from mathutils import Vector
 A,C,B,W,K,_=frames(c)
 expected_names=set(contract['object_names'])
 if set(o.name for o in bpy.context.scene.objects)!=expected_names:raise ValueError('Unexpected scene objects/full anatomy forbidden')
 if bpy.context.scene.get('dataset_generation_allowed') is not False or bpy.context.scene.get('assembly_status')!=STATUS:raise ValueError('Diagnostic boundary changed')
 if bpy.context.scene.unit_settings.system!='METRIC' or bpy.context.scene.unit_settings.scale_length!=1:raise ValueError('Metric units')
 for name,expected in contract['objects'].items():
  o=bpy.data.objects[name]
  if (o.parent.name if o.parent else None)!=expected['parent'] or not np.array_equal(np.array(o.matrix_parent_inverse),np.eye(4)):raise ValueError('Parent/reference frame altered')
  if len(o.constraints) or o.animation_data is not None or tuple(o.scale)!=(1.,1.,1.) or len(o.modifiers):raise ValueError('Static scale1 objects only: '+o.name+' scale='+str(tuple(o.scale)))
  rigid(o.matrix_world)
  if np.max(abs(np.array(o.matrix_world)-np.array(expected['world_matrix'])))>2e-6:raise ValueError('Effective world transform mismatch')
  if np.max(abs(np.array(o.matrix_basis)-np.array(expected['basis_matrix'])))>2e-6:raise ValueError('Basis transform mismatch')
  if o.type=='MESH' and env.digest(mesh_state(o))!=env.digest(expected['mesh_state']):raise ValueError('Frozen mesh/shading inputs changed')
 band=bpy.data.objects['brace_band'];wr=bpy.data.objects['right_wrist_module_root'];br=bpy.data.objects['brace_module_root'];core=bpy.data.objects['right_Action4_core']
 if np.max(abs(np.array(br.matrix_world)-W@B))>2e-6 or np.max(abs(np.array(wr.matrix_world)-W@B))>2e-6 or np.max(abs(np.array(core.matrix_world)-W@C))>2e-6:raise ValueError('Shared W C B transform broken')
 actualA=np.linalg.inv(np.array(br.matrix_world))@np.array(core.matrix_world)
 if np.max(abs(actualA-A))>2e-6:raise ValueError('C=BA violated')
 canonical=read(c['sources']['camera_source_contract']['path'])['canonical8']
 if canonical!=contract['canonical_definition']:raise ValueError('Canonical definition/order altered')
 V=np.array(canonical['corners_3d_mm'])*.001
 pc=transform(np.linalg.inv(W),transform(core.matrix_world,V));pw=transform(core.matrix_world,V);uv=project(pc,K);ideal=project(transform(C,V),K)
 if np.max(abs(uv-ideal))>.002:raise ValueError('Actual8 projection mismatch')
 observer=bpy.data.objects['assembly_observer'];width,height=c['native_resolution_wh']
 if bpy.context.scene.camera!=observer or bpy.context.scene.render.resolution_x!=width or bpy.context.scene.render.resolution_y!=height or bpy.context.scene.render.resolution_percentage!=100 or observer.data.sensor_fit!='HORIZONTAL' or abs(observer.data.lens-.8125*36)>1e-7 or observer.data.sensor_width!=36 or observer.data.shift_x!=0 or observer.data.shift_y!=0:raise ValueError('Capture camera parameters changed')
 uv_bpy=[]
 for point in pw:
  q=world_to_camera_view(bpy.context.scene,observer,Vector(point));uv_bpy.append([q.x*width-.5,(1-q.y)*height-.5])
 err=float(np.max(abs(np.array(uv_bpy)-uv)))
 if err>.003:raise ValueError('Actual Blender camera/native K projection mismatch')
 wc=read(c['sources']['wrist_contract']['path']);archive=np.load(bounded(c['sources']['wrist_geometry']['path']))
 skin=bpy.data.objects[wc['object_name']];skinworld,_=world_geometry(skin);expectedskin=transform(W@B,archive['brace_vertices_m'])
 skinerr=float(np.max(abs(skinworld-expectedskin)))
 if skinerr>1e-6:raise ValueError('Actual posed skin module/source frame mismatch')
 bc=read(c['sources']['brace_contract']['path']);bv=np.array([v.co[:] for v in band.data.vertices]);bandmodule=transform(np.linalg.inv(np.array(br.matrix_world)),transform(band.matrix_world,bv))
 from data.action4_brace_module import expected as brace_expected
 _,expectedv,_,_,_,rims=brace_expected(brace.load_config(c['sources']['brace_config']['path']))
 if np.max(abs(bandmodule-expectedv))>1e-6:raise ValueError('Posed actual brace metric geometry changed')
 return {'status':'PASS_TRANSFORM_AND_FROZEN_COMPONENT_INVARIANTS_ONLY','instance_id':c['instance_id'],'frame_id':960,'corner_ids':list(range(8)),'corner_order':canonical['corner_order'],'canonical8_local_m':V.tolist(),'canonical8_world_m':pw.tolist(),'canonical8_capture_m':pc.tolist(),'canonical8_native_px':uv.tolist(),'capture_from_world':np.linalg.inv(W).tolist(),'W_from_capture':W.tolist(),'C_capture_from_CAD':C.tolist(),'A_brace_from_CAD_failed_candidate':A.tolist(),'B_capture_from_brace':B.tolist(),'K_native':K.tolist(),'native_resolution_wh':[width,height],'actualC_world':np.array(core.matrix_world).tolist(),'C_equals_BA_error':float(np.max(abs(actualA-A))),'actual8_vs_fixedC_projection_max_px':float(np.max(abs(uv-ideal))),'blender_camera_projection_max_px':err,'posed_skin_vs_library_max_m':skinerr,'posed_band_vs_analytic_max_m':float(np.max(abs(bandmodule-expectedv))),'rim_native_px':{k:project(transform(np.linalg.inv(W),transform(band.matrix_world,bv[np.r_[v,v[0]]])),K).tolist() for k,v in rims.items()},'source_camera_dimensions_xyz_m':np.ptp(np.array([v.co[:] for v in core.data.vertices]),axis=0).tolist(),'dataset_generation_allowed':False}
def geometry_diagnostics(c,export):
 import bpy
 from mathutils.bvhtree import BVHTree
 from mathutils import Vector
 names=['right_Action4_core','right_Action4_core_top_id_decal','right_forearm_natural_hand','brace_band']
 geometries={n:world_geometry(bpy.data.objects[n]) for n in names};trees={n:BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0) for n,(v,f) in geometries.items()}
 pairs={}
 for a,b in [('right_Action4_core','right_forearm_natural_hand'),('right_Action4_core','brace_band'),('brace_band','right_forearm_natural_hand')]:
  hits=trees[a].overlap(trees[b]);pairs[a+'__'+b]={'surface_triangle_pairs_count':len(hits),'pairs':[[int(x),int(y)] for x,y in hits],'claim':'Surface intersection only; absence does not establish no containment'}
 wc=read(c['sources']['wrist_contract']['path']);a=np.load(bounded(c['sources']['wrist_geometry']['path']));idx=wrist.boundary_loop(a['faces']);world=geometries['right_forearm_natural_hand'][0][idx]
 center=world.mean(0);samples=np.r_[center[None],center+.35*(world-center),center+.70*(world-center)]
 W=np.array(c['W_from_capture']);capture=transform(np.linalg.inv(W),samples);K=np.array(export['K_native']);uv=np.full((len(samples),2),np.nan);positive=capture[:,2]>0;uv[positive]=project(capture[positive],K)
 eye=W[:3,3];rows=[];visible=[]
 for i,(point,pc,pix) in enumerate(zip(samples,capture,uv)):
  inframe=bool(pc[2]>0 and 0<=pix[0]<3248 and 0<=pix[1]<2464)
  distance=float(np.linalg.norm(point-eye));direction=(point-eye)/distance;nearest=None
  if inframe:
   for n,t in trees.items():
    hit=t.ray_cast(Vector(eye),Vector(direction),max(0,distance-1e-5))
    if hit[0] is not None and (nearest is None or hit[3]<nearest['distance_m']):nearest={'object':n,'triangle':int(hit[2]),'distance_m':float(hit[3])}
  clear=bool(inframe and nearest is None)
  if clear:visible.append(i)
  rows.append({'id':i,'native_px':pix.tolist() if pc[2]>0 else None,'positive_depth':bool(pc[2]>0),'in_native_frame':inframe,'foreground_blocker':nearest,'clear_ray_to_open_aperture_proxy':clear})
 return {'surface_intersections':pairs,'elbow_opening':{'edges':44,'world_boundary_m':world.tolist(),'native_boundary_px':project(transform(np.linalg.inv(W),world),K).tolist() if np.all(transform(np.linalg.inv(W),world)[:,2]>0) else None,'sample_method':'Source44boundary centroid and0.35/0.70 radial interior samples; nonplanar aperture proxy, not added cap','samples':rows,'clear_inframe_sample_ids':visible,'status':'VISIBLE_APERTURE_PROXY_NO_PRODUCTION' if visible else 'NO_CLEAR_SAMPLED_APERTURE_NOT_CONTINUOUS_VISIBILITY_PROOF'},'missing_support_status':'PENDING_NOT_MECHANICALLY_ACCEPTED_NO_SUPPORT_CONSTRUCTED','containment_accepted':False,'production_accepted':False},geometries
def build(config_path):
 import bpy
 from mathutils import Matrix
 from data import action4_global_scene as source_scene
 c,registry=load_config(config_path);A,C,B,W,K,candidate=frames(c)
 if (OUTPUT/'diagnostic_scene.blend').exists() or (OUTPUT/'camera_components.blend').exists():raise FileExistsError('Fresh diagnostic output required')
 sourcecontract=read(c['sources']['camera_source_contract']['path'])
 bpy.ops.wm.open_mainfile(filepath=str(bounded(c['sources']['camera_source']['path'])),load_ui=False,use_scripts=False);source_scene.verify(bpy.context.scene,sourcecontract)
 core=bpy.data.objects['right_Action4_core'];targets=[core]+list(core.children_recursive)
 if set(o.name for o in targets)!=set(c['allowed_camera_object_names']):raise ValueError('Unexpected camera component scope')
 states={o.name:mesh_state(o) for o in targets}
 originalsourceA=np.asarray(core.matrix_basis).tolist()
 for o in targets:
  if o.type!='MESH' or len(o.modifiers) or len(o.constraints) or o.animation_data is not None:raise ValueError('Static original target components required')
 relative={core.name:np.eye(4).tolist()}
 for o in targets[1:]:
  if o.parent!=core:raise ValueError('Explicit direct original decal ownership expected')
  relative[o.name]=(np.array(o.matrix_parent_inverse)@np.array(o.matrix_basis)).tolist()
  rigid(relative[o.name])
 cad=np.load(bounded(c['sources']['CAD_geometry']['path']))
 if not np.array_equal(np.array([v.co[:] for v in core.data.vertices]),(cad['vertices_mm']*.001).astype(np.float32).astype(float)) or not np.array_equal(np.array([p.vertices[:] for p in core.data.polygons]),cad['faces']):raise ValueError('Original CAD core does not match registry source')
 core.parent=None;core.matrix_parent_inverse=Matrix.Identity(4);core.matrix_basis=Matrix.Identity(4)
 for o in targets[1:]:o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_basis=Matrix(relative[o.name])
 collection=bpy.data.collections.new('diagnostic_original_right_camera_components')
 for o in targets:collection.objects.link(o)
 bpy.context.view_layer.update()
 for o in targets:
  if env.digest(mesh_state(o))!=env.digest(states[o.name]):raise ValueError('Camera extraction changed data/material inputs')
 bpy.data.libraries.write(str(OUTPUT/'camera_components.blend'),{collection},fake_user=True,path_remap='NONE')
 camera_ref=ref(OUTPUT/'camera_components.blend')
 # Library validators run at their original identity roots before assembly.
 bc=brace.load_config(c['sources']['brace_config']['path']);be=read(c['sources']['brace_contract']['path'])
 bpy.ops.wm.open_mainfile(filepath=str(bounded(c['sources']['brace_asset']['path'])),load_ui=False,use_scripts=False);brace.validate_loaded(bc,be)
 wc=read(c['sources']['wrist_contract']['path']);wa=np.load(bounded(c['sources']['wrist_geometry']['path']))
 bpy.ops.wm.open_mainfile(filepath=str(bounded(c['sources']['wrist_asset']['path'])),load_ui=False,use_scripts=False)
 wrist._verify_mesh(bpy.data.objects[wc['object_name']],wc,wa)
 # Exact independently validated original libraries, not reconstructed meshes.
 bpy.ops.wm.read_factory_settings(use_empty=True)
 scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
 load_objects(c['sources']['wrist_asset']['path'],[wc['root_name'],wc['object_name']])
 wrist._verify_mesh(bpy.data.objects[wc['object_name']],wc,wa)
 load_objects(c['sources']['brace_asset']['path'],['brace_module_root','brace_band','wrist_reference'])
 load_objects(camera_ref['path'],c['allowed_camera_object_names'])
 actions=[]
 for name,state in states.items():share_exact_materials(bpy.data.objects[name],state['material_states'],actions)
 share_exact_materials(bpy.data.objects['brace_band'],be['material_states'],actions)
 share_exact_materials(bpy.data.objects[wc['object_name']],[wc['material_state']],actions)
 root=bpy.data.objects['brace_module_root'];set_rigid_basis(root,W@B)
 wr=bpy.data.objects[wc['root_name']];set_rigid_basis(wr,W@B)
 core=bpy.data.objects['right_Action4_core'];core.parent=root;core.matrix_parent_inverse=Matrix.Identity(4);set_rigid_basis(core,A)
 for name,M in relative.items():
  if name!=core.name:
   o=bpy.data.objects[name];o.parent=core;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_basis=Matrix(M)
 data=bpy.data.cameras.new('assembly_observer');cam=bpy.data.objects.new('assembly_observer',data);scene.collection.objects.link(cam);set_rigid_basis(cam,W@np.diag([1,-1,-1,1]));data.type='PERSP';data.sensor_fit='HORIZONTAL';data.sensor_width=36;data.lens=.8125*36;data.shift_x=data.shift_y=0;data.clip_start=.001;data.clip_end=100
 scene.camera=cam;scene.render.resolution_x=3248;scene.render.resolution_y=2464;scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
 scene['dataset_generation_allowed']=False;scene['assembly_status']=STATUS
 bpy.context.view_layer.update()
 objects={}
 for o in scene.objects:
  objects[o.name]={'parent':o.parent.name if o.parent else None,'world_matrix':np.asarray(o.matrix_world).tolist(),'basis_matrix':np.asarray(o.matrix_basis).tolist(),'mesh_state':mesh_state(o) if o.type=='MESH' else None}
 contract={'schema':'action4.modular_assembly_diagnostic.v1','status':STATUS,'dataset_generation_allowed':False,'config':ref(config_path),'source_module':ref(__file__),'sources':c['sources'],'registry_validation':registry,'camera_component_library':camera_ref,'original_source_A_separate_from_candidate':originalsourceA,'candidate_A':A.tolist(),'camera_component_relativecore_matrices':relative,'camera_source_mesh_states':states,'canonical_definition':sourcecontract['canonical8'],'object_names':sorted(objects),'objects':objects,'material_collision_resolution':actions,'local_geometry_is_fixed':True,'wholebody_imported':False,'pending_components':c['pending_components'],'production_accepted':False}
 for name,state in states.items():
  if env.digest(objects[name]['mesh_state'])!=env.digest(state):raise ValueError('Final camera differs from original source')
 export=verify_scene(c,contract);diagnostics,geometries=geometry_diagnostics(c,export)
 bpy.data.texts.new(TEXT).write(json.dumps(contract,sort_keys=True))
 bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT/'diagnostic_scene.blend'),check_existing=False)
 write(OUTPUT/'assembly_contract.json',contract)
 write(OUTPUT/'actual_geometry_and_canonical8.json',export)
 write(OUTPUT/'geometry_diagnostics.json',diagnostics)
 np.savez_compressed(OUTPUT/'actual_world_geometry.npz',**{n+'_'+k:a for n,arrays in geometries.items() for k,a in zip(['vertices_m','triangles'],arrays)})
 load_config(config_path)
 write(OUTPUT/'build_receipt.json',{'status':'BUILT_DIAGNOSTIC_NOT_TRAINABLE','asset':ref(OUTPUT/'diagnostic_scene.blend'),'contract':ref(OUTPUT/'assembly_contract.json'),'sources_unchanged':True,'CPU_only':True,'RGB_rendered':False,'training_allowed':False,'calibration_pass':False,'mechanical_support_accepted':False,'elbow_status':diagnostics['elbow_opening']['status'],'surface_intersection_counts':{k:v['surface_triangle_pairs_count'] for k,v in diagnostics['surface_intersections'].items()}})
 return export
def verify(config_path):
 import bpy
 c,_=load_config(config_path);receipt=read(OUTPUT/'build_receipt.json');contract=read(OUTPUT/'assembly_contract.json')
 for r in [receipt['asset'],receipt['contract'],contract['source_module'],contract['config'],contract['camera_component_library']]:
  if sha(r['path'])!=r['sha256']:raise ValueError('Frozen assembly identity mismatch')
 bpy.ops.wm.open_mainfile(filepath=str(bounded(receipt['asset']['path'])),load_ui=False,use_scripts=False)
 if json.loads(bpy.data.texts[TEXT].as_string())!=contract:raise ValueError('Embedded assembly contract mismatch')
 report=verify_scene(c,contract)
 old=read(OUTPUT/'actual_geometry_and_canonical8.json')
 if env.digest(report)!=env.digest(old):raise ValueError('Fresh scene reload changed exported coordinates')
 load_config(config_path)
 write(OUTPUT/'reload_receipt.json',{'status':'PASS_FRESH_RELOAD_INVARIANTS_ONLY','source_asset_hashes_unchanged':True,'geometry_export_exact':True,'report':report,'production_accepted':False})
 return report
def main():
 import argparse
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','verify']);parser.add_argument('--config',required=True);a=parser.parse_args(args)
 print((build if a.mode=='build' else verify)(a.config)['status'])
if __name__=='__main__':main()
