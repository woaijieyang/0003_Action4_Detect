
'''Independent derived right-brace geometry, exactly matching frozen analytic34mm rings.'''
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
ROOT=Path('/data/zy/vln/a-test')
KEYS={'schema','status','dataset_generation_allowed','side','width_m','proximal_offset_from_wrist_m','segments','sources','output_dir','report_dir'}
def bounded(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p
def sha(path):return hashlib.sha256(bounded(path).read_bytes()).hexdigest()
def load(path):return json.loads(bounded(path).read_text())
def write_new(path,value):
 p=bounded(path)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def load_config(path):
 c=load(path)
 if set(c)!=KEYS or c['schema']!='action4.modular_brace.v1' or c['status']!='diagnostic_unassembled' or c['dataset_generation_allowed'] is not False:raise ValueError('Exact diagnostic-only config required')
 if c['side']!='right' or c['segments']!=72 or c['width_m']!=.034 or c['proximal_offset_from_wrist_m']!=.032:raise ValueError('Only frozen right34 reference is approved')
 for ref in c['sources'].values():
  if set(ref)!={'path','sha256'} or sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen source mismatch')
 for key,root in [('output_dir',ROOT/'data/action4_scene_assets/modular_brace_v001'),('report_dir',ROOT/'outputs/action4_realism_reassessment_v001/modular_brace_build_v001')]:
  if not bounded(c[key]).is_relative_to(root):raise ValueError('Wrong module output scope')
 return c
def parameters(c):
 rig=load(c['sources']['rig_contract']['path']);g1=load(c['sources']['G1_config']['path']);scene=load(c['sources']['scene_config']['path']);analytic=load(c['sources']['analytic34_reference']['path'])
 inner=np.array(rig['fixed_brace_and_camera_attachment'][c['side']]['brace_inner_radii_m'],float);t=float(g1['brace']['thickness_m']);outer=inner+t
 if not np.allclose(outer,analytic['ring_radii_m'],atol=1e-14,rtol=0):raise ValueError('Analytic radius mismatch')
 if abs(g1['brace']['proximal_offset_m']-c['proximal_offset_from_wrist_m']-.003)>1e-12:raise ValueError('Reference shift mismatch')
 trim=2*scene['initial_priors']['green_edge_radius_m']
 return {'inner':inner,'outer':outer,'thickness':t,'trim_width':trim,'scene':scene,'source_F_old':np.array(rig['fixed_brace_and_camera_attachment'][c['side']]['rest_world_frame'])}
def annular_geometry(inner,thickness,width,segments,trim):
 inner=np.asarray(inner,float)
 if inner.shape!=(2,) or not np.isfinite(inner).all() or min(inner)<=0 or not all(math.isfinite(x) and x>0 for x in [thickness,width,trim]) or not isinstance(segments,int) or segments<8 or 2*trim>=width:raise ValueError('Invalid metric geometry')
 outer=inner+thickness;theta=np.arange(segments)*2*np.pi/segments
 V=[]
 for radius,ys in [(outer,[-width/2,-width/2+trim,width/2-trim,width/2]),(inner,[-width/2,width/2])]:
  for y in ys:V.extend(np.c_[radius[0]*np.cos(theta),np.full(segments,y),radius[1]*np.sin(theta)].tolist())
 faces=[];materials=[];roles=[]
 def add(face,mat,role):faces.append(face);materials.append(mat);roles.append(role)
 n=segments
 for station in range(3):
  for i in range(n):
   j=(i+1)%n;add([station*n+i,(station+1)*n+i,(station+1)*n+j,station*n+j],1 if station!=1 else 0,'outer')
 for i in range(n):
  j=(i+1)%n;add([4*n+i,4*n+j,5*n+j,5*n+i],0,'inner')
  add([i,j,4*n+j,4*n+i],1,'proximal_cap')
  add([3*n+i,5*n+i,5*n+j,3*n+j],1,'distal_cap')
 return np.array(V),np.array(faces,int),np.array(materials,int),roles,{'proximal':np.arange(n),'distal':np.arange(3*n,4*n)}
def expected(c):
 p=parameters(c);v,f,mat,roles,rims=annular_geometry(p['inner'],p['thickness'],c['width_m'],c['segments'],p['trim_width'])
 return p,v,f,mat,roles,rims
def material_states(c,band,contract=None):
 from data.action4_global_environment import _material_state,digest
 if len(band.data.materials)!=2:raise ValueError('Exactly gray weave and green stripe materials required')
 prior=parameters(c)['scene']['initial_priors']
 for index,mat in enumerate(band.data.materials):
  if not mat.use_nodes or mat.animation_data is not None:raise ValueError('Static node material required')
  nodes=list(mat.node_tree.nodes);links=list(mat.node_tree.links)
  types=['BSDF_PRINCIPLED','OUTPUT_MATERIAL']+(['TEX_COORD','VECT_MATH','TEX_IMAGE','MIX_RGB'] if index==0 else [])
  if sorted(n.type for n in nodes)!=sorted(types) or any(n.mute for n in nodes):raise ValueError('Exact supported unmuted material graph required')
  by={n.type:n for n in nodes};shader=by['BSDF_PRINCIPLED'];output=by['OUTPUT_MATERIAL']
  if not output.is_active_output:raise ValueError('Expected active material output')
  expected_links=[(shader.outputs['BSDF'],output.inputs['Surface'])]
  def close(a,b):return np.allclose(a,b,rtol=0,atol=1e-7)
  if not close(shader.inputs['Roughness'].default_value,.87 if index==0 else .80) or not close(shader.inputs['Metallic'].default_value,0) or not close(shader.inputs['Specular'].default_value,.24):raise ValueError('Source material surface parameters differ')
  color=prior['brace_linear_rgb'] if index==0 else prior['green_linear_rgb']
  if not close(shader.inputs['Base Color'].default_value,(*color,1)):raise ValueError('Source material base prior differs')
  if index==0:
   coord,vector,image,mix=(by[k] for k in ['TEX_COORD','VECT_MATH','TEX_IMAGE','MIX_RGB'])
   expected_links.extend([(coord.outputs['Object'],vector.inputs[0]),(vector.outputs['Vector'],image.inputs['Vector']),(image.outputs['Color'],mix.inputs[2]),(mix.outputs[0],shader.inputs['Base Color'])])
   if vector.operation!='SCALE' or not np.isclose(vector.inputs['Scale'].default_value,1/prior['brace_photo_tile_m'],rtol=1e-7,atol=1e-7):raise ValueError('Weave metric texture scale differs')
   if mix.blend_type!='MIX' or not close(mix.inputs[0].default_value,1) or not close(mix.inputs[1].default_value,(*color,1)):raise ValueError('Weave source mix differs')
   if image.projection!='BOX' or not close(image.projection_blend,.25) or image.interpolation!='Linear' or image.extension!='REPEAT':raise ValueError('Weave mapping differs')
  if len(links)!=len(expected_links) or any(not any(l.from_socket==a and l.to_socket==b for l in links) for a,b in expected_links):raise ValueError('Active texture-to-material-surface chain differs')
 states=[_material_state(m) for m in band.data.materials]
 if contract is not None:
  if 'material_states' not in contract or digest(states)!=digest(contract['material_states']):raise ValueError('Frozen complete supported material state differs')
 return states

def validate_loaded(c,contract=None):
 import bpy
 p,V,F,M,roles,rims=expected(c);root=bpy.data.objects['brace_module_root'];band=bpy.data.objects['brace_band'];wrist=bpy.data.objects['wrist_reference']
 if len(bpy.context.scene.objects)!=3 or len([o for o in bpy.context.scene.objects if o.type=='MESH'])!=1:raise ValueError('Independent module must have one mesh and two reference empties')
 if band.parent!=root or wrist.parent!=root or not np.array_equal(np.array(root.matrix_world),np.eye(4)) or not np.array_equal(np.array(band.matrix_basis),np.eye(4)):raise ValueError('Unexpected module transform')
 if tuple(band.scale)!=(1.,1.,1.) or tuple(root.scale)!=(1.,1.,1.) or len(band.modifiers):raise ValueError('No scale/modifier geometry allowed')
 if abs(wrist.location.y-.032)>1e-8 or abs(wrist.location.x)+abs(wrist.location.z)>1e-12:raise ValueError('Wristreference offset')
 bpy.context.view_layer.update()
 identity=np.eye(4);wrist_expected=np.eye(4);wrist_expected[1,3]=.032
 if root.parent is not None or any(len(o.constraints) or o.animation_data is not None for o in (root,band,wrist)):raise ValueError('Static independent module transforms required')
 if any(not np.array_equal(np.array(o.matrix_parent_inverse),identity) for o in (band,wrist)):raise ValueError('Parent inverse must be identity')
 module_from_world=np.linalg.inv(np.array(root.matrix_world))
 effective_band=module_from_world@np.array(band.matrix_world);effective_wrist=module_from_world@np.array(wrist.matrix_world)
 if not np.array_equal(effective_band,identity) or not np.allclose(effective_wrist,wrist_expected,atol=2e-9,rtol=0):raise ValueError('Effective module-frame transforms differ')
 states=material_states(c,band,contract)
 if bpy.context.scene.unit_settings.system!='METRIC' or bpy.context.scene.unit_settings.scale_length!=1:raise ValueError('Meter units required')
 ev=band.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
 try:
  actual=np.array([list(v.co) for v in mesh.vertices]);faces=np.array([list(x.vertices) for x in mesh.polygons]);materials=np.array([x.material_index for x in mesh.polygons])
  if actual.shape!=V.shape or not np.array_equal(faces,F) or not np.array_equal(materials,M):raise ValueError('Actual topology/material assignment differs')
  local_actual=actual.copy()
  evaluated_to_module=module_from_world@np.array(ev.matrix_world)
  actual=local_actual@evaluated_to_module[:3,:3].T+evaluated_to_module[:3,3]
  world_actual=local_actual@np.array(ev.matrix_world)[:3,:3].T+np.array(ev.matrix_world)[:3,3]
  if not np.array_equal(evaluated_to_module,identity) or abs(np.ptp(world_actual[:,1])-c['width_m'])>1e-8:raise ValueError('Evaluated effective transform/world width differs')
  maxerr=float(abs(actual-V).max())
  if maxerr>1e-8:raise ValueError('Actual metric coordinates differ')
  edge=np.concatenate([faces[:,[i,(i+1)%4]] for i in range(4)]);u,inv,count=np.unique(np.sort(edge,axis=1),axis=0,return_inverse=True,return_counts=True);bal=np.bincount(inv,weights=np.where(edge[:,0]<edge[:,1],1,-1))
  if not np.all(count==2) or not np.all(bal==0):raise ValueError('Not closed consistentlyoriented manifold')
  normals=np.array([list(x.normal) for x in mesh.polygons]);centers=actual[faces].mean(1);sign=[]
  for n,center,role in zip(normals,centers,roles):
   radial=np.array([center[0],0,center[2]])
   value=float(n@radial)*(1 if role=='outer' else -1) if role in ['outer','inner'] else float(n[1])*(-1 if role=='proximal_cap' else 1)
   sign.append(value)
  if min(sign)<=0:raise ValueError('Incorrect outer/inner/cap normal direction')
  mesh.calc_loop_triangles();tri=np.array([list(t.vertices) for t in mesh.loop_triangles])
  volume=float(np.einsum('ij,ij->i',actual[tri[:,0]],np.cross(actual[tri[:,1]],actual[tri[:,2]])).sum()/6)
  expectedvolume=c['width_m']*c['segments']/2*math.sin(2*math.pi/c['segments'])*(np.prod(p['outer'])-np.prod(p['inner']))
  if volume<=0 or abs(volume-expectedvolume)>1e-11:raise ValueError('Closed shell volume mismatch')
  analytic=load(c['sources']['analytic34_reference']['path']);poses=load(c['sources']['projection_hypotheses']['path']);hyp=next(x for x in poses['hypotheses'] if x['K_ratio']==.8125 and x['f960_rim_hypothesis']=='c11_front_patch_inner');A=np.array(hyp['A_initial_shared']);projection=[];coords={}
  for name,ids in rims.items():
   indices=np.r_[ids,ids[0]];ar=actual[indices];reference=np.array(analytic['rings34_in_newreference'][name])
   difference=float(abs(ar-reference).max());coords[name]={'actual_vertex_indices_withclosure':indices.tolist(),'actual_points_local_m':ar.tolist(),'analytic_points_local_m':reference.tolist(),'max_difference_m':difference}
   if difference>1e-8:raise ValueError('Actual rim does not equal analytic sampled ring')
   for frame,C in hyp['C_fixed'].items():
    B=np.array(C)@np.linalg.inv(A)
    def project(x):
     v=x@B[:3,:3].T+B[:3,3];assert np.min(v[:,2])>.005
     return v[:,:2]/v[:,2,None]*(.8125*3248)+[1623.5,1231.5]
    delta=float(abs(project(ar)-project(reference)).max());assert delta<.001
    projection.append({'frame':int(frame),'rim':name,'points_including_closure':73,'maximum_projection_difference_px':delta,'B_capture_from_brace_reference':B.tolist()})
  packed=[]
  for material in band.data.materials:
   for node in material.node_tree.nodes:
    if node.type=='TEX_IMAGE':
     if node.image.packed_file is None or node.image.colorspace_settings.name!='sRGB':raise ValueError('Weave must be packed original sRGB')
     h=hashlib.sha256(bytes(node.image.packed_file.data)).hexdigest()
     if h!=c['sources']['weave_photo']['sha256']:raise ValueError('Packedweave source mismatch')
     packed.append(h)
  if len(packed)!=1 or band.pass_index!=0 or bool(band.get('target_geometry',True)):raise ValueError('Non-target/materialcontract')
  return {'status':'PASS_LOADED_MODULAR_BRACE_METRIC_AND_POINTWISE_RINGS','vertices':len(actual),'quads':len(faces),'triangles':len(tri),'edges':len(u),'Euler':len(actual)-len(u)+len(faces),'closed_edge_incidence2':True,'consistent_winding':True,'all_normal_roles_correct':True,'positive_volume_m3':volume,'expected_polygonal_shell_volume_m3':expectedvolume,'max_expected_vertex_error_m':maxerr,'dimensions_xyz_m':np.ptp(actual,axis=0).tolist(),'scale1_and_proper_identity_frame':True,'effective_band_module_matrix':effective_band.tolist(),'effective_wrist_module_matrix':effective_wrist.tolist(),'world_dimensions_xyz_m':np.ptp(world_actual,axis=0).tolist(),'material_states':states,'active_material_surface_chain_verified':True,'units_m':True,'rim_checks':coords,'threeframe_projection_checks':projection,'wrist_point_module_m':list(wrist.location),'target_geometry':False,'canonical8':None,'packed_weave_sha256':packed[0]},actual,faces,materials
 finally:ev.to_mesh_clear()
def build(config_path):
 import bpy
 from mathutils import Matrix
 from data.action4_global_rig import _mesh
 from data.action4_render import _material
 from data.action4_blender_foundation import apply_photo_material
 c=load_config(config_path);out=bounded(c['output_dir']);reports=bounded(c['report_dir'])
 if out.exists():raise FileExistsError('Fresh module build required; originalasset untouched')
 out.mkdir(parents=True);reports.mkdir(parents=True,exist_ok=True)
 bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1.
 p,V,F,M,roles,rims=expected(c);prior=p['scene']['initial_priors'];gray=_material('modular_brace_gray_weave',prior['brace_linear_rgb'],.87);apply_photo_material(gray,bounded(c['sources']['weave_photo']['path']),prior['brace_photo_tile_m'],strength=1)
 green=_material('modular_brace_green_stripe',prior['green_linear_rgb'],.80)
 root=bpy.data.objects.new('brace_module_root',None);scene.collection.objects.link(root);root.matrix_world=Matrix.Identity(4)
 band=_mesh('brace_band',V.tolist(),F.tolist(),root,gray);band.matrix_parent_inverse=Matrix.Identity(4);band.matrix_basis=Matrix.Identity(4);band.data.materials.append(green)
 for face,mat in zip(band.data.polygons,M):face.material_index=int(mat)
 band.pass_index=0;band['target_geometry']=False;band['dataset_generation_allowed']=False;band['canonical8_owner']=False
 for name,ids in rims.items():group=band.vertex_groups.new(name=name+'_outer_rim');group.add(ids.tolist(),1.,'REPLACE')
 wrist=bpy.data.objects.new('wrist_reference',None);scene.collection.objects.link(wrist);wrist.parent=root;wrist.matrix_parent_inverse=Matrix.Identity(4);wrist.location=(0,.032,0)
 root['frame_definition']='Origin brace34 centre; +Y distal toward wrist; +X/+Z frozen source radial axes; meters';root['assembly_status']='diagnostic_unassembled';root['target_geometry']=False
 bpy.context.view_layer.update();report,actual,faces,mats=validate_loaded(c)
 S=np.eye(4);S[1,3]=.003;Fnew=p['source_F_old']@S
 contract={'schema':'action4.modular_brace.asset.v2','verifier_revision':'effective_frame_and_active_material_v2','material_states':report['material_states'],'status':'diagnostic_unassembled','side':'right','dataset_generation_allowed':False,'units':'m','target_geometry':False,'canonical8':None,'root_object':'brace_module_root','mesh_object':'brace_band','wrist_reference_object':'wrist_reference','width_m':.034,'inner_radii_m':p['inner'].tolist(),'outer_radii_m':p['outer'].tolist(),'radial_corresponding_sample_thickness_m':p['thickness'],'constant_normal_Solidify_thickness_claim':False,'green_axial_stripe_width_m':p['trim_width'],'green_representation':'Sameclosed-shellendmaterialstrips, notBezier tubes; noadditionalgeometryoverhang','green_width_source':'twiceoriginalgreen_edge_radius prior; appearanceprior notmeasured seam','segments':72,'rim_vertex_indices':{k:v.tolist() for k,v in rims.items()},'source_old28_frame_in_source_body':p['source_F_old'].tolist(),'new34_frame_in_source_body':Fnew.tolist(),'source_body_to_module':np.linalg.inv(Fnew).tolist(),'wrist_origin_in_module_m':[0,.032,0],'proximal_boundary_from_wrist_m':-.049,'distal_boundary_from_wrist_m':-.015,'reference_shift_old_to_new_m':.003,'source_asset_width_m':.028,'sources':c['sources'],'module_source_sha256':sha(Path(__file__).resolve()),'config_sha256':sha(config_path),'geometry_scope':'Onlyclosedband+greenmaterialstrips; noarm/skin/camera/whitepuck/support/crosslinkassembly','assembly_mount_A':None,'assembly_or_production_accepted':False}
 text=json.dumps(contract,sort_keys=True,separators=(',',':'));bpy.data.texts.new('Action4BraceModuleContract.json').write(text)
 asset=out/'brace_module_right.blend';bpy.ops.wm.save_as_mainfile(filepath=str(asset),check_existing=False)
 np.savez_compressed(out/'mesh_geometry.npz',vertices_module_m=actual,faces=faces,material_indices=mats)
 external={**contract,'saved_asset':{'path':str(asset.relative_to(ROOT)),'sha256':sha(asset)},'embedded_contract_sha256':hashlib.sha256(text.encode()).hexdigest()}
 write_new(out/'module_contract.json',external);write_new(reports/'build_geometry_report.json',report)
 for ref in c['sources'].values():assert sha(ref['path'])==ref['sha256']
 return external
def verify(config_path):
 import bpy
 c=load_config(config_path);out=bounded(c['output_dir']);reports=bounded(c['report_dir']);ext=load(out/'module_contract.json')
 if sha(ext['saved_asset']['path'])!=ext['saved_asset']['sha256'] or sha(Path(__file__).resolve())!=ext['module_source_sha256']:raise ValueError('Savedasset/sourceidentity mismatch')
 bpy.ops.wm.open_mainfile(filepath=str(bounded(ext['saved_asset']['path'])),load_ui=False,use_scripts=False)
 embedded=bpy.data.texts['Action4BraceModuleContract.json'].as_string();assert hashlib.sha256(embedded.encode()).hexdigest()==ext['embedded_contract_sha256']
 assert json.loads(embedded)=={k:v for k,v in ext.items() if k not in ['saved_asset','embedded_contract_sha256']}
 report,V,F,M=validate_loaded(c,ext);cache=np.load(out/'mesh_geometry.npz')
 assert np.array_equal(V,cache['vertices_module_m']) and np.array_equal(F,cache['faces']) and np.array_equal(M,cache['material_indices'])
 report['fresh_process_reloaded']=True;report['saved_asset_sha256']=sha(ext['saved_asset']['path']);report['asset_changed_by_verify']=False
 write_new(reports/'reload_geometry_report.json',report)
 for ref in c['sources'].values():assert sha(ref['path'])==ref['sha256']
 return report
def main():
 import argparse
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','verify']);p.add_argument('--config',required=True);a=p.parse_args(args)
 result=build(a.config) if a.mode=='build' else verify(a.config);print(result.get('status'))
if __name__=='__main__':main()
