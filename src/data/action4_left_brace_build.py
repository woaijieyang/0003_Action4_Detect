"""Reviewed-input left34 builder adapter. Actual execution requires bound approval."""
from pathlib import Path
import json,hashlib,sys,numpy as np
from data import action4_left_brace_module as left
from data import action4_brace_module as old
ROOT=Path('/data/zy/vln/a-test')
OWN=ROOT/'outputs/action4_left_brace_build_v001'
TARGET=ROOT/'data/action4_scene_assets/modular_left_brace_v001/build_v001'
NAMES={'brace_module_root','brace_band','wrist_reference'}

def target_path(value=TARGET):
 p=old.bounded(value)
 if p!=TARGET:raise ValueError('Only new left build_v001 output allowed')
 return p

def prepare():
 if Path.cwd().resolve()!=ROOT:raise ValueError('Physical project root required')
 spec=old.load(OWN/'build_spec.json')
 if spec['target']!=str(TARGET.relative_to(ROOT)) or spec['mode']!='LEFT34_ONLY_PENDING_APPROVAL':raise ValueError('Fixed build scope')
 for p,h in spec['sources'].items():
  if old.sha(p)!=h:raise ValueError('Bound source changed '+p)
 c=left.load_config();p,V,F,M,roles,rims=left.expected(c);report,T=left.verify_geometry(c,V,F,M)
 if target_path().exists():raise FileExistsError('New asset target already exists; no overwrite/retry')
 return spec,c,p,V,F,M,roles,rims

def check_approval(a):
 if a.get('status')!='APPROVED_ONE_LEFT34_BUILD_AND_REOPEN' or a.get('preparation_handoff_sha256')!=old.sha(OWN/'preparation_handoff.json'):raise ValueError('Exact preparation approval required')
 for key in ['spec_review','root_review']:
  if a.get(key+'_pass') is not True or old.sha(a[key+'_path'])!=a[key+'_sha256']:raise ValueError('Independent/root review binding required')
 for p,h in old.load(OWN/'preparation_handoff.json')['files'].items():
  if old.sha(p)!=h:raise ValueError('Prepared file changed')

def validate(c,expected_contract=None):
 import bpy
 scene=bpy.context.scene;p,V,F,M,roles,rims=left.expected(c)
 if set(scene.objects.keys())!=NAMES or scene.unit_settings.system!='METRIC' or scene.unit_settings.scale_length!=1:raise ValueError('Independent metric left module only')
 root,band,wrist=(scene.objects[n] for n in ['brace_module_root','brace_band','wrist_reference']);I=np.eye(4);W=I.copy();W[1,3]=.032
 if root.parent is not None or band.parent!=root or wrist.parent!=root or root.type!='EMPTY' or wrist.type!='EMPTY' or band.type!='MESH':raise ValueError('Fixed parents/types')
 for o in [root,band,wrist]:
  if tuple(o.scale)!=(1.,1.,1.) or len(o.constraints) or len(o.modifiers) or o.animation_data is not None:raise ValueError('Static unit-scale components')
  if not np.array_equal(np.array(o.matrix_parent_inverse),I):raise ValueError('Parent inverse must be identity')
 bpy.context.view_layer.update()
 for actual,ref in [(root.matrix_world,I),(band.matrix_basis,I),(band.matrix_world,I),(wrist.matrix_basis,W),(wrist.matrix_world,W)]:
  if not np.allclose(actual,ref,rtol=0,atol=2e-9):raise ValueError('Effective module transform mismatch')
 if len(band.data.uv_layers)!=0:raise ValueError('Source mesh has no UV; Object mapped material required')
 if band.pass_index!=0 or band.get('target_geometry') is not False or band.get('canonical8_owner') is not False or band.get('dataset_generation_allowed') is not False:raise ValueError('Non-target contract')
 if scene.get('dataset_generation_allowed') is not False or scene.get('module_side')!='left':raise ValueError('Scene diagnostic identity')
 # This reused material validator accepts explicit LEFT parameters; never calls right.load_config.
 states=old.material_states(c,band,expected_contract)
 packed=[]
 for mat in band.data.materials:
  for node in mat.node_tree.nodes:
   if node.type=='TEX_IMAGE':
    image=node.image
    if image is None or image.packed_file is None or image.colorspace_settings.name!='sRGB':raise ValueError('Packed sRGB source weave required')
    h=hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
    if h!=c['sources']['weave_photo']['sha256']:raise ValueError('Packed source bytes differ')
    packed.append(h)
 if len(packed)!=1:raise ValueError('Exactly one weave source')
 ev=band.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
 try:
  actual=np.array([list(v.co) for v in mesh.vertices]);faces=np.array([list(f.vertices) for f in mesh.polygons]);mats=np.array([f.material_index for f in mesh.polygons]);report,T=left.verify_geometry(c,actual,faces,mats)
  if not np.allclose(ev.matrix_world,I,rtol=0,atol=2e-9) or abs(np.ptp(actual[:,1])-.034)>1e-8:raise ValueError('Actual effective width')
  normals=np.array([list(f.normal) for f in mesh.polygons]);centers=actual[faces].mean(1);values=[]
  for n,center,role in zip(normals,centers,roles):
   values.append(float(n@np.array([center[0],0,center[2]]))*(1 if role=='outer' else -1) if role in ['outer','inner'] else float(n[1])*(-1 if role=='proximal_cap' else 1))
  if min(values)<=0:raise ValueError('Normal orientation')
 finally:ev.to_mesh_clear()
 report.update(status='PASS_ACTUAL_LEFT34_METRIC_MATERIAL_NON_TARGET',material_states=states,packed_weave_sha256=packed[0],UV_layers=0,Object_mapping=True,normals_correct=True,scale1=True,effective_width_m=float(np.ptp(actual[:,1])),source_side='left',actual_asset_built=True,training_accepted=False)
 if expected_contract is not None and expected_contract['side']!='left':raise ValueError('Wrong side contract')
 return report,actual,faces,mats

def build(approval_path):
 spec,c,p,V,F,M,roles,rims=prepare();check_approval(old.load(approval_path))
 import bpy
 from mathutils import Matrix
 from data.action4_global_rig import _mesh
 from data.action4_render import _material
 from data.action4_blender_foundation import apply_photo_material
 out=target_path();out.mkdir(parents=True,exist_ok=False);run=OWN/'actual_build_v001';run.mkdir(exist_ok=False)
 old.write_new(run/'started.json',{'approval_sha256':old.sha(approval_path),'actual_builds_max':1})
 bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1.;scene['dataset_generation_allowed']=False;scene['module_side']='left'
 prior=p['scene']['initial_priors'];gray=_material('left34_gray_weave',prior['brace_linear_rgb'],.87);apply_photo_material(gray,old.bounded(c['sources']['weave_photo']['path']),prior['brace_photo_tile_m'],strength=1);green=_material('left34_green_stripe',prior['green_linear_rgb'],.80)
 root=bpy.data.objects.new('brace_module_root',None);scene.collection.objects.link(root);root.matrix_world=Matrix.Identity(4)
 band=_mesh('brace_band',V.tolist(),F.tolist(),root,gray);band.matrix_parent_inverse=Matrix.Identity(4);band.matrix_basis=Matrix.Identity(4);band.data.materials.append(green)
 for face,index in zip(band.data.polygons,M):face.material_index=int(index)
 band.pass_index=0
 for key in ['target_geometry','canonical8_owner','dataset_generation_allowed']:band[key]=False
 for name,ids in rims.items():band.vertex_groups.new(name=name+'_outer_rim').add(ids.tolist(),1.,'REPLACE')
 wrist=bpy.data.objects.new('wrist_reference',None);scene.collection.objects.link(wrist);wrist.parent=root;wrist.matrix_parent_inverse=Matrix.Identity(4);wrist.location=(0,.032,0);root['side']='left';root['assembly_status']='diagnostic_unassembled'
 bpy.context.view_layer.update();report,actual,faces,mats=validate(c);shift=np.eye(4);shift[1,3]=.003;Fnew=p['source_F_old']@shift
 contract={'schema':'action4.left_brace.asset.v1','side':'left','status':'diagnostic_unassembled','units':'m','dataset_generation_allowed':False,'target_geometry':False,'canonical8':None,'width_m':.034,'reference_m':.032,'inner_radii_m':p['inner'].tolist(),'outer_radii_m':p['outer'].tolist(),'radial_corresponding_thickness_m':p['thickness'],'segments':72,'source_old_frame':p['source_F_old'].tolist(),'new34_frame':Fnew.tolist(),'wrist_reference_m':[0,.032,0],'material_states':report['material_states'],'sources':spec['sources'],'builder_sha256':old.sha(Path(__file__).resolve()),'left_preparation_config':{'path':left.CONFIG,'sha256':old.sha(left.CONFIG)},'assembly_A':None,'assembly_S':None,'production_accepted':False}
 text=json.dumps(contract,sort_keys=True,separators=(',',':'));bpy.data.texts.new('Action4LeftBraceContract.json').write(text);asset=out/'brace_module_left.blend';bpy.ops.wm.save_as_mainfile(filepath=str(asset),check_existing=False)
 np.savez_compressed(out/'mesh_geometry.npz',vertices_module_m=actual,faces=faces,material_indices=mats);external={**contract,'saved_asset':{'path':str(asset.relative_to(ROOT)),'sha256':old.sha(asset)},'embedded_contract_sha256':hashlib.sha256(text.encode()).hexdigest()};old.write_new(out/'module_contract.json',external);old.write_new(run/'build_report.json',report)
 # Clean database then reopen same artifact; this is same-process clean reload, not a fresh OS process.
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.open_mainfile(filepath=str(asset),load_ui=False,use_scripts=False);embedded=bpy.data.texts['Action4LeftBraceContract.json'].as_string()
 if hashlib.sha256(embedded.encode()).hexdigest()!=external['embedded_contract_sha256'] or json.loads(embedded)!=contract:raise ValueError('Embedded/external contract divergence')
 reopened,RV,RF,RM=validate(c,contract);cache=np.load(out/'mesh_geometry.npz',allow_pickle=False)
 if not np.array_equal(RV,cache['vertices_module_m']) or not np.array_equal(RF,cache['faces']) or not np.array_equal(RM,cache['material_indices']) or old.sha(asset)!=external['saved_asset']['sha256']:raise ValueError('Reload geometry/file changed')
 for source,h in spec['sources'].items():
  if old.sha(source)!=h:raise ValueError('Original frozen input changed')
 reopened.update(clean_same_process_reload=True,fresh_process_reload=False,source_files_unchanged=True,asset_sha256=external['saved_asset']['sha256']);old.write_new(run/'reload_report.json',reopened)
 return external

def main():
 import argparse
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:];parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--approval');a=parser.parse_args(args)
 if a.prepare_only:prepare();print('PREPARED_LEFT_BUILD_NO_ACTUAL')
 elif a.approval:build(a.approval)
 else:raise ValueError('Prepare-only or approved explicit build required')
