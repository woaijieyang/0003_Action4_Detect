"""Current derived10deg full garment. Historical metadata only; no old source_prepare."""
from pathlib import Path
from copy import deepcopy
import importlib.util,json,hashlib,math
import numpy as np
from data import action4_modular_assembly as assembly
from data import action4_modular_bilateral_scene as bilateral
from data import action4_modular_pose_controls as controls
from data import action4_global_environment as env
ROOT=Path('/data/zy/vln/a-test')
REFS={'prescription': {'path': 'outputs/action4_rigid_garment_lift_robust_v003/prescription.json', 'sha256': 'fc7680303ebe4c04a5417f2b6de9394fb014454a9dfcc1a04d323ad8e7414e69'}, 'packets': {'path': 'outputs/action4_rigid_garment_lift_robust_v003/scene_packets.json', 'sha256': '17cb096d6f4202dfb1275c15b95ce7c4fb80ae83092a5732024aaea5bfd4200c'}, 'observer': {'path': 'outputs/action4_parametric_presence_actual_prepare_v001/actual_presence.py', 'sha256': '90be2dab4bf066083fb177938ed96e6f69cf2b6767a50fffebf4f9575ac9e1eb'}}
GEOMETRY_SPEC={'path':'outputs/action4_forearm_continuation_spec_v001/SPEC_REVIEW.json','sha256':'f180f03c58d96970e9e0b85999dbe101fe42da0629ff47cfd8c80708f25871c2'}
PRESETS=('cloth_lift10_source540','cloth_lift10_source960')
def check(r):
 p=bilateral.bounded(r['path'])
 if Path(r['path']).is_absolute() or '..' in Path(r['path']).parts or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Exact current lift source reference')
 return p
def read(r):return json.loads(check(r).read_text())
def load(r,name):
 spec=importlib.util.spec_from_file_location(name,check(r));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
from data import action4_formal_environment as environment
from data import action4_formal_composition as router
background=environment.garment
interface=router.load_fixed('interface','formal_lift_derived_binding')
observer=load(REFS['observer'],'formal_lift_actor_observer')
def prepare(config):
 view=environment.prepare(config);source_descriptor,source=environment.provider.descriptor(config['environment_source_ref'])
 p=read(REFS['prescription']);packets=read(REFS['packets']);q=read(config['request_ref'])
 if view['execution_source_context_sha256']!=q['source_context_sha256'] or p['source_context_sha256']!=view['historical_source_context_sha256']:raise ValueError('Explicit current/evidence physical source bridge')
 if p['target_margin_m']!=.020 or p['physical_minimum_margin_m']!=.002 or p['actual_float32_numeric_budget_m']!=2e-6 or p['thresholds']['maximum_common_actor_translation_m']!=.16:raise ValueError('Original20mm/2um/160mm lift policy')
 recipe=next(x for x in packets['candidates'] if x['id']=='rigid_lift_10deg_margin20mm')['recipe']
 if recipe['angle_deg']!=10 or recipe['table_capture_m']!=view['table_profile']['object_to_capture_matrix_m']:raise ValueError('Fixed10deg/table profile')
 if any(q[k]!=source_descriptor[k] for k in ('factory_ref','source_context_ref','source_context_sha256','bridge_ref')):raise ValueError('Actual lift request differs from current source-only environment')
 return {'environment_view':view,'prescription':p,'recipe_geometry':{k:deepcopy(recipe[k]) for k in ('angle_deg','garment_pose_capture_m','table_capture_m')}}
def validate_binding(config,context):
 required={'entry_authority_ref','baseline_scene_contract_ref','baseline_source_contract_ref'}
 if not isinstance(config,dict) or not required.issubset(config):raise ValueError('Complete authority and both baseline aliases required before creation')
 q=read(config['request_ref']);p=read(config['parameters_override_ref'])
 if config['entry_authority_ref']!=q.get('entry_authority_ref') or config['entry_authority_ref']!=p.get('provenance',{}).get('entry_source_ref'):raise ValueError('Layer authority must equal request and current parameter provenance')
 if config['baseline_scene_contract_ref']!=config['baseline_source_contract_ref'] or config['baseline_scene_contract_ref']!=q.get('baseline_scene_contract_ref'):raise ValueError('Both layer baseline aliases must equal the request geometry baseline')
 bundle=prepare(config)
 if config['spec_review']!=GEOMETRY_SPEC or read(GEOMETRY_SPEC)['reviewed_interface_ref']!=interface.FIXED:raise ValueError('Exact independently reviewed original continuation geometry')
 if config['interface']!=router.REVIEWED['interface'] or config['fixed_interface']!=interface.FIXED or config['geometry_checks']!=interface.GEOMETRY_GUARDS or config['selection_handoff']!=interface.SELECTION:raise ValueError('Exact current derived binding/geometry modules')
 if q.get('bank_id') not in PRESETS or q.get('bank_id')!=config['bank_id'] or q.get('geometry_bank_id')!=config['geometry_bank_id'] or q.get('parameters_ref')!=config['parameters_override_ref']:raise ValueError('Explicit current preset/geometry/parameters binding')
 reader=interface._ReadSet();validated=interface._validate(reader,config['request_ref'],p,context);reader.close()
 current=validated[1];loaded,_,_=controls.load_config(q['config_ref']['path'],context);bank=loaded['banks'][q['bank_id']]
 if current['target_presence']!=[True,True] or current['entry']['scene_family']!='cloth_lift':raise ValueError('Exact dual cloth_lift current entry')
 if bank['derived_preset']['angle_deg']!=10 or bank['geometry_bank_id']!=q['geometry_bank_id']:raise ValueError('Fixed derived10deg preset')
 for side in ('left','right'):
  if not np.allclose(current['actor_pose_capture_by_side'][side]['brace_B_capture'],bank['B_capture_by_side'][side],atol=1e-12,rtol=0):raise ValueError('Actual parent B must equal current derived anchor')
 for key in ('candidate_refs','baseline_scene_contract_ref'):
  target='candidates' if key=='candidate_refs' else key
  if config[target]!=q[key]:raise ValueError('Geometry-bank selection differs from request')
 if bank['derived_preset']['table_profile_ref']!=bundle['environment_view']['historical_profile_ref']:raise ValueError('Preset/table profile identity')
 if bank['derived_preset']['garment_geometry_ref']!=bundle['prescription']['sources']['v002_garment_10']:raise ValueError('Exact full10deg panel/seam cache')
 return {'current_parameters':current,'bank':bank,'geometry_binding':validated[-1],**bundle}
def frame_from_angle(angle,table_capture,config):
 p=prepare(config)['prescription']
 if type(angle) not in (float,int) or not math.isfinite(angle) or angle!=10:raise ValueError('Only reviewed discrete10deg')
 a=math.radians(angle);R=np.eye(4);R[:3,:3]=[[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]];P=np.eye(4);P[:3,3]=p['pivot_garment_local_m'];placement=np.eye(4);placement[:3,3]=p['source_garment_placement_table_local_m'];return assembly.rigid(table_capture)@placement@P@R@np.linalg.inv(P)
def propose(parent,config):
 checked=validate_binding(config,parent.session.context);parent.verifier(parent.scene)
 if parent.session.parameters!=checked['current_parameters']:raise ValueError('Actual parent entry/parameters differ from exact bound current request')
 p=background.propose(parent,'table_garment',check(config['garment_config_ref']))
 pose=frame_from_angle(10,checked['recipe_geometry']['table_capture_m'],config)
 if np.max(abs(pose-np.asarray(checked['recipe_geometry']['garment_pose_capture_m'])))>1e-12:raise ValueError('Rigid frame differs from unchanged10deg geometry')
 p.update(schema='action4.current_derived_rigid_lift_proposal.v1',pose_capture_m=pose.tolist(),rigid_lift_binding={'preset_bank_id':config['bank_id'],'geometry_bank_id':config['geometry_bank_id'],'parameters_ref':config['parameters_override_ref'],'request_ref':config['request_ref'],'garment_geometry_ref':checked['bank']['derived_preset']['garment_geometry_ref'],'table_capture_m':checked['recipe_geometry']['table_capture_m'],'angle_deg':10,'legacy_actor_recipe_used':False,'full20_acceptance_pending':True},foreground_occlusion_verified=False)
 return refresh_lifted_outline_metadata(p)
def verify_exact_garment_cache(scene,contract,cache_ref):
 cache=np.load(check(cache_ref),allow_pickle=False);rows={}
 for role,key in [('garment_panel','panel_name'),('garment_seam','seam_name')]:
  V,F=assembly.world_geometry(scene.objects[contract[key]])
  U=cache[role+'_vertices_world_m'];G=cache[role+'_triangles']
  if V.shape!=U.shape or not np.array_equal(F,G) or np.max(abs(V-U))>2e-6:raise ValueError('Actual full garment geometry differs from unchanged10deg cache')
  rows[role]={'vertices':len(V),'triangles':len(F),'world_vertex_error_max_m':float(np.max(abs(V-U)))}
 return rows
class CurrentRigidGarmentLift(background.GarmentBackground):
 def __init__(self,parent,proposal,config):super().__init__(parent,proposal,check(config['garment_config_ref']));self.layer_config=deepcopy(config)
 def verifier(self,scene):
  prepare(self.layer_config)
  if self.session.parameters!=read(self.proposal['rigid_lift_binding']['parameters_ref']):raise ValueError('Live garment parent parameters drifted from current binding')
  result=super().verifier(scene)
  if self.contract['schema']!='action4.garment_background.contract.v1' or self.contract.get('rigid_lift_kind')!='current_derived_10deg' or self.contract['foreground_occlusion_verified'] is not False:raise ValueError('Explicit rigid lift generic garment owner contract')
  return {**result,'status':'PASS_CURRENT_RIGID_LIFT_GARMENT_CONTRACT_ONLY'}
 def geometry_diagnostics(self):
  report=super().geometry_diagnostics()
  report['unchanged_full_garment_cache']=verify_exact_garment_cache(self.scene,self.contract,self.proposal['rigid_lift_binding']['garment_geometry_ref'])
  report.update(status='ACTUAL_PRECONTINUATION_LIFT_GARMENT_ONLY',full20_acceptance_pending=True,foreground_occlusion_verified=False)
  return report

def create_source_geometry(name,material):
 """Only own original fullpanel/seam. Caller owns exception-safe cleanup."""
 import bpy
 panel=background.synth._garment_panel(name,[tuple(x) for x in background.OUTLINE],material,0.,0.);seam=bpy.data.objects[name+'_boundary_stitch'];old=seam.data.splines[0];points=[p.co.copy() for p in old.bezier_points]
 if not np.array_equal(points[0],points[-1]):raise ValueError('Original duplicate endpoint')
 seam.data.splines.remove(old);sp=seam.data.splines.new('BEZIER');sp.bezier_points.add(len(points)-2);sp.use_cyclic_u=True;seam.data.use_fill_caps=False
 for v,co in zip(sp.bezier_points,points[:-1]):v.co=co;v.handle_left_type=v.handle_right_type='AUTO'
 background.cloth._update();return panel,seam

def refresh_lifted_outline_metadata(proposal):
 """Project the unchanged coarse source outline using this explicit lifted pose.
 This is descriptive geometry, never a visible mask or evaluated solidified mesh.
 """
 xy=np.asarray(proposal['source_outline_m'],dtype=float)
 z=.005*np.sin(30*xy[:,0])*np.cos(24*xy[:,1])+.003*np.sin(60*xy[:,1])
 points=assembly.transform(assembly.rigid(proposal['pose_capture_m']),np.c_[xy,z])
 if not np.isfinite(points).all() or points[:,2].min()-.012<=.001:raise ValueError('Lifted garment finite nearplane bound')
 uv=assembly.project(points,np.asarray(proposal['K_native'],dtype=float))
 proposal['projected_outline_native_px']=uv.tolist()
 proposal['outline_bbox_native_xyxy']=np.r_[uv.min(0),uv.max(0)].tolist()
 proposal['outline_projection_semantics']='Current lifted pose of coarse source outline/folds; not evaluated Solidify/seam geometry or visible mask'
 return proposal

def materialize(parent,proposal,config):
 import bpy
 from mathutils import Matrix
 p=propose(parent,config)
 if env.digest(p)!=env.digest(proposal):raise ValueError('Exact explicit new rigid frame proposal required')
 background.require_table_coupling(p);c=background.prepare(check(config['garment_config_ref']));scene=parent.scene;name='wearing_garment_table_garment'
 if scene!=bpy.context.scene or any(n in bpy.data.objects for n in [name,name+'_frame',name+'_boundary_stitch']):raise ValueError('Fresh own names and actual scene required')
 layer=CurrentRigidGarmentLift(parent,p,config)
 try:
  material=background._material(name+'_denim',p,c);panel,seam=create_source_geometry(name,material);seam.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*p['stitch_rgb'],1.);frame=bpy.data.objects.new(name+'_frame',None);scene.collection.objects.link(frame);world=np.asarray(p['W_from_capture'])@p['pose_capture_m'];assembly.set_rigid_basis(frame,world)
  for o in (panel,seam):o.parent=frame;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_basis=Matrix.Identity(4)
  for o in (frame,panel,seam):o.pass_index=0;o.hide_render=False;o.hide_viewport=False;o.visible_camera=True
  background.cloth._update();layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before}
  if {k:len(v) for k,v in layer.owned.items()}!={'objects':3,'meshes':1,'curves':1,'materials':2,'images':3}:raise ValueError('Exact own garmentframe/panel/seam/images')
  objects={o.name:{'type':o.type,'parent':o.parent.name if o.parent else None,'basis':world.tolist() if o==frame else np.eye(4).tolist(),'world':world.tolist(),'data_sha256':env.digest(background._new_state(o))} for o in (frame,panel,seam)}
  layer.contract={'schema':'action4.garment_background.contract.v1','rigid_lift_kind':'current_derived_10deg','proposal':p,'proposal_sha256':env.digest(p),'objects':objects,'objects_sha256':env.digest(objects),'panel_name':panel.name,'seam_name':seam.name,'source_footprint_m':[.9,.5],'new_owned_images_only':True,'no_panel_seam_geometry_deformation':True,'foreground_occlusion_verified':False,'environment_background_complete':False,'family_effect_render_verified':False,'dataset_generation_allowed':False,'training_accepted':False};layer.verifier(scene);layer.contract['geometry']=layer.geometry_diagnostics();return layer
 except Exception as exc:
  layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before}
  try:layer.remove()
  except Exception as restoration:raise RuntimeError('Rigidlift creation failed: '+repr(exc)+'; restoration='+repr(restoration)) from exc
  raise
