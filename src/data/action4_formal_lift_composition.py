"""base -> exact table -> complete garment -> continuation, explicit ownership."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import importlib.util,json
import numpy as np
from data import action4_modular_scene_sampling as sampling
from data import action4_modular_assembly as assembly
from data import action4_modular_cloth_effects as cloth
from data import action4_modular_render as camera
from data import action4_modular_bilateral_scene as bilateral
from data import action4_global_environment as env
ROOT=Path('/data/zy/vln/a-test')
from data import action4_formal_composition as router
from data import action4_formal_lift as lift
base_cont=router.load_fixed('dual_base','formal_lift_base_continuation')
registry_module=router.load_fixed('registry','formal_lift_ownership')
check=base_cont.check;ids=base_cont.ids;remove_owned=base_cont.remove_owned;skin_locals=base_cont.skin_locals
ContinuationCollisionError=base_cont.ContinuationCollisionError
def ref(path):
 import hashlib
 p=bilateral.bounded(path);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load_ref(ref,name):return base_cont.load(ref,name)
def validate_selection(config,context):
 checked=lift.validate_binding(config,context)
 router.validate_layer_config(config)
 if config['composition_module_ref']!=ref(__file__) or config['lift_module_ref']!=ref(lift.__file__):raise ValueError('Exact new composition/lift implementation binding')
 if config['environment_profile']!='cloth_lift10_full_garment' or config['angle_deg']!=10:raise ValueError('Only declared current derived full10deg lift')
 for r in config['dependencies']:check(r)
 return checked

def full20_halfspace(actor_vertices,expected_actor_names,garment_parts,table_vertices,wall_vertices,table_capture,near_plane):
 names=list(expected_actor_names)
 if len(names)!=20 or len(set(names))!=20 or set(actor_vertices)!=set(names):raise ValueError('Exactly original18 plus both continuation objects required')
 if set(garment_parts)!={'panel','seam'}:raise ValueError('Complete original garment panel and seam required')
 T=assembly.rigid(table_capture);n=T[:3,2]
 def finite(v):
  V=np.asarray(v,float)
  if V.ndim!=2 or V.shape[1]!=3 or not len(V) or not np.isfinite(V).all():raise ValueError('Complete finite metric geometry')
  return V
 actors={k:finite(v) for k,v in actor_vertices.items()};garment=np.concatenate([finite(garment_parts[k]) for k in ('panel','seam')]);table=finite(table_vertices);wall=finite(wall_vertices)
 rows={name:float(np.min((V-T[:3,3])@n)-.0125) for name,V in actors.items()};height=(garment-T[:3,3])@n-.0125
 allV=np.concatenate([*actors.values(),garment,table]);gap=float(wall[:,2].min()-allV[:,2].max());margin=min(rows.values())-float(height.max())
 if min(rows.values())<.032-2e-6 or height.min()<=0 or margin<.020-2e-6 or gap<=0:raise ValueError('Complete20 actual10deg lift separation failed: margin_m='+str(margin)+', min_top_m='+str(min(rows.values()))+', wall_gap_m='+str(gap))
 if not np.isfinite(near_plane) or near_plane<=0 or allV[:,2].min()<=near_plane:raise ValueError('All20 and environment nearplane failure')
 return {'all20_actor_and_continuation_top_gaps_m':rows,'full_actor_piece_count':20,'minimum_top_gap_m':min(rows.values()),'garment_bottom_above_table_m':float(height.min()),'garment_top_above_table_m':float(height.max()),'actor_garment_plane_margin_m':margin,'required_nominal_margin_m':.020,'actual_float32_numeric_budget_m':2e-6,'all20_table_garment_wall_depth_gap_m':gap,'minimum_all_capture_depth_m':float(allV[:,2].min()),'triangle_and_convex_hull_bound':True,'new_table_garment_containment_excluded_by_strict_disjoint_halfspaces':True,'original_open_actor_general_containment_accepted':False,'actual_visible_mask_measured':False}

def full_inventory():
 import bpy
 return {k:set(getattr(bpy.data,k)) for k in ('objects','meshes','curves','materials','images')}

class EnvironmentParent:
 def __init__(self,session,table,full,registry,environment):
  self.session=session;self.scene=session.scene;self.context=session.context;self.parameters=session.parameters;self.table=table;self.full=full;self.registry=registry;self.environment=environment
  self.contract={'schema':'action4.owned_table_full_garment_parent.v1','source_contract_sha256':env.digest(session.contract),'actor_collision_diagnostics':deepcopy(session.contract['actor_collision_diagnostics']),'canonical_definition':deepcopy(session.contract['canonical_definition']),'table_contract_sha256':env.digest(table.contract),'garment_contract_sha256':env.digest(full.contract),'parent_owners':registry.manifest()}
 def targets(self):return self.session.targets()
 def verifier(self,scene):
  if scene!=self.scene:raise ValueError('Same composition scene')
  self.registry.verify();self.full.verifier(scene)
  return {'status':'PASS_LEGAL_TABLE_GARMENT_PARENT_PHASE'}
 def complete_halfspace_guard(self,data):
  observed=lift.observer.inventory(self.session,list(registry_module.EXPECTED['continuation']))
  if observed['requirements']['original_total_pair_count']!=153:raise ValueError('Original actual153 named pairs required')
  inverse=np.linalg.inv(np.array(self.parameters['W_from_capture']));table_name=self.table.contract['table_name'];T=inverse@np.array(self.scene.objects[table_name].matrix_world)
  actor_names=self.session.contract['actor_collision_diagnostics']['retained_actor_surface_names'];names=actor_names['left']+actor_names['right']+list(registry_module.EXPECTED['continuation'])
  actors={name:assembly.transform(inverse,data[name][0]) for name in names}
  garment={key:assembly.transform(inverse,data[self.full.contract[key+'_name']][0]) for key in ('panel','seam')}
  return full20_halfspace(actors,names,garment,assembly.transform(inverse,data[table_name][0]),assembly.transform(inverse,data[camera.ENV_NAMES[0]][0]),T,self.scene.camera.data.clip_start)

class ComposedContinuation(base_cont.ContinuationLayer):
 def __init__(self,parent,config):
  if parent.contract['schema']!='action4.owned_table_full_garment_parent.v1':raise ValueError('Exact full environment parent')
  validate_selection(config,parent.context);parent.verifier(parent.scene);self.parent=parent;self.scene=parent.scene;self.config=deepcopy(config);self.config_digest=env.digest(self.config);self.parent_contract_digest=env.digest(parent.contract);self.source=deepcopy(parent.registry.expected_source);self.before=ids();self.owned={};self.removed=False;self.contract=None;self.expected={};self.names=[]
  self.interface=load_ref(config['interface'],'composition_payload_interface');self.fixed=load_ref(config['fixed_interface'],'composition_fixed_geometry');self.math=load_ref(config['geometry_checks'],'composition_geometry_checks');parent.registry.begin_continuation(self)
 def source_guard(self):
  if env.digest(self.config)!=self.config_digest or env.digest(self.parent.contract)!=self.parent_contract_digest:raise ValueError('Bound composition parent/config drift')
  checked=validate_selection(self.config,self.parent.context)
  if self.parent.parameters!=checked['current_parameters']:raise ValueError('Live parent entry/preset parameters drifted')
  sampling.verify_sources(self.parent.context);self.parent.registry.verify_pending_continuation(self)
 def remove(self):
  if self.removed:raise ValueError('One composed continuation removal')
  errors=[]
  try:remove_owned(self.owned)
  except Exception as exc:errors.append('owned continuation removal: '+str(exc))
  cloth._update()
  try:
   registry=self.parent.registry
   if 'continuation' in registry.owners:registry.unregister_removed('continuation',self)
   else:registry.pending_child=None;registry.verify()
   self.parent.full.verifier(self.scene)
  except Exception as exc:errors.append('restored full garment parent: '+str(exc))
  if ids()!=self.before:errors.append('Strict precontinuation inventories not restored')
  if errors:raise RuntimeError('Composed continuation cleanup failed: '+str(errors))
  self.removed=True;return {'status':'REMOVED_CONTINUATION_FULL_GARMENT_PARENT_VERIFIED'}

 def geometry_diagnostics(self):
  from mathutils.bvhtree import BVHTree
  self.verifier(self.scene);names=self.names;original=self.parent.contract['actor_collision_diagnostics']['retained_actor_surface_names'];actors=original['left']+original['right']
  if len(actors)!=18 or len(set(actors))!=18:raise ValueError('Exact original18 actor pieces required')
  others=actors+[camera.ENV_NAMES[0],self.parent.table.contract['table_name'],self.parent.full.contract['panel_name'],self.parent.full.contract['seam_name']];data={n:assembly.world_geometry(self.scene.objects[n]) for n in set(names+others)};trees={};rows={};pairs=[(a,b) for a in names for b in others]+[(names[0],names[1])];unresolved=0
  for a,b in pairs:
   V,F=data[a];U,G=data[b];sep=bilateral.aabb_separation(V,U);raw=[];nonsep=[];separated=[];seams=[];inside=[]
   if not sep['strictly_disjoint']:
    for n in (a,b):
     if n not in trees:trees[n]=BVHTree.FromPolygons(data[n][0].tolist(),data[n][1].tolist(),all_triangles=True,epsilon=0)
    raw=trees[a].overlap(trees[b])
    for i,j in raw:
     if self.math.triangle_separated(V[F[i]],U[G[j]]):separated.append([int(i),int(j)]);continue
     witness=None
     if b==self.expected[a]['parent']:
      rim=self.expected[a]['source_rim_ids'];shared={rim[int(k)] for k in F[i] if k<44}&set(map(int,G[j]))
      if len(shared)==2 and list(sorted(shared)) not in self.expected[a]['source_boundary_edges']:witness={'proved':False,'reason':'shared_pair_not_actual_boundary_edge'}
      elif shared:witness=self.math.seam_witness(V[F[i]],U[G[j]],U[sorted(shared)])
     if witness is not None and witness['proved']:seams.append({'pair':[int(i),int(j)],**witness})
     else:nonsep.append({'pair':[int(i),int(j)],'seam_witness':witness})
   if b in (camera.ENV_NAMES[0],self.parent.table.contract['table_name']):
    obj=self.scene.objects[b];local=np.array([v.co[:] for v in obj.data.vertices]);q=assembly.transform(np.linalg.inv(np.array(obj.matrix_world)),V);inside=np.flatnonzero(np.all((q>local.min(0)+1e-9)&(q<local.max(0)-1e-9),axis=1)).tolist()
   rows[a+'__'+b]={**sep,'BVH_candidate_count':len(raw),'SAT_proved_separated':separated,'exact_rim_witness_contacts':seams,'outside_or_unresolved_pairs':nonsep,'environment_inside_vertices':inside};unresolved+=len(nonsep)+len(inside)
  selfrows={}
  for name in names:
   V,F=data[name];tree=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True,epsilon=0);raw=[(i,j) for i,j in tree.overlap(tree) if i<j and not set(F[i])&set(F[j])];kept=[list(p) for p in raw if not self.math.triangle_separated(V[F[p[0]]],V[F[p[1]]])];selfrows[name]={'nonadjacent_BVH_candidates':len(raw),'outside_or_unresolved_pairs':kept,'adjacent_faces_follow_exact_verified_manifold_topology':True};unresolved+=len(kept)
  report={'status':'ACTUAL_CONTINUATION_GEOMETRY_ONLY','pair_count':len(pairs),'original_actor_piece_count':len(actors),'pairs':rows,'self_pairs':selfrows,'unresolved_count':unresolved,'original153_preserved_by_source_snapshot':True,'seam_policy':'only actual intersection witnesses on exact original44-vertex rim; no skin-pair exemption','whole_actor_containment_accepted':False,'new_whitelist_count':0,'training_accepted':False}
  report['composition_halfspace_and_containment']=self.parent.complete_halfspace_guard(data)
  if report['pair_count']!=45:raise ValueError('Complete45 extension-to-actor/environment/garment pairs required')
  if unresolved:raise ContinuationCollisionError(report)
  return report

def materialize_continuation(parent,config):
 import bpy
 from mathutils import Matrix
 layer=ComposedContinuation(parent,config)
 try:
  actual_skins={};skin_rows={}
  for side in ('left','right'):
   meta=json.loads(check(config['candidates'][side]).read_text());skin=layer.scene.objects[meta['source_skin_object']];V,G=skin_locals(skin);state=assembly.mesh_state(skin)
   actual_skins[side]={'vertices_local_m':V,'matrix_world':np.array(skin.matrix_world),'mesh_geometry_uv_sha256':state['geometry_uv_sha256']};skin_rows[side]=(meta,skin,V,G)
  pair=layer.interface.payloads_for_verified_skins(config['request_ref'],actual_skins,parameters=parent.parameters,context=parent.context)
  for side in ('left','right'):
   meta,skin,V,G=skin_rows[side];payload=pair['payloads'][side];name=payload['name']
   if name in bpy.data.objects:raise FileExistsError('Continuation name already exists')
   if len(skin.data.vertices)!=len(V) or np.max(abs(V-np.array([v.co[:] for v in skin.data.vertices])))>1e-8:raise ValueError('Source evaluated/local vertex identity required for normal binding')
   mesh=bpy.data.meshes.new(name+'_mesh');mesh.from_pydata(payload['vertices'].tolist(),[],payload['triangles'].tolist());mesh.update();obj=bpy.data.objects.new(name,mesh);layer.scene.collection.objects.link(obj);obj.parent=skin;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4);obj.pass_index=0;obj.hide_render=False;obj.hide_viewport=False;obj.visible_camera=True;layer.names.append(name)
   if len(skin.data.materials)!=1:raise ValueError('Exactly one source skin material required')
   mesh.materials.append(skin.data.materials[0]);mesh.use_auto_texspace=False;mesh.texspace_location=skin.data.texspace_location;mesh.texspace_size=skin.data.texspace_size
   for poly in mesh.polygons:poly.use_smooth=True
   skin.data.calc_normals_split();rim=layer.fixed.load_candidate(meta['candidate_id'])[1]['source_boundary_vertex_ids'];lookup={int(v):i for i,v in enumerate(rim)};normals={i:[] for i in range(44)}
   for loop in skin.data.loops:
    if loop.vertex_index in lookup:normals[lookup[loop.vertex_index]].append(np.array(loop.normal))
   counts=Counter(tuple(sorted((int(a),int(b)))) for f in G for a,b in zip(f,np.roll(f,-1)));boundary_edges=[list(e) for e,n in counts.items() if n==1]
   if len(boundary_edges)!=44 or set(rim)!=set(v for e in boundary_edges for v in e) or max(counts.values())!=2:raise ValueError('Exact actual source44 open boundary required')
   source_normals={}
   for i,values in normals.items():
    if not values:raise ValueError('Missing source rim split normal')
    avg=np.mean(values,axis=0);avg/=np.linalg.norm(avg)
    if max(np.linalg.norm(v-avg) for v in values)>1e-4:raise ValueError('Source rim has split normals requiring per-loop mapping')
    source_normals[i]=avg
   mesh.use_auto_smooth=True;mesh.calc_normals_split();custom=[source_normals[l.vertex_index].tolist() if l.vertex_index<44 else list(l.normal) for l in mesh.loops];mesh.normals_split_custom_set(custom);mesh.update();cloth._update()
   actualV,actualF=assembly.world_geometry(obj)
   if not np.array_equal(actualF,payload['triangles']):raise ValueError('Blender triangulation changes candidate face order')
   U,_=assembly.world_geometry(skin);error=float(np.max(abs(actualV[:44]-U[rim])))
   if error>1e-7:raise ValueError('Created exact rim world discrepancy')
   mesh.calc_normals_split();normalerror=max(np.linalg.norm(np.array(l.normal)-source_normals[l.vertex_index]) for l in mesh.loops if l.vertex_index<44)
   if normalerror>2e-4:raise ValueError('Created seam split normal mismatch')
   layer.expected[name]={'parent':skin.name,'world':np.array(obj.matrix_world).tolist(),'mesh_sha256':env.digest(assembly.mesh_state(obj)),'vertex_count':len(actualV),'triangles':actualF.tolist(),'source_rim_ids':rim.tolist(),'source_boundary_edges':boundary_edges,'seam_world_error_m':error,'seam_normal_max_error':float(normalerror),'seam_roundoff_correction_max_m':payload['seam_roundoff_correction_max_m'],'candidate_ref':config['candidates'][side],'typed_pose_proof':payload['typed_pose_proof'],'material_shared_original_unchanged':True,'generated_coordinate_texspace_copied_from_source':True}
  layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before}
  if {k:len(v) for k,v in layer.owned.items()}!={'objects':2,'meshes':2,'materials':0,'images':0}:raise ValueError('Exactly2 added meshes/objects and no material/image mutation')
  layer.contract={'schema':'action4.composed_continuation.contract.v1','request_ref':config['request_ref'],'entry_authority_ref':config['entry_authority_ref'],'parametric_binding_proof':pair['proof'],'parameters_override_ref':config['parameters_override_ref'],'baseline_source_contract_ref':config['baseline_source_contract_ref'],'parent_contract_sha256':layer.parent_contract_digest,'objects':deepcopy(layer.expected),'objects_sha256':env.digest(layer.expected),'canonical_definition':deepcopy(parent.contract['canonical_definition']),'source_snapshot_sha256':env.digest(layer.source),'dataset_generation_allowed':False,'training_accepted':False}
  layer.verifier(layer.scene);layer.contract['geometry_diagnostics']=layer.geometry_diagnostics();return layer
 except Exception as exc:
  layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before}
  try:layer.remove()
  except Exception as restoration:raise RuntimeError('Continuation create failed: '+repr(exc)+'; cleanup='+repr(restoration)) from exc
  raise

class SceneComposition:
 def __init__(self,session,config):
  self.session=session;self.scene=session.scene;self.context=session.context;self.parameters=session.parameters;self.config=deepcopy(config);self.config_sha=env.digest(config);self.before=full_inventory();self.table=None;self.full=None;self.continuation=None;self.registry=None;self.removed=False;self.contract=None
 def targets(self):return self.session.targets()
 def verifier(self,scene):
  if scene!=self.scene or self.removed or self.continuation is None:raise ValueError('Complete active composition required')
  if env.digest(self.config)!=self.config_sha:raise ValueError('Composition config drift')
  for r in self.config['dependencies']:check(r)
  self.registry.verify();self.continuation.verifier(scene);sampling.pixels._validate_targets(scene,self.parameters['entry'],self.targets())
  return {'status':'PASS_COMPLETE_TABLE_GARMENT_CONTINUATION_OWNERSHIP','composition_kind':'table_rigid_lift10_full_garment_dual_worn_cameras','scene_family_acceptance':False}
 def geometry_diagnostics(self):
  self.verifier(self.scene);return self.continuation.geometry_diagnostics()
 def remove(self):
  if self.removed:raise ValueError('One composition removal')
  receipts={};errors=[]
  for role,layer in (('continuation',self.continuation),('garment',self.full),('table',self.table)):
   if layer is None:continue
   try:
    receipts[role]=layer.remove()
    if role!='continuation' and self.registry is not None and role in self.registry.owners:self.registry.unregister_removed(role,layer)
   except Exception as exc:errors.append({'stage':role,'error':str(exc)})
  try:
   self.session.verifier(self.scene)
   if full_inventory()!=self.before:raise ValueError('Strict base object/mesh/curve/material/image inventories not restored')
  except Exception as exc:errors.append({'stage':'base_source_inventory','error':str(exc)})
  if errors:raise RuntimeError('Composition reverse cleanup failed: '+str(errors))
  self.removed=True;return {'status':'REMOVED_CONTINUATION_GARMENT_TABLE_BASE_VERIFIED','reverse_order':['continuation','garment','table'],'receipts':receipts}

def create(session,config):
 session.verifier(session.scene);validate_selection(config,session.context)
 bound=json.loads(check(config['parameters_override_ref']).read_text())
 if session.parameters!=bound or session.parameters['target_presence']!=[True,True] or session.parameters['entry']['scene_family']!='cloth_lift':raise ValueError('Actual parent must own exact current dual derived cloth_lift parameters')
 composition=SceneComposition(session,config)
 try:
  from data import action4_formal_environment as environment
  environment.prepare(config);composition.registry=registry_module.OwnershipRegistry(session.scene,environment.garment)
  composition.table=environment.materialize_table(session,config);composition.registry.source_table_hidden(composition.table);composition.registry.register('table',composition.table)
  proposal=lift.propose(composition.table,config);composition.full=lift.materialize(composition.table,proposal,config);composition.full.verifier(session.scene);composition.registry.register('garment',composition.full)
  parent=EnvironmentParent(session,composition.table,composition.full,composition.registry,environment);composition.continuation=materialize_continuation(parent,config);composition.registry.register('continuation',composition.continuation)
  composition.contract={'schema':'action4.complete_table_garment_continuation.v1','composition_kind':'table_rigid_lift10_full_garment_dual_worn_cameras','original_entry_preserved':deepcopy(session.parameters['entry']),'source_contract_sha256':env.digest(session.contract),'canonical_definition':deepcopy(session.contract['canonical_definition']),'owner_contracts':{'table':deepcopy(composition.table.contract),'garment':deepcopy(composition.full.contract),'continuation':deepcopy(composition.continuation.contract)},'ownership':composition.registry.manifest(),'actual_geometry':composition.geometry_diagnostics(),'source_context_provider':deepcopy(config['environment_source_ref']),'rigid_lift':{'preset_bank_id':config['bank_id'],'geometry_bank_id':config['geometry_bank_id'],'angle_deg':10,'full20_nominal_margin_m':.020,'not_measured_grasp':True,'foreground_occlusion_verified':False},'dataset_generation_allowed':False,'training_accepted':False}
  composition.verifier(session.scene);return composition
 except Exception as exc:
  try:composition.remove()
  except Exception as restoration:raise RuntimeError('Composition create failed: '+repr(exc)+'; reverse cleanup='+repr(restoration)) from exc
  raise
