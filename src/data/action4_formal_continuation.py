"""Two owned continuation meshes over frozen actual palmar-wearing sources.
No action on import. No render/GPU. Creation is explicit and transactional.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import importlib.util,hashlib,json
import numpy as np
from data import action4_modular_scene_sampling as sampling
from data import action4_modular_assembly as assembly
from data import action4_modular_bilateral_scene as bilateral
from data import action4_modular_render as camera
from data import action4_modular_cloth_effects as cloth
from data import action4_global_environment as env
ROOT=Path('/data/zy/vln/a-test')

def check(ref):
 if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or Path(ref['path']).is_absolute() or '..' in Path(ref['path']).parts:raise ValueError('Exact relative ref required')
 p=bilateral.bounded(ref['path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Continuation source identity changed: '+ref['path'])
 return p

def load(ref,name):
 p=check(ref);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate_selection(config):
 from data import action4_formal_composition as router
 return router.validate_layer_config(config)

def ids():
 import bpy
 return {k:set(getattr(bpy.data,k)) for k in ('objects','meshes','materials','images')}

def remove_owned(owned):
 import bpy
 for key in ('objects','meshes','materials','images'):
  for obj in owned.get(key,set()):
   if key=='objects':getattr(bpy.data,key).remove(obj,do_unlink=True)
   elif obj.users==0:getattr(bpy.data,key).remove(obj)
   else:raise RuntimeError('Owned continuation data still used '+obj.name)

def source_snapshot(scene,excluded=()):return cloth._source_snapshot(scene,excluded)
def same(a,b):
 if env.digest(a)!=env.digest(b):raise ValueError('Frozen source snapshot changed')

def skin_locals(obj):
 import bpy
 e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=e.to_mesh()
 try:
  mesh.calc_loop_triangles();V=np.array([v.co[:] for v in mesh.vertices]);F=np.array([t.vertices[:] for t in mesh.loop_triangles],int)
 finally:e.to_mesh_clear()
 return V,F

class ContinuationCollisionError(ValueError):
 def __init__(self,report):self.report=report;super().__init__('Continuation unresolved actual geometry: '+str(report['unresolved_count']))

class ContinuationLayer:
 def __init__(self,parent,config):
  if parent.contract['schema']!='action4.modular_dynamic_scene.v1' or parent.context.get('base_kind')!='palmar_wearing':raise ValueError('One dynamic palmar wearing source required')
  validate_selection(config);parent.verifier(parent.scene);self.parent=parent;self.scene=parent.scene;self.config=deepcopy(config);self.config_digest=env.digest(self.config);self.parent_contract_digest=env.digest(parent.contract);self.source=source_snapshot(self.scene);self.before=ids();self.owned={};self.removed=False;self.contract=None;self.expected={};self.names=[]
  self.interface=load(config['interface'],'parametric_continuation_interface_bound');self.fixed=load(config['fixed_interface'],'continuation_fixed_geometry_bound');self.math=load(config['geometry_checks'],'continuation_geometry_checks_bound');self.presence=load(config['actual_presence_ref'],'actual_presence_scene_guard')
 def targets(self):return self.parent.targets()
 def source_guard(self):
  if env.digest(self.config)!=self.config_digest:raise ValueError('Layer config drift')
  validate_selection(self.config)
  sampling.verify_sources(self.parent.context)
  for r in self.config['dependencies']:check(r)
  if env.digest(self.parent.contract)!=self.parent_contract_digest:raise ValueError('Parent contract mutated')
  same(source_snapshot(self.scene,self.names),self.source)
 def verifier(self,scene):
  if scene!=self.scene or self.removed or self.contract is None:raise ValueError('Active continuation layer required')
  self.source_guard()
  if set(self.names)!=set(self.contract['objects']) or env.digest(self.expected)!=self.contract['objects_sha256'] or env.digest(self.contract['objects'])!=self.contract['objects_sha256']:raise ValueError('Layer expected contract changed')
  for name,row in self.expected.items():
   o=scene.objects[name];bilateral._check_static(o)
   if o.type!='MESH' or o.parent is None or o.parent.name!=row['parent'] or o.pass_index!=0 or not sampling.pixels._camera_render_enabled(scene,o) or tuple(o.scale)!=(1.,1.,1.) or not np.array_equal(o.matrix_basis,np.eye(4)) or not np.array_equal(o.matrix_parent_inverse,np.eye(4)) or np.max(abs(np.array(o.matrix_world)-row['world']))>2e-6:raise ValueError('Independent continuation role/transform changed')
   if env.digest(assembly.mesh_state(o))!=row['mesh_sha256']:raise ValueError('Continuation mesh/material/normals/texture-space changed')
   V,F=assembly.world_geometry(o)
   if len(V)!=row['vertex_count'] or not np.array_equal(F,np.array(row['triangles'])):raise ValueError('Actual continuation topology changed')
   skin=scene.objects[row['parent']];U,G=assembly.world_geometry(skin);seam_error=float(np.max(abs(V[:44]-U[row['source_rim_ids']])))
   if seam_error>1e-7:raise ValueError('Actual world rim no longer coincides with unchanged skin')
   self.math.terminal_guard(assembly.transform(np.linalg.inv(np.array(self.parent.parameters['W_from_capture'])),V),camera.actual_K(scene),scene.camera.data.clip_start)
  self.presence.inventory(self.parent,self.names)
  sampling.pixels._validate_targets(scene,self.parent.parameters['entry'],self.targets())
  return {'status':'PASS_CONTINUATION_SOURCE_AND_OBJECT_CONTRACT_ONLY','source_unchanged':True,'training_accepted':False}
 def geometry_diagnostics(self):
  from mathutils.bvhtree import BVHTree
  self.verifier(self.scene);names=self.names;original=self.parent.contract['actor_collision_diagnostics']['retained_actor_surface_names'];actors=original['left']+original['right']
  observed=self.presence.inventory(self.parent,names)
  if original!=observed['retained_actor_surface_names'] and any(set(original[s])!=set(observed['retained_actor_surface_names'][s]) for s in ('left','right')):raise ValueError('Actual retained actor object inventory changed')
  others=actors+[camera.ENV_NAMES[0],camera.ENV_NAMES[1]];data={n:assembly.world_geometry(self.scene.objects[n]) for n in set(names+others)};trees={};rows={};pairs=[(a,b) for a in names for b in others]+[(names[0],names[1])];unresolved=0
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
   if b in camera.ENV_NAMES[:2]:
    obj=self.scene.objects[b];local=np.array([v.co[:] for v in obj.data.vertices]);q=assembly.transform(np.linalg.inv(np.array(obj.matrix_world)),V);inside=np.flatnonzero(np.all((q>local.min(0)+1e-9)&(q<local.max(0)-1e-9),axis=1)).tolist()
   rows[a+'__'+b]={**sep,'BVH_candidate_count':len(raw),'SAT_proved_separated':separated,'exact_rim_witness_contacts':seams,'outside_or_unresolved_pairs':nonsep,'environment_inside_vertices':inside};unresolved+=len(nonsep)+len(inside)
  selfrows={}
  for name in names:
   V,F=data[name];tree=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True,epsilon=0);raw=[(i,j) for i,j in tree.overlap(tree) if i<j and not set(F[i])&set(F[j])];kept=[list(p) for p in raw if not self.math.triangle_separated(V[F[p[0]]],V[F[p[1]]])];selfrows[name]={'nonadjacent_BVH_candidates':len(raw),'outside_or_unresolved_pairs':kept,'adjacent_faces_follow_exact_verified_manifold_topology':True};unresolved+=len(kept)
  presence_inventory=self.presence.validate_bare_pairs(self.parent,names,pairs)
  report={'status':'ACTUAL_CONTINUATION_GEOMETRY_ONLY','pair_count':len(pairs),'original_actor_piece_count':len(actors),'pairs':rows,'self_pairs':selfrows,'unresolved_count':unresolved,'original_presence_pairs_preserved_by_source_snapshot':True,'presence_pair_inventory':presence_inventory,'seam_policy':'only actual intersection witnesses on exact original44-vertex rim; no skin-pair exemption','whole_actor_containment_accepted':False,'new_whitelist_count':0,'training_accepted':False}
  if unresolved:raise ContinuationCollisionError(report)
  return report
 def remove(self):
  if self.removed:raise ValueError('One continuation removal only')
  errors=[]
  try:remove_owned(self.owned)
  except Exception as exc:errors.append('owned cleanup: '+str(exc))
  cloth._update()
  try:same(source_snapshot(self.scene),self.source);self.parent.verifier(self.scene)
  except Exception as exc:errors.append('parent restore: '+str(exc))
  if ids()!=self.before:errors.append('original bpy object/mesh/material/image inventories not exactly restored')
  if errors:raise RuntimeError('Continuation cleanup failed: '+str(errors))
  self.removed=True;return {'status':'REMOVED_CONTINUATION_ORIGINAL_PARENT_VERIFIED','source_and_datablocks_restored':True,'training_accepted':False}

def create(parent,config):
 import bpy
 from mathutils import Matrix
 layer=ContinuationLayer(parent,config)
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
  layer.contract={'schema':'action4.parametric_presence_continuation_layer.v1','request_ref':config['request_ref'],'entry_authority_ref':config['entry_authority_ref'],'parametric_binding_proof':pair['proof'],'parameters_override_ref':config['parameters_override_ref'],'baseline_source_contract_ref':config['baseline_source_contract_ref'],'current_factory_ref':config['current_factory_ref'],'current_config_ref':config['current_config_ref'],'parent_contract_sha256':layer.parent_contract_digest,'objects':deepcopy(layer.expected),'objects_sha256':env.digest(layer.expected),'canonical_definition':deepcopy(parent.contract['canonical_definition']),'source_snapshot_sha256':env.digest(layer.source),'dataset_generation_allowed':False,'training_accepted':False}
  layer.verifier(layer.scene);layer.contract['geometry_diagnostics']=layer.geometry_diagnostics();return layer
 except Exception as exc:
  layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before}
  try:layer.remove()
  except Exception as restoration:raise RuntimeError('Continuation create failed: '+repr(exc)+'; cleanup='+repr(restoration)) from exc
  raise
