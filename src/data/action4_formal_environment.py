"""Prepared table-only adapter. No Blender initialization or render on import.
Ownership: garment.remove -> table.remove -> outer session.restore.
"""
from pathlib import Path
from copy import deepcopy
import importlib.util
import numpy as np
from data import action4_modular_cloth_effects as cloth
from data import action4_modular_scene_sampling as sampling
from data import action4_modular_assembly as assembly
from data import action4_modular_render as neutral
from data import action4_modular_bilateral_scene as bilateral
from data import action4_global_environment as env
ROOT=Path('/data/zy/vln/a-test')
from data import action4_formal_composition as router
effects=router.load_fixed('environment_effects','formal_environment_effects')
garment=router.load_fixed('garment_module_ref','formal_full_garment')
provider=router.load_fixed('environment_provider_module_ref','formal_environment_provider')
def prepare(config):
 for key in ('environment_module','environment_source_ref','environment_binding_ref','garment_config_ref'):router.backend()._ref(config[key])
 if config['environment_module']!={'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':router.synth._hash(__file__)}:raise ValueError('Exact live formal environment consumer')
 view=provider.environment_view(config['environment_source_ref'],router.backend()._ref(config['environment_binding_ref']),__file__,garment.__file__)
 garment.prepare(router.backend()._ref(config['garment_config_ref']))
 return view
class TableOnly:
 def __init__(self,session,config):
  self.config=deepcopy(config);m=prepare(self.config);session.verifier(session.scene)
  if session.context['source_context_sha256']!=m['execution_source_context_sha256']:raise ValueError('Exact actor context required')
  self.session,self.scene=session,session.scene;self.before=cloth._ids();self.original=cloth._source_snapshot(self.scene);self.expected=deepcopy(self.original);self.old_table=self.scene.objects[neutral.ENV_NAMES[1]];self.old_hide=self.old_table.hide_render;self.expected['objects'][self.old_table.name]['hide_render']=True;self.owned=None;self.removed=False;self.contract=None
 def targets(self):return self.session.targets()
 def verifier(self,scene):
  if scene!=self.scene or self.removed:raise ValueError('Active table identity')
  prepare(self.config);sampling.verify_sources(self.session.context)
  if env.digest(self.session.contract)!=self.contract['parent_contract_sha256']:raise ValueError('Parent contract drift')
  effects._verify_source(cloth._source_snapshot(scene,self.contract['objects']),self.expected)
  o=scene.objects[self.contract['table_name']];r=self.contract['objects'][o.name]
  if env.digest(self.contract['objects'])!=self.contract['objects_sha256'] or env.digest(assembly.mesh_state(o))!=r['mesh_sha256'] or np.max(abs(np.asarray(o.matrix_world)-r['world']))>2e-6 or o.parent is not None or tuple(o.scale)!=(1.,1.,1.) or len(o.modifiers) or len(o.constraints) or o.animation_data is not None or o.pass_index!=0 or not sampling.pixels._camera_render_enabled(scene,o):raise ValueError('Table metric geometry/material/pose drift')
  effects.grid_contract(o,.025)
  if np.max(abs(np.ptp(np.asarray([v.co[:] for v in o.data.vertices]),axis=0)-[1.4,.65,.025]))>2e-7:raise ValueError('Table dimensions')
  return {'status':'PASS_TABLE_ONLY_CONTRACT','training_accepted':False}
 def geometry_diagnostics(self):
  self.verifier(self.scene);inverse=np.linalg.inv(self.session.context['source_context']['W_from_capture']);T=inverse@np.asarray(self.scene.objects[self.contract['table_name']].matrix_world);n=T[:3,2];rows={}
  for side in ('left','right'):
   for name in self.session.bindings[side]['names']:
    o=self.scene.objects[name]
    if o.type=='MESH' and sampling.pixels._camera_render_enabled(self.scene,o):
     V,F=assembly.world_geometry(o);V=assembly.transform(inverse,V);rows[name]={'vertices':len(V),'triangles':len(F),'top_plane_lower_bound_m':float(np.min((V-T[:3,3])@n)-.0125)}
  low=min(v['top_plane_lower_bound_m'] for v in rows.values())
  if low<.032-2e-6:raise ValueError('Actual actor/table clearance below declared32mm: '+str(low))
  V,F=assembly.world_geometry(self.scene.objects[self.contract['table_name']]);V=assembly.transform(inverse,V)
  wall=self.scene.objects[neutral.ENV_NAMES[0]];U,G=assembly.world_geometry(wall);U=assembly.transform(inverse,U);gap=float(U[:,2].min()-V[:,2].max())
  if gap<=0 or V[:,2].min()<=self.scene.camera.data.clip_start:raise ValueError('Table wall/nearplane separation')
  return {'status':'ACTUAL_TABLE_FULL_VERTEX_HALFSPACE_ONLY','actor_components':rows,'minimum_actor_table_gap_m':low,'table_wall_depth_gap_m':gap,'triangle_interior_bound':True,'actual_visible_mask_measured':False,'training_accepted':False}
 def remove(self):
  if self.removed:raise ValueError('One removal only')
  cloth._remove_created(self.before,self.owned);self.old_table.hide_render=self.old_hide;cloth._update();effects._verify_source(cloth._source_snapshot(self.scene),self.original);self.session.verifier(self.scene);self.removed=True
  return {'status':'REMOVED_TABLE_PARENT_VERIFIED'}
 def __enter__(self):return self
 def __exit__(self,kind,error,tb):self.remove();return False

def materialize_table(session,config):
 import bpy
 from data import action4_render as base
 m=prepare(config);name='modular_environment_v002_table'
 if name in bpy.data.objects:raise ValueError('Fresh table required')
 layer=TableOnly(session,config)
 try:
  effects.grid_contract(layer.old_table,.025);p=m['table_profile'];o=base._cube(name,[0,0,0],p['dimensions_m'],layer.old_table.data.materials[0]);world=np.asarray(session.context['source_context']['W_from_capture'])@p['object_to_capture_matrix_m'];assembly.set_rigid_basis(o,world);o.pass_index=0;o.hide_render=False;o.visible_camera=True;layer.old_table.hide_render=True;cloth._update();layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before}
  if {k:len(v) for k,v in layer.owned.items()}!={'objects':1,'meshes':1,'materials':0}:raise ValueError('One table only, shared unchanged material')
  objects={name:{'world':world.tolist(),'mesh_sha256':env.digest(assembly.mesh_state(o))}}
  layer.contract={'schema':'action4.wearing_table_cloth.contract.v1','table_only':True,'table_name':name,'objects':objects,'objects_sha256':env.digest(objects),'parent_contract_sha256':env.digest(session.contract),'training_accepted':False,'dataset_generation_allowed':False};layer.verifier(session.scene);layer.contract['geometry']=layer.geometry_diagnostics();return layer
 except Exception:
  layer.owned={k:set(getattr(bpy.data,k))-layer.before[k] for k in layer.before};layer.remove();raise

def materialize_garment(table):
 p=garment.propose(table,'table_garment',router.backend()._ref(table.config['garment_config_ref']));layer=garment.materialize(table,p,router.backend()._ref(table.config['garment_config_ref']))
 try:
  inverse=np.linalg.inv(table.session.context['source_context']['W_from_capture']);T=inverse@np.asarray(table.scene.objects[table.contract['table_name']].matrix_world);parts=[]
  for name in (layer.contract['panel_name'],layer.contract['seam_name']):parts.append(assembly.transform(inverse,assembly.world_geometry(table.scene.objects[name])[0]))
  V=np.concatenate(parts);height=(V-T[:3,3])@T[:3,2]-.0125;actor_gap=table.contract['geometry']['minimum_actor_table_gap_m'];wall=assembly.transform(inverse,assembly.world_geometry(table.scene.objects[neutral.ENV_NAMES[0]])[0]);wall_gap=float(wall[:,2].min()-V[:,2].max())
  if height.min()<=0 or actor_gap-height.max()<.002 or wall_gap<=0:raise ValueError('Actual garment table/actor/wall halfspace margin failed')
  layer.contract['actual_halfspace_checks']={'garment_bottom_above_table_m':float(height.min()),'garment_top_above_table_m':float(height.max()),'actor_garment_plane_margin_m':float(actor_gap-height.max()),'garment_wall_depth_gap_m':wall_gap,'triangle_interior_bound':True,'actual_mask_verified':False};return layer
 except Exception:
  layer.remove();raise
