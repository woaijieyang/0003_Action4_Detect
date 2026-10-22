"""Actual CPU-only v004 current-contributor visibility regressions."""
import bpy,json,sys
from pathlib import Path
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_global_environment as e
O=R/'outputs/action4_realism_reassessment_v001/global_environment_init/fix_v004'
c=json.loads((R/'data/action4_scene_assets/global_environment_init_v001/build_v004/environment_contract.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(R/c['saved_asset']['path']),load_ui=False,use_scripts=False)
scene=bpy.context.scene;bpy.context.view_layer.update();e.verify_environment(scene,c)
results=[]
def check(owner,key,value,label):
 old=getattr(owner,key)
 try:
  setattr(owner,key,value);bpy.context.view_layer.update()
  assert getattr(owner,key)==value,('Mutation not effective',label)
  try:e.verify_environment(scene,c)
  except ValueError as ex:results.append({'case':label,'rejected':str(ex)})
  else:raise AssertionError('Verifier accepted '+label)
 finally:
  setattr(owner,key,old);bpy.context.view_layer.update()
 e.verify_environment(scene,c)
check(scene.world,'use_nodes',False,'world_use_nodes')
for key in ('camera','diffuse','glossy','transmission','shadow','scatter'):
 check(scene.world.cycles_visibility,key,False,'world_ray_'+key)
table=scene.objects[e.PREFIX+'table_surface']
for key in ('visible_camera','visible_diffuse','visible_glossy','visible_transmission','visible_volume_scatter','visible_shadow'):
 check(table,key,False,'object_'+key)
for key in ('is_holdout','is_shadow_catcher'):check(table,key,True,'object_'+key)
check(scene.view_layers[0],'use',False,'view_layer_disabled')
# Existing important material/light repros on the exact new saved contract.
grid=bpy.data.materials[e.PREFIX+'table_grid_material']
check(next(n for n in grid.node_tree.nodes if n.type=='MIX_RGB'),'mute',True,'old_grid_mute')
check(next(n for n in grid.node_tree.nodes if n.type=='TEX_COORD'),'object',scene.camera,'old_texcoord_object')
check(scene.objects[e.PREFIX+'area'].data,'spread',.5,'old_area_spread')
# Current master collection has no mutable render hiding and no children.
# A new contributing child is rejected by the original contract, then tested
# under an explicitly in-memory fixture contract (never saved as an asset).
child=bpy.data.collections.new('visibility_fixture_child');scene.collection.children.link(child)
child.objects.link(table);scene.collection.objects.unlink(table);bpy.context.view_layer.update()
try:e.verify_environment(scene,c)
except ValueError as ex:results.append({'case':'new_collection_membership','rejected':str(ex)})
else:raise AssertionError('New collection accepted')
c=dict(c);c['fixed_snapshot']=e.fixed_snapshot(scene);c['fixed_snapshot_sha256']=e.digest(c['fixed_snapshot'])
check(child,'hide_render',True,'fixture_child_hide_render')
layer=scene.view_layers[0].layer_collection.children[child.name]
for key in ('exclude','holdout','indirect_only'):
 check(layer,key,True,'fixture_layer_'+key)
result={'status':'PASS_CPU_CURRENT_CONTRIBUTOR_VISIBILITY','cases':results,'render_calls':0,'gpu_used':False,'asset_sha256':e.sha(R/c['saved_asset']['path']),'dataset_generation_allowed':False}
(O/'visibility_regression.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
