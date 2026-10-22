import bpy,sys,json
from pathlib import Path
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'));from data import action4_global_environment as e
O=R/'outputs/action4_realism_reassessment_v001/global_environment_init/fix_v003';base=json.loads((R/'data/action4_scene_assets/global_environment_init_v001/build_v003/environment_contract.json').read_text());results=[]
def baseline():
 bpy.ops.wm.open_mainfile(filepath=str(R/base['saved_asset']['path']),load_ui=False,use_scripts=False);scene=bpy.context.scene;bpy.context.view_layer.update();c=dict(base);e.verify_environment(scene,c);return scene,c
for case in ('mute','coordinate_object','spread','extra_light','ramp','image_interpolation'):
 scene,c=baseline();grid=bpy.data.materials['g1b_env_table_grid_material'];nodes=grid.node_tree.nodes
 if case=='mute':next(n for n in nodes if n.type=='MIX_RGB').mute=True
 if case=='coordinate_object':next(n for n in nodes if n.type=='TEX_COORD').object=scene.camera
 if case=='spread':scene.objects['g1b_env_area'].data.spread=.1
 if case=='extra_light':
  d=bpy.data.lights.new('test_added_light','POINT');d.energy=500;ob=bpy.data.objects.new('test_added_light',d);scene.collection.objects.link(ob)
 if case=='ramp':
  n=next(n for m in bpy.data.materials if m.use_nodes and m.name.startswith('connected_jacket') for n in m.node_tree.nodes if n.type=='VALTORGB');n.color_ramp.elements[0].color=(.8,.2,.2,1)
 if case=='image_interpolation':
  n=next(n for m in bpy.data.materials if m.use_nodes and m.name.startswith('connected_jacket') for n in m.node_tree.nodes if n.type=='TEX_IMAGE');n.interpolation='Closest'
 bpy.context.view_layer.update()
 try:e.verify_environment(scene,c)
 except ValueError as ex:results.append({'case':case,'rejected':str(ex)})
 else:raise AssertionError(case+' still bypasses')
(O/'q1_regression_v003.json').write_text(json.dumps({'cases':results,'scope':'CPU serializer regressions on saved v003 exact contract; no asset save/render'},indent=2));print(json.dumps(results))
