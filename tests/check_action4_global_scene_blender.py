"""Actual CPU saved-scene checks, no render/acceptance."""
import bpy,json,sys
from pathlib import Path
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_global_scene as m,action4_finger_rig as finger,action4_global_rig as rigapi
O=R/'outputs/action4_realism_reassessment_v001/global_scene_init_v001'
c=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v003/integrated_contract.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(R/c['saved_asset']['path']),load_ui=False,use_scripts=False)
scene=bpy.context.scene;rig=scene.objects['global_metric_armature'];bpy.context.view_layer.update();m.verify(scene,c)
f=json.loads((R/c['config']['sources']['finger_contract']['path']).read_text())
finger.apply_pose(rig,{'left_index':[12,12,8],'right_thumb':[5,8,0]},f['mapping']);m.verify(scene,c)
finger.apply_pose(rig,{},f['mapping']);m.verify(scene,c)
# Original rig API config is embedded in its source initialization report.
g=json.loads(bpy.data.texts['GLOBAL_RIG_INITIALIZATION.json'].as_string())
print('G1_REPORT_KEYS',list(g))
cfg=g.get('source_inputs')
if cfg is None:raise ValueError('Missing original G1 config for original pose API')
rigapi.apply_pose(rig,{'left_forearm':[5,0,0],'right_hand':[0,5,0]},cfg);m.verify(scene,c)
rigapi.apply_pose(rig,{},cfg);m.verify(scene,c)
negative=[]
def reject(label,mutate,restore):
 mutate();bpy.context.view_layer.update()
 try:m.verify(scene,c)
 except ValueError as ex:negative.append({'case':label,'rejected':str(ex)})
 else:raise AssertionError(label)
 finally:restore();bpy.context.view_layer.update()
 m.verify(scene,c)
proxy=scene.objects['left_mount_initial_support']
reject('proxy_unhidden',lambda:setattr(proxy,'hide_render',False),lambda:setattr(proxy,'hide_render',True))
usb=scene.objects['integrated_left_usb_connector'];old=usb.location.copy()
reject('accessory_reattached',lambda:setattr(usb,'location',(0,0,0)),lambda:setattr(usb,'location',old))
core=scene.objects['left_Action4_core'];old=core.data.vertices[0].co.copy()
reject('target_mesh_changed',lambda:setattr(core.data.vertices[0],'co',old*1.01),lambda:setattr(core.data.vertices[0],'co',old))
assert all(o.pass_index==0 for o in scene.objects if o.name.startswith(m.PREFIX))
assert all(o.type in ('MESH','EMPTY') for o in scene.objects if o.name.startswith(m.PREFIX))
assert scene.objects['global_source_anatomy'].data.materials[0].get('photo_texture_source')
out={'status':'PASS_INTEGRATED_CPU_INTERFACES_AND_NEGATIVES','negative_cases':negative,'finger_API':True,'G1_API':True,'environment_API':'build/reopen two observer poses','bone_count':len(rig.data.bones),'render_calls':0,'gpu_used':False,'dataset_generation_allowed':False}
(O/'actual_interfaces_v003.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
