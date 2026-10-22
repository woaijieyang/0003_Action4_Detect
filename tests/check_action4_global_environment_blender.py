"""Actual CPU Blender reopen/pose invariants; never render."""
import bpy,json,sys,hashlib
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data.action4_global_environment import (bounded,sha,object_record,verify_environment,apply_capture_pose,add_environment,PREFIX)
OUT=ROOT/'outputs/action4_realism_reassessment_v001/global_environment_init'
cfg=json.loads((ROOT/'configs/data/action4_global_environment_init_v001.json').read_text())
contract=json.loads((ROOT/'data/action4_scene_assets/global_environment_init_v001/build_v003/environment_contract.json').read_text())
before_hash={ref['path']:sha(ref['path']) for ref in cfg['sources'].values()}
bpy.ops.wm.open_mainfile(filepath=str(bounded(cfg['sources']['rig_asset']['path'],True)),load_ui=False,use_scripts=False);bpy.context.view_layer.update()
original={o.name:object_record(o) for o in bpy.context.scene.objects}
bpy.ops.wm.open_mainfile(filepath=str(bounded(contract['saved_asset']['path'],True)),load_ui=False,use_scripts=False);scene=bpy.context.scene;bpy.context.view_layer.update()
assert original=={name:object_record(scene.objects[name]) for name in original},'Existing rig changed after saved reopen'
verify_environment(scene,contract)
static_names=sorted(name for name,o in scene.objects.items() if o.get('g1b_static'))
assert all(scene.objects[name].pass_index==0 for name in static_names)
assert scene.objects['left_Action4_core'].pass_index==1 and scene.objects['right_Action4_core'].pass_index==2
assert len([o for o in scene.objects if o.type=='CAMERA'])==1
camera=scene.objects[PREFIX+'observer'];K=contract['K_prior'];w,h=cfg['native_image']['resolution_wh'];errors=[]
for pose in cfg['engineering_capture_poses']:
 apply_capture_pose(scene,pose,contract)
 assert original=={name:object_record(scene.objects[name]) for name in original},'Observer pose mutated rig'
 for point in ((.05,.03,-1),(-.1,-.05,-1.5),(0,0,-2)):
  world=camera.matrix_world@Vector(point);view=world_to_camera_view(scene,camera,world)
  actual=(view.x*w-.5,(1-view.y)*h-.5);expected=(K[0][0]*point[0]/-point[2]+K[0][2],K[1][1]*-point[1]/-point[2]+K[1][2]);error=max(abs(a-b) for a,b in zip(actual,expected));assert error<.001;errors.append(error)
negative=[]
for label,mutate,restore in [
 ('focal',lambda:setattr(camera.data,'lens',22),lambda:setattr(camera.data,'lens',26)),
 ('resolution',lambda:setattr(scene.render,'resolution_x',812),lambda:setattr(scene.render,'resolution_x',3248)),
 ('table_transform',lambda:setattr(scene.objects[PREFIX+'table_surface'].location,'x',.2),lambda:setattr(scene.objects[PREFIX+'table_surface'].location,'x',0)),
 ('light_energy',lambda:setattr(scene.objects[PREFIX+'area'].data,'energy',36),lambda:setattr(scene.objects[PREFIX+'area'].data,'energy',18)),
 ('generation_scope',lambda:scene.__setitem__('g1b_dataset_generation_allowed',True),lambda:scene.__setitem__('g1b_dataset_generation_allowed',False))]:
 mutate()
 try:verify_environment(scene,contract)
 except ValueError as e:negative.append({'case':label,'rejected':str(e)})
 else:raise AssertionError('Invariant accepted '+label)
 restore();bpy.context.view_layer.update();verify_environment(scene,contract)
mat=bpy.data.materials[PREFIX+'wall_material'];socket=mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'];old=socket.default_value;socket.default_value=.2
try:verify_environment(scene,contract)
except ValueError as e:negative.append({'case':'material_change','rejected':str(e)})
else:raise AssertionError('Material invariant bypass')
socket.default_value=old;verify_environment(scene,contract)
for override in ('focal_mm','table_size_xy_m','camera_scale'):
 pose={**cfg['engineering_capture_poses'][0],override:2};matrix=[list(x) for x in camera.matrix_world]
 try:apply_capture_pose(scene,pose,contract)
 except ValueError as e:negative.append({'case':'pose_override_'+override,'rejected':str(e)})
 else:raise AssertionError('Override accepted')
 assert matrix==[list(x) for x in camera.matrix_world]
names=set(scene.objects.keys())
try:add_environment(scene,cfg)
except ValueError as e:negative.append({'case':'duplicate_initialize','rejected':str(e)})
else:raise AssertionError('Duplicate accepted')
assert names==set(scene.objects.keys())
# A fresh neutral scene with an occupied reserved name rejects before mutation.
bpy.ops.wm.open_mainfile(filepath=str(bounded(cfg['sources']['rig_asset']['path'],True)),load_ui=False,use_scripts=False);scene=bpy.context.scene
sentinel=bpy.data.objects.new(PREFIX+'table_surface',None);scene.collection.objects.link(sentinel);names=set(scene.objects.keys())
try:add_environment(scene,cfg)
except ValueError as e:negative.append({'case':'reserved_name_collision','rejected':str(e)})
else:raise AssertionError('Existing named object overwritten')
assert names==set(scene.objects.keys()) and scene.objects[PREFIX+'table_surface']==sentinel
assert all(sha(path)==h for path,h in before_hash.items())
result={'status':'PASS_CPU_BLENDER_INVARIANTS','dataset_generation_allowed':False,'fitted':False,'source_inputs_unchanged':True,'existing_G1_objects_exact_after_reopen_and_capture_poses':True,'static_object_count':len(static_names),'static_object_names':static_names,'actual_projection_max_error_px':max(errors),'actual_capture_pose_cases':2,'negative_cases':negative,'render_calls':0,'gpu_used':False,'limitation':'Initial priors and static engineering consistency only; no video fit, realism, garment equilibrium or dataset acceptance'}
with (OUT/'actual_invariants_v003.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
