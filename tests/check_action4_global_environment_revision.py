"""CPU-only actual Blender counterexamples for G1B quality revision."""
import bpy,json,sys,copy
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test');sys.path.insert(0,str(ROOT/'src'))
from data import action4_global_environment as m
OUT=ROOT/'outputs/action4_realism_reassessment_v001/global_environment_init/fix_v003'
cfg=json.loads((ROOT/'configs/data/action4_global_environment_init_v001.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(m.bounded(cfg['sources']['rig_asset']['path'],True)),load_ui=False,use_scripts=False)
scene=bpy.context.scene;bpy.context.view_layer.update()
original={o.name:m.object_record(o) for o in scene.objects}
before=m._creation_state(scene)
def unchanged():
 after=m._creation_state(scene)
 for key in before:
  assert after[key]==before[key],(key,repr(before[key]),repr(after[key]))
 assert original=={o.name:m.object_record(o) for o in scene.objects}
cases=[]
for group,key,value in [('lighting','look','INVALID_TEST_LOOK'),('capture','focal_mm',.5),('capture','sensor_width_mm',.5),('capture','clip_start_m',1e-9),('lighting','gamma',1e100)]:
 bad=copy.deepcopy(cfg);bad['initial_priors'][group][key]=value
 try:m.add_environment(scene,bad)
 except ValueError as e:cases.append({'case':key,'rejected':str(e),'unchanged':True})
 else:raise AssertionError('Accepted invalid '+key)
 unchanged()
original_snapshot=m.fixed_snapshot
def fail_late(s):
 assert len(s.objects)>len(original) and s.world is not None and s.camera is not None
 raise RuntimeError('injected late snapshot failure')
m.fixed_snapshot=fail_late
try:
 try:m.add_environment(scene,cfg)
 except RuntimeError as e:assert 'injected late' in str(e)
 else:raise AssertionError('No injected failure')
finally:m.fixed_snapshot=original_snapshot
unchanged();cases.append({'case':'late_failure_full_rollback','unchanged':True})
contract=m.add_environment(scene,cfg)
m.verify_environment(scene,contract)
cases.append({'case':'retry_after_rollback','passed':True})
camera=scene.camera;matrix=[list(r) for r in camera.matrix_world]
bad=copy.deepcopy(cfg['engineering_capture_poses'][0]);bad['location_m'][0]=1e100
try:m.apply_capture_pose(scene,bad,contract)
except ValueError as e:cases.append({'case':'overflow_pose','rejected':str(e)})
else:raise AssertionError('Accepted overflow pose')
assert matrix==[list(r) for r in camera.matrix_world]
old_assert=m._assert_numeric
def fail_pose(actual,requested,label):
 if label=='capture matrix':raise ValueError('injected postassignment mismatch')
 return old_assert(actual,requested,label)
m._assert_numeric=fail_pose
try:
 try:m.apply_capture_pose(scene,cfg['engineering_capture_poses'][0],contract)
 except ValueError as e:assert 'injected postassignment' in str(e)
 else:raise AssertionError('No injected mismatch')
finally:m._assert_numeric=old_assert
assert matrix==[list(r) for r in camera.matrix_world]
m.verify_environment(scene,contract)
cases.append({'case':'postassignment_pose_failure_exact_restore','passed':True})
for pose in cfg['engineering_capture_poses']:m.apply_capture_pose(scene,pose,contract)
assert original=={name:m.object_record(scene.objects[name]) for name in original}
result={'status':'PASS_CPU_BLENDER_Q2_Q3','cases':cases,'actual_K':m.actual_K(scene),'contract_K':contract['K_prior'],'render_calls':0,'gpu_used':False,'dataset_generation_allowed':False}
(OUT/'q2_q3_regression.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
