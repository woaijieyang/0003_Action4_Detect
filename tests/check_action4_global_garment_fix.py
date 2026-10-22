"""Regression of actual stale-matrix asset and physical-support/visibility gates; no GPU."""
import bpy,json,sys
from pathlib import Path
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_global_environment as env,action4_global_render as adapter,action4_global_scene as integrated
O=adapter.OUTPUTS/'geometry_fix_v003';cases=[]
old=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v004/integrated_contract.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(R/old['saved_asset']['path']),load_ui=False,use_scripts=False)
scene=bpy.context.scene;bpy.context.view_layer.update()
try:env.garment_geometry_checks(scene,old['environment']['config'],.01701172045432031)
except ValueError as ex:cases.append({'case':'old_2m_button_asset','rejected':str(ex)})
else:raise AssertionError('Old giant button accepted')
preview=json.loads((R/'configs/data/action4_global_preview_v001.json').read_text())
# Old asset observer is known inside giant button; no source contract mutation needed.
targets=adapter.target_parts(scene)
try:adapter.inspection_visibility_guard(scene,targets,old['canonical8']['corners_3d_m'],[1])
except ValueError as ex:cases.append({'case':'old_camera_inside_button','rejected':str(ex)})
else:raise AssertionError('Observer inside giant geometry accepted')
c=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v005/integrated_contract.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(R/c['saved_asset']['path']),load_ui=False,use_scripts=False)
scene=bpy.context.scene;bpy.context.view_layer.update();integrated.verify(scene,c)
cfg=c['environment']['config'];span=c['environment']['garment_geometry_checks']['source_primary_height_span_m']
def reject(label,mutate,restore):
 mutate();bpy.context.view_layer.update()
 try:env.garment_geometry_checks(scene,cfg,span)
 except ValueError as ex:cases.append({'case':label,'rejected':str(ex)})
 else:raise AssertionError(label)
 finally:restore();bpy.context.view_layer.update()
 integrated.verify(scene,c)
button=scene.objects['g1b_env_garment_jacket_metal_button'];oldscale=button.scale.copy()
reject('button_scale_changed',lambda:setattr(button,'scale',(1.1,1,1)),lambda:setattr(button,'scale',oldscale))
primary=scene.objects['g1b_env_garment_jacket_connected_front_sleeves'];oldloc=primary.location.copy()
reject('main_cloth_floating_despite_other_group_members',lambda:setattr(primary,'location',(oldloc.x,oldloc.y,oldloc.z+.1)),lambda:setattr(primary,'location',oldloc))
reject('main_cloth_outside_table_XY',lambda:setattr(primary,'location',(oldloc.x+1,oldloc.y,oldloc.z)),lambda:setattr(primary,'location',oldloc))
env.apply_capture_pose(scene,preview['inspection_poses'][1]['pose'],c['environment']);targets=adapter.target_parts(scene)
baseline=adapter.inspection_visibility_guard(scene,targets,c['canonical8']['corners_3d_m'],[1])
hidden=[(o,o.hide_render,o.hide_viewport) for o in targets[1]['parts']]
try:
 for obj,_,_ in hidden:obj.hide_render=True;obj.hide_viewport=True
 bpy.context.view_layer.update()
 try:adapter.inspection_visibility_guard(scene,targets,c['canonical8']['corners_3d_m'],[1])
 except ValueError as ex:cases.append({'case':'inspection_target_completely_hidden','rejected':str(ex)})
 else:raise AssertionError('Hidden wrist target accepted')
finally:
 for obj,render,viewport in hidden:obj.hide_render=render;obj.hide_viewport=viewport
 bpy.context.view_layer.update()
integrated.verify(scene,c)
assert old['original_invariants']==c['original_invariants']
assert old['canonical8']==c['canonical8']
assert old['assembly']['assembly_priors']==c['assembly']['assembly_priors']
result={'status':'PASS_GARMENT_SOURCE_SCALE_SUPPORT_AND_INSPECTION_GATES','negative_cases':cases,'new_actual_geometry_checks':env.garment_geometry_checks(scene,cfg,span),'left_wrist_actual_rays':baseline,'original_body_bones_weights_CAD_canonical8_A_unchanged':True,'original_accessory_assembly_unchanged':True,'render_calls':0,'GPU_used':False}
(O/'geometry_regressions.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
