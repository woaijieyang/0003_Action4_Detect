"""CPU-only state failure, canonical projection and ownership counterexamples."""
import bpy,json,sys,numpy as np
from pathlib import Path
from mathutils import Vector
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_global_render as m,action4_render as base
cfg,c=m.validate_config(json.loads((R/'configs/data/action4_global_preview_v001.json').read_text()))
O=m.OUTPUTS
bpy.ops.wm.open_mainfile(filepath=str(R/cfg['sources']['integrated_asset']['path']),load_ui=False,use_scripts=False)
scene=bpy.context.scene;bpy.context.view_layer.update();m.integrated.verify(scene,c)
targets=m.target_parts(scene);before=m.state_snapshot(scene);K=np.asarray(c['environment']['K_prior']);maxerror=0.
for shot in cfg['inspection_poses']:
 with m.temporary_render_state(scene):
  m.env.apply_capture_pose(scene,shot['pose'],c['environment'])
  for target in targets.values():
   points,_=m.project_canonical(scene,target['core'],c['canonical8']['corners_3d_m'])
   inverse=scene.camera.matrix_world.inverted()
   for corner,observed in zip(c['canonical8']['corners_3d_m'],points):
    q=inverse@(target['core'].matrix_world@Vector(corner))
    expected=[K[0,0]*q.x/-q.z+K[0,2],K[1,1]*-q.y/-q.z+K[1,2]]
    maxerror=max(maxerror,max(abs(a-b) for a,b in zip(expected,observed)))
 assert m.state_snapshot(scene)==before
assert maxerror<.003,maxerror
# Actual failure after all mask/camera/hide/border state mutations; no render call.
try:
 with m.temporary_render_state(scene):
  m.env.apply_capture_pose(scene,cfg['inspection_poses'][1]['pose'],c['environment'])
  scene.cycles.samples=64;nodes=base._mask_nodes(O/'unused_mask_test')
  with m.isolated_expanded(scene,targets[1]['parts']):
   with m.amodal.temporary_border(scene,nodes,O/'unused_border_test',(10,10,100,100),scene.render.resolution_x,scene.render.resolution_y):
    scene.render.filepath='injected_temporary_path'
    raise RuntimeError('injected pre-render failure')
except RuntimeError as ex:assert str(ex)=='injected pre-render failure'
else:raise AssertionError('Expected failure')
assert m.state_snapshot(scene)==before
m.integrated.verify(scene,c)
accessory=scene.objects['integrated_left_usb_connector'];accessory.pass_index=1
try:m.target_parts(scene)
except ValueError as ex:ownership=str(ex)
else:raise AssertionError('Accessory fused into target')
finally:accessory.pass_index=0
m.integrated.verify(scene,c)
result={'status':'PASS_CPU_ADAPTER_FAILURE_AND_PROJECTION','render_calls':0,'gpu_used':False,'max_analytic_projection_error_px':maxerror,'projection_cases':3*2*8,'exact_render_camera_hide_state_restored_after_injected_failure':True,'integrated_contract_after_failure':True,'accessory_label_fusion_rejected':ownership,'target_parts':{j:[o.name for o in t['parts']] for j,t in targets.items()},'dataset_generation_allowed':False}
(O/'cpu_regressions.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
