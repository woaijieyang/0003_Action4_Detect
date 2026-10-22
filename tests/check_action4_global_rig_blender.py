"""Actual Blender reopen/FK check; no render and no saved-asset mutation."""
import json,sys,hashlib
from pathlib import Path
import bpy,numpy as np
root=Path("/data/zy/vln/a-test");assert Path.cwd().resolve()==root
sys.path.insert(0,str(root/"src"))
from data.action4_global_rig import apply_pose,JOINTS,_evaluated_vertices,sha
reportdir=root/"outputs/action4_realism_reassessment_v001/global_rig_init"
r=json.loads((reportdir/"build_v002/report.json").read_text())
asset=root/r["saved_blend"]["path"];before=sha(asset)
assert before==r["saved_blend"]["sha256"]
bpy.ops.wm.open_mainfile(filepath=str(asset),load_ui=False,use_scripts=False)
scene=bpy.context.scene;rig=bpy.data.objects["global_metric_armature"];body=bpy.data.objects["global_source_anatomy"]
assert scene["status"]=="initialization_pending_fit" and scene["dataset_generation_allowed"] is False
assert scene.unit_settings.system=="METRIC" and scene.unit_settings.scale_length==1
assert len(rig.data.bones)==7
assert all(abs(float(v))<1e-8 for n in JOINTS for v in rig.pose.bones[n].rotation_euler)
assert all(abs(float(v)-1)<1e-7 for v in rig.scale)
for vertex in body.data.vertices:
 assert abs(sum(g.weight for g in vertex.groups)-1)<1e-6
assert next(m for m in body.modifiers if m.type=="ARMATURE").object==rig
cfg=r["source_inputs"];initial,_=_evaluated_vertices(body)
baseline={s:np.array(bpy.data.objects[s+"_Action4_core"].matrix_world) for s in ("left","right")}
checks=[]
for case in cfg["validation_poses"]:
 apply_pose(rig,case["joints"],cfg);current,_=_evaluated_vertices(body)
 delta=float(np.linalg.norm(current-initial,axis=1).max());assert delta>1e-4
 item={"name":case["name"],"skin_displacement_m":delta,"cameras":{}}
 for side in ("left","right"):
  core=bpy.data.objects[side+"_Action4_core"];brace=bpy.data.objects[side+"_brace_frame"]
  assert core.parent==brace and brace.parent==rig and brace.parent_bone==side+"_forearm"
  A=np.array(brace.matrix_world.inverted()@core.matrix_world);expected=np.array(r["brace_and_fixed_attachment"][side]["A_brace_camera"])
  assert np.max(abs(A-expected))<2e-6
  assert abs(np.linalg.det(np.array(core.matrix_world)[:3,:3])-1)<2e-6
  for part in ("upper_arm","forearm","hand"):
   bone=rig.pose.bones[side+"_"+part];assert abs((bone.tail-bone.head).length-r["rest_lengths_m"][side+"_"+part])<1e-6
  if case["name"]=="hand_only":assert np.max(abs(np.array(core.matrix_world)-baseline[side]))<2e-6
  item["cameras"][side]={"constant_A_maxerror":float(np.max(abs(A-expected))),"hand_only_camera_unchanged":case["name"]=="hand_only"}
 checks.append(item)
for invalid in [{"camera_scale":[1,1,1]},{"left_brace_A":[0,0,0]},{"family":[0,0,0]},{"left_forearm":[26,0,0]}]:
 try:apply_pose(rig,invalid,cfg)
 except ValueError:pass
 else:raise AssertionError("Forbidden pose input accepted")
apply_pose(rig,{},cfg)
restored,_=_evaluated_vertices(body);assert np.max(abs(restored-initial))<1e-7
assert sha(asset)==before
for ref in (cfg["body"],cfg["camera"]["foundation"],cfg["camera"]["asset_manifest"]):assert sha(ref["path"])==ref["sha256"]
result={"status":"PASS_ACTUAL_REOPEN_FK_AND_CONTRACT","saved_neutral":True,"skin_weights_normalized":True,
        "checks":checks,"negative_pose_cases":4,"asset_unchanged_sha256":before,
        "scope":"CPU Blender engineering initialization checks; not fitted anatomy, scene realism or training acceptance"}
with (reportdir/"reopen_validation_v002.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
