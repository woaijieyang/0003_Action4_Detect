"""Actual shading-only regression. CPU Blender, no render or asset saving."""
import bpy,json,sys,hashlib
from pathlib import Path
import numpy as np
R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_realism_reassessment_v001/global_scene_init_v001/smooth_fix_v006';sys.path.insert(0,str(R/'src'))
from data import action4_global_scene as m
old=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v005/integrated_contract.json').read_text());new=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v006/integrated_contract.json').read_text())
def load(c):
 bpy.ops.wm.open_mainfile(filepath=str(R/c['saved_asset']['path']),load_ui=False,use_scripts=False);bpy.context.view_layer.update();return bpy.context.scene

def geometry(body,evaluated=False):
 owner=None
 if evaluated:owner=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=owner.to_mesh()
 else:mesh=body.data
 try:
  return {'vertices':np.array([list(v.co) for v in mesh.vertices]),'faces':[tuple(p.vertices) for p in mesh.polygons],
   'UV':{u.name:np.array([list(v.uv) for v in u.data]) for u in mesh.uv_layers},'smooth_count':sum(p.use_smooth for p in mesh.polygons)}
 finally:
  if owner:owner.to_mesh_clear()
scene=load(old);body=scene.objects['global_source_anatomy'];old_control=geometry(body);old_eval=geometry(body,True);old_skin=m.env._material_state(body.data.materials[0]);old_shading=m.body_shading_state(scene)
scene=load(new);body=scene.objects['global_source_anatomy'];m.verify(scene,new);control=geometry(body);ev=geometry(body,True)
for a,b in [(old_control,control),(old_eval,ev)]:
 assert np.array_equal(a['vertices'],b['vertices']) and a['faces']==b['faces']
 assert set(a['UV'])==set(b['UV'])
assert all(np.array_equal(old_control['UV'][k],control['UV'][k]) for k in old_control['UV'])
eval_uv_diagnostic={k:{'exact':bool(np.array_equal(old_eval['UV'][k],ev['UV'][k])),'max_abs':float(np.max(abs(old_eval['UV'][k]-ev['UV'][k]))),'different_components':int(np.sum(old_eval['UV'][k]!=ev['UV'][k]))} for k in old_eval['UV']}
assert new['body_shading']['source']==old_shading
assert old['original_invariants']==new['original_invariants'] and old['canonical8']==new['canonical8']
assert old['config']==new['config'] and old['assembly']['assembly_priors']==new['assembly']['assembly_priors'] and old['attachment_state']==new['attachment_state']
assert old_skin==m.env._material_state(body.data.materials[0])
skin=body.data.materials[0]
coordinate_links=[link for link in skin.node_tree.links if link.from_node.type=='TEX_COORD']
assert coordinate_links and all(link.from_socket.name=='Object' for link in coordinate_links)
assert not any(node.type in ('UVMAP','ATTRIBUTE') for node in skin.node_tree.nodes)
assert len(scene.objects['global_metric_armature'].data.bones)==35
negative=[]
# The old geometry/environment checks pass, proving the new explicit guard matters.
face_sets=body.data.attributes['.sculpt_face_set'];face_id=next(i for i,v in enumerate(face_sets.data) if v.value==102)
body.data.polygons[face_id].use_smooth=False;body.data.update();bpy.context.view_layer.update()
assert m.env.digest(m._original_invariants(scene,new['original_names']))==new['original_invariants_sha256']
m.env.verify_environment(scene,new['environment'])
changed=m.body_shading_state(scene)
try:m.verify(scene,new)
except ValueError as e:negative.append({'case':'one_forearm_face_flat','face_id':face_id,'legacy_geometry_and_environment_pass':True,'actual_changed_shading':changed,'rejected':str(e)})
else:raise AssertionError('Explicit shading guard missed flat face')
body.data.polygons[face_id].use_smooth=True;body.data.update();bpy.context.view_layer.update();m.verify(scene,new)
for case,get_owner,key,value in [('auto_smooth_changed',lambda:body.data,'use_auto_smooth',True),('sharp_edge_changed',lambda:body.data.edges[0],'use_edge_sharp',True)]:
 owner=get_owner();previous=getattr(owner,key);setattr(owner,key,value);body.data.update();bpy.context.view_layer.update()
 assert getattr(get_owner(),key)==value,case+' mutation ineffective'
 try:m.verify(scene,new)
 except ValueError as e:negative.append({'case':case,'rejected':str(e)})
 else:raise AssertionError(case+' accepted')
 setattr(get_owner(),key,previous);body.data.update();bpy.context.view_layer.update();m.verify(scene,new)
for shot in new['environment']['config']['engineering_capture_poses']:
 m.env.apply_capture_pose(scene,shot,new['environment']);m.verify(scene,new)
assert m.env.sha(old['saved_asset']['path'])==old['saved_asset']['sha256'] and m.env.sha(new['saved_asset']['path'])==new['saved_asset']['sha256']
result={'status':'PASS_SHADING_GEOMETRY_GUARDS_WITH_EVAL_UV_DIAGNOSTIC_LIMITATION','old_asset':old['saved_asset'],'new_asset':new['saved_asset'],
 'control_vertices_faces_UV_exact':True,'evaluated_vertices_faces_exact':True,'evaluated_UV_diagnostic_not_claimed_exact':eval_uv_diagnostic,'original_bones_weights_modifiers_A_CAD8_invariants_exact':True,
 'skin_material_graph_and_patch_mix_exact':True,'configuration_lighting_appearance_priors_unchanged':True,'accessory_assembly_priors_and_attachments_exact':True,
 'body_shading':new['body_shading'],'negative_cases':negative,'legal_capture_pose_cases':2,'render_calls':0,'GPU_used':False,'asset_saves':0,'CPU_threads':2,
 'skin_Object_mapping_does_not_consume_evaluated_UV':True,'evaluated_UV_bitwise_preservation_not_accepted':True,'known_UV_diagnostic_evidence':'UV_smooth_toggle_causality.json','raw_photo_periodic_bands_still_unaddressed':True,'visual_smooth_effect_not_rendered':True}
with (O/'actual_shading_checks.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'status':result['status'],'control_smooth':control['smooth_count'],'evaluated_smooth':ev['smooth_count'],'geometry_exact':True,'negative_cases':len(negative)}))
