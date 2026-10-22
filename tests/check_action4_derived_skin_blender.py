"""Actual v006 raw/v007 Linear input comparison, no render/asset writes."""
import bpy,json,sys,copy,hashlib,numpy as np
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_global_scene as m
O=R/'outputs/action4_realism_reassessment_v001/skin_texture_periodic_integration_v001'
old=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v006/integrated_contract.json').read_text());new=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v007/integrated_contract.json').read_text())
preview=json.loads((R/'configs/data/action4_global_preview_periodic_skin_v001.json').read_text())
def load(c):
 bpy.ops.wm.open_mainfile(filepath=str(R/c['saved_asset']['path']),load_ui=False,use_scripts=False);bpy.context.view_layer.update();return bpy.context.scene
def evaluated(body):
 obj=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
 try:return {'vertices':np.asarray([list(v.co) for v in mesh.vertices]),'faces':[tuple(p.vertices) for p in mesh.polygons],'UV':{u.name:np.asarray([list(p.uv) for p in u.data]) for u in mesh.uv_layers}}
 finally:obj.to_mesh_clear()
def projections(scene,c):
 rows=[]
 for shot in preview['inspection_poses']:
  m.env.apply_capture_pose(scene,shot['pose'],c['environment']);row=[]
  for side in ['left','right']:
   core=scene.objects[side+'_Action4_core']
   for point in c['canonical8']['corners_3d_m']:
    v=world_to_camera_view(scene,scene.camera,core.matrix_world@Vector(point));row.append([float(v.x*3248-.5),float((1-v.y)*2464-.5),float(v.z)])
  rows.append(row)
 return rows
scene=load(old);m.verify(scene,old)
oldskin=m.env._material_state(scene.objects['global_source_anatomy'].data.materials[0])
oldbody=evaluated(scene.objects['global_source_anatomy']);oldproj=projections(scene,old)
oldmaterials={o.name:m.env.object_record(o).get('materials') for o in scene.objects if o.type in ('MESH','CURVE') and o.name!='global_source_anatomy'}
scene=load(new);m.verify(scene,new)
body=scene.objects['global_source_anatomy'];material=body.data.materials[0];newbody=evaluated(body)
assert old['original_invariants']==new['original_invariants'] and old['body_shading']==new['body_shading'] and old['canonical8']==new['canonical8']
assert old['assembly']['assembly_priors']==new['assembly']['assembly_priors'] and old['attachment_state']==new['attachment_state']
assert old['environment']['config']==new['environment']['config']
assert np.array_equal(oldbody['vertices'],newbody['vertices']) and oldbody['faces']==newbody['faces']
uv={k:{'max_abs_delta':float(np.max(abs(oldbody['UV'][k]-newbody['UV'][k]))),'bitexact':bool(np.array_equal(oldbody['UV'][k],newbody['UV'][k]))} for k in oldbody['UV']}
assert oldmaterials=={o.name:m.env.object_record(o).get('materials') for o in scene.objects if o.type in ('MESH','CURVE') and o.name!='global_source_anatomy'}
newskin=m.env._material_state(material);expected=copy.deepcopy(oldskin)
image_rows=[r for r in newskin['nodes']['nodes'] if r['type']=='ShaderNodeTexImage'];assert len(image_rows)==1
newrow=image_rows[0];oldrow=next(r for r in expected['nodes']['nodes'] if r['name']==newrow['name'])
oldrow['rna']['image']=newrow['rna']['image'];oldrow['image']=newrow['image']
assert expected==newskin,'More than active image identity/colour-space changed'
state=m.skin_input_state(scene,new['config']);assert state==new['skin_color_input'] and state['actual_color_space']=='Linear'
assert oldproj==projections(scene,new)
embedded=json.loads(bpy.data.texts[m.TEXT].as_string());assert embedded=={k:v for k,v in new.items() if k!='saved_asset'}
node=material.node_tree.nodes[state['image_node']];original_image=node.image
raw_png=next(im for im in bpy.data.images if im.packed_file and hashlib.sha256(bytes(im.packed_file.data)).hexdigest()==new['config']['sources']['skin_forearm']['sha256'])
assert raw_png.colorspace_settings.name=='sRGB' and raw_png.use_fake_user
brace_material=scene.objects['left_brace_initial_geometry'].data.materials[0]
brace_images=[n.image for n in brace_material.node_tree.nodes if n.type=='TEX_IMAGE']
assert len(brace_images)==1 and brace_images[0].colorspace_settings.name=='sRGB'
assert hashlib.sha256(bytes(brace_images[0].packed_file.data)).hexdigest()==new['config']['sources']['wristband_weave']['sha256']
negative=[]
def reject(label,mutate,restore):
 mutate();bpy.context.view_layer.update()
 try:m.verify(scene,new)
 except ValueError as ex:negative.append({'case':label,'rejected':str(ex)})
 else:raise AssertionError(label)
 finally:restore();bpy.context.view_layer.update()
 m.verify(scene,new)
reject('Linear_EXR_retagged_sRGB',lambda:setattr(original_image.colorspace_settings,'name','sRGB'),lambda:setattr(original_image.colorspace_settings,'name','Linear'))
reject('active_EXR_replaced_by_original_PNG',lambda:setattr(node,'image',raw_png),lambda:setattr(node,'image',original_image))
mix=material.node_tree.nodes[state['mix_node']];link=mix.inputs[2].links[0]
reject('active_colour_link_disconnected',lambda:material.node_tree.links.remove(link),lambda:material.node_tree.links.new(node.outputs['Color'],mix.inputs[2]))
for c in (old,new):assert m.env.sha(c['saved_asset']['path'])==c['saved_asset']['sha256']
report={'status':'PASS_CPU_DERIVED_LINEAR_SKIN_V007_BINDING_AND_GUARDS','raw_v1_existing_asset_verify_pass':True,'active_skin_input':state,'material_change_only_image_identity_colour_space':True,'wristband_sRGB_hash_unchanged':True,'raw_source_PNG_packed_inactive_provenance_preserved':True,'all_other_materials_exact':True,'control_geometry_weights_35bones_A_ordered8_smooth_exact':True,'evaluated_vertices_faces_exact':True,'evaluated_UV_diagnostic_not_claimed_stable':uv,'all48_target_projections_exact':True,'environment_light_config_exact':True,'external_embedded_contract_consistent':True,'negative_cases':negative,'old_v006_and_new_v007_asset_bytes_preserved':True,'render_calls':0,'GPU_used':False,'CPU_threads':2,'asset_saves':False}
(O/'actual_binding_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
