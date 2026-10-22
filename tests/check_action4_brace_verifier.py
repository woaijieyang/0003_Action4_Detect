'''Actual CPU Blender mutation regression for the fixed modular brace verifier.'''
from pathlib import Path
import json,hashlib
import numpy as np
import bpy
from mathutils import Matrix
from data import action4_brace_module as m
from data.action4_global_environment import _material_state,digest
BASE=Path('outputs/action4_realism_reassessment_v001/modular_brace_build_v001/verifier_fix_v002')
c=m.load_config('configs/data/action4_brace_module_v002.json')
ext=m.load(Path(c['output_dir'])/'module_contract.json')
old=m.load_config('configs/data/action4_brace_module_v001.json')
oldext=m.load(Path(old['output_dir'])/'module_contract.json')
protected={ref['path']:m.sha(ref['path']) for ref in c['sources'].values()}
protected[oldext['saved_asset']['path']]=m.sha(oldext['saved_asset']['path'])
protected[ext['saved_asset']['path']]=m.sha(ext['saved_asset']['path'])
def reopen(cfg,contract):
 bpy.ops.wm.open_mainfile(filepath=str(m.bounded(contract['saved_asset']['path'])),load_ui=False,use_scripts=False)
 bpy.context.view_layer.update()
 return bpy.data.objects['brace_band'],bpy.data.objects['brace_module_root'],bpy.data.objects['wrist_reference']
band,root,wrist=reopen(old,oldext)
oldreport,ov,of,om=m.validate_loaded(old)
oldstates=[_material_state(x) for x in band.data.materials]
band,root,wrist=reopen(c,ext)
baseline,nv,nf,nm=m.validate_loaded(c,ext)
assert np.array_equal(ov,nv) and np.array_equal(of,nf) and np.array_equal(om,nm)
assert digest(oldstates)==digest([_material_state(x) for x in band.data.materials])
rows=[]
def run(name,mutate):
 band,root,wrist=reopen(c,ext)
 mutate(band,root,wrist)
 bpy.context.view_layer.update()
 try:m.validate_loaded(c,ext)
 except ValueError as exc:rows.append({'name':name,'rejected':True,'reason':str(exc)})
 else:raise AssertionError('Mutation silently accepted: '+name)
 reopen(c,ext);m.validate_loaded(c,ext)
def node(b,kind,index=0):return next(n for n in b.data.materials[index].node_tree.nodes if n.type==kind)
def disconnect(b,kind):
 tree=b.data.materials[0].node_tree;n=node(b,kind)
 tree.links.remove(next(l for l in tree.links if l.from_node==n))
run('BRACES1_parent_inverse_y_scale1p1',lambda b,r,w:setattr(b,'matrix_parent_inverse',Matrix.Diagonal((1.,1.1,1.,1.))))
run('parent_inverse_translation',lambda b,r,w:setattr(b,'matrix_parent_inverse',Matrix.Translation((.003,0,0))))
run('parent_inverse_rotation',lambda b,r,w:setattr(b,'matrix_parent_inverse',Matrix.Rotation(.1,4,'Y')))
run('wrist_parent_inverse_scale',lambda b,r,w:setattr(w,'matrix_parent_inverse',Matrix.Diagonal((1.,1.1,1.,1.))))
run('wrist_rotation',lambda b,r,w:setattr(w,'rotation_euler',(.1,0,0)))
run('wrist_scale',lambda b,r,w:setattr(w,'scale',(1,1.1,1)))
run('root_translation',lambda b,r,w:setattr(r,'location',(.001,0,0)))
run('BRACES2_disconnect_image',lambda b,r,w:disconnect(b,'TEX_IMAGE'))
run('disconnect_mix',lambda b,r,w:disconnect(b,'MIX_RGB'))
run('disconnect_BSDF_surface',lambda b,r,w:disconnect(b,'BSDF_PRINCIPLED'))
run('mix_factor_zero',lambda b,r,w:setattr(node(b,'MIX_RGB').inputs[0],'default_value',0))
run('mix_clamp_parameter',lambda b,r,w:setattr(node(b,'MIX_RGB'),'use_clamp',not node(b,'MIX_RGB').use_clamp))
run('texture_scale',lambda b,r,w:setattr(node(b,'VECT_MATH').inputs['Scale'],'default_value',1))
run('texture_mapping_rotation',lambda b,r,w:setattr(node(b,'TEX_IMAGE').texture_mapping,'rotation',(.1,0,0)))
run('texture_projection',lambda b,r,w:setattr(node(b,'TEX_IMAGE'),'projection','FLAT'))
run('texture_muted',lambda b,r,w:setattr(node(b,'TEX_IMAGE'),'mute',True))
run('green_roughness',lambda b,r,w:setattr(node(b,'BSDF_PRINCIPLED',1).inputs['Roughness'],'default_value',.2))
run('green_color',lambda b,r,w:setattr(node(b,'BSDF_PRINCIPLED',1).inputs['Base Color'],'default_value',(.1,.1,.1,1)))
run('gray_unlinked_subsurface_parameter',lambda b,r,w:setattr(node(b,'BSDF_PRINCIPLED').inputs['Subsurface'],'default_value',.2))
run('extra_unconnected_supported_node',lambda b,r,w:b.data.materials[0].node_tree.nodes.new('ShaderNodeTexNoise'))
run('vertex_coordinate_tamper',lambda b,r,w:setattr(b.data.vertices[0],'co',(0,0,0)))
for path,h in protected.items():assert m.sha(path)==h
result={'status':'PASS_ACTUAL_RELOAD_AND_21_REJECTED_MUTATIONS','old_new_vertices_exact':True,'old_new_faces_exact':True,'old_new_material_indices_exact':True,'old_new_complete_supported_material_states_exact':True,'material_state_digest':digest(oldstates),'mutations':rows,'mutation_count':len(rows),'all_restores_revalidated':True,'sources_and_both_assets_unchanged':protected,'new_world_dimensions_m':baseline['world_dimensions_xyz_m'],'maximum_rim_difference_m':max(x['max_difference_m'] for x in baseline['rim_checks'].values()),'maximum_projection_difference_px':max(x['maximum_projection_difference_px'] for x in baseline['threeframe_projection_checks']),'CPU_only':True,'saved_assets':False,'RGB_rendered':False,'production_approval':False}
m.write_new(BASE/'actual_mutation_and_equivalence.json',result)
print(result['status'])
