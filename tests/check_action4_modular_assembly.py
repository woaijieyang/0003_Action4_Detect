import bpy,json,copy,numpy as np
from mathutils import Matrix
from data import action4_modular_assembly as m
c,_=m.load_config('configs/data/action4_modular_assembly_probe_v001.json')
contract=m.read(m.OUTPUT/'assembly_contract.json');receipt=m.read(m.OUTPUT/'build_receipt.json')
before={r['path']:m.sha(r['path']) for r in c['sources'].values()};before[receipt['asset']['path']]=m.sha(receipt['asset']['path'])
def reopen():
 bpy.ops.wm.open_mainfile(filepath=str(m.bounded(receipt['asset']['path'])),load_ui=False,use_scripts=False)
 return bpy.data.objects['right_Action4_core'],bpy.data.objects['brace_module_root'],bpy.data.objects['right_wrist_module_root']
rows=[]
def run(name,mutate):
 core,br,wr=reopen();cc=copy.deepcopy(contract);mutate(core,br,wr,cc);bpy.context.view_layer.update()
 try:m.verify_scene(c,cc)
 except ValueError as exc:rows.append({'name':name,'rejected':True,'reason':str(exc)})
 else:raise AssertionError('Unexpected acceptance: '+name)
 reopen();m.verify_scene(c,contract)
run('camera_independent_world_translation',lambda core,b,w,cc:setattr(core,'location',core.location+__import__('mathutils').Vector((.01,0,0))))
run('wrist_world_frame_disagrees',lambda core,b,w,cc:setattr(w,'location',w.location+__import__('mathutils').Vector((0,.01,0))))
run('band_hidden_parent_inverse',lambda core,b,w,cc:setattr(bpy.data.objects['brace_band'],'matrix_parent_inverse',Matrix.Diagonal((1,1.1,1,1))))
run('camera_scale',lambda core,b,w,cc:setattr(core,'scale',(1.01,1,1)))
run('mirror',lambda core,b,w,cc:setattr(w,'scale',(-1,1,1)))
run('camera_source_A_used_instead_of_candidate',lambda core,b,w,cc:m.set_rigid_basis(core,cc['original_source_A_separate_from_candidate']))
run('canonical_corner_reorder',lambda core,b,w,cc:cc['canonical_definition']['corner_order'].reverse())
run('CAD_vertex_change',lambda core,b,w,cc:setattr(core.data.vertices[0],'co',(0,0,0)))
run('decal_separate_relative_shift',lambda core,b,w,cc:setattr(bpy.data.objects['right_Action4_core_top_id_decal'],'location',(.001,0,0)))
def disconnect(core,b,w,cc):
 mat=bpy.data.objects['right_forearm_natural_hand'].data.materials[0];link=next(l for l in mat.node_tree.links if l.from_node.type=='TEX_IMAGE');mat.node_tree.links.remove(link)
run('skin_active_texture_disconnect',disconnect)
run('observer_focal_change',lambda core,b,w,cc:setattr(bpy.data.objects['assembly_observer'].data,'lens',30))
run('dataset_permission_true',lambda core,b,w,cc:bpy.context.scene.__setitem__('dataset_generation_allowed',True))
for p,h in before.items():assert m.sha(p)==h
m.write(m.OUTPUT/'actual_counterexamples.json',{'status':'PASS12_ACTUAL_MUTATIONS_REJECTED','mutations':rows,'restored_each_case_and_reverified':True,'source_and_snapshot_hashes_unchanged':before,'saved_assets':False,'production_accepted':False})
print('PASS',len(rows))
