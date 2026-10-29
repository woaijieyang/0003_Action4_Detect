from pathlib import Path
import sys,json
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
from data.action4_arm_endcap_candidate import boundary_components
r=CoverageArmRuntime('table_full_garment');rows={}
import bpy
for side in ['left','right']:
    hand=r.scene.objects[side+'_forearm_natural_hand'];obj=r.scene.objects[side+'_forearm_continuation']
    mat=hand.data.materials[0];nodes=[]
    for node in mat.node_tree.nodes:
        nodes.append({'name':node.name,'type':node.bl_idname,'image':getattr(getattr(node,'image',None),'name',None),
            'inputs':{s.name:list(s.default_value) if hasattr(s.default_value,'__len__') else s.default_value for s in node.inputs if hasattr(s,'default_value') and not isinstance(s.default_value,str)}})
    links=[(l.from_node.name,l.from_socket.name,l.to_node.name,l.to_socket.name) for l in mat.node_tree.links]
    normrows=[];obj.data.calc_normals_split();vertices=np.array([v.co[:] for v in obj.data.vertices]).reshape(24,44,3)
    for i in [0,1,2,8,14,23]:
        points=vertices[i];_,_,basis=np.linalg.svd(points-points.mean(0));axis=(vertices[1].mean(0)-vertices[0].mean(0));axis/=np.linalg.norm(axis)
        normal=basis[-1];n=[list(l.normal) for l in obj.data.loops if i*44<=l.vertex_index<(i+1)*44]
        normrows.append({'ring':i,'plane_normal_dot_axis':float(abs(normal@axis)),'loop_normal_axis_absmean':float(np.mean(abs(np.array(n)@axis)))})
    fresh=bpy.data.meshes.new('normal_reference_'+side);fresh.from_pydata(np.array([v.co[:] for v in obj.data.vertices]).tolist(),[],[list(p.vertices) for p in obj.data.polygons]);fresh.update()
    for face in fresh.polygons:face.use_smooth=True
    fresh.use_auto_smooth=True;fresh.auto_smooth_angle=np.pi;fresh.calc_normals_split()
    a=np.array([l.normal[:] for l in obj.data.loops]);b=np.array([l.normal[:] for l in fresh.loops]);cos=np.einsum('ij,ij->i',a,b)
    different={'mean_cosine':float(cos.mean()),'max_angle_deg':float(np.degrees(np.arccos(np.clip(cos,-1,1))).max()),'fraction_angle_over_30deg':float((cos<np.cos(np.pi/6)).mean()),'fraction_opposite':float((cos<0).mean())}
    from collections import Counter
    slots={a.name:[m.name if m else None for m in a.data.materials] for a in [hand,obj]}
    counts={a.name:dict(Counter(p.material_index for p in a.data.polygons)) for a in [hand,obj]}
    actual={a.name:[{'link':s.link,'material':s.material.name if s.material else None} for s in a.material_slots] for a in [hand,obj]}
    rows[side]={'actual_object_material_slots':actual,'material_slots':slots,'material_indices':counts,'fresh_geometric_normal_comparison':different,'nodes':nodes,'links':links,'hand_uv':[x.name for x in hand.data.uv_layers],'continuation_uv':[x.name for x in obj.data.uv_layers],
        'hand_texspace':[list(hand.data.texspace_location),list(hand.data.texspace_size),hand.data.use_auto_texspace],
        'continuation_texspace':[list(obj.data.texspace_location),list(obj.data.texspace_size),obj.data.use_auto_texspace],'rings':normrows}
(R/'outputs/coverage1000_v001/arm_repair_v001/SHADING_AUDIT_FRESH_NORMALS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
