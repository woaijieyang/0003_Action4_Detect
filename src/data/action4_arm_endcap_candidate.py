"""Independent terminal-cap candidate; not a body-attachment or realism solution."""
from pathlib import Path
import sys,json
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
import bpy,bmesh,numpy as np
from data.action4_template_runtime import TemplateRuntime,read


def world_points(obj,indices):
    v=np.array([obj.data.vertices[i].co[:] for i in indices]);m=np.array(obj.matrix_world)
    return v@m[:3,:3].T+m[:3,3]


def boundary_components(mesh):
    counts={}
    for p in mesh.polygons:
        v=list(p.vertices)
        for a,b in zip(v,v[1:]+v[:1]):
            e=tuple(sorted((a,b)));counts[e]=counts.get(e,0)+1
    edges=[e for e,c in counts.items() if c==1];remaining=set(i for e in edges for i in e);groups=[]
    while remaining:
        found={remaining.pop()};todo=list(found)
        while todo:
            u=todo.pop()
            for a,b in edges:
                v=b if a==u else a if b==u else None
                if v in remaining:remaining.remove(v);found.add(v);todo.append(v)
        groups.append(found)
    return groups,edges


def run():
    out=R/'outputs/action4_arm_endcap_candidate_v001';out.mkdir(exist_ok=True)
    case=read('configs/data/action4_counterfactual_diagnostic_v002.json')['cases'][0]
    runtime=TemplateRuntime(case['template']);runtime.apply(read(case['parameters_ref']['path']));before=runtime.export_corners();rows=[]
    for side in ['left','right']:
        hand=runtime.scene.objects[side+'_forearm_natural_hand'];obj=runtime.scene.objects[side+'_forearm_continuation']
        loops,edges=boundary_components(obj.data);hg,_=boundary_components(hand.data)
        assert len(loops)==2 and len(hg)==1
        hp=world_points(hand,sorted(hg[0]));dist=[]
        for loop in loops:
            vv=world_points(obj,sorted(loop));dist.append(float(np.linalg.norm(vv[:,None]-hp[None],axis=-1).min(1).mean()))
        near=int(np.argmin(dist));far=1-near
        if dist[near]>.002 or dist[far]<.01:raise ValueError('Cannot safely identify junction and terminal ring: '+str(dist))
        original=np.array([v.co[:] for v in obj.data.vertices]);obj.data=obj.data.copy()
        bm=bmesh.new();bm.from_mesh(obj.data);bm.verts.ensure_lookup_table()
        terminal=[e for e in bm.edges if e.is_boundary and all(v.index in loops[far] for v in e.verts)]
        made=bmesh.ops.holes_fill(bm,edges=terminal,sides=0);assert made['faces']
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free();obj.data.update()
        assert np.array_equal(original,np.array([v.co[:] for v in obj.data.vertices]))
        after_loops,after_edges=boundary_components(obj.data);assert len(after_loops)==1 and len(after_edges)==44
        rows.append({'side':side,'junction_distance_m':dist[near],'terminal_distance_from_hand_m':dist[far],'boundary_edges_before':len(edges),'boundary_edges_after':len(after_edges),'all_vertex_positions_unchanged':True,'junction_ring_preserved':True})
    runtime.verify(runtime.scene);after=runtime.export_corners()
    assert before['canonical8_by_instance']==after['canonical8_by_instance'],'Corner or visibility changed; inspect before publishing even diagnostic'
    asset=R/'data/action4_scene_assets/arm_endcap_candidate_v001';asset.mkdir(exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(asset/'wall_diagnostic.blend'))
    (out/'REPORT.json').write_text(json.dumps({'status':'TOPOLOGY_ONLY_CANDIDATE_NOT_REALISM_ACCEPTED','rows':rows,'canonical_corner_and_visibility_exports_unchanged':True,'asset':str((asset/'wall_diagnostic.blend').relative_to(R)),'training_use':False,'remaining':'terminal may still be visible as a stump; needs body/sleeve attachment and visual review'},indent=2))
if __name__=='__main__':run()
