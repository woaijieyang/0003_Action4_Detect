"""Build a small, versioned upper-arm mesh overlay over admitted templates.

The hand/forearm, CAD, brace, materials and mounting matrices are untouched.
Only the old folded-back upper-arm centreline is replaced by a continuous
36 cm continuation. Cross-section sizes and mesh topology are retained.
The unchanged 44-vertex source rim is used exactly, including split normals.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build():
    import bpy
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import align
    from data.action4_arm_endcap_candidate import boundary_components
    from data.action4_modular_assembly import world_geometry
    from data.action4_phrase2_collision import joined_skin
    from data.action4_template_collision import Geometry,triangle_contact
    out=R/'data/action4_scene_templates/coverage1000_arm_v001';out.mkdir(exist_ok=False)
    reports=[]
    for scene in ['table_full_garment','cloth_lift10_full_garment','wall']:
        runtime=Phrase2Runtime(scene);meta=json.loads(json.dumps(runtime.metadata));asset=out/scene;asset.mkdir()
        report={'scene':scene,'source_blend':meta['blend_ref'],'sides':{}};arrays={}
        for side in ['left','right']:
            obj=runtime.scene.objects[side+'_forearm_continuation'];hand=runtime.scene.objects[side+'_forearm_natural_hand']
            groups,_=boundary_components(obj.data)
            if len(groups)!=1 or sorted(groups[0])!=list(range(44)):raise ValueError('Expected exact existing 44-point start rim')
            original=np.array([v.co[:] for v in obj.data.vertices]);polygons=[list(p.vertices) for p in obj.data.polygons]
            if len(original)!=24*44:raise ValueError('Unexpected admitted continuation ring topology')
            rings=original.reshape(24,44,3);centers=rings.mean(1);offsets=rings-centers[:,None]
            # Existing object and hand share the same local coordinate frame.
            if not np.allclose(obj.matrix_world,hand.matrix_world,atol=1e-8,rtol=0):raise ValueError('Source skin and continuation frames differ')
            segments=np.linalg.norm(np.diff(centers,axis=0),axis=1)
            length=float(np.linalg.norm(centers[-1]-centers[0]))
            if not .35<=length<=.37:raise ValueError('Preserve admitted ~36cm upper-arm bone length')
            tangent=(centers[2]-centers[0]);tangent/=np.linalg.norm(tangent)
            s=np.r_[0.,np.cumsum(segments)];s=s/s[-1]*length
            replacement=[]
            for i in range(24):
                local_tangent=centers[min(i+1,23)]-centers[max(0,i-1)]
                rotate=align(local_tangent,tangent)
                replacement.append(centers[0]+tangent*s[i]+offsets[i]@rotate.T)
            new=np.array(replacement).reshape(-1,3)
            # Exact seam ownership and normals are inherited, not approximated.
            new[:44]=original[:44]
            mesh=obj.data.copy();mesh.clear_geometry();mesh.from_pydata(new.tolist(),[],polygons);mesh.update()
            for poly in mesh.polygons:poly.use_smooth=True
            mesh.use_auto_smooth=True;mesh.calc_normals_split();hand.data.calc_normals_split()
            hverts=np.array([v.co[:] for v in hand.data.vertices]);hg,_=boundary_components(hand.data)
            if len(hg)!=1 or len(hg[0])!=44:raise ValueError('Unexpected hand rim')
            ids=np.array(sorted(hg[0]));dist=np.linalg.norm(original[:44,None]-hverts[ids][None],axis=-1);mapping=ids[dist.argmin(1)]
            if dist.min(1).max()>1e-7 or len(set(mapping))!=44:raise ValueError('Source rim mismatch')
            handnormals={int(v):[] for v in mapping}
            for loop in hand.data.loops:
                if loop.vertex_index in handnormals:handnormals[loop.vertex_index].append(np.array(loop.normal))
            normals={i:np.mean(handnormals[int(v)],axis=0) for i,v in enumerate(mapping)}
            normals={i:n/np.linalg.norm(n) for i,n in normals.items()}
            loops=[normals[l.vertex_index].tolist() if l.vertex_index<44 else list(l.normal) for l in mesh.loops]
            mesh.normals_split_custom_set(loops);obj.data=mesh;bpy.context.view_layer.update()
            v,f=world_geometry(obj);hv,hf=world_geometry(hand)
            body=Geometry.create(v,f);skin=Geometry.create(hv,hf);joined=joined_skin(body,skin)
            # Explicit nonadjacent self-contact check on the welded actual skin.
            candidates=joined.bvh().overlap(joined.bvh());bad=[]
            for a,b in candidates:
                if a>=len(f) and b>=len(f):continue  # Unchanged source hand-only pairs are outside this repair.
                if a>=b or set(joined.faces[a])&set(joined.faces[b]):continue
                if triangle_contact(joined.vertices[joined.faces[a]],joined.vertices[joined.faces[b]]):bad.append([int(a),int(b)]);break
            if bad:raise ValueError('Repaired joined arm self-intersection: '+str(bad))
            arrays[side+'_vertices_local_m']=new
            arrays[side+'_loop_normals']=np.array(loops)
            arrays[side+'_source_vertices_local_m']=original
            terminal=new[-44:].mean(0)
            meta['arm_repairs'][side].update(terminal_local=terminal.tolist(),terminal_vertex_indices=list(range(23*44,24*44)),
                original_upperarm_endpoint_distance_m=length,repaired_upperarm_endpoint_distance_m=float(np.linalg.norm(terminal-new[:44].mean(0))),
                source_rim_vertices_unchanged=True,hand_and_forearm_vertices_unchanged=True,continuous_visible_junction_candidate=True)
            report['sides'][side]={'source_rim_error_m':float(np.max(abs(new[:44]-original[:44]))),'joined_skin_oriented_closed':bool(joined.closed),
                'new_continuation_self_or_hand_intersections':len(bad),'upperarm_bone_length_before_m':length,'upperarm_bone_length_after_m':float(np.linalg.norm(terminal-new[:44].mean(0))),
                'source_cross_section_radii_preserved':bool(np.allclose(np.linalg.norm(np.array(replacement)-np.array(replacement).mean(1)[:,None],axis=2),np.linalg.norm(offsets,axis=2),atol=1e-8)),
                'hand_vertices_unchanged':True,'DJI_and_installation_unchanged':True}
        path=asset/'upperarm_overlay.npz';np.savez_compressed(path,**arrays)
        meta.update(phrase2_arm_geometry_version='coverage1000_continuous_upperarm_v001',coverage_arm_overlay_ref={'path':str(path.relative_to(R)),'sha256':sha(path)},
            coverage_arm_parent_metadata=runtime.metadata_path,coverage_arm_candidate=True,coverage_arm_visual_admitted=False)
        (asset/'template.json').write_text(json.dumps(meta,indent=2));reports.append(report)
    manifest={'status':'GEOMETRY_PASS_VISUAL_REVIEW_REQUIRED','schema':'action4.coverage_arm_overlay.v1','rows':reports,
        'source_builder':{'path':'src/data/action4_coverage1000_arm_asset.py','sha256':sha(R/'src/data/action4_coverage1000_arm_asset.py')},
        'formal_generation_allowed':False,'preserved':'DJI, canonical8, mount, wrist, natural hand/forearm, scene, camera and materials',
        'change':'Only static upper-arm continuation: same 36cm endpoint length, cross sections and topology; folded-back centreline replaced by continuous outgoing centreline.'}
    (out/'ASSET_GEOMETRY.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

if __name__=='__main__':build()
