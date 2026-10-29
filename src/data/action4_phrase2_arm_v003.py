"""Restore original22cm forearm+36cm upper-arm proportions; cap only off-frame terminal."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
import bpy,bmesh
from data.action4_arm_endcap_candidate import boundary_components,world_points

def run():
    out=R/"outputs/phrase2_generation_v001/assets/arm_v003";out.mkdir(parents=True,exist_ok=True);rows=[]
    sources={"wall":"wall","table_full_garment":"table_v002","cloth_lift10_full_garment":"cloth_v002"}
    for name,base in sources.items():
        src=R/"data/action4_scene_templates/phrase2_v002"/name;meta=json.loads((src/"template.json").read_text());bpy.ops.wm.open_mainfile(filepath=str(R/meta["blend_ref"]["path"]))
        original_meta=json.loads((R/"data/action4_scene_templates/v007_candidate"/base/"template.json").read_text())
        with bpy.data.libraries.load(str(R/original_meta["blend_ref"]["path"]),link=False) as (available,selected):
            selected.objects=[s+"_forearm_continuation" for s in ["left","right"]]
        repairs={}
        for side,loaded in zip(["left","right"],selected.objects):
            obj=bpy.context.scene.objects[side+"_forearm_continuation"];obj.data=loaded.data.copy();original=np.array([v.co[:] for v in obj.data.vertices]);bpy.data.objects.remove(loaded,do_unlink=True)
            groups,_=boundary_components(obj.data);hand=bpy.context.scene.objects[side+"_forearm_natural_hand"];hg,_=boundary_components(hand.data);hp=world_points(hand,sorted(hg[0]))
            distance=[np.linalg.norm(world_points(obj,sorted(g))[:,None]-hp[None],axis=-1).min(1).mean() for g in groups];near=int(np.argmin(distance));far=1-near
            bm=bmesh.new();bm.from_mesh(obj.data);bm.verts.ensure_lookup_table();terminal=[e for e in bm.edges if e.is_boundary and all(v.index in groups[far] for v in e.verts)];made=bmesh.ops.holes_fill(bm,edges=terminal,sides=0);assert made["faces"];bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()
            assert np.array_equal(original,np.array([v.co[:] for v in obj.data.vertices]))
            repairs[side]={"original_vertices_exact":True,"length_extension_m":0.,"terminal_local":original[sorted(groups[far])].mean(0).tolist(),"terminal_vertex_indices":sorted(groups[far]),"junction_local":original[sorted(groups[near])].mean(0).tolist(),"junction_error_m":float(distance[near]),"collision_volume":"source hand+forearm exact seam welded only in collision representation"}
        destination=R/"data/action4_scene_templates/phrase2_v003"/name;destination.mkdir(parents=True,exist_ok=False);blend=destination/"scene.blend";bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        meta.update(phrase2_arm_geometry_version="original_proportions_capped_v003",arm_repairs=repairs,parent_phrase2_asset=str((src/"template.json").relative_to(R)),blend_ref={"path":str(blend.relative_to(R)),"sha256":hashlib.sha256(blend.read_bytes()).hexdigest()})
        (destination/"template.json").write_text(json.dumps(meta,indent=2));rows.append({"template":name,"repairs":repairs})
    (out/"REPORT.json").write_text(json.dumps({"rows":rows,"prior_extended_candidate_rejected":True,"formal_accepted":False},indent=2))
if __name__=="__main__":run()
