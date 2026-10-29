"""One scene-level support-plane correction, preserving environment size and arm/CAD geometry."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
import bpy
from mathutils import Matrix

def run():
    out=R/"outputs/phrase2_generation_v001/assets/layout_v002";out.mkdir(parents=True,exist_ok=True);rows=[]
    for name in ["wall","table_full_garment","cloth_lift10_full_garment"]:
        src=R/"data/action4_scene_templates/phrase2_v001"/name;meta=json.loads((src/"template.json").read_text());bpy.ops.wm.open_mainfile(filepath=str(R/meta["blend_ref"]["path"]));scene=bpy.context.scene;W=np.array(meta["W_from_capture"])
        if name=="wall":roots=["modular_preview_table"];translation=[0,.10,.18]
        else:roots=["modular_environment_v002_table","wearing_garment_table_garment_frame"];translation=[0,-.10,.18]
        delta=np.eye(4);delta[:3,3]=W[:3,:3]@np.array(translation);changes=[]
        actors={n:np.array(scene.objects[n].matrix_world) for group in meta["actor_groups"].values() for n in group["objects"]}
        for key in roots:
            obj=scene.objects[key];before=np.array(obj.matrix_world);obj.matrix_world=Matrix((delta@before).tolist());changes.append({"root":key,"before":before.tolist(),"after":np.array(obj.matrix_world).tolist()})
        bpy.context.view_layer.update()
        assert all(np.array_equal(np.array(scene.objects[n].matrix_world),v) for n,v in actors.items())
        destination=R/"data/action4_scene_templates/phrase2_v002"/name;destination.mkdir(parents=True,exist_ok=False)
        scene.render.use_compositing=True;scene.render.resolution_percentage=100
        blend=destination/"scene.blend";bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        meta.update(scene_layout_version="v002",layout_translation_capture_m=translation,layout_rationale="finite tabletop working clearance plus larger projected occupancy; not old1.05m far shift",parent_phrase2_asset=str((src/"template.json").relative_to(R)),blend_ref={"path":str(blend.relative_to(R)),"sha256":hashlib.sha256(blend.read_bytes()).hexdigest()})
        (destination/"template.json").write_text(json.dumps(meta,indent=2));rows.append({"template":name,"translation_capture_m":translation,"changes":changes,"actor_matrices_exact":True})
    (out/"REPORT.json").write_text(json.dumps({"rows":rows,"body_scale_and_shape_unchanged":True,"native_render_review_required":True},indent=2))
if __name__=="__main__":run()
