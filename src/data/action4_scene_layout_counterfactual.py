"""Independent layout/occluder counterfactual; never publishes training samples."""
from pathlib import Path
import json,sys,hashlib
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))

def run():
    import bpy,numpy as np
    from mathutils import Matrix
    from data.action4_template_runtime import TemplateRuntime,read
    cfg=read("configs/data/action4_scene_layout_counterfactual_v001.json")
    out=R/cfg["output"];out.mkdir(exist_ok=False)
    record=next(r for r in map(json.loads,(R/"data/train/annotations.jsonl").read_text().splitlines()) if r["sample_id"]==cfg["sample_id"])
    parameters=read(record["template_parameters_ref"]["path"])
    runtime=TemplateRuntime("table_full_garment");runtime.apply(parameters)
    actor_names=set(n for group in runtime.metadata["actor_groups"].values() for n in group["objects"])
    actor_world={n:np.array(runtime.scene.objects[n].matrix_world).copy() for n in actor_names}
    meshes={}
    for n in actor_names:
        o=runtime.scene.objects[n]
        if o.type=="MESH":meshes[n]=np.array([v.co[:] for v in o.data.vertices])
    before=runtime.export_corners();scene=runtime.scene
    def render(name):
        scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=8
        scene.cycles.use_denoising=False;scene.render.threads_mode="FIXED";scene.render.threads=2
        scene.render.use_compositing=False;scene.render.resolution_percentage=25
        scene.render.image_settings.file_format="PNG";scene.render.filepath=str(out/name)
        bpy.ops.render.render(write_still=True)
        scene.render.resolution_percentage=100
    render("before.png")
    changes=[]
    baseline=runtime.metadata["baseline_scene_state"]["objects"]
    for name in cfg["restore_static_roots"]:
        if name in actor_names:raise ValueError("May not restore an actor")
        obj=scene.objects[name];old=np.array(obj.matrix_world)
        expected=np.array(baseline[name]["world"]);obj.matrix_world=Matrix(expected.tolist())
        changes.append({"object":name,"before":old.tolist(),"after":expected.tolist()})
    for patch in runtime.patches.values():patch.hide_render=True
    bpy.context.view_layer.update()
    for name,matrix in actor_world.items():
        if not np.array_equal(np.array(scene.objects[name].matrix_world),matrix):raise ValueError("Actor transform changed")
    for name,vertices in meshes.items():
        if not np.array_equal(np.array([v.co[:] for v in scene.objects[name].data.vertices]),vertices):raise ValueError("Frozen actor geometry changed")
    after=runtime.export_corners()
    render("candidate.png")
    asset=R/cfg["asset_output"];asset.mkdir(parents=True,exist_ok=False)
    scene.render.filepath=str(out/"candidate.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(asset/"candidate.blend"))
    report={"status":"COUNTERFACTUAL_RENDER_ONLY_NOT_TRAINING_ACCEPTED","sample_id":record["sample_id"],
        "static_changes":changes,"independent_patches_hidden":True,"frozen_actor_geometry_and_transforms_exact":True,
        "before_corner_projection_error_px":before["max_proposal_vs_actual_error_px"],
        "after_corner_projection_error_px":after["max_proposal_vs_actual_error_px"],
        "actor_metadata_unchanged":True,"camera_unchanged":True,
        "preview_render":{"resolution_percentage":25,"samples":8,"device":"CPU","threads":2},
        "physics_collision_pass":False,"training_allowed":False,
        "remaining":"requires joint support-plane sampling, connected garment occlusion and arm-end handling; not a corrected dataset",
        "asset":str((asset/"candidate.blend").relative_to(R))}
    (out/"REPORT.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":run()
