"""One-time independent environment/arm derivative; frozen DJI parts unchanged."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))

def run():
    import bpy
    from data.action4_phrase2_runtime import Phrase2Runtime
    out=R/"outputs/phrase2_generation_v001/assets";out.mkdir(exist_ok=True);rows=[]
    cad=np.load(R/"data/DJI_Action4_construct/r3_v001/mesh_v001.npz")["vertices_mm"]/1000
    for name in ["wall","table_full_garment","cloth_lift10_full_garment"]:
        destination=R/"data/action4_scene_templates/phrase2_v001"/name
        if destination.exists():raise FileExistsError("Existing phrase2 asset must not be overwritten")
        runtime=Phrase2Runtime(name);scene=runtime.scene;source_meta=runtime.metadata_path
        core_checks=[]
        for side in ["left","right"]:
            core=scene.objects[side+"_Action4_core"];v=np.array([x.co[:] for x in core.data.vertices]);assert np.allclose(v,cad,atol=1e-7,rtol=0)
            B=np.array(runtime.metadata["actor_groups"][side]["base_brace_capture_m"]);end=B@np.r_[runtime.arm_terminal_B[side],1];h=runtime.K@end[:3]
            core_checks.append({"side":side,"frozen_core_max_vertex_error_m":float(np.max(np.abs(v-cad))),"arm_terminal_uv":(h[:2]/h[2]/[scene.render.resolution_x,scene.render.resolution_y]).tolist(),"arm_axis_B":runtime.arm_axis_B[side].tolist(),"arm_repair":runtime.arm_repairs[side]})
        destination.mkdir(parents=True,exist_ok=False)
        scene.render.filepath=str(out/(name+".png"));scene.render.resolution_percentage=25;scene.render.engine="CYCLES";scene.cycles.device="GPU";scene.cycles.samples=16;scene.cycles.use_denoising=True;scene.render.threads_mode="FIXED";scene.render.threads=2;scene.render.use_compositing=False
        prefs=bpy.context.preferences.addons["cycles"].preferences;prefs.compute_device_type="CUDA";prefs.get_devices()
        for device in prefs.devices:device.use=device.type=="CUDA"
        bpy.ops.render.render(write_still=True);scene.render.resolution_percentage=100
        blend=destination/"scene.blend";bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        keep=["canonical_definition","K_native","W_from_capture","camera_name","near_plane_m","actor_groups","baseline_parameters_ref","parent_manifest_ref","sample_id"]
        metadata={k:runtime.metadata[k] for k in keep};metadata.update(schema="action4.phrase2_scene.v1",phrase2_arm_geometry_version="v001",arm_repairs=runtime.arm_repairs,blend_ref={"path":str(blend.relative_to(R)),"sha256":hashlib.sha256(blend.read_bytes()).hexdigest()},source_template_ref=source_meta,occlusion_mode="arm_geometry",floating_patches_allowed=False,table_layout="original source support plane, no far translation",status="CANDIDATE_VISUAL_AND_GEOMETRY_REVIEW_REQUIRED")
        (destination/"template.json").write_text(json.dumps(metadata,indent=2));rows.append({"template":name,"asset":str(blend.relative_to(R)),"checks":core_checks,"preview_only":{"samples":16,"resolution_percentage":25}})
    (out/"BUILD_REPORT.json").write_text(json.dumps({"rows":rows,"frozen_DJI_exact":True,"formal_accepted":False,"requires":"arm-end view rejection, fresh changed-geometry collision proof, native render quality"},indent=2));print("PHRASE2_ASSET_CANDIDATES_BUILT")
if __name__=="__main__":run()
