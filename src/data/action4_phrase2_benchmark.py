"""One representative slot per scene, native32 vs64 RGB, immutable parameters."""
from pathlib import Path
import sys,json,hashlib,time,os,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))

def run():
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data import action4_phrase2_pixels as pixels
    from data import action4_global_render as old
    from data import action4_phrase2_amodal as new
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    cfg=json.loads((R/"configs/data/action4_phrase2_v001.json").read_text());ready=json.loads((R/cfg["output_root"]/"PREFLIGHT_GEOMETRY.json").read_text())
    out=R/cfg["output_root"]/"render_benchmark_v003";out.mkdir(exist_ok=False);rows=[]
    ids={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in pixels.REQUIRED_SOURCE_PATHS}
    for spec in ready["templates"]:
        row=next(x for x in spec["found"] if not x.get("unfilled") and x["slot"]["slot_id"] not in [745,2283]);slot=row["slot"];proposal=row["parameters"]
        runtime=Phrase2Runtime(spec["template"]);checker=Phrase2CollisionChecker(runtime);runtime.apply(proposal);collision=checker.check()
        assert collision["accepted"];actual=runtime.export_corners();canonical=runtime.metadata["canonical_definition"];target=runtime.targets()[1]
        with old.isolated_expanded(runtime.scene,target["parts"]):
            obj=target["core"];values=np.array([v.co[:] for v in obj.data.vertices]);m=np.array(obj.matrix_world);world=values@m[:3,:3].T+m[:3,3];vectorized=new.project_world(world,runtime.scene)
            index=np.linspace(0,len(world)-1,64,dtype=int);legacy=[]
            for point in world[index]:
                p=world_to_camera_view(runtime.scene,runtime.scene.camera,Vector(point));legacy.append([p.x*runtime.scene.render.resolution_x,p.y*runtime.scene.render.resolution_y,p.z])
            error=float(np.max(np.abs(vectorized[index]-legacy)));assert error<.005
            a=old.crop_geometry_preflight(runtime.scene,obj,target["parts"],np.array(canonical["corners_3d_m"]));b=new.geometry_preflight(runtime.scene,obj,target["parts"],np.array(canonical["corners_3d_m"]));assert a==b
        pixel_ids=dict(ids);pixel_ids[canonical["source_manifest"]["path"]]=canonical["source_manifest"]["sha256"]
        for samples in [32,64]:
            dest=out/(runtime.name+f"_{samples}");dest.mkdir();data=dest/"data";data.mkdir();meta=dest/"runtime";meta.mkdir()
            settings={"device":"CUDA","denoiser":"OPTIX","samples":samples,"mask_samples":1,"amodal_mode":"crop","amodal_margin_px":16,"seed":proposal["entry"]["seed"],"worker_gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"]}
            begin=time.monotonic()
            result=pixels.render_frame(scene=runtime.scene,entry=proposal["entry"],targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=data,runtime_out=meta,output_role="diagnostic",source_identities=pixel_ids,render_settings=settings,verify_scene=runtime.verify)
            rows.append({"template":runtime.name,"samples":samples,"slot_id":slot["slot_id"],"attempt":row["attempt"],"seconds":time.monotonic()-begin,"record_ref":result["record_ref"],"source_parameters":proposal,"actual_projection_error":actual["max_proposal_vs_actual_error_px"],"vectorized_projection_error_px":error})
            (out/"PROGRESS.json").write_text(json.dumps({"rows":rows},indent=2));print(runtime.name,samples,rows[-1]["seconds"],flush=True)
    (out/"COMPLETE.json").write_text(json.dumps({"rows":rows,"formal_accepted":False},indent=2))
if __name__=="__main__":run()
