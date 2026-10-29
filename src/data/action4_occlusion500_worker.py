"""Additional500 batch: derived from src/data/action4_phrase2_worker.py; see DERIVATION.json."""
"""Persistent one-template Blender worker; bounded attempts and immutable frame packets."""
from pathlib import Path
import sys,json,os,time,hashlib,collections,traceback
import numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))

def run(template,worker_id):
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_occlusion500_sampling import propose,occlusion_bin
    from data.action4_phrase2_surface import SurfaceDeriver
    from data.action4_phrase2_scene_visibility import SceneVisibility
    from data.action4_occlusion500_contract import eligible_sources
    from data import action4_phrase2_pixels as pixels
    from data import action4_occlusion500_queue as queue
    cfg=json.loads((R/"configs/data/action4_phrase2_occlusion500_v001.json").read_text());out=R/cfg["output_root"];catalog=json.loads((R/cfg["occlusion_catalog"]).read_text())
    from data.action4_occlusion500_contract import assert_configuration
    assert_configuration(cfg)
    runtime=Phrase2Runtime(template)
    if runtime.metadata.get("phrase2_arm_geometry_version")!="original_proportions_capped_v003":raise ValueError("Formal phrase2 worker requires corrected original arm proportions")
    checker=Phrase2CollisionChecker(runtime);trees={s:LocalArmVisibility(runtime,s) for s in ["left","right"]};deriver=SurfaceDeriver();scene_visibility=SceneVisibility(runtime,trees)
    canonical=runtime.metadata["canonical_definition"];ids={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in pixels.REQUIRED_SOURCE_PATHS};ids[canonical["source_manifest"]["path"]]=canonical["source_manifest"]["sha256"]
    for path in ["src/data/action4_occlusion500_contract.py","configs/data/action4_phrase2_occlusion500_v001.json"]:ids[path]=hashlib.sha256((R/path).read_bytes()).hexdigest()
    worker=out/"workers"/worker_id;worker.mkdir(parents=True,exist_ok=False)
    sources={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in ["src/data/action4_occlusion500_worker.py","src/data/action4_phrase2_runtime.py","src/data/action4_occlusion500_sampling.py","src/data/action4_phrase2_geometry.py","src/data/action4_phrase2_collision.py","src/data/action4_phrase2_surface.py","src/data/action4_phrase2_scene_visibility.py",runtime.metadata_path]}
    (worker/"INITIALIZED.json").write_text(json.dumps({"pid":os.getpid(),"template":template,"loads":1,"source_identities":sources,"asset":runtime.metadata["blend_ref"],"mesh_prepared_once":True,"random_cloth_objects":False},indent=2))
    counters=collections.Counter();rendered=0;start=time.monotonic()
    while not (out/"STOP.json").exists():
        claim=queue.claim(template)
        if claim is None:
            db=queue.connect();remaining=db.execute("SELECT count(*) FROM slots WHERE scene=? AND status!='accepted'",(template,)).fetchone()[0];db.close()
            if not remaining:break
            time.sleep(2);continue
        slot=json.loads(claim["spec"]);submitted=False
        try:
            for _ in range(cfg["max_attempts_per_claim"]):
                attempt=queue.next_attempt(slot["slot_id"])
                if attempt>=cfg["max_total_attempts_per_slot"]:queue.transition(slot["slot_id"],"unfilled",error="bounded_total_attempts_exhausted");submitted=True;break
                proposal,error=propose(runtime,cfg,slot,attempt,catalog,local_visibility=trees)
                if error:counters[error]+=1;continue
                estimates={}
                for side in ["left","right"]:
                    pose=proposal["actor_pose_capture_by_side"][side]
                    estimates[side]=trees[side].estimate(np.array(pose["brace_B_capture"]),np.array(pose["core_C_capture"]),28,return_sources=True)
                if any(m["occlusion_estimate"] is None or occlusion_bin(m["occlusion_estimate"])!=slot["occlusion_bins"][s] or (slot["occlusion_bins"][s]>0 and m["primary_class"]!="own_wrist_forearm") for s,m in estimates.items()):counters["estimated_occlusion_or_source"]+=1;continue
                runtime.apply(proposal);collision=checker.check()
                if not collision["accepted"]:counters[collision["status"]]+=1;continue
                source_measurements=scene_visibility.measure(proposal)
                source_ok,source_reason=eligible_sources(source_measurements,slot["occlusion_bins"])
                if not source_ok:counters[source_reason]+=1;continue
                actual=runtime.export_corners()
                case=worker/f"{slot['sample_id']}_a{attempt:05d}";case.mkdir()
                data=R/"data/phrase2/.staging_batches/occlusion500_v001"/slot["split"]/".staging"/slot["sample_id"]/f"a{attempt:05d}";data.mkdir(parents=True,exist_ok=False)
                (case/"FRAME_SNAPSHOT.json").write_text(json.dumps({"slot":slot,"parameters":proposal,"collision":collision,"actual_corner_export":actual,"estimated_occlusion":estimates,"generation_sources":sources},indent=2))
                settings={"device":"CUDA","denoiser":"OPTIX","samples":cfg["rgb_samples"],"mask_samples":1,"amodal_mode":"crop","amodal_margin_px":16,"seed":proposal["entry"]["seed"],"worker_gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"]}
                begin=time.monotonic();result=pixels.render_frame(scene=runtime.scene,entry=proposal["entry"],targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=data,runtime_out=case,output_role="dataset_candidate",source_identities=ids,render_settings=settings,verify_scene=runtime.verify)
                record=result["record"];record["occlusion_sources"]=source_measurements
                for obj,side in zip(record["instances"],["left","right"]):
                    source=source_measurements[side]
                    if source["sampled_amodal_body_rays"]>50 and abs(obj["occlusion_inframe"]-source["estimated_external_occlusion"])>.15:
                        raise ValueError("Actual mask vs scene-ray visibility has a major engineering mismatch")
                enough=(len(record["instances"])==2 and not record["excluded_instances"] and
                        all(not obj.get("ignore",False) and occlusion_bin(obj["occlusion_inframe"])==slot["occlusion_bins"][side]
                            for obj,side in zip(record["instances"],["left","right"])))
                if enough:record=deriver.derive(record,data/"geometry")
                record["frame_snapshot_ref"]=str((case/"FRAME_SNAPSHOT.json").relative_to(R));record["asset_version"]="phrase2_scene_layout_v002_original_arms_v003"
                payload={"record":record,"proposal":proposal,"slot":slot,"collision":collision,"actual_projection_error_px":actual["max_proposal_vs_actual_error_px"],"runtime_dir":str(case.relative_to(R)),"staging_dir":str(data.relative_to(R)),"render_seconds":time.monotonic()-begin}
                (case/"CANDIDATE.json").write_text(json.dumps(payload,allow_nan=False))
                queue.transition(slot["slot_id"],"rendered",record={"candidate":str((case/"CANDIDATE.json").relative_to(R))});rendered+=1;submitted=True;break
            if not submitted:queue.transition(slot["slot_id"],"pending")
        except BaseException:
            failure={"worker":worker_id,"pid":os.getpid(),"slot":slot,"traceback":traceback.format_exc()}
            (worker/"FAILURE.json").write_text(json.dumps(failure,indent=2));queue.transition(slot["slot_id"],"failed",error=failure["traceback"]);(out/"STOP.json").write_text(json.dumps(failure,indent=2));raise
        (worker/"STATUS.json").write_text(json.dumps({"rendered":rendered,"rejections":dict(counters),"elapsed_seconds":time.monotonic()-start}))
    (worker/"COMPLETE.json").write_text(json.dumps({"rendered":rendered,"rejections":dict(counters),"seconds":time.monotonic()-start}))
if __name__=="__main__":
    i=sys.argv.index("--");run(sys.argv[i+1],sys.argv[i+2])
