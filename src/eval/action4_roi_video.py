"""Development video orchestration using the existing resolved-ROI pipeline."""
import copy,json,hashlib,time,itertools,os
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_video_roi_context import VideoROIContext
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_staged_pipeline import StagedPipeline
from eval.action4_sam_full_review import ReceiptVerifier

def bootstrap_context(provider,frame_id,rgb,K,baseline,seeds,bootstrap_identity):
    """Replay the existing five-frame offline confirmation using real current B0 boxes."""
    row=copy.deepcopy(provider.rows[frame_id]);tracks=[]
    import re
    bygrid={d["grid_index"]:d for d in baseline["prediction"]["detections"]}
    for seed in seeds:
        refs=[re.fullmatch(r"(?:b0:)?frame(\d+):grid:(\d+)",v) for v in seed["evidence_ids"]]
        grids=[int(m[2]) for m in refs if m and int(m[1])==frame_id]
        if len(grids)>1:raise ValueError("Ambiguous bootstrap source")
        if not grids or grids[0] not in bygrid:continue
        raw=copy.deepcopy(bygrid[grids[0]])
        if raw["score"]<.3:continue
        raw["candidate_id"]=f"b0:frame{frame_id}:grid:{grids[0]}"
        tracks.append({"track_id":seed["track_id"],"status":"observed","candidate_id":raw["candidate_id"],
            "raw_detection":raw,"bbox_for_roi":raw["bbox_full_xyxy"],"sam_current_support":False,
            "source":"offline_confirmation_at_frame4_from_actual_B0","calibration_frames":[]})
    row.update(fused={"mode":"video_inference","frame_id":frame_id,"tracks":tracks},time_base=provider.region["time_base"])
    return VideoROIContext.from_replay(row,rgb,K,b0_sha256=provider.b0_sha,replay_identity=bootstrap_identity)

def run_part(config_path,part):
    import torch
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text());job=cfg["parts"][part]
    from eval.action4_video_roi_contract import config_ratio
    selected_ratio=config_ratio(cfg) if cfg.get("corner_trial_binding") is not None or cfg.get("video_sam_features") is not None else None
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=job["gpu_uuid"] or set(os.sched_getaffinity(0))!=set(job["cpus"]):
        raise ValueError("Assigned video worker resource differs")
    from model.training.action4_resource_policy import validate_device_policy
    validate_device_policy({"training_gpus":[],"eval_gpus":[job["gpu"]]},R)
    import subprocess
    info=subprocess.check_output(["nvidia-smi","-i",str(job["gpu"]),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if info[0].strip()!=job["gpu_uuid"] or int(info[1])<16384:raise ValueError("Fresh video GPU headroom differs")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(float(cfg.get("allocator_gib",8))*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.use_deterministic_algorithms(True)
    lock=read_binding(cfg["input_lock"]);gate=read_binding(lock["formal_pair_gate"])
    v=ReceiptVerifier();provider=ReanchorROIProvider(**dict(lock["video"]["track_replay"],
        register_project_bodies=cfg.get("video_sam_features")=="current_native_v1"))
    for x in gate["models"].values():v.verify(x["path"],x["sha256"])
    corner_trial=cfg.get("corner_trial_binding")
    if corner_trial is not None:
        if cfg["ratios"]!=[selected_ratio]:
            raise ValueError("Corner video trial has one selected ROI")
        acfg=read_binding(cfg["corner_training_config"])
        pipe=StagedPipeline.from_corner_bundles(gate["models"]["A0"]["path"],gate["models"]["B0"]["path"],
            acfg["sam_eval_cache"],device="cuda:0",full_box_quality_receipt=acfg["b0_full_box_quality_receipt"])
        from eval.action4_corner_video import validate_corner_trial_row
    else:
        pipe=StagedPipeline.from_bundles(gate["models"]["A0"]["path"],gate["models"]["B0"]["path"],device="cuda:0")
    pipe.b0.eval().requires_grad_(False)
    enhanced_video=pipe.a0_kind in ("joint_pose_sam","joint_pose_sam_corner_candidate")
    if enhanced_video:
        if cfg.get("video_sam_features")!="current_native_v1" or cfg["ratios"]!=[pipe.roi_expansion]:
            raise ValueError("SAM-enhanced video requires explicit current native features and selected model ROI")
        if not pipe.acfg.get("sam_cache",{}).get("expected_sources",{}).get("sam_model_sha256"):
            raise ValueError("Enhanced model lacks frozen SAM input identity")
    elif cfg.get("video_sam_features") is not None:
        raise ValueError("Baseline video cannot ignore an enhanced feature request")
    pipe.b0_scoring_policy=lock["image_eval"]["B0_scoring_policy"]
    if corner_trial is None and pipe.pose_method!="network":raise ValueError("No unadmitted post-A0 geometry in this sweep")
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"]);cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000.
    seeds=read_binding(cfg["bootstrap_seeds"])["seeds"]
    with v.verify(lock["video"]["raw_B0_operating_reference"]["path"],lock["video"]["raw_B0_operating_reference"]["sha256"]).open() as f:
        first=[json.loads(line) for line in itertools.islice(f,4)]
    bootstrap_id=hashlib.sha256(json.dumps({"B0":lock["video"]["raw_B0_operating_reference"],"seeds":cfg["bootstrap_seeds"],"policy":"shared_existing_offline_confirmation_at4"},sort_keys=True).encode()).hexdigest()
    out=contained_path(R,cfg["output"],exists=True)/f"part_{part:03d}";out.mkdir(exist_ok=False)
    files={e:(out/("e"+format(e,".2f").replace(".","p")+".jsonl")).open("x") for e in cfg["ratios"]}
    reference_files=read_binding(cfg["reference_225_files"]) if 2.25 in cfg["ratios"] else {}
    calls={"B0":0,"backbone":0};hooks=[]
    for name,module in [("B0",pipe.b0),("backbone",pipe.base.backbone)]:
        def counter(*args,name=name):calls[name]+=1
        hooks.append(module.register_forward_hook(counter))
    n=0;beg=time.monotonic()
    try:
        for fid in range(job["start"],job["end"]):
            sam_features=None
            if enhanced_video:
                from eval.action4_video_sam_inputs import VideoSamFeatureContext
                bootstrap=({"raw_B0_operating_reference":lock["video"]["raw_B0_operating_reference"],
                    "bootstrap_seeds":cfg["bootstrap_seeds"]} if fid<4 else None)
                rgb,context,sam_features=VideoSamFeatureContext.load(provider,fid,K,
                    expected_b0_sha256=pipe.identity["B0_bundle_sha256"],
                    expected_sam_model_sha256=pipe.acfg["sam_cache"]["expected_sources"]["sam_model_sha256"],bootstrap=bootstrap)
            else:
                rgb,context=provider.load(fid,K)
                if fid<4:
                    if first[fid]["frame_id"]!=fid or first[fid]["rgb_sha256"]!=provider.rows[fid]["frame"]["sha256"]:raise ValueError("Bootstrap RGB source differs")
                    context=bootstrap_context(provider,fid,rgb,K,first[fid],seeds,bootstrap_id)
            resolved=context.select_for_a0(context.consume(rgb,K,b0_sha256=provider.b0_sha))
            expected={d["track_id"]:d for d in resolved}
            for e in cfg["ratios"]:
                if e==2.25:
                    old=read_binding(reference_files[str(fid)])
                    if old["RGB_sha256"]!=provider.rows[fid]["frame"]["sha256"]:raise ValueError("Reused frame RGB differs")
                    pred=copy.deepcopy(old["fused"])
                    if pred["identity"]["A0_step"]!=gate["models"]["A0"]["selected_step"] or pred["identity"]["B0_step"]!=gate["models"]["B0"]["selected_step"]:raise ValueError("Reused models differ")
                    if {x["track_id"] for x in pred["instances"]}!=set(expected):raise ValueError("Reused track domain differs")
                    for instance in pred["instances"]:
                        d=expected[instance["track_id"]]
                        if instance["bbox_full_xyxy"]!=d["bbox_for_a0"]:raise ValueError("Reused A0 ROI differs")
                        instance.update(copy.deepcopy(d))
                        instance["corners_raw_from_resolved_roi"]=copy.deepcopy(instance["corners_px"])
                        instance["corners_final"]=copy.deepcopy(instance["corners_px"]);instance["post_A0_geometry"]="none"
                    pred["roi_expansion"]=2.25;origin="reused_existing_complete_225_prediction"
                else:
                    export_surface=fid in cfg.get("surface_export_frames",[])
                    kwargs={}
                    if export_surface:
                        from data.action4_sam_roi_inputs import ImageFrame,array_digest
                        pixels=np.moveaxis(rgb.cpu().numpy(),0,-1)
                        kwargs=dict(return_surface=True,surface_frame=ImageFrame(
                            f"frame{fid}",pixels,provider.region["pixel_space_id"],
                            provider.rows[fid]["frame"]["sha256"],"original",array_digest(pixels)))
                        if enhanced_video:kwargs["surface_frame"]=sam_features.frame
                    if enhanced_video:kwargs["sam_video_context"]=sam_features
                    if corner_trial is not None:kwargs["corner_trial_binding"]=corner_trial
                    pred=pipe.predict_video(rgb,K,cad,video_context=context,roi_expansion=e,**kwargs)
                    if export_surface:
                        from data.action4_surface_snapshot import save_snapshot
                        snapshots=pred.pop("surface_snapshots")
                        folder=out/"surface";folder.mkdir(exist_ok=True)
                        refs=[]
                        for j,snapshot in enumerate(snapshots):
                            path=folder/f"frame_{fid:08d}_e{e:g}_{j}.npz"
                            save_snapshot(path,snapshot);refs.append(binding(path))
                        pred["surface_snapshot_refs"]=refs
                    if {x["track_id"] for x in pred["instances"]}!=set(expected):raise ValueError("Ratio changed selected instances")
                    origin="current_A0_from_shared_resolved_ROI"
                for instance in pred["instances"]:
                    if instance["bbox_for_a0"]!=expected[instance["track_id"]]["bbox_for_a0"]:raise ValueError("Actual resolved ROI altered")
                    if corner_trial is None and instance["corners_raw_from_resolved_roi"]!=instance["corners_final"]:raise ValueError("Unadmitted post-A0 geometry")
                    if fid<4 and corner_trial is None:instance["offline_confirmation_frame"]=4
                row={"frame_id":fid,"pts":provider.rows[fid]["frame"]["pts"],"RGB_sha256":provider.rows[fid]["frame"]["sha256"],
                    "roi_expansion":e,"prediction":pred,"prediction_origin":origin,"track_replay_identity":context.replay_identity}
                if corner_trial is not None:validate_corner_trial_row(row,K,cad,corner_trial)
                files[e].write(json.dumps(row,allow_nan=False)+chr(10))
            n+=1
            if n%25==0:(out/"STATUS.json").write_text(json.dumps({"frames":n,"expected":job["end"]-job["start"],"calls":calls}))
        if calls["B0"]!=0 or n!=job["end"]-job["start"]:raise ValueError("Unexpected inference or missing frame")
        (out/"COMPLETE.json").write_text(json.dumps({"status":("SIX_RATIO_PART_COMPLETE" if len(cfg["ratios"])==6 else "FIXED_RATIO_PART_COMPLETE"),"start":job["start"],"end":job["end"],
            "frames":n,"calls":calls,"seconds":time.monotonic()-beg,"models":gate["models"],"input_lock":cfg["input_lock"],
            "peak_allocated_bytes":torch.cuda.max_memory_allocated()},indent=2))
    finally:
        for f in files.values():f.close()
        for h in hooks:h.remove()


def render_standard_six(video_root,input_lock,demo_reference,output_root,*,stabilization_receipt=None,corner_trial_receipt=None,encoder="h264_nvenc",encoder_threads=2,image_prefetch_workers=0):
    """Read saved predictions only; shared frame decode, no model inference."""
    import subprocess,datetime
    from fractions import Fraction
    from PIL import Image,ImageDraw,ImageFont
    from eval.action4_video import draw_cuboids
    from data.action4_dataset import virtual_roi
    root=contained_path(R,video_root,exists=True)
    complete=json.loads((root/"COMPLETE.json").read_text())
    stabilization=None;corner_trial=None
    if stabilization_receipt is not None and corner_trial_receipt is not None:
        raise ValueError("Do not stack unrelated pose consumers in renderer")
    if corner_trial_receipt is not None:
        from eval.action4_corner_video_receipt import validate_trial_render
        from eval.action4_corner_video import validate_corner_trial_row
        corner_trial=validate_trial_render(video_root,input_lock,corner_trial_receipt)
        if complete["ratios"]!=[corner_trial.get("roi_expansion",1.5)]:raise ValueError("Corner receipt ROI differs")
    if stabilization_receipt is not None:
        from eval.action4_video_stabilization import validate_render_receipt,validate_processed_row
        stabilization=validate_render_receipt(video_root,input_lock,stabilization_receipt)
        if complete["ratios"]!=[stabilization.get("roi_expansion",1.5)]:raise ValueError("Stabilization receipt ROI differs")
    elif corner_trial is None and complete["status"] not in ("SIX_FULL_VIDEO_PREDICTION_DOMAINS_COMPLETE","FIXED_RATIO_FULL_VIDEO_PREDICTIONS_COMPLETE"):
        raise ValueError("Complete prediction domains required")
    if encoder not in ("h264_nvenc","libx264") or (stabilization is None and corner_trial is None and encoder!="h264_nvenc"):
        raise ValueError("Explicit stabilized CPU encoding only; raw path unchanged")
    if type(encoder_threads) is not int or not 1<=encoder_threads<=4:raise ValueError("Bounded encoder thread count required")
    lock=read_binding(input_lock);manifest=read_binding(lock["video"]["frames_manifest"])
    frames=manifest["frames"];source=manifest["source"];fps=source["avg_frame_rate"]
    raw_stream=None
    if stabilization is not None or corner_trial is not None:
        K_stable=np.asarray(read_binding(lock["video"]["K"])["K_native"],float)
        cad_stable=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"],float)/1000

    if len(frames)!=2850 or [f["frame_id"] for f in frames]!=list(range(2850)):raise ValueError("Complete source frame domain required")
    step=frames[1]["pts"]-frames[0]["pts"]
    if any(b["pts"]-a["pts"]!=step for a,b in zip(frames,frames[1:])) or Fraction(source["time_base"])*step!=1/Fraction(fps):
        raise ValueError("This renderer requires the verified exact-CFR source")
    reference=contained_path(R,demo_reference["path"],exists=True)
    if binding(reference)["sha256"]!=demo_reference["sha256"]:raise ValueError("Demo reference changed")
    ffmpeg=str(R/"runtime/ffmpeg/bin/ffmpeg");ffprobe=str(R/"runtime/ffmpeg/bin/ffprobe")
    ref=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=width,height,avg_frame_rate,nb_frames","-of","json",str(reference)],text=True))["streams"][0]
    if (ref["width"],ref["height"],int(ref["nb_frames"]),Fraction(ref["avg_frame_rate"]))!=(950,720,2850,Fraction(fps)):raise ValueError("Demo alignment differs")
    out=contained_path(R,output_root,exists=False);out.mkdir(exist_ok=False)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    logs=[];streams={};encoders={};folders={};decoder=None;scale=np.array([950/source["width"],720/source["height"]]);font=ImageFont.load_default(size=15)
    try:
        if stabilization is not None:raw_stream=contained_path(R,stabilization["raw_predictions"]["path"],exists=True).open()
        log=(out/"decode.log").open("x");logs.append(log)
        decoder=subprocess.Popen([ffmpeg,"-v","error","-nostdin","-threads","2","-i",str(reference),"-map","0:v:0","-fps_mode","passthrough","-frames:v","2850","-f","rawvideo","-pix_fmt","rgb24","-threads","2","pipe:1"],stdout=subprocess.PIPE,stderr=log)
        for e in complete["ratios"]:
            tag="e"+format(e,".2f").replace(".","p");part=root/tag
            if json.loads((part/"COMPLETE.json").read_text())["frames"]!=2850:raise ValueError("Ratio frame domain incomplete")
            streams[e]=(part/"predictions.jsonl").open()
            folder=R/"outputs/videos"/(stamp+"-"+complete.get("video_stage","roi-sweep")+"-"+tag+"-track");folder.mkdir(exist_ok=False);(folder/"debug").mkdir();folders[e]=folder
            log=(folder/"encode.log").open("x");logs.append(log)
            codec=(["-c:v","h264_nvenc","-gpu","0","-preset","p4","-rc","vbr","-cq","18","-b:v","0"] if encoder=="h264_nvenc"
                   else ["-c:v","libx264","-preset","veryfast","-crf","18","-threads",str(encoder_threads)])
            command=[ffmpeg,"-v","error","-nostdin","-n","-f","rawvideo","-pix_fmt","rgb24","-s","1900x750","-r",fps,"-i","pipe:0","-an"]+codec+["-pix_fmt","yuv420p","-video_track_timescale","11988","-movie_timescale","11988",str(folder/"comparison.mp4")]
            encoders[e]=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        from data.action4_ordered_image_prefetch import iter_rgb_previews
        for f,base in iter_rgb_previews(frames,image_size=(source["width"],source["height"]),workers=image_prefetch_workers):
            fid=f["frame_id"];raw=decoder.stdout.read(950*720*3)
            if len(raw)!=950*720*3:raise RuntimeError("Demo frame missing")
            demo=Image.frombytes("RGB",(950,720),raw)
            raw_prediction_row=json.loads(next(raw_stream)) if raw_stream is not None else None
            for e,stream in streams.items():
                row=json.loads(next(stream))
                if row["frame_id"]!=fid or row["pts"]!=f["pts"] or row["RGB_sha256"]!=f["image_sha256"] or row["roi_expansion"]!=e:raise ValueError("Prediction/frame/PTS mismatch")
                if stabilization is not None:
                    validate_processed_row(raw_prediction_row,row,K_stable,cad_stable,stabilization["filter_id"])
                if corner_trial is not None:validate_corner_trial_row(row,K_stable,cad_stable,corner_trial["trial_binding"])
                instances=row["prediction"]["instances"];draw_rows=[]
                for instance in instances:
                    if stabilization is None and corner_trial is None and (instance["post_A0_geometry"]!="none" or instance["corners_final"]!=instance["corners_raw_from_resolved_roi"]):raise ValueError("Unadmitted geometry in render input")
                    if instance["pose_valid"]:
                        draw_rows.append(dict(corners_px=(np.asarray(instance["corners_final"])*scale).tolist()))
                right=draw_cuboids(base,draw_rows,line_width=2,debug_numbers=True);d=ImageDraw.Draw(right)
                for instance in instances:
                    box=np.asarray(instance["bbox_for_a0"])*np.tile(scale,2)
                    d.rectangle(box.tolist(),outline="lime",width=2)
                    label=instance["track_id"]+(" SAM" if instance["temporal_status"]=="sam_supported" else " B0/track")
                    d.text((max(0,box[0]),max(0,box[1]-16)),label,fill="lime",font=font)
                canvas=Image.new("RGB",(1900,750),"black");canvas.paste(demo,(0,30));canvas.paste(right,(950,30));d=ImageDraw.Draw(canvas)
                d.text((5,5),f"Demo reference (not GT) | frame {fid}",fill="white",font=font)
                mode_label="corner research candidate" if corner_trial is not None else ("stable rigid A0" if stabilization is not None else "raw A0")
                d.text((955,5),f"{complete.get('video_label','development_sweep')} | ROI {e:g} | B0+SAM track | {mode_label}",fill="white",font=font)
                encoders[e].stdin.write(canvas.tobytes())
                if fid in (0,4,1195,1289,1636,1950,2400,2849):
                    canvas.save(folders[e]/"debug"/f"comparison_{fid:08d}.jpg",quality=93)
                    debug=right.copy();dd=ImageDraw.Draw(debug)
                    for instance in instances:
                        box=np.asarray(virtual_roi(instance["bbox_for_a0"],e))*np.tile(scale,2);dd.rectangle(box.tolist(),outline="orange",width=2)
                    debug.save(folders[e]/"debug"/f"actual_roi_{fid:08d}.jpg",quality=93)
            if fid%100==0:(out/"STATUS.json").write_text(json.dumps({"rendered_frames_per_ratio":fid+1,"ratios":complete["ratios"]}))
        for stream in streams.values():
            if stream.read().strip():raise ValueError("Unexpected trailing prediction frames")
        if raw_stream is not None and raw_stream.read().strip():raise ValueError("Unexpected trailing raw frames")
        for proc in encoders.values():proc.stdin.close()
        codes=[p.wait() for p in encoders.values()]
        if decoder.wait()!=0 or any(codes):raise RuntimeError("Actual encoding failure retained")
        results=[]
        for e,folder in folders.items():
            video=folder/"comparison.mp4"
            probe=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=nb_frames,avg_frame_rate,time_base,duration_ts,width,height","-of","json",str(video)],text=True))["streams"][0]
            if int(probe["nb_frames"])!=2850 or Fraction(probe["avg_frame_rate"])!=Fraction(fps) or Fraction(probe["time_base"])*int(probe["duration_ts"])!=Fraction(2850)/Fraction(fps):
                raise ValueError("Encoded frame/time identity differs")
            results.append({"roi_expansion":e,"video":binding(video),"probe":probe,"prediction_source":str((root/("e"+format(e,".2f").replace(".","p"))/"predictions.jsonl").relative_to(R))})
        (out/"COMPLETE.json").write_text(json.dumps({"status":("SIX_DEVELOPMENT_VIDEOS_ENCODED_NOT_ACCURACY_ACCEPTANCE" if len(complete["ratios"])==6 else "DEVELOPMENT_VIDEO_ENCODED_NOT_ACCURACY_ACCEPTANCE"),"results":results,"model_calls":0,"demo_is_GT":False},indent=2))
        return results
    finally:
        if raw_stream is not None:raw_stream.close()
        for stream in streams.values():stream.close()
        for proc in encoders.values():
            if proc.poll() is None:proc.terminate()
        if decoder is not None and decoder.poll() is None:decoder.terminate()
        for log in logs:log.close()


def render_saved_development_clip(payload_ref):
    """Explicit short diagnostic entry; never weakens full-video publication gates."""
    import os,subprocess
    from fractions import Fraction
    from PIL import Image,ImageDraw,ImageFont
    from eval.action4_video import draw_cuboids
    payload=read_binding(payload_ref)
    if payload.get("schema")!="action4.saved_pose_development_clip.v1" or payload.get("accuracy_accepted") is not False:
        raise ValueError("Explicit unaccepted development clip payload required")
    rows=payload["frames"]
    if not 1<=len(rows)<=120 or [r["frame_id"] for r in rows]!=list(range(rows[0]["frame_id"],rows[0]["frame_id"]+len(rows))):
        raise ValueError("Bounded contiguous complete clip required")
    manifest=read_binding(payload["source_manifest"]);source=manifest["source"];native={f["frame_id"]:f for f in manifest["frames"]}
    fps=source["avg_frame_rate"];tb=Fraction(source["time_base"]);step=1/Fraction(fps)/tb
    if step.denominator!=1:raise ValueError("Exact source PTS interval required")
    for i,row in enumerate(rows):
        frame=native[row["frame_id"]]
        if row["pts"]!=frame["pts"] or row["RGB_sha256"]!=frame["image_sha256"] or row["pts"]!=rows[0]["pts"]+i*int(step):
            raise ValueError("Saved pose/frame/PTS mismatch")
        for side in ["baseline","candidate"]:
            if len({p["track_id"] for p in row[side]})!=len(row[side]):raise ValueError("Duplicate render track")
            for inst in row[side]:
                if inst["corner_ids"]!=list(range(8)):raise ValueError("Canonical ID order changed")
                if inst["corners_px"] is not None and (np.asarray(inst["corners_px"]).shape!=(8,2) or not np.isfinite(inst["corners_px"]).all()):
                    raise ValueError("Invalid saved render corners")
    if payload["resources"]["physical_gpu"]!=7 or os.environ.get("CUDA_VISIBLE_DEVICES")!="7":
        raise ValueError("This registered clip encoder requires explicit physical GPU7 mapping")
    out=contained_path(R,payload["output"],exists=False);out.mkdir(exist_ok=False)
    ffmpeg=str(R/"runtime/ffmpeg/bin/ffmpeg");ffprobe=str(R/"runtime/ffmpeg/bin/ffprobe")
    video=out/"comparison.mp4";scale=np.array([950/source["width"],720/source["height"]]);font=ImageFont.load_default(size=15)
    command=[ffmpeg,"-v","error","-nostdin","-n","-f","rawvideo","-pix_fmt","rgb24","-s","1900x750","-r",fps,"-i","pipe:0","-an","-c:v","h264_nvenc","-gpu","0","-preset","p4","-rc","vbr","-cq","18","-b:v","0","-pix_fmt","yuv420p","-video_track_timescale","11988","-movie_timescale","11988",str(video)]
    log=(out/"encode.log").open("x");proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
    from eval.action4_sam_full_review import ReceiptVerifier
    verifier=ReceiptVerifier();rendered=0
    try:
        for row in rows:
            f=native[row["frame_id"]];path=verifier.verify(f["image"],f["image_sha256"])
            with Image.open(path) as im:
                if im.size!=(source["width"],source["height"]):raise ValueError("Native image size changed")
                base=im.convert("RGB").resize((950,720))
            canvas=Image.new("RGB",(1900,750),"black")
            for index,side in enumerate(["baseline","candidate"]):
                drawings=[dict(corners_px=(np.asarray(p["corners_px"])*scale).tolist()) for p in row[side] if p["corners_px"] is not None]
                image=draw_cuboids(base,drawings,line_width=2,debug_numbers=True);draw=ImageDraw.Draw(image)
                for p in row[side]:
                    box=np.asarray(p["bbox_for_a0"])*np.tile(scale,2);draw.rectangle(box.tolist(),outline="lime",width=2)
                    draw.text((max(0,box[0]),max(0,box[1]-16)),p["track_id"]+" "+p["status"],fill="lime",font=font)
                canvas.paste(image,(index*950,30))
                label="A0 baseline / B0+SAM / ROI1.5" if side=="baseline" else "Part geometry + recent initialization / NOT ACCEPTED"
                ImageDraw.Draw(canvas).text((index*950+5,5),label+" | frame "+str(row["frame_id"]),fill="white",font=font)
            proc.stdin.write(canvas.tobytes());rendered+=1
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError("Development clip encoder failed")
    except BaseException:
        if proc.poll() is None:
            proc.stdin.close();proc.wait()
        (out/"FAILED.json").write_text(json.dumps({"rendered":rendered,"expected":len(rows),"payload":payload_ref}));raise
    finally:log.close()
    probe=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames,avg_frame_rate,time_base,duration_ts","-of","json",str(video)],text=True))["streams"][0]
    if int(probe["nb_frames"])!=len(rows) or Fraction(probe["avg_frame_rate"])!=Fraction(fps) or Fraction(probe["time_base"])*int(probe["duration_ts"])!=Fraction(len(rows))/Fraction(fps):
        raise ValueError("Encoded clip frame/time identity differs")
    result=dict(status="DEVELOPMENT_CLIP_ENCODED_NOT_ACCEPTANCE",payload=payload_ref,video=binding(video),probe=probe,
        source_frame_ids=[r["frame_id"] for r in rows],source_first_pts=rows[0]["pts"],model_calls=0,geometry_recomputed=False,
        renderer_pose_selection=False,full_video=False,accuracy_accepted=False)
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));return result
