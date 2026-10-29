"""Bounded parallel dispatch and read-only rendering of the existing ROI video path."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_video import render_standard_six

def validate_parts(cfg):
    cursor=0;used=set();counts={}
    for job in cfg["parts"]:
        if job["start"]!=cursor or not job["start"]<job["end"]<=2850:
            raise ValueError("Non-contiguous frame partition")
        cursor=job["end"]
        if job["gpu"] not in (4,5,6,7) or not isinstance(job["gpu_uuid"],str):
            raise ValueError("Unauthorized physical GPU")
        cpus=set(job["cpus"])
        if len(cpus)!=len(job["cpus"]) or not cpus or not cpus<=set(range(24,96)) or cpus&used:
            raise ValueError("Overlapping or unauthorized worker CPUs")
        used|=cpus;counts[job["gpu"]]=counts.get(job["gpu"],0)+1
    if cursor!=2850 or len(used)>72:raise ValueError("Incomplete frame domain or CPU budget")
    limit=cfg.get("allocator_gib",8)
    if not 0<limit<=8 or any(n*limit>40 for n in counts.values()):
        raise ValueError("Aggregate allocator reservation exceeds safe card budget")
    from eval.action4_video_roi_contract import config_ratio
    config_ratio(cfg)
    return True

def verify_source(source):
    m=json.loads((source/"SOURCE_MANIFEST.json").read_text())
    for name,sha in m["files"].items():
        p=source/name
        if not p.resolve().is_relative_to(source.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError("Immutable source differs: "+name)
    if m.get("configuration") is not None:
        ref=m["configuration"];cfg_path=contained_path(R,ref["path"],exists=True)
        if cfg_path.stat().st_mode & 0o222 or hashlib.sha256(cfg_path.read_bytes()).hexdigest()!=ref["sha256"]:
            raise ValueError("Immutable video configuration changed or writable")
    return m


def validate_reuse_identity(cfg, old_cfg, current_source, old_source):
    keys=("input_lock","corner_trial_binding","corner_training_config","video_sam_features","ratios","bootstrap_seeds")
    if any(cfg.get(k)!=old_cfg.get(k) for k in keys):raise ValueError("Reused inference inputs differ")
    def files(m):return {k:v for k,v in m["files"].items() if k.startswith("src/") and k!="src/eval/action4_roi_video_job.py"}
    if files(current_source)!=files(old_source):raise ValueError("Reused inference code differs")


def reuse_part(cfg, index, source):
    from eval.action4_roi_sweep import read_binding,binding
    from eval.action4_corner_video import validate_corner_trial_row
    import numpy as np
    refs=cfg.get("reuse_parts",{}).get(str(index))
    if refs is None:return False
    if index==0:raise ValueError("Initial live part cannot be reused")
    old_manifest=read_binding(refs["source_manifest"])
    old_source_path=contained_path(R,refs["source_manifest"]["path"],exists=True).parent
    verify_source(old_source_path)
    validate_reuse_identity(cfg,read_binding(old_manifest["configuration"]),source,old_manifest)
    done=read_binding(refs["completion"]);job=cfg["parts"][index]
    if done["start"]!=job["start"] or done["end"]!=job["end"] or done["status"]!="FIXED_RATIO_PART_COMPLETE" or done["input_lock"]!=cfg["input_lock"]:raise ValueError("Reused interval/input differs")
    lock=read_binding(cfg["input_lock"]);pair=read_binding(lock["formal_pair_gate"])
    if done["models"]!=pair["models"]:raise ValueError("Reused model pair differs")
    pred=contained_path(R,refs["predictions"]["path"],exists=True)
    if binding(pred)!=refs["predictions"]:raise ValueError("Reused predictions changed")
    frames=read_binding(lock["video"]["frames_manifest"])["frames"]
    K=np.asarray(read_binding(lock["video"]["K"])["K_native"]);cad=np.asarray(read_binding(lock["CAD"])["corners_3d_mm"])/1000
    lines=pred.read_text().splitlines()
    if len(lines)!=job["end"]-job["start"]:raise ValueError("Reused frame count differs")
    for fid,line in zip(range(job["start"],job["end"]),lines):
        row=json.loads(line);f=frames[fid]
        if row["frame_id"]!=fid or row["pts"]!=f["pts"] or row["RGB_sha256"]!=f["image_sha256"]:raise ValueError("Reused frame identity differs")
        validate_corner_trial_row(row,K,cad,cfg["corner_trial_binding"])
    out=contained_path(R,cfg["output"],exists=True)/("part_%03d"%index);out.mkdir(exist_ok=False)
    with (out/"e1p50.jsonl").open("xb") as f:f.write(pred.read_bytes())
    (out/"COMPLETE.json").write_text(json.dumps(dict(done,reused_from=refs,new_model_calls=0),indent=2))
    return True

def run_job(config_path):
    cfg=json.loads(contained_path(R,config_path,exists=True).read_text())
    validate_parts(cfg)
    out=contained_path(R,cfg["output"],exists=True);source=out/"source"
    manifest=verify_source(source)
    if manifest.get("configuration") is not None and contained_path(R,manifest["configuration"]["path"],exists=True)!=contained_path(R,config_path,exists=True):
        raise ValueError("Video source manifest bound a different configuration")
    if (out/"DISPATCH.json").exists():raise ValueError("Existing dispatch cannot be launched twice")
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1",
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    jobs={}
    def launch(i):
        job=cfg["parts"][i];child=dict(env,CUDA_VISIBLE_DEVICES=job["gpu_uuid"])
        for key in ["TMPDIR","CUDA_CACHE_PATH","TORCH_HOME","XDG_CACHE_HOME","MPLCONFIGDIR"]:
            cache=out/"runtime_cache"/str(i)/key;cache.mkdir(parents=True,exist_ok=True);child[key]=str(cache)
        with (out/("worker_%02d.log"%i)).open("x") as log:
            p=subprocess.Popen(["taskset","-c",",".join(map(str,job["cpus"])),str(R/".venv/bin/python"),
                "-I","-B",str(source/"worker.py"),str(i)],cwd=R,env=child,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[i]=p
    def receipt(stage):
        (out/"DISPATCH.json").write_text(json.dumps({"stage":stage,"pids":{i:p.pid for i,p in jobs.items()},
            "reused_parts":list(cfg.get("reuse_parts",{})),"parts":len(cfg["parts"]),"CPU_worker_total":sum(len(x["cpus"]) for x in cfg["parts"])},indent=2))
    try:
        launch(0);receipt("FIRST_REAL_PART")
        while jobs[0].poll() is None:time.sleep(5)
        if jobs[0].returncode:raise RuntimeError("First actual part failed; no other part launched")
        for i in range(1,len(cfg["parts"])):
            if not reuse_part(cfg,i,manifest):launch(i)
        receipt("ALL_PARTS_RUNNING")
        while any(p.poll() is None for p in jobs.values()):time.sleep(15)
        failures={i:p.returncode for i,p in jobs.items() if p.returncode}
        if failures:raise RuntimeError("Video part failures: "+str(failures))
        from eval.action4_video_roi_contract import config_ratio,ratio_tag
        selected_ratio=config_ratio(cfg);tag=ratio_tag(selected_ratio);dest=out/tag;dest.mkdir()
        with (dest/"predictions.jsonl").open("x") as f:
            expected=0
            for i,job in enumerate(cfg["parts"]):
                part=out/("part_%03d"%i);done=json.loads((part/"COMPLETE.json").read_text())
                if done["start"]!=job["start"] or done["end"]!=job["end"]:raise ValueError("Part coverage differs")
                for line in (part/(tag+".jsonl")).open():
                    row=json.loads(line)
                    if row["frame_id"]!=expected or row["roi_expansion"]!=selected_ratio:raise ValueError("Frame identity differs")
                    f.write(line);expected+=1
            if expected!=2850:raise ValueError("Incomplete full video")
        (dest/"COMPLETE.json").write_text(json.dumps({"status":"FULL2850_DEVELOPMENT_PREDICTIONS",
            "frames":2850,"roi_expansion":selected_ratio,"tracking":True,"baseline_reused":False},indent=2))
        (out/"COMPLETE.json").write_text(json.dumps({"status":"FIXED_RATIO_FULL_VIDEO_PREDICTIONS_COMPLETE",
            "frames_per_ratio":2850,"ratios":[selected_ratio],"B0_and_SAM_new_calls":0,"rendering_complete":False,
            "video_stage":cfg["video_stage"],"video_label":cfg["video_label"]},indent=2))
        if cfg.get("native_candidate_attachment") is not None:
            from eval.action4_native_candidate_bridge import attach_saved_native_candidates
            attach_saved_native_candidates(str(dest/"predictions.jsonl"),cfg["native_candidate_attachment"],
                str(out/"native_candidate_review"))
        os.environ["CUDA_VISIBLE_DEVICES"]=cfg["parts"][0]["gpu_uuid"]
        os.sched_setaffinity(0,set(cfg["render_cpus"]))
        render_args={}
        if cfg.get("corner_trial_binding") is not None:
            from eval.action4_corner_video_receipt import seal_trial_video
            ref={"path":str((source/"SOURCE_MANIFEST.json").relative_to(R)),
                 "sha256":hashlib.sha256((source/"SOURCE_MANIFEST.json").read_bytes()).hexdigest()}
            render_args["corner_trial_receipt"]=seal_trial_video(cfg["output"],cfg["input_lock"],cfg["corner_trial_binding"],ref)
            render_args["encoder"]=cfg.get("encoder","libx264")
            render_args["encoder_threads"]=cfg.get("encoder_threads",2)
        result=render_standard_six(cfg["output"],cfg["input_lock"],cfg["demo_reference"],cfg["render_output"],**render_args)
        (out/"CONTROLLER_COMPLETE.json").write_text(json.dumps({"status":"VIDEO_ENCODED_REVIEW_REQUIRED",
            "results":result,"raw_predictions_preserved":True,"accuracy_accepted":False},indent=2))
    except BaseException:
        (out/"FAILED.json").write_text(json.dumps({"traceback":traceback.format_exc(),
            "jobs":{i:{"pid":p.pid,"returncode":p.poll()} for i,p in jobs.items()}},indent=2))
        raise
