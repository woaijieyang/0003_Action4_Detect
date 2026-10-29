"""Owned process runner for prepared B0-only SAM prefix discovery; no A0 use."""
import json,os,subprocess,time,traceback
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding
from eval.action4_sam_image_eval import validate_live_authority,validate_source_snapshot
from eval.action4_clip_replay import record_process_exit

def validate_prefix(cfg):
    if (cfg.get("schema")!="action4.sam.b0_video_sequence.v1" or cfg.get("discovery_seed_only") is not True
        or cfg.get("discovery_decode_prefix") is not True or cfg.get("seed_mode")!="box_text"
        or cfg.get("sequence",{}).get("direction")!="both"):
        raise ValueError("Only explicit prefix seed discovery is authorized by this runner")

def validate_forward(cfg,manifest):
    if (cfg.get("schema")!="action4.sam.b0_video_sequence.v1" or cfg.get("discovery_seed_only") is not False
        or cfg.get("discovery_decode_prefix") is not False or cfg.get("seed_mode")!="instance_points"
        or cfg.get("instance_output_bridge") is not True
        or cfg.get("temporal_mode")!="causal_forward_B0_assist"
        or cfg.get("sequence",{}).get("direction")!="forward"
        or not cfg.get("instance_seed_sources")
        or manifest.get("preparation_provenance",{}).get("kind")!="completed_prefix_discovery_immutable_inheritance"):
        raise ValueError("Explicit authenticated prefix-to-forward preparation required")

def run(prepared_ref,control_output,*,mode="prefix"):

    prepared=read_binding(prepared_ref)
    if prepared.get("status")!="PREPARED_NOT_EXECUTED":raise ValueError("Prepared source required")
    cfg=read_binding(prepared["config_ref"])
    manifest=validate_source_snapshot(cfg)
    if mode=="prefix":validate_prefix(cfg)
    elif mode=="forward":validate_forward(cfg,manifest)
    else:raise ValueError("Unknown explicit dispatch mode")
    validate_live_authority(cfg)
    if prepared["source_manifest_ref"]!={"path":cfg["source_snapshot"]+"/SNAPSHOT_MANIFEST.json",
      "sha256":__import__("hashlib").sha256((R/cfg["source_snapshot"]/"SNAPSHOT_MANIFEST.json").read_bytes()).hexdigest()}:
        raise ValueError("Prepared source manifest differs")
    expected=[str(R/".venv/sam31/bin/python"),"-B",str(R/cfg["source_snapshot"]/"scripts/action4_run_frozen_sequence.py"),
        "--config",str(R/cfg["source_snapshot"]/"config.json"),"--mode","sam"]
    if prepared["command"]!=expected:raise ValueError("Unexpected native producer command")
    native=contained_path(R,cfg["output"],exists=True)
    if any(native.iterdir()):raise ValueError("Never restart or overwrite a native producer")
    resources=cfg["resources"]
    info=subprocess.check_output(["nvidia-smi","-i",str(resources["physical_gpu"]),
        "--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if info[0].strip()!=resources["gpu_uuid"] or int(info[1])<resources["fresh_free_mib"]:
        raise RuntimeError("Native headroom unavailable; no process launched")
    out=contained_path(R,control_output,exists=False);out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=resources["gpu_uuid"],ACTION4_PROJECT_ROOT=str(R),
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS=str(resources["torch_threads"]),
        MKL_NUM_THREADS=str(resources["torch_threads"]),OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command=["nice","-n","10","taskset","-c",",".join(map(str,resources["cpu_affinity"]))]+expected
    child=None
    try:
        with (out/"native.log").open("xb") as log:
            child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/"LAUNCH.json").write_text(json.dumps({"pid":child.pid,"command":command,
            "prepared":prepared_ref,"source_identity":manifest["identity"],"time_unix":time.time()},indent=2))
        code=child.wait()
        receipt=record_process_exit(child,prepared["config_ref"],out/"PROCESS_EXIT.json")
        (out/"RESULT.json").write_text(json.dumps({"status":"PROCESS_EXITED_REVIEW_REQUIRED" if code==0 else "PROCESS_FAILED",
            "returncode":code,"producer_exit_ref":receipt,"accuracy_accepted":False},indent=2))
    except BaseException:
        (out/"CONTROL_FAILURE.json").write_text(json.dumps({"traceback":traceback.format_exc(),
            "child_pid":None if child is None else child.pid,"child_returncode":None if child is None else child.poll(),
            "no_automatic_restart_or_stop":True},indent=2))
        raise
