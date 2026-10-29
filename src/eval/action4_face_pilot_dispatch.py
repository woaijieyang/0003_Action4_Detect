"""Launch already prepared immutable face pilots under per-job resource contracts."""
import json,os,subprocess,traceback
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_sam_image_eval import validate_source_snapshot,validate_live_authority

def run(jobs_ref):
    spec=read_binding(jobs_ref);root=contained_path(R,jobs_ref["path"],exists=True).parent
    if not 1<=len(spec["jobs"])<=4:raise ValueError("Bounded independent native pilots only")
    configs=[]
    for job in spec["jobs"]:
        cfg=read_binding(job["config"]);validate_source_snapshot(cfg);validate_live_authority(cfg)
        if cfg["resources"]!=job["resources"]:raise ValueError("Resource declaration differs")
        expected=[str(R/".venv/sam31/bin/python"),"-I","-B",str(R/cfg["source_snapshot"]/"scripts/action4_run_frozen_face.py"),
                  "--config",str(R/cfg["source_snapshot"]/"config.json"),"--execute-native"]
        if job["command"]!=expected:raise ValueError("Native command identity differs")
        out=contained_path(R,cfg["output"],exists=False)
        if out.exists() and any(out.iterdir()):raise ValueError("Do not duplicate or overwrite native run")
        configs.append(cfg)
    cpus=[c for cfg in configs for c in cfg["resources"]["cpu_affinity"]]
    if len(cpus)!=len(set(cpus)) or len(cpus)>72:raise ValueError("Overlapping or excess native CPU reservation")
    if len({cfg["resources"]["physical_gpu"] for cfg in configs})!=len(configs):
        raise ValueError("This dispatcher requires separate physical GPUs")
    with (root/"DISPATCH.json").open("x") as f:json.dump(dict(status="ADMITTED_PREPARED_JOBS",jobs=jobs_ref),f)
    processes=[]
    for job,cfg in zip(spec["jobs"],configs):
        r=cfg["resources"];folder=root/job["label"];folder=contained_path(R,folder,exists=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=r["gpu_uuid"],PYTHONDONTWRITEBYTECODE="1",ACTION4_PROJECT_ROOT=str(R),
                 OMP_NUM_THREADS=str(r["torch_threads"]),MKL_NUM_THREADS=str(r["torch_threads"]),OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
        for key in ["TMPDIR","XDG_CACHE_HOME","TORCH_HOME","CUDA_CACHE_PATH","MPLCONFIGDIR"]:
            p=folder/"runtime_cache"/key;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
        command=["nice","-n","10","taskset","-c",",".join(map(str,r["cpu_affinity"]))]+job["command"]
        with (folder/"native.log").open("x") as log:
            child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (folder/"LAUNCH.json").write_text(json.dumps(dict(pid=child.pid,command=command,config=job["config"],source=job["source"]),indent=2))
        processes.append((job,child))
    exits=[]
    for job,child in processes:
        code=child.wait();entry=dict(label=job["label"],pid=child.pid,returncode=code)
        (root/job["label"]/"PROCESS_EXIT.json").write_text(json.dumps(entry,indent=2));exits.append(entry)
    result=dict(status="ALL_NATIVE_PILOTS_EXITED_REVIEW_REQUIRED",exits=exits,accuracy_accepted=False)
    (root/"CONTROLLER_COMPLETE.json").write_text(json.dumps(result,indent=2));return result
