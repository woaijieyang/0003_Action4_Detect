"""Bounded parallel frame-interval video generation and exact ordered merge."""
import json,os,subprocess,time,math,traceback,signal
from pathlib import Path
from fractions import Fraction
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

def stop_owned_workers(processes):
    # Each worker was created with start_new_session=True; only its own group.
    for process in processes:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    for process in processes:
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()

def run(selection,output,devices,limit=None,reserved_cpu=0):
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    selection=contained_path(R,selection,exists=True)
    if not devices or len(devices)>10 or any(g not in (1,4,5,6,7) for g in devices):raise ValueError("Explicit approved video GPU list required")
    if devices.count(1)>8 or len(devices)*6+reserved_cpu+2>72:raise ValueError("CPU/GPU1 concurrency budget exceeded")
    cfg=json.loads(selection.read_text())
    if limit is None:
        receipts=cfg.get("training_completion_receipts",[])
        if len(receipts)<2:raise ValueError("Full video requires completed selected training receipts")
        for name in receipts:
            receipt=json.loads(contained_path(R,name,exists=True).read_text())
            if receipt.get("epochs",0)<80:raise ValueError("Selected training has not completed80epochs")
    count=2850 if limit is None else limit
    if not 1<=count<=2850 or count<len(devices):raise ValueError("Invalid frame count/worker count")
    uuids={}
    for device in set(devices):
        record=subprocess.check_output(["nvidia-smi","-i",str(device),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if int(record[1])<devices.count(device)*4000:raise RuntimeError("Insufficient free GPU memory")
        uuids[device]=record[0].strip()
    intervals=[(i*count//len(devices),(i+1)*count//len(devices)) for i in range(len(devices))]
    env=dict(os.environ,PYTHONPATH=str(R/"runtime/surface_deps")+":"+str(R/"src"),TMPDIR=str(R/"runtime"),
             CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
    workers=[];processes=[]
    try:
        for index,(device,(start,end)) in enumerate(zip(devices,intervals)):
            folder=out/f"part_{index:03d}"
            command=[str(R/".venv/bin/python"),"-B","-m","eval.action4_staged_video","--selection",str(selection),
                     "--output",str(folder),"--start",str(start),"--end",str(end)]
            with (out/f"part_{index:03d}.log").open("x") as log:
                proc=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=uuids[device]),stdout=log,stderr=log,start_new_session=True)
            workers.append({"pid":proc.pid,"gpu":device,"start":start,"end":end,"output":str(folder.relative_to(R))});processes.append(proc)
        atomic_json(out/"DISPATCH.json",{"workers":workers,"cpu_worker_budget":len(devices)*6,"reserved_cpu":reserved_cpu,"frames":count})
        while any(p.poll() is None for p in processes):
            if any(p.poll() is not None and p.returncode!=0 for p in processes):
                raise RuntimeError("Owned video worker failed; no automatic retry")
            time.sleep(5)
    except BaseException:
        stop_owned_workers(processes)
        raise
    identity=None;next_frame=0;invalid=0;concat=out/"concat.txt"
    with (out/"predictions.jsonl").open("x") as result,concat.open("x") as listing:
        for worker in workers:
            folder=R/worker["output"];receipt=json.loads((folder/"COMPLETE.json").read_text())
            if receipt["frames"]!=worker["end"]-worker["start"] or receipt["start_frame"]!=worker["start"] or receipt["end_frame_exclusive"]!=worker["end"]:
                raise ValueError("Worker interval mismatch")
            if identity is None:identity=receipt["identity"]
            if identity!=receipt["identity"]:raise ValueError("Workers used different model identities")
            invalid+=receipt["invalid_pose_detections"]
            for line in (folder/"predictions.jsonl").read_text().splitlines():
                prediction=json.loads(line)
                if prediction["frame_id"]!=next_frame or prediction["pts"]!=400*next_frame or prediction["prediction"]["identity"]!=identity:
                    raise ValueError("Missing/repeated/reordered frame or model identity")
                result.write(line+"\n");next_frame+=1
            video=str(folder/"comparison.mp4").replace("'","'\\''")
            listing.write("file '"+video+"'\n")
    if next_frame!=count:raise ValueError("Incomplete prediction coverage")
    ffmpeg=str(R/"runtime/ffmpeg/bin/ffmpeg");ffprobe=str(R/"runtime/ffmpeg/bin/ffprobe")
    partial=out/"comparison.partial.mp4"
    with (out/"merge.log").open("x") as log:
        subprocess.run([ffmpeg,"-v","error","-nostdin","-n","-f","concat","-safe","0","-i",str(concat),"-c","copy",
             "-video_track_timescale","11988","-movie_timescale","11988","-movflags","+faststart",str(partial)],stderr=log,check=True)
    probe=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=nb_read_frames,avg_frame_rate,time_base,duration_ts,width,height","-of","json",str(partial)],text=True))["streams"][0]
    if int(probe["nb_read_frames"])!=count or Fraction(probe["avg_frame_rate"])!=Fraction(2997,100):
        raise ValueError("Merged frame/rate mismatch")
    if Fraction(probe["time_base"])*int(probe["duration_ts"])!=Fraction(count*100,2997):
        raise ValueError("Merged exact duration mismatch")
    partial.rename(out/"comparison.mp4")
    atomic_json(out/"COMPLETE.json",{"status":"PARALLEL_VIDEO_IO_COMPLETE_NOT_ACCURACY_ACCEPTED","full_video":count==2850,
          "frames":count,"workers":len(workers),"identity":identity,"probe":probe,"invalid_pose_detections":invalid,"parts_preserved_for_release_cleanup":True})

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--output",required=True)
    p.add_argument("--devices",required=True);p.add_argument("--limit",type=int);p.add_argument("--reserved-cpu",type=int,default=0);a=p.parse_args()
    try:run(a.selection,a.output,[int(x) for x in a.devices.split(",")],a.limit,a.reserved_cpu)
    except BaseException:
        out=contained_path(R,a.output,exists=False)
        if out.exists():atomic_json(out/"FAILURE.json",{"traceback":traceback.format_exc()})
        raise
