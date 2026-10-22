"""Stream fixed independent checkpoints into demo-aligned comparison video."""
import json,subprocess,time
from fractions import Fraction
import numpy as np
import torch
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_staged_pipeline import load_pipeline
from eval.action4_video import draw_cuboids
from model.training.action4_runtime import atomic_json

def run(selection,output,limit=None,start=0,end=None):
    torch.set_num_threads(2)
    selection=contained_path(R,selection,exists=True)
    cfg=json.loads(selection.read_text())
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    manifest=json.loads((R/"data/action4_real/cache_v001/frames_manifest.json").read_text())
    frames=manifest["frames"];source=manifest["source"]
    if len(frames)!=2850 or not manifest["complete"]:raise ValueError("Expected complete2850frames")
    if any(f["frame_id"]!=i or f["pts"]!=i*400 or f["duration_pts"]!=400 for i,f in enumerate(frames)):
        raise ValueError("Frame identity/timing discontinuity")
    fps=source["observed_cfr_fps"]
    if Fraction(source["time_base"])*400!=1/Fraction(fps):raise ValueError("CFR mismatch")
    if limit is not None and end is not None:raise ValueError("Use limit or end, not both")
    end=start+limit if limit is not None else (len(frames) if end is None else end)
    if not 0<=start<end<=len(frames):raise ValueError("Invalid frame interval")
    frames=frames[start:end]
    pipeline=load_pipeline(cfg)
    K=np.array(json.loads(contained_path(R,cfg["intrinsics_context"],exists=True).read_text())["K_native"])
    cad=np.array(json.loads((R/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_text())["corners_3d_mm"])/1000
    ffmpeg=str(R/"runtime/ffmpeg/bin/ffmpeg");ffprobe=str(R/"runtime/ffmpeg/bin/ffprobe")
    reference=contained_path(R,cfg["demo_reference"],exists=True)
    ref_probe=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames","-of","json",str(reference)],text=True))["streams"][0]
    if (ref_probe["width"],ref_probe["height"])!=(950,720) or Fraction(ref_probe["avg_frame_rate"])!=Fraction(fps) or int(ref_probe["nb_frames"])!=2850:
        raise ValueError("Demo reference geometry/timing mismatch")
    partial=out/"comparison.partial.mp4"
    encode=[ffmpeg,"-v","error","-nostdin","-n","-f","rawvideo","-pix_fmt","rgb24","-s","1900x750","-r",fps,
            "-i","pipe:0","-an","-c:v","libx264","-threads","2","-preset","fast","-crf","18","-pix_fmt","yuv420p",
            "-video_track_timescale","11988","-movie_timescale","11988","-movflags","+faststart",str(partial)]
    decode=[ffmpeg,"-v","error","-nostdin","-threads","2","-i",str(reference),"-map","0:v:0","-vf",f"select='between(n,{start},{end-1})'","-fps_mode","passthrough","-frames:v",str(len(frames)),"-f","rawvideo","-pix_fmt","rgb24","-threads","2","pipe:1"]
    atomic_json(out/"SELECTION.json",dict(cfg,actual_identity=pipeline.identity,scope="fixed-weight video; accuracy not inferred from encoding"))
    begin=time.monotonic();invalid=0
    with (out/"encode.log").open("x") as elog,(out/"decode.log").open("x") as dlog,(out/"predictions.jsonl").open("x") as predictions:
        decoder=subprocess.Popen(decode,stdout=subprocess.PIPE,stderr=dlog)
        encoder=subprocess.Popen(encode,stdin=subprocess.PIPE,stderr=elog)
        try:
            for i,frame in enumerate(frames):
                # Buffered read returns exactly one full RGB reference frame or EOF.
                raw=decoder.stdout.read(950*720*3)
                if len(raw)!=950*720*3:raise RuntimeError("Demo ended before paired source frame")
                demo=Image.frombytes("RGB",(950,720),raw)
                with Image.open(contained_path(R,frame["image"],exists=True)) as im:rgb=im.convert("RGB")
                if rgb.size!=(3248,2464):raise ValueError("Source frame dimensions changed")
                result=pipeline.predict(torch.from_numpy(np.array(rgb)).permute(2,0,1),K,cad)
                invalid+=sum(not p["pose_valid"] for p in result["instances"])
                predictions.write(json.dumps({"frame_id":frame["frame_id"],"pts":frame["pts"],"time_base":source["time_base"],"prediction":result},allow_nan=False)+"\n")
                visible=[p for p in result["instances"] if p["corners_px"] is not None]
                rendered=draw_cuboids(rgb,visible,line_width=6).resize((950,720))
                draw=ImageDraw.Draw(rendered);scale=np.array([950/rgb.width,720/rgb.height]*2)
                for p in result["instances"]:
                    box=(np.array(p["bbox_full_xyxy"])*scale).tolist();draw.rectangle(box,outline="lime",width=2)
                    if not p["pose_valid"]:draw.text((box[0],box[1]),"POSE UNAVAILABLE",fill="yellow")
                panel=Image.new("RGB",(1900,750));panel.paste(demo,(0,30));panel.paste(rendered,(950,30))
                title=ImageDraw.Draw(panel);title.text((8,8),f"Demo reference | frame {frame['frame_id']}",fill="white")
                title.text((958,8),f"Model prediction | frame {frame['frame_id']}",fill="white")
                encoder.stdin.write(panel.tobytes())
                if i==0 or (i+1)%25==0 or i+1==len(frames):
                    predictions.flush();atomic_json(out/"STATUS.json",{"completed_frames":i+1,"total_frames":len(frames),"elapsed_seconds":time.monotonic()-begin})
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError("Encoder failed")
            if decoder.stdout.read(1):raise RuntimeError("Demo interval contains extra decoded frames")
            if decoder.wait()!=0:raise RuntimeError("Demo interval decoder failed")
        except BaseException:
            for process in (encoder,decoder):
                if process.poll() is None:process.terminate()
            for process in (encoder,decoder):process.wait()
            raise
    stream=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-count_frames","-show_entries",
         "stream=nb_read_frames,avg_frame_rate,time_base,duration_ts,width,height","-of","json",str(partial)],text=True))["streams"][0]
    if int(stream["nb_read_frames"])!=len(frames) or Fraction(stream["avg_frame_rate"])!=Fraction(fps):
        raise ValueError("Output frame count/rate mismatch")
    if Fraction(stream["time_base"])*int(stream["duration_ts"])!=len(frames)/Fraction(fps):
        raise ValueError("Output exact duration mismatch")
    partial.rename(out/"comparison.mp4")
    atomic_json(out/"COMPLETE.json",{"status":"VIDEO_IO_COMPLETE_NOT_ACCURACY_ACCEPTED","full_video":len(frames)==2850,
        "frames":len(frames),"start_frame":start,"end_frame_exclusive":end,"probe":stream,"invalid_pose_detections":invalid,"demo_is_GT":False,
        "assumed_intrinsics":True,"temporal_smoothing":False,"identity":pipeline.identity})

if __name__=="__main__":
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument("--selection",required=True);p.add_argument("--output",required=True);p.add_argument("--limit",type=int);p.add_argument("--start",type=int,default=0);p.add_argument("--end",type=int);a=p.parse_args()
    try:run(a.selection,a.output,a.limit,a.start,a.end)
    except BaseException:
        out=contained_path(R,a.output,exists=False)
        if out.exists():atomic_json(out/"FAILURE.json",{"traceback":traceback.format_exc()})
        raise
