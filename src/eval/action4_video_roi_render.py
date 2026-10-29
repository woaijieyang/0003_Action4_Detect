"""Render paired original and tracked-ROI A0 predictions without new inference."""
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_video import draw_cuboids

def run(predictions_root,output,*,panel_size=(640,486),header_height=24):
    pw,ph=panel_size
    if (pw,ph,header_height) not in ((640,486,24),(950,720,30)):
        raise ValueError("Explicit registered diagnostic/full-video render size required")
    source=contained_path(R,predictions_root,exists=True)
    done=json.loads((source/"COMPLETE.json").read_text())
    frame_ids=[r["frame_id"] for r in done["frames"]]
    if (not frame_ids or frame_ids!=list(range(frame_ids[0],frame_ids[-1]+1))
        or frame_ids[0]<0 or frame_ids[-1]>=2850):
        raise ValueError("Registered video frames must be complete, ordered and contiguous")
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    command=[str(R/"runtime/ffmpeg/bin/ffmpeg"),"-v","error","-nostdin","-n",
       "-f","rawvideo","-pix_fmt","rgb24","-s",f"{2*pw}x{ph+header_height}","-r","2997/100","-i","pipe:0",
       "-an","-c:v","libx264","-threads","2","-preset","fast","-crf","18","-pix_fmt","yuv420p",
       str(out/"comparison.mp4")]
    previous={};changes=[];invalid=0
    with (out/"encode.log").open("x") as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for fid in frame_ids:
                row=json.loads((source/f"frame_{fid:08d}.json").read_text())
                image=contained_path(R,f"data/action4_real/cache_v001/frames/frame_{fid:08d}.png",exists=True)
                rgb=Image.open(image).convert("RGB")
                canvas=Image.new("RGB",(2*pw,ph+header_height),"black");draw=ImageDraw.Draw(canvas)
                for index,key in enumerate(["baseline","fused"]):
                    instances=row[key]["instances"];valid=[p for p in instances if p["pose_valid"]]
                    im=draw_cuboids(rgb,valid,line_width=5);d=ImageDraw.Draw(im)
                    for det in instances:d.rectangle(det["bbox_full_xyxy"],outline="lime",width=5)
                    canvas.paste(im.resize((pw,ph)),(index*pw,header_height))
                    draw.text((index*pw+5,5),f"{key} A0 {done['models']['A0_step']} / B0 {done['models']['B0_step']} | frame {fid}",fill="white")
                proc.stdin.write(np.asarray(canvas).tobytes())
                for instance in row["fused"]["instances"]:
                    tid=instance["track_id"]
                    if not instance["pose_valid"]:invalid+=1;previous.pop(tid,None);continue
                    uv=np.asarray(instance["corners_px"]);box=np.asarray(instance["bbox_full_xyxy"])
                    if tid in previous:
                        pf,puv,pbox=previous[tid]
                        scale=np.linalg.norm(box[2:]-box[:2])
                        changes.append({"frame":fid,"previous_frame":pf,"track":tid,
                            "max_corner_displacement_per_box_diagonal":float(np.linalg.norm(uv-puv,axis=1).max()/scale),
                            "box_center_displacement_per_diagonal":float(np.linalg.norm((box[:2]+box[2:]-pbox[:2]-pbox[2:])/2)/scale)})
                    previous[tid]=(fid,uv,box)
                if fid in [825,850,880,910,939]:canvas.save(out/f"frame_{fid:08d}.png")
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError("Encoder failed")
        except BaseException:
            if proc.poll() is None:proc.kill();proc.wait()
            raise
    changes.sort(key=lambda x:x["max_corner_displacement_per_box_diagonal"],reverse=True)
    summary={"status":"PAIRED_VIDEO_COMPLETE_NOT_ACCURACY_ACCEPTANCE","frames":len(frame_ids),
        "A0_ROI_consumed":True,"SAM_features_in_A0":False,"invalid_fused_poses":invalid,
        "motion_diagnostics_not_error_metrics":True,"largest_changes":changes[:10],
        "prediction_root":str(source.relative_to(R)),"no_new_model_inference":True}
    probe=json.loads(subprocess.check_output([str(R/"runtime/ffmpeg/bin/ffprobe"),"-v","error","-count_frames",
        "-select_streams","v:0","-show_entries","stream=nb_read_frames,avg_frame_rate,width,height",
        "-of","json",str(out/"comparison.mp4")],text=True))
    if int(probe["streams"][0]["nb_read_frames"])!=len(frame_ids):raise ValueError("Missing encoded frames")
    summary["probe"]=probe
    (out/"COMPLETE.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary
