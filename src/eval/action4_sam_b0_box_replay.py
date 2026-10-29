"""Identity-checked offline SAM/B0 replay; no model calls or GT inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R, contained_path
from model.action4_video_box_history import VideoBoxHistory

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_bound(ref):
    p=contained_path(R,ref["path"],exists=True)
    if digest(p)!=ref["sha256"]:raise ValueError("Changed input: "+ref["path"])
    return json.loads(p.read_text())

def published_baseline(pool):
    """Match StagedPipeline's frozen publication threshold, not Soft-NMS's .001 floor."""
    return [p for p in pool["single_frame_selected"] if p["score"]>=.3]

def support_box(mask, expected_shape):
    mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.shape!=tuple(expected_shape):
        raise ValueError("Native boolean mask geometry differs")
    yy,xx=np.nonzero(mask)
    if not len(xx):raise ValueError("Available mask has zero area")
    # Continuous pixel boundaries, used ONLY as visible support for association.
    return [float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)]

def independent_supports(supports,evidence):
    """A copied mask is one observation, even under distinct session/object IDs."""
    groups={}
    for tid in supports:
        sha=evidence[tid].get("pixel_sha256")
        if not isinstance(sha,str) or len(sha)!=64 or any(x not in "0123456789abcdef" for x in sha):
            raise ValueError("Actual mask pixel identity required")
        groups.setdefault(sha,[]).append(tid)
    rejected={tid for ids in groups.values() if len(ids)>1 for tid in ids}
    return ({tid:box for tid,box in supports.items() if tid not in rejected},
            [{"track_id":tid,"reason":"shared_identical_mask_pixels"} for tid in sorted(rejected)])

def prefix_record(row,seeds,start):
    """Keep every initialization frame; never invent pre-confirmation tracks."""
    frame=row["frame"];fid=frame["global_frame_id"]
    if fid>start:raise ValueError("Prefix builder cannot replace tracked frames")
    tracks=[]
    if fid==start:
        pool={p["grid_index"]:p for p in row["pool"]["association_candidates"]}
        for seed in seeds:
            grid=int(seed["candidate_id"].rsplit(":",1)[1])
            if seed["global_frame_id"]!=fid or grid not in pool:
                raise ValueError("Confirmed seed is not a current native candidate")
            candidate=dict(pool[grid],candidate_id=seed["candidate_id"])
            if candidate["bbox_full_xyxy"]!=seed["bbox_native"]:
                raise ValueError("Seed box differs from current full-body candidate")
            tracks.append({"track_id":seed["track_id"],"candidate_id":seed["candidate_id"],
                "status":"observed","bbox_for_roi":seed["bbox_native"],"raw_detection":candidate,
                "history_length":1,"sam_support":"seed_binding_only","reason":None,
                "redetection_required":False,"box_quality_scope":"B0_seed_not_temporal_quality"})
    return {"frame":frame,"fused":{"frame_id":fid,"mode":"video_inference","tracks":tracks,
            "initialization_status":"seed_confirmed" if fid==start else "unconfirmed",
            "baseline_detection_fallback":published_baseline(row["pool"]) if fid<start else [],
            "automatic_reinitialization":False},"history_only":None,"sam_evidence":{},
            "initialization_prefix":True}

def replay(sequence_root, output, *, include_prefix=False):
    if type(include_prefix) is not bool:raise ValueError("Explicit prefix policy required")
    root=contained_path(R,sequence_root,exists=True)
    cfg=json.loads((root/"source/config.json").read_text())
    complete=json.loads((root/"run/NATIVE_COMPLETE.json").read_text())
    manifest=json.loads((root/"source/SNAPSHOT_MANIFEST.json").read_text())
    if complete["source_identity"]!=manifest["identity"]:
        raise ValueError("SAM completion source differs")
    region=read_bound(cfg["region"]);b0=read_bound(cfg["b0_evidence"])
    if b0["b0"]!=cfg["b0"] or b0["region"]!=cfg["region"]:
        raise ValueError("B0/SAM frame or model source differs")
    poolpath=contained_path(R,b0["pools"]["path"],exists=True)
    if digest(poolpath)!=b0["pools"]["sha256"]:raise ValueError("B0 pool changed")
    pools=[json.loads(line) for line in poolpath.read_text().splitlines()]
    if [r["frame"] for r in pools]!=region["frames"]:raise ValueError("B0 frame identity differs")
    seed_doc=json.loads((root/"run/SEEDS.json").read_text())
    seeds=seed_doc["seeds"]
    if not seeds or len({s["global_frame_id"] for s in seeds})!=1:
        raise ValueError("This replay requires a common confirmed seed frame")
    snapshots={}
    for item in complete["tracks"]:
        doc=read_bound({"path":item["snapshot"],"sha256":item["snapshot_sha256"]})
        snap=doc["snapshot"];tid=item["track_id"]
        if (snap["track_id"]!=tid or snap["dependency_identity"]!=complete["source_identity"]
            or snap["region_identity"]!=complete["region_identity"]):
            raise ValueError("SAM snapshot ownership differs")
        by_frame={f["global_frame_id"]:f for f in snap["frames"]}
        if len(by_frame)!=len(region["frames"]):raise ValueError("Incomplete SAM frame domain")
        snapshots[tid]=(doc,by_frame)
    if set(snapshots)!={s["track_id"] for s in seeds}:raise ValueError("Seed/track mismatch")
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    start=seeds[0]["global_frame_id"]
    timebase=region["time_base"][0]/region["time_base"][1]
    seed_frame=next(f for f in region["frames"] if f["global_frame_id"]==start)
    histories=[VideoBoxHistory(mode="video_inference") for _ in range(2)]
    for history in histories:
        history.initialize(start,seed_frame["pts"]*timebase,
            [{"track_id":s["track_id"],"bbox_full_xyxy":s["bbox_native"]} for s in seeds])
    prefix_count=0
    counts={"frames":0,"associated":0,"predicted":0,"lost":0,"sam_used":0,
            "candidate_changed_by_sam":0,"differs_from_single_frame_top2":0}
    with (out/"FUSED_BOXES.jsonl").open("x") as writer:
        for row in pools:
            frame=row["frame"];fid=frame["global_frame_id"]
            if fid<=start:
                if include_prefix:
                    writer.write(json.dumps(prefix_record(row,seeds,start),allow_nan=False)+"\n")
                    prefix_count+=1
                continue
            supports={};evidence={}
            for tid,(doc,index) in snapshots.items():
                f=index[fid]
                if f["pts"]!=frame["pts"] or f["local_index"]!=frame["local_index"]:
                    raise ValueError("Mask timing differs")
                if not f["available"]:continue
                reg=doc["registrations"][f["mask_identity"]];binding=reg["binding"]
                for key,value in [("track_id",tid),("global_frame_id",fid),("pts",frame["pts"]),
                                  ("generation",doc["snapshot"]["generation"]),
                                  ("obj_id",doc["snapshot"]["obj_id"])]:
                    if binding[key]!=value:raise ValueError("Mask binding differs: "+key)
                path=contained_path(R,reg["path"],exists=True)
                if digest(path)!=reg["sha256"]:raise ValueError("Mask file changed")
                with np.load(path,allow_pickle=False) as arrays:mask=arrays["mask"]
                header=json.dumps({"dtype":"bool","shape":list(mask.shape)},separators=(",",":")).encode()
                if hashlib.sha256(header+b"\0"+mask.tobytes(order="C")).hexdigest()!=reg["pixel_sha256"]:
                    raise ValueError("Mask pixels changed")
                supports[tid]=support_box(mask,(region["height"],region["width"]))
                evidence[tid]={"mask_identity":f["mask_identity"],"binding_identity":reg["binding_identity"],"pixel_sha256":reg["pixel_sha256"],
                               "visible_support_box":supports[tid],"full_box":False}
            supports,support_rejections=independent_supports(supports,evidence)
            candidates=[dict(c,candidate_id=f"b0:frame{fid}:grid:{c['grid_index']}")
                        for c in row["pool"]["association_candidates"]]
            fused=histories[0].update(fid,frame["pts"]*timebase,candidates,sam_support_boxes=supports)
            counter=histories[1].update(fid,frame["pts"]*timebase,candidates)
            baseline={c["grid_index"] for c in published_baseline(row["pool"])}
            plain={t["track_id"]:t for t in counter["tracks"]}
            counts["frames"]+=1
            for track in fused["tracks"]:
                state=track["status"];counts[{"observed":"associated","predicted":"predicted","lost":"lost"}[state]]+=1
                counts["sam_used"]+=track["sam_support"]=="used_for_association_only"
                counts["candidate_changed_by_sam"]+=track["candidate_id"]!=plain[track["track_id"]]["candidate_id"]
                if track["raw_detection"] is not None:
                    counts["differs_from_single_frame_top2"]+=track["raw_detection"]["grid_index"] not in baseline
            writer.write(json.dumps({"frame":frame,"fused":fused,"history_only":counter,"sam_evidence":evidence,"sam_support_rejections":support_rejections},
                                    allow_nan=False)+"\n")
    summary={"status":"OFFLINE_BOX_ASSOCIATION_REPLAY_COMPLETE_NOT_ACCURACY_ACCEPTANCE","counts":counts,
             "sam_source_identity":complete["source_identity"],"b0":cfg["b0"],"start_after_seed_frame":start,
             "initialization_frames":prefix_count,"output_frames":counts["frames"]+prefix_count,
             "preseed_frames_not_scored":sum(f["global_frame_id"]<=start for f in region["frames"]),"temporal_scope":"offline bidirectional SAM; not causal latency evidence",
             "A0_ROI_consumed":False,"GT_used":False,"weights_modified":False}
    (out/"COMPLETE.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def _display_frame(row):
    from PIL import Image
    frame=row["frame"];p=contained_path(R,frame["image"],exists=True)
    if digest(p)!=frame["sha256"]:raise ValueError("Source RGB changed")
    with Image.open(p) as source:
        image=source.convert("RGB");w,h=image.size
        return image.resize((640,486)),w,h

def ordered_display(rows,workers=0,loader=None):
    """Bounded ordered prefetch; never materialize the whole decoded video."""
    loader=_display_frame if loader is None else loader
    if type(workers) is not int or not 0<=workers<=8:raise ValueError("Bounded worker count required")
    if workers==0:
        for row in rows:yield row,loader(row)
        return
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    iterator=iter(rows);pending=deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(2*workers):
            try:row=next(iterator)
            except StopIteration:break
            pending.append((row,pool.submit(loader,row)))
        while pending:
            row,future=pending.popleft();value=future.result()
            try:
                next_row=next(iterator);pending.append((next_row,pool.submit(loader,next_row)))
            except StopIteration:pass
            yield row,value

def render(replay_root, sequence_root, output, *, preview_frames=(825,850,880,910,939), tracking_label="SAM + 5-frame B0 association", decode_workers=0):
    """Same-frame box-only video. Explicitly no A0/corner predictions."""
    import subprocess
    from PIL import Image, ImageDraw
    replay_root=contained_path(R,replay_root,exists=True)
    sequence_root=contained_path(R,sequence_root,exists=True)
    receipt=json.loads((replay_root/"COMPLETE.json").read_text())
    cfg=json.loads((sequence_root/"source/config.json").read_text())
    if "b0_evidence" in cfg:
        b0=read_bound(cfg["b0_evidence"])
    else:
        b0=json.loads((sequence_root/"run/B0_COMPLETE.json").read_text())
        manifest=json.loads((sequence_root/"source/SNAPSHOT_MANIFEST.json").read_text())
        if b0["source_identity"]!=manifest["identity"] or b0["b0"]!=cfg["b0"]:
            raise ValueError("B0-only renderer source differs")
    pools={r["frame"]["global_frame_id"]:r["pool"] for r in
           (json.loads(line) for line in contained_path(R,b0["pools"]["path"],exists=True).read_text().splitlines())}
    rows=[json.loads(line) for line in (replay_root/"FUSED_BOXES.jsonl").read_text().splitlines()]
    if len(rows)!=receipt.get("output_frames",receipt["counts"]["frames"]):raise ValueError("Replay count changed")
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    command=[str(R/"runtime/ffmpeg/bin/ffmpeg"),"-v","error","-nostdin","-n",
             "-f","rawvideo","-pix_fmt","rgb24","-s","1280x510","-r","2997/100",
             "-i","pipe:0","-an","-c:v","libx264","-threads","2","-preset","fast",
             "-crf","18","-pix_fmt","yuv420p",str(out/"comparison.mp4")]
    with (out/"encode.log").open("x") as log:
        worker=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for row,(image,w,h) in ordered_display(rows,decode_workers):
                frame=row["frame"]
                canvas=Image.new("RGB",(1280,510),"black")
                canvas.paste(image,(0,24));canvas.paste(image,(640,24));draw=ImageDraw.Draw(canvas)
                draw.text((5,5),f'B0 independent top2 | frame {frame["global_frame_id"]}',fill="white")
                draw.text((645,5),tracking_label+" | box-only, no A0",fill="white")
                def rectangle(box,x,color):
                    b=[box[0]*640/w+x,box[1]*486/h+24,box[2]*640/w+x,box[3]*486/h+24]
                    draw.rectangle(b,outline=color,width=2)
                for det in published_baseline(pools[frame["global_frame_id"]]):
                    rectangle(det["bbox_full_xyxy"],0,"lime")
                if row["fused"].get("initialization_status")=="unconfirmed":
                    draw.text((645,25),"INITIALIZING: raw B0 fallback, no confirmed tracks",fill="yellow")
                    for det in row["fused"]["baseline_detection_fallback"]:
                        rectangle(det["bbox_full_xyxy"],640,"gray")
                for track,color in zip(row["fused"]["tracks"],["lime","orange"]):
                    if track["bbox_for_roi"] is not None:rectangle(track["bbox_for_roi"],640,color)
                worker.stdin.write(np.asarray(canvas).tobytes())
                if frame["global_frame_id"] in preview_frames:
                    canvas.save(out/f'frame_{frame["global_frame_id"]:08d}.png')
            worker.stdin.close()
            if worker.wait()!=0:raise RuntimeError("Video encoder failed")
        except BaseException:
            if worker.poll() is None:worker.kill();worker.wait()
            raise
    probe=json.loads(subprocess.check_output([str(R/"runtime/ffmpeg/bin/ffprobe"),
        "-v","error","-count_frames","-select_streams","v:0","-show_entries",
        "stream=nb_read_frames,width,height,avg_frame_rate","-of","json",str(out/"comparison.mp4")],text=True))
    if int(probe["streams"][0]["nb_read_frames"])!=len(rows):raise ValueError("Encoded frame count differs")
    (out/"COMPLETE.json").write_text(json.dumps({"frames":len(rows),"frame_range":[rows[0]["frame"]["global_frame_id"],rows[-1]["frame"]["global_frame_id"]+1],
        "replay":str(replay_root.relative_to(R)),"replay_sha256":digest(replay_root/"FUSED_BOXES.jsonl"),
        "A0_used":False,"tracking_label":tracking_label,"decode_workers":decode_workers,"test_acceptance":False,"probe":probe},indent=2)+"\n")
    return str(out.relative_to(R))
