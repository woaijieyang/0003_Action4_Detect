"""Read-only native clip evidence review; geometric consistency is not real accuracy."""
import json,hashlib
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_face_entity_registry import build_registry,label_pose_records
from model.action4_native_recovery import box_iou

def review_clip(output_directory,visual_frames):
    from PIL import Image,ImageDraw
    from eval.action4_video import draw_cuboids
    out=contained_path(R,output_directory,exists=True)
    done=json.loads((out/"TRACK_COMPLETE.json").read_text());manifest=json.loads((out/"INPUT_MANIFEST.json").read_text())
    records=[json.loads(l) for l in (out/"POSES.jsonl").read_text().splitlines()]
    events=json.loads((out/"FRAME_EVENTS.json").read_text())
    expected=[(f,t) for f in manifest["frame_ids"] for t in manifest["tracks"]]
    if [(r["frame_id"],r["track_id"]) for r in records]!=expected:raise ValueError("Missing/swapped native body records")
    if any(r["final_corners"]!=r["raw_baseline_corners"] for r in records):raise ValueError("Unadmitted candidate overwrote baseline")
    old_registry=json.loads((R/"outputs/action4_persistent_face_identity_v001/REGISTRY.json").read_text())
    registry=build_registry([b["source_seed"] for b in old_registry["bodies"].values()],old_registry["contract"])
    if registry!=old_registry:raise ValueError("Persistent body registry differs")
    labels=label_pose_records(records,registry)
    (out/"PERSISTENT_ENTITIES.jsonl").write_text(''.join(json.dumps(r)+chr(10) for r in labels))
    inputs={(f["frame_id"],i["track_id"]):i for f in manifest["rows"] for i in f["instances"]}
    missing=[];reference_iou=[];raw_iou=[];raw_delta=[]
    for r in records:
        inp=inputs[r["frame_id"],r["track_id"]]
        if inp["raw_B0_box"] is None:missing.append(r)
        if r.get("corners_candidate") is None:continue
        c=np.array(r["corners_candidate"]);a=np.array(r["raw_baseline_corners"]);box=np.r_[c.min(0),c.max(0)]
        ref=inp.get("bbox_for_a0",inp["raw_B0_box"])
        if ref is not None:
            reference_iou.append(box_iou(box,ref));raw_iou.append(box_iou(np.r_[a.min(0),a.max(0)],ref))
        raw_delta.extend(np.linalg.norm(c-a,axis=1).tolist())
    report=dict(status="REAL_NATIVE_CLIP_ENGINEERING_COMPLETE_VISUAL_REVIEW_REQUIRED",frames=len(events),records=len(records),
        B0_missing_instances=len(missing),SAM_only_geometry_continuations=sum((r.get("recovery_decision") or {}).get("reason")=="current_SAM_texture_support_without_B0" for r in missing),
        missing_B0_instances_still_have_A0_baseline=len(missing),native_numeric_invalid=done["numeric_invalid"],native_call_failures=done["native_call_failed_frames"],
        recovery_counts=done["recovery_counts"],region_disabled=sum(not g["enabled"] for e in events for g in e["region_observation_gate"]),
        fixed_face_ID_count=len({f["persistent_face_id"] for l in labels for f in l["faces"]}),fixed_corner_ID_count=len({c["persistent_corner_id"] for l in labels for c in l["corners"]}),
        native_resolved_box_IoU_median=float(np.median(reference_iou)) if reference_iou else None,
        raw_A0_resolved_box_IoU_median=float(np.median(raw_iou)) if raw_iou else None,
        native_vs_raw_corner_distance_median_px=float(np.median(raw_delta)) if raw_delta else None,
        comparison_is_not_GT_accuracy=True,raw_final_unchanged=True,whole_video_accepted=False)
    frame_map={f["frame_id"]:f for f in manifest["rows"]};canvas=Image.new("RGB",(1280,510*len(visual_frames)),"white")
    for index,fid in enumerate(visual_frames):
        f=frame_map[fid];path=contained_path(R,f["source_png"]["path"],exists=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=f["source_png"]["sha256"]:raise ValueError("Visual source changed")
        with Image.open(path) as im:base=im.convert("RGB").resize((640,486))
        scale=np.array([640/f["shape"][1],486/f["shape"][0]])
        selected=[r for r in records if r["frame_id"]==fid]
        for side,key in enumerate(["raw_baseline_corners","corners_candidate"]):
            rows=[dict(corners_px=(np.asarray(r[key])*scale).tolist()) for r in selected if r.get(key) is not None]
            image=draw_cuboids(base,rows,line_width=2,debug_numbers=True);d=ImageDraw.Draw(image)
            for r in selected:
                i=inputs[fid,r["track_id"]];box=i.get("bbox_for_a0",i["raw_B0_box"])
                if box is not None:
                    b=np.asarray(box)*np.tile(scale,2);d.rectangle(b.tolist(),outline="lime",width=1)
                    d.text((b[0],max(0,b[1]-12)),r["track_id"]+(" SAM ROI" if i["raw_B0_box"] is None else " fused ROI"),fill="yellow")
            canvas.paste(image,(side*640,index*510+24))
            ImageDraw.Draw(canvas).text((side*640+4,index*510+4),f"frame {fid} | "+("raw A0" if side==0 else "native candidate NOT ACCEPTED"),fill="black")
    canvas.save(out/"PAIRED_CONTACT.jpg",quality=92)
    (out/"ENGINEERING_REVIEW.json").write_text(json.dumps(report,indent=2));return report


def compare_saved_clips(parent_directory,candidate_directory,frame_ids,*,labels=("parent native weights","derivative weights candidate"),artifact_name="WEIGHT_COMPARISON.jpg"):
    from PIL import Image,ImageDraw
    from eval.action4_video import draw_cuboids
    if len(labels)!=2 or any(not isinstance(x,str) for x in labels) or Path(artifact_name).name!=artifact_name or not artifact_name.endswith(".jpg"):raise ValueError("Explicit two labels and local JPG name required")
    parent=contained_path(R,parent_directory,exists=True);candidate=contained_path(R,candidate_directory,exists=True)
    pm=json.loads((parent/"INPUT_MANIFEST.json").read_text());cm=json.loads((candidate/"INPUT_MANIFEST.json").read_text())
    if pm["rows"]!=cm["rows"]:raise ValueError("Paired native experiment inputs changed")
    maps=[]
    for d in [parent,candidate]:
        rows=[json.loads(l) for l in (d/"POSES.jsonl").read_text().splitlines()]
        maps.append({(r["frame_id"],r["track_id"]):r for r in rows})
    frames={f["frame_id"]:f for f in pm["rows"]};canvas=Image.new("RGB",(1400,424*len(frame_ids)),"white")
    for index,fid in enumerate(frame_ids):
        f=frames[fid];boxes=np.array([i["bbox_for_a0"] for i in f["instances"]]);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0)
        margin=(hi-lo)*.25;lo=np.maximum(0,np.floor(lo-margin)).astype(int);hi=np.minimum([f["shape"][1],f["shape"][0]],np.ceil(hi+margin)).astype(int)
        path=contained_path(R,f["source_png"]["path"],exists=True)
        with Image.open(path) as im:base=im.convert("RGB").crop((*lo,*hi))
        scale=min(700/base.width,400/base.height);base=base.resize((round(base.width*scale),round(base.height*scale)))
        for side,lookup in enumerate(maps):
            rows=[lookup[fid,i["track_id"]] for i in f["instances"]]
            drawn=[dict(corners_px=((np.asarray(r["corners_candidate"])-lo)*scale).tolist()) for r in rows if r.get("corners_candidate") is not None]
            panel=draw_cuboids(base,drawn,line_width=2,debug_numbers=True);draw=ImageDraw.Draw(panel)
            for inst in f["instances"]:
                box=(np.asarray(inst["bbox_for_a0"])-np.tile(lo,2))*scale;draw.rectangle(box.tolist(),outline="lime",width=1)
                draw.text((box[0],max(0,box[1]-13)),inst["track_id"],fill="yellow")
            canvas.paste(panel,(700*side,424*index+24))
            ImageDraw.Draw(canvas).text((700*side+3,424*index+4),f"frame {fid} | "+labels[side],fill="black")
    canvas.save(candidate/artifact_name,quality=94)
