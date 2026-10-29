"""Render paired saved surface diagnostics, with explicit source-frame identities."""
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R,contained_path
from eval.action4_video import draw_cuboids

def render(reference,candidate,frame_manifest,frame_ids,output,labels):
 a=json.loads(contained_path(R,reference,exists=True).read_text())
 b=json.loads(contained_path(R,candidate,exists=True).read_text())
 if a["models"]!=b["models"]:raise ValueError("Paired frozen models differ")
 maps=[{(r["frame_id"],r["track_id"]):r for r in x["rows"]} for x in [a,b]]
 if set(maps[0])!=set(maps[1]):raise ValueError("Paired instance domains differ")
 manifest=json.loads(contained_path(R,frame_manifest,exists=True).read_text())
 fm={f["frame_id"]:f for f in manifest["rows"]}
 canvas=Image.new("RGB",(1400,424*len(frame_ids)),"white")
 for index,fid in enumerate(frame_ids):
  keys=[k for k in maps[0] if k[0]==fid]
  rows=[maps[0][k] for k in keys];boxes=np.asarray([r["bbox_for_a0"] for r in rows])
  if any(maps[0][k]["bbox_for_a0"]!=maps[1][k]["bbox_for_a0"] for k in keys):raise ValueError("Paired ROI changed")
  lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0);pad=(hi-lo)*.35
  lo=np.maximum(0,np.floor(lo-pad)).astype(int);hi=np.minimum([fm[fid]["shape"][1],fm[fid]["shape"][0]],np.ceil(hi+pad)).astype(int)
  ref=fm[fid]["source_png"];path=contained_path(R,ref["path"],exists=True)
  if hashlib.sha256(path.read_bytes()).hexdigest()!=ref["sha256"]:raise ValueError("Frame bytes changed")
  with Image.open(path) as im:base=im.convert("RGB").crop((*lo,*hi))
  scale=min(700/base.width,400/base.height);base=base.resize((round(base.width*scale),round(base.height*scale)))
  for side,lookup in enumerate(maps):
   values=[dict(corners_px=((np.asarray(lookup[k]["surface_corners_native"])-lo)*scale).tolist()) for k in keys if lookup[k].get("surface_corners_native") is not None]
   panel=draw_cuboids(base,values,line_width=2,debug_numbers=True);draw=ImageDraw.Draw(panel)
   for row in rows:
    box=(np.asarray(row["bbox_for_a0"])-np.tile(lo,2))*scale
    draw.rectangle(box.tolist(),outline="lime",width=1);draw.text((box[0],max(0,box[1]-12)),row["track_id"],fill="yellow")
   canvas.paste(panel,(700*side,424*index+24))
   ImageDraw.Draw(canvas).text((700*side+3,424*index+4),f"frame{fid} | "+labels[side]+" NOT accepted",fill="black")
 target=contained_path(R,output,exists=False)
 if target.exists():raise FileExistsError("Preserve prior review")
 canvas.save(target,quality=94)
