"""Preview contacts, native target crops and actual B0/A0 training-input views."""
import json,numpy as np,torch
from pathlib import Path
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R
from data import action4_occlusion500_queue as queue
from data.action4_dataset import prepare_rois,prepare_full_images
from data import DataAdapter

def rgb_image(tensor):
    a=tensor.cpu().numpy()*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
    return Image.fromarray(np.uint8(np.clip(a.transpose(1,2,0),0,1)*255))

def run(slots=None,output_name="preview_review",status="preview_checked"):
    torch.set_num_threads(2)
    if Path(output_name).name!=output_name or not output_name.startswith("preview_review"):raise ValueError("Review output must be a local named child")
    if status not in ["preview_checked","accepted"]:raise ValueError("Explicit review status required")
    out=R/"outputs/phrase2_occlusion500_v001"/output_name;out.mkdir(exist_ok=True)
    db=queue.connect();rows=db.execute("SELECT * FROM slots WHERE status=? ORDER BY scene,slot",(status,)).fetchall();db.close()
    if slots is not None:rows=[row for row in rows if row["slot"] in set(slots)]
    if not rows:raise ValueError("No accepted review records in requested scope")
    images=[];identities={};stats=[];inputs=[];mask_seen=set()
    cfg=json.loads((R/"configs/model/action4_b0_photometric_v001.json").read_text());fullsize=cfg["preprocess"]["full_size"]
    for row in rows:
        entry=json.loads(row["record"])
        pointer=entry["candidate"] if status=="preview_checked" else entry["source_candidate_record"]
        payload=json.loads((R/pointer).read_text());record=payload["record"] if status=="preview_checked" else entry;slot=payload["slot"]
        identities[str(slot["slot_id"])]=record["rgb_sha256"]
        with Image.open(R/record["image"]) as source:raw=source.convert("RGB")
        visual=raw.copy();draw=ImageDraw.Draw(visual)
        for obj in record["instances"]:
            draw.rectangle(obj["bbox_full_xyxy"],outline="lime",width=5)
            pts=obj["corners_px"]
            for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:draw.line([tuple(pts[a]),tuple(pts[b])],fill="red",width=4)
            draw.text(tuple(pts[0]),f"ID{obj['instance_id']}",fill="yellow",stroke_width=1,stroke_fill="black")
        small=visual.resize((400,303));panel=Image.new("RGB",(400,329));panel.paste(small,(0,26))
        ImageDraw.Draw(panel).text((3,3),f"slot{slot['slot_id']} {slot['scene_id']} outer={slot['outer_coordinates'] or 'none'} occ={slot['occlusion_bins']}",fill="white")
        images.append(panel)
        image=torch.from_numpy(np.array(raw)).permute(2,0,1)
        full,matrices=prepare_full_images([image],tuple(fullsize),"cpu");b0=rgb_image(full[0])
        rois,_=prepare_rois([image],[x["bbox_full_xyxy"] for x in record["instances"]],[0,0],roi_size=448,expansion=2.25,device="cpu")
        for j,obj in enumerate(record["instances"]):
            bbox=np.array(obj["bbox_full_xyxy"]);bounds=np.r_[np.floor(bbox[:2]-8),np.ceil(bbox[2:]+8)].astype(int)
            crop=raw.crop(tuple(bounds));crop.save(out/f"slot{slot['slot_id']:05d}_i{j+1}_native.png")
            from data.action4_phrase2_visibility import canonical_visibility
            visible=canonical_visibility(np.array(record["corners_3d_mm"])/1000,np.array(obj["canonical8_geometry"]["object_to_capture_cv_m"]),obj["canonical8_geometry"]["ray_clear_to_virtual_point"],obj["canonical8_geometry"]["point_inframe"])["visible"]
            overlay=crop.copy();od=ImageDraw.Draw(overlay)
            for point_id,(point,shown) in enumerate(zip(obj["corners_px"],visible)):
                x,y=np.array(point)-bounds[:2];color="lime" if shown else "red";od.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2);od.text((x+4,y),str(point_id),fill=color,stroke_width=1,stroke_fill="black")
            overlay.save(out/f"slot{slot['slot_id']:05d}_i{j+1}_fixed_ID_overlay.png")
            key=(slot["scene_id"],slot["occlusion_bins"][["left","right"][j]])
            if key not in mask_seen:
                mask_seen.add(key)
                with Image.open(R/obj["mask_visible"]) as source:vm=np.array(source.crop(tuple(bounds)))>0
                origin=np.array(obj["amodal_expanded_origin_px"]);amodal_bounds=bounds-np.tile(origin,2)
                with Image.open(R/obj["mask_amodal_expanded"]) as source:am=np.array(source.crop(tuple(amodal_bounds)))>0
                colored=np.array(crop).copy();colored[am&~vm]=(np.array([255,80,0])*.45+colored[am&~vm]*.55).astype(np.uint8);colored[vm]=(np.array([0,255,0])*.35+colored[vm]*.65).astype(np.uint8)
                Image.fromarray(colored).save(out/f"slot{slot['slot_id']:05d}_i{j+1}_mask_overlay.png")
            a0=rgb_image(rois[j]);a0.save(out/f"slot{slot['slot_id']:05d}_i{j+1}_A0_ROI448.png")
            b=DataAdapter.transform_points(bbox.reshape(2,2),matrices[0]);b=np.r_[np.floor(b[0]-2),np.ceil(b[1]+2)].astype(int);b0crop=b0.crop(tuple(b));b0crop.save(out/f"slot{slot['slot_id']:05d}_i{j+1}_B0_native_input_crop.png")
            stats.append({"slot":slot["slot_id"],"side":j+1,"occlusion":obj["occlusion_inframe"],"native_size":crop.size,"B0_input_size":b0crop.size,"A0_ROI_size":a0.size})
            if len(inputs)<36 or slot["occlusion_bins"][["left","right"][j]]>0:
                thumb=Image.new("RGB",(740,245),(35,35,35));nc=crop.copy();nc.thumbnail((240,215));thumb.paste(nc,(0,28));thumb.paste(a0.resize((215,215)),(250,28));thumb.paste(b0crop,(470,28))
                ImageDraw.Draw(thumb).text((3,5),f"slot{slot['slot_id']} ID{j+1} | raw crop downsample-only | A0 ROI display | B0 native pixels",fill="white");inputs.append(thumb)
    for start in range(0,len(images),12):
        page=Image.new("RGB",(1600,329*3))
        for k,panel in enumerate(images[start:start+12]):page.paste(panel,((k%4)*400,(k//4)*329))
        page.save(out/f"contact_{start//12:02d}.jpg")
    for start in range(0,len(inputs),6):
        page=Image.new("RGB",(1480,245*3))
        for k,panel in enumerate(inputs[start:start+6]):page.paste(panel,((k%2)*740,(k//2)*245))
        page.save(out/f"inputs_{start//6:02d}.jpg")
    report={"count":len(rows),"slot_ids":[row["slot"] for row in rows],"rgb_sha256_by_slot":identities,"configuration_sha256":__import__("hashlib").sha256((R/"configs/data/action4_phrase2_occlusion500_v001.json").read_bytes()).hexdigest(),"input_checks":stats,"B0_full_size":fullsize,"A0_ROI":[448,448],"scope":"GT training/input and same-frame synthetic geometry review; no model accuracy claim"}
    (out/"REVIEW_INPUTS.json").write_text(json.dumps(report,indent=2));print("REVIEW_READY",len(rows),"contacts",(len(images)+11)//12,"input_pages",(len(inputs)+5)//6)
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--slots",type=int,nargs="+");parser.add_argument("--output",default="preview_review");parser.add_argument("--status",default="preview_checked");a=parser.parse_args();run(a.slots,a.output,a.status)
