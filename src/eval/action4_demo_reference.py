"""Extract demo wireframe envelopes independently of model predictions.

Reference consistency only: compression, fragmented lines and occlusion make
these envelopes unsuitable as training labels or certified real mAP ground truth.
"""
import json,subprocess,numpy as np
from scipy import ndimage
from data.action4_paths import PROJECT_ROOT as R
from model.training.action4_runtime import atomic_json

def extract(demo,raw):
    low=demo.min(2);spread=demo.max(2).astype(np.int16)-low.astype(np.int16)
    difference=demo.astype(np.float32).mean(2)-raw.astype(np.float32).mean(2)
    mask=(low>=165)&(spread<=65)&(difference>=35)
    grown=ndimage.binary_dilation(mask,structure=np.ones((3,3)),iterations=1)
    labels,count=ndimage.label(grown,structure=np.ones((3,3)))
    objects=ndimage.find_objects(labels)
    components=[]
    for index,region in enumerate(objects,1):
        if region is None:continue
        sy,sx=region;inside=(labels[region]==index)&mask[region]
        ys,xs=np.where(inside)
        if len(xs)<40:continue
        components.append({"bbox_demo_px":[int(xs.min()+sx.start),int(ys.min()+sy.start),int(xs.max()+sx.start+1),int(ys.max()+sy.start+1)],"bright_difference_pixels":len(xs)})
    components.sort(key=lambda c:c["bbox_demo_px"][0])
    error=np.abs(demo.astype(np.int16)-raw.astype(np.int16)).mean(2).ravel()
    # Bounded alignment diagnostic; no model output or score contributes.
    threshold=np.percentile(error,95)
    trimmed=float(error[error<=threshold].mean())
    return components,trimmed

def run(output):
    out=R/output;out.mkdir(exist_ok=False)
    ffmpeg=str(R/"runtime/ffmpeg/bin/ffmpeg")
    videos=[R/"data/action4_reference/demo_standard.mp4",R/"data/final_test_data/head_left_rgb_raw.mp4"]
    procs=[];logs=[]
    try:
        for index,video in enumerate(videos):
            log=(out/f"decode_{index}.log").open("x");logs.append(log)
            cmd=[ffmpeg,"-v","error","-nostdin","-threads","2","-i",str(video),"-vf","scale=950:720:flags=bicubic","-frames:v","2850","-f","rawvideo","-pix_fmt","rgb24","-threads","2","pipe:1"]
            procs.append(subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log))
        rows=[]
        for fid in range(2850):
            buffers=[p.stdout.read(950*720*3) for p in procs]
            if any(len(b)!=950*720*3 for b in buffers):raise RuntimeError("Video frame coverage mismatch")
            demo,raw=[np.frombuffer(b,dtype=np.uint8).reshape(720,950,3) for b in buffers]
            components,mae=extract(demo,raw)
            rows.append({"frame_id":fid,"components":components,"component_count":len(components),"trimmed_background_MAE":mae,
                "reference_candidate":len(components)==2 and mae<8,"status":"REFERENCE_CANDIDATE_NOT_GT"})
            if (fid+1)%250==0:atomic_json(out/"STATUS.json",{"frames":fid+1,"total":2850})
        if any(p.stdout.read(1) for p in procs):raise RuntimeError("Unexpected additional decoded frame")
        if any(p.wait()!=0 for p in procs):raise RuntimeError("Decoder failed")
        reference=json.loads((R/"outputs/action4_demo_alignment_v001/REFERENCE_OVERLAY_EXTRACTION.json").read_text())["frames"]
        checks=[]
        for old in reference:
            new=rows[old["frame_id"]]["components"]
            boxes_old=sorted([x["bbox_demo_px"] for x in old["reference_overlay_components"]],key=lambda box:box[0])
            checks.append({"frame_id":old["frame_id"],"old_components":len(boxes_old),"new_components":len(new),
                "max_bbox_delta_px":float(np.abs(np.array([x["bbox_demo_px"] for x in new])-boxes_old).max()) if len(new)==len(boxes_old) else None})
        report={"frames":2850,"parameters":{"min_demo_channel":165,"max_channel_range":65,"min_brightness_increase":35,"minimum_original_pixels":40,"dilation":1},
            "independent_of_model_predictions":True,"training_labels_allowed":False,"certified_GT":False,"old_six_reference_checks":checks,
            "reference_candidate_frames":sum(r["reference_candidate"] for r in rows),"rows":rows}
        atomic_json(out/"REFERENCE.json",report);print(json.dumps({k:v for k,v in report.items() if k!="rows"}))
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:p.wait()
        for log in logs:log.close()
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)
