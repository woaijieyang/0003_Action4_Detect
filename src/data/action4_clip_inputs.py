"""Prepare isolated clip inputs by reusing frozen native frames and B0 evidence."""
import hashlib,json,os
from pathlib import Path
from .action4_paths import PROJECT_ROOT as R,contained_path

def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def bind(p):return {"path":str(p.relative_to(R)),"sha256":digest(p)}
def read(ref):
    p=contained_path(R,ref["path"],exists=True)
    if digest(p)!=ref["sha256"]:raise ValueError("Clip input reference changed")
    return json.loads(p.read_text())

def prepare(protocol_ref,input_lock_ref,output):
    protocol=read(protocol_ref);lock=read(input_lock_ref)
    if protocol["source_manifest"]!=lock["video"]["frames_manifest"] or protocol["roi_expansion"]!=1.5:
        raise ValueError("Registered clip source/ROI differs")
    source=read(protocol["source_manifest"]);frames=source["frames"]
    if len(frames)!=2850 or [f["frame_id"] for f in frames]!=list(range(2850)):
        raise ValueError("Original2850 frame mapping required")
    evidence=read(lock["video"]["raw_B0_pool_evidence"])
    if evidence["pools"]!=lock["video"]["raw_B0_pools"] or evidence["b0"]["sha256"]!=lock["video"]["track_replay"]["b0_sha256"]:
        raise ValueError("Frozen B0 evidence differs")
    native_region=read(evidence["region"])
    if (native_region["width"],native_region["height"])!=(lock["video"]["width"],lock["video"]["height"]):
        raise ValueError("Native source dimensions differ")
    out=contained_path(R,output,exists=False);out.mkdir(exist_ok=False)
    regions=[];streams=[];owners={}
    try:
        for c in protocol["clips"]:
            if c["reuse_whole_video_track_state"] or c["session_policy"]!="independent_clip_cold_start":
                raise ValueError("Independent clip session contract required")
            a,b=c["context"];lo,hi=c["core"]
            if not 0<=a<=lo<hi<=b<=2850 or c["input_frame_ids"]!=list(range(a,b)):
                raise ValueError("Invalid registered clip domain")
            folder=contained_path(R,out/c["clip_id"],exists=False)
            if folder.parent!=out:raise ValueError("Clip directory must be a direct child")
            folder.mkdir();png=folder/"frames";png.mkdir()
            local=[]
            for i,f in enumerate(frames[a:b]):
                src=contained_path(R,f["image"],exists=True);dst=png/("%06d.png"%i)
                os.link(src,dst)
                if not os.path.samefile(src,dst):raise ValueError("Staged native bytes differ")
                local.append(dict(local_index=i,global_frame_id=f["frame_id"],pts=f["pts"],image=f["image"],sha256=f["image_sha256"]))
                owners.setdefault(f["frame_id"],[]).append((len(regions),i))
            region=dict(region_id=c["clip_id"],frames=local,source_manifest=protocol["source_manifest"],
                staged_frames=str(png.relative_to(R)),pixel_space_id=native_region["pixel_space_id"],
                width=lock["video"]["width"],height=lock["video"]["height"],
                time_base=native_region["time_base"],core=[lo,hi],session_policy="independent_clip_cold_start",
                inherited_session=None,core_only_scoring=True)
            rp=folder/"REGION.json";rp.write_text(json.dumps(region,indent=2))
            bp=folder/"B0_POOLS.jsonl";streams.append(bp.open("x"))
            regions.append(dict(folder=folder,region=region,region_ref=bind(rp),pools=bp,count=0))
        pools=contained_path(R,evidence["pools"]["path"],exists=True);sha=hashlib.sha256();total=0
        with pools.open("rb") as f:
            for raw in f:
                sha.update(raw);row=json.loads(raw);frame=row["frame"];fid=frame["global_frame_id"]
                if fid!=total:raise ValueError("Original B0 order differs")
                expected=frames[fid]
                if (frame["image"],frame["sha256"],frame["pts"])!=(expected["image"],expected["image_sha256"],expected["pts"]):
                    raise ValueError("B0 RGB/PTS mismatch")
                for j,i in owners.get(fid,[]):
                    streams[j].write(json.dumps(dict(frame=regions[j]["region"]["frames"][i],pool=row["pool"]),allow_nan=False)+chr(10))
                    regions[j]["count"]+=1
                total+=1
        if total!=2850 or sha.hexdigest()!=evidence["pools"]["sha256"]:raise ValueError("Original frozen B0 content differs")
        for f in streams:f.close()
        results=[]
        for item in regions:
            if item["count"]!=len(item["region"]["frames"]):raise ValueError("Incomplete clip B0 domain")
            proof=dict(status="B0_REGION_EVIDENCE_COMPLETE",frames=item["count"],b0=evidence["b0"],
                region=item["region_ref"],pools=bind(item["pools"]),source_identity=evidence["source_identity"],
                provenance="exact_subset_of_frozen_single_frame_B0_no_new_inference",
                parent_evidence=lock["video"]["raw_B0_pool_evidence"],GT_used=False)
            cp=item["folder"]/"B0_COMPLETE.json";cp.write_text(json.dumps(proof,indent=2))
            results.append(dict(clip_id=item["region"]["region_id"],region=item["region_ref"],B0=bind(cp)))
        result=dict(status="CLIP_INPUTS_PREPARED_NOT_TRACKING_RUN",protocol=protocol_ref,input_lock=input_lock_ref,
            clips=results,model_calls=0,native_SAM_sessions=0,A0_calls=0,
            PNG_content_rehashed=False,PNG_identity="project-contained samefile hardlinks to immutable native source; inference rechecks declared SHA")
        (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));return result
    finally:
        for f in streams:
            if not f.closed:f.close()
