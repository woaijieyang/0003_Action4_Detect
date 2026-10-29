"""CPU-only complete joint-native SAM + frozen B0 replay and diagnostics.

Reads immutable evidence once per consumer, preserves four distinct tracking
statuses, and never executes B0/SAM/A0 models or treats availability as accuracy.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT
from model.action4_sam_supported_boxes import SamSupportedBoxes


def checked_path(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Project path escape or symlink")
    return p


def finite(value):
    if isinstance(value,np.ndarray):
        if value.dtype.kind=="O" or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError("Nonfinite/unsafe array")
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,(list,tuple)):
        for x in value:finite(x)
    elif isinstance(value,(float,np.floating)) and not math.isfinite(value):
        raise ValueError("Nonfinite value")


def read_json(path):
    def reject(value):raise ValueError("Nonfinite JSON:"+value)
    return json.loads(checked_path(path).read_text(),parse_constant=reject)


def atomic_json(path,value):
    path=checked_path(path);temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)


class ReceiptVerifier:
    """Hash each newly consumed file once; detect changes on subsequent access."""
    def __init__(self):self.verified={}
    def verify(self,path,sha):
        p=checked_path(path)
        if not isinstance(sha,str) or len(sha)!=64:raise ValueError("Receipt SHA required")
        st=p.stat();fingerprint=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        key=str(p.resolve())
        if key in self.verified:
            if self.verified[key]!=(sha,fingerprint):raise ValueError("Consumed file changed")
        else:
            with p.open("rb") as f:actual=hashlib.file_digest(f,"sha256").hexdigest()
            after=p.stat()
            if actual!=sha or fingerprint!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError("Receipt SHA mismatch or concurrent mutation")
            self.verified[key]=(sha,fingerprint)
        return p
    def bound_json(self,ref):return read_json(self.verify(ref["path"],ref["sha256"]))
    def raw(self,receipt,frame_dir):
        frame_dir=checked_path(frame_dir)
        arr=self.verify(receipt["arrays"],receipt["sha256"])
        header=self.verify(receipt["header"],receipt["header_sha256"])
        if arr!=frame_dir/"raw.npz" or header!=frame_dir/"raw.json":
            raise ValueError("Raw receipt outside exact frame directory")
        meta=read_json(header)
        with np.load(arr,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        used=set()
        def restore(v):
            if isinstance(v,dict) and "array_key" in v:
                if set(v)!={"array_key","shape","dtype"} or v["array_key"] not in arrays or v["array_key"] in used:
                    raise ValueError("Malformed/duplicate raw array descriptor")
                a=arrays[v["array_key"]];used.add(v["array_key"])
                if list(a.shape)!=v["shape"] or str(a.dtype)!=v["dtype"]:raise ValueError("Raw descriptor differs")
                return a
            if isinstance(v,dict):return {k:restore(x) for k,x in v.items()}
            if isinstance(v,list):return [restore(x) for x in v]
            return v
        raw=restore(meta)
        if used!=set(arrays):raise ValueError("Unreferenced raw arrays")
        finite(raw)
        return raw


def validate_completion(complete,close):
    expected={"status":"FULL2850_COVERAGE_JOINT_NATIVE_BASELINE_NOT_ACCURACY","original_frames":2850,
              "propagated_frames":2846,"preseed_unconfirmed":[0,1,2,3],"first_frame":4,"last_frame":2849,
              "joint_obj_map":{"0":"track_00","1":"track_01"},"causal_forward":True,"auto_correction":False,
              "closed":True,"accuracy_accepted":False}
    if any(complete.get(k)!=v or type(complete.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError("Actual complete2850/2846 causal closed native run required")
    if not isinstance(complete.get("single_session"),str) or not complete["single_session"]:raise ValueError("Session missing")
    if close.get("is_success") is not True or close.get("gpu_mem",{}).get("active_session_count")!=0:
        raise ValueError("Native close/idle proof missing")
    finite(complete);finite(close)
    return complete["single_session"]


def candidates_for_frame(pool_row,frame):
    if pool_row.get("frame")!=frame:raise ValueError("B0 frame/RGB/source mapping differs")
    pool=pool_row["pool"]
    if pool.get("schema")!="action4.b0_association_pool.v1":raise ValueError("B0 pool schema differs")
    candidates=[];seen=set()
    for raw in pool["association_candidates"]:
        finite(raw);grid=raw["grid_index"]
        if type(grid) is not int or grid<0 or grid in seen:raise ValueError("Duplicate/invalid B0 grid")
        seen.add(grid);item=copy.deepcopy(raw)
        cid=f"b0:frame{frame['global_frame_id']}:grid:{grid}"
        if "candidate_id" in item and item["candidate_id"]!=cid:raise ValueError("B0 candidate frame identity differs")
        item["candidate_id"]=cid;candidates.append(item)
    return candidates


def masks_for_frame(status,raw,frame,region,session):
    fid=frame["global_frame_id"]
    expected={"frame":fid,"global_frame_id":fid,"pts":frame["pts"],"time_base":region["time_base"],
              "pixel_space_id":region["pixel_space_id"],"RGB_sha256":frame["sha256"],"session_id":session,
              "generation":0,"accuracy_accepted":False}
    if any(status.get(k)!=v or type(status.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError("Native frame/RGB/PTS/session/generation mismatch")
    finite(status);finite(raw)
    if raw.get("frame_index")!=fid:raise ValueError("Raw frame mismatch")
    output=raw["outputs"];ids=np.asarray(output["out_obj_ids"])
    masks=np.asarray(output["out_binary_masks"]);probs=np.asarray(output["out_probs"]);boxes=np.asarray(output["out_boxes_xywh"])
    if ids.ndim!=1 or ids.dtype.kind not in "iu" or len(set(ids.tolist()))!=len(ids) or not set(ids.tolist())<={0,1}:
        raise ValueError("Native IDs duplicated/unknown")
    if masks.dtype!=np.bool_ or masks.shape!=(len(ids),region["height"],region["width"]) or probs.shape!=(len(ids),) or boxes.shape!=(len(ids),4):
        raise ValueError("Native arrays shape/pixel space differs")
    if ((probs<0)|(probs>1)).any():raise ValueError("Native probability invalid")
    entries=status["instances"]
    if len(entries)!=2 or {(e["obj_id"],e["track_id"]) for e in entries}!={(0,"track_00"),(1,"track_01")}:
        raise ValueError("Fixed project/native IDs differ")
    scores=status["native_scores"];history=status["history"]
    if scores.get("frame")!=fid or scores.get("session_id")!=session or scores.get("available") is not True or set(scores.get("objects",{}))!={"0","1"}:
        raise ValueError("Actual per-frame native scores missing")
    if history.get("frame")!=fid or history.get("reverse") is not False or not history.get("counts") or max(history["counts"])>15:
        raise ValueError("Native15-history proof differs")
    result={};unavailable=[]
    for e in entries:
        oid=e["obj_id"];tid=e["track_id"];where=np.flatnonzero(ids==oid)
        index=int(where[0]) if len(where) else None
        mask=None if index is None else masks[index]
        available=mask is not None and bool(mask.any())
        if (e.get("generation")!=0 or e.get("class_quality_certified") is not False
                or e.get("public_response_present")!=(index is not None)
                or e.get("mask_pixels")!=(0 if mask is None else int(mask.sum()))
                or e.get("status")!=("available" if available else "target_visibility_unknown")
                or e.get("public_out_prob")!=(None if index is None else float(probs[index]))
                or e.get("native_tracking_score")!=scores["objects"][str(oid)]):
            raise ValueError("Native per-instance status contradicts raw output")
        result[tid]=mask if available else None
        if not available:unavailable.append(tid)
    return result,unavailable


class ReplayAudit:
    def __init__(self,frame_count=2850):
        self.frame_count=frame_count;self.next_frame=0
        self.counts={tid:Counter() for tid in ("track_00","track_01")}
        self.native_unavailable=Counter();self.open_lost={};self.lost_spans=[];self.ambiguous=[]
        self.previous={};self.motion=[];self.initialization_frames=[]
    def add(self,row):
        frame=row["frame"];fid=frame["global_frame_id"];fused=row["fused"]
        if fid!=self.next_frame or fused["frame_id"]!=fid:raise ValueError("Absent/duplicate/out-of-order replay frame")
        finite(row);self.next_frame+=1
        tracks=fused["tracks"]
        unavailable=row.get("native_unavailable")
        if (not isinstance(unavailable,list) or any(t not in self.counts for t in unavailable)
                or len(unavailable)!=len(set(unavailable))):
            raise ValueError("Native unavailable labels must be unique fixed IDs")
        if fid<4:
            if tracks or unavailable or row.get("status")!="initialization":raise ValueError("Frames0..3 must be explicit initialization tracks[]")
            self.initialization_frames.append(fid);return
        if len(tracks)!=2 or {x["track_id"] for x in tracks}!={"track_00","track_01"}:raise ValueError("Fixed ID coverage required")
        used=[x["candidate_id"] for x in tracks if x["candidate_id"] is not None]
        if len(used)!=len(set(used)):raise ValueError("One B0 candidate used by two IDs")
        for tid in row["native_unavailable"]:self.native_unavailable[tid]+=1
        for x in tracks:
            tid=x["track_id"];status=x["status"]
            if status not in ("observed","sam_supported","predicted","lost"):raise ValueError("Unknown tracker state")
            if (status=="observed")!=(x["candidate_id"] is not None):raise ValueError("SAM support cannot be relabeled B0-observed")
            support=x.get("sam_current_support")
            if type(support) is not bool:raise ValueError("Explicit current SAM support boolean required")
            box=x.get("bbox_for_roi")
            if (status=="lost")!=(box is None):raise ValueError("Lost must have null box; all nonlost states require box")
            raw=x.get("raw_detection")
            if status=="observed":
                if not isinstance(raw,dict):raise ValueError("Observed state requires actual B0 detection")
                grid=raw.get("grid_index")
                if type(grid) is not int or grid<0:raise ValueError("Observed B0 grid invalid")
                expected=f"b0:frame{fid}:grid:{grid}"
                if x["candidate_id"]!=expected or raw.get("candidate_id")!=expected:
                    raise ValueError("Observed candidate/frame/grid identity mismatch")
                raw_box=np.asarray(raw.get("bbox_full_xyxy"),float)
                if raw_box.shape!=(4,) or not np.isfinite(raw_box).all() or not (raw_box[2:]>raw_box[:2]).all():
                    raise ValueError("Observed raw B0 box invalid")
            elif raw is not None:
                raise ValueError("Nonobserved state cannot carry B0 detection")
            if status=="sam_supported" and support is not True:
                raise ValueError("SAM-supported state requires actual current support")
            if status in ("predicted","lost") and support is not False:
                raise ValueError("History-only state cannot claim current SAM support")
            self.counts[tid][status]+=1
            if status=="lost":
                if tid not in self.open_lost:self.open_lost[tid]=fid
            elif tid in self.open_lost:self.lost_spans.append({"track_id":tid,"start":self.open_lost.pop(tid),"end":fid-1})
            if x.get("support_rejection")=="ambiguous_shared_instance_support":
                self.ambiguous.append({"frame":fid,"track_id":tid,"reason":x["support_rejection"]})
            box=x["bbox_for_roi"]
            if box is None:self.previous.pop(tid,None);continue
            box=np.asarray(box,float)
            if box.shape!=(4,) or not (box[2:]>box[:2]).all():raise ValueError("Invalid replay box")
            if tid in self.previous:
                pf,pb=self.previous[tid]
                if pf==fid-1:
                    jump=float(np.linalg.norm((box[:2]+box[2:]-pb[:2]-pb[2:])/2))
                    self.motion.append({"frame":fid,"previous_frame":pf,"track_id":tid,"center_jump_px":jump,
                        "center_jump_per_previous_box_diagonal":jump/float(np.linalg.norm(pb[2:]-pb[:2]))})
            self.previous[tid]=(fid,box)
    def finish(self):
        if self.next_frame!=self.frame_count:raise ValueError("Full frame domain incomplete")
        spans=self.lost_spans+[{"track_id":tid,"start":start,"end":self.frame_count-1} for tid,start in self.open_lost.items()]
        for x in spans:x["length"]=x["end"]-x["start"]+1
        jumps=sorted(self.motion,key=lambda x:(-x["center_jump_per_previous_box_diagonal"],x["frame"],x["track_id"]))[:20]
        frames={x for x in (0,4,116,1195,1289,1374,2400,2849) if x<self.frame_count}
        frames.update(x["start"] for x in spans);frames.update(x["frame"] for x in jumps)
        return {"status":"COMPLETE_REPLAY_AUDIT_NOT_GT_ACCURACY","frames":self.frame_count,
            "initialization_frames":self.initialization_frames,"counts":{tid:{k:counts[k] for k in ("observed","sam_supported","predicted","lost")} for tid,counts in self.counts.items()},
            "native_unavailable":{tid:self.native_unavailable[tid] for tid in self.counts},
            "total_counts":{state:sum(c[state] for c in self.counts.values()) for state in ("observed","sam_supported","predicted","lost")},"lost_spans":sorted(spans,key=lambda x:(x["start"],x["track_id"])),
            "ambiguous_mask_events":self.ambiguous,"largest_center_motion_diagnostics":jumps,
            "diagnostic_frames":sorted(frames),"accuracy_accepted":False,"GT_accuracy_measured":False,
            "A0_connected":False,"model_calls":0,"raw_B0_SAM_modified":False}


def run(config):
    if Path.cwd().resolve()!=ROOT.resolve():raise ValueError("Expected project cwd")
    native=checked_path(config["native_job"]);out=checked_path(config["output"])
    if (native/"run/FAILED.json").exists():raise ValueError("Native run failed; no replay admission")
    verifier=ReceiptVerifier()
    manifest=verifier.bound_json(config["native_source_manifest"])
    cfg=read_json(verifier.verify(native/"source/config.json",manifest["files_sha256"]["config.json"]))
    if checked_path(cfg["source_snapshot"])!=native/"source" or checked_path(cfg["output"])!=native/"run":
        raise ValueError("Native config belongs to another job")
    region=verifier.bound_json(cfg["region"])
    if len(region["frames"])!=2850 or [(r["local_index"],r["global_frame_id"]) for r in region["frames"]]!=[(i,i) for i in range(2850)]:
        raise ValueError("Original2850 frame domain required")
    verifier.bound_json(region["source_manifest"])
    complete=read_json(native/"run/COMPLETE.json");session=validate_completion(complete,read_json(native/"run/CLOSE.json"))
    points=read_json(verifier.verify(native/"source/inputs/POINT_SPECS.json",manifest["files_sha256"]["inputs/POINT_SPECS.json"]))
    if [(p["obj_id"],p["track_id"],p["frame"]) for p in points]!=[(0,"track_00",4),(1,"track_01",4)]:
        raise ValueError("Initial joint ID mapping differs")
    seed_refs=[cfg["instance_seed_sources"][tid]["seeds"] for tid in ("track_00","track_01")]
    if seed_refs[0]!=seed_refs[1]:raise ValueError("Bootstrap seed source differs")
    seeds=verifier.bound_json(seed_refs[0])["seeds"]
    if [(s["track_id"],s["global_frame_id"],s["local_index"]) for s in seeds]!=[("track_00",4,4),("track_01",4,4)]:
        raise ValueError("Original frame4 seed identities required")
    for p,s in zip(points,seeds):
        if p["candidate_id"]!=s["candidate_id"] or p["proof"]["B0_box"]!=s["bbox_native"] or p["proof"]["GT_used"] is not False:
            raise ValueError("Point source/seed box changed")
    b0=verifier.bound_json(cfg["b0_evidence"])
    if b0.get("status")!="B0_REGION_EVIDENCE_COMPLETE" or b0.get("frames")!=2850 or b0["b0"]!=cfg["b0"] or b0["region"]["sha256"]!=cfg["region"]["sha256"]:
        raise ValueError("Frozen B0 source/model/region mismatch")
    pools=verifier.verify(b0["pools"]["path"],b0["pools"]["sha256"])
    tracker=SamSupportedBoxes(seeds,window=5,geometry_mode=config.get("geometry_mode","legacy"))
    audit=ReplayAudit();out.mkdir(parents=True,exist_ok=False)
    partial=out/"FUSED_BOXES.partial.jsonl"
    try:
        with pools.open() as source,partial.open("x") as destination:
            for fid in range(2850):
                line=source.readline()
                if not line:raise ValueError("B0 pool frame missing")
                pool=json.loads(line);finite(pool);frame=region["frames"][fid]
                candidates=candidates_for_frame(pool,frame)
                if fid<4:
                    pre=read_json(native/f"run/preseed_{fid:06d}.json")
                    if pre.get("frame")!=fid or pre.get("pts")!=frame["pts"] or pre.get("time_base")!=region["time_base"] or pre.get("status")!="unconfirmed_preseed":
                        raise ValueError("Native preseed identity mismatch")
                    if {(x.get("obj_id"),x.get("track_id"),x.get("status")) for x in pre.get("instances",[])}!={(0,"track_00","unconfirmed_preseed"),(1,"track_01","unconfirmed_preseed")} or len(pre["instances"])!=2:
                        raise ValueError("Native preseed slots differ")
                    row={"frame":frame,"status":"initialization","fused":{"frame_id":fid,"tracks":[],"mode":"video_inference"},"native_unavailable":[]}
                else:
                    directory=native/f"run/frame_{fid:06d}";status=read_json(directory/"STATUS.json")
                    raw=verifier.raw(status["raw"],directory)
                    masks,unavailable=masks_for_frame(status,raw,frame,region,session)
                    seconds=frame["pts"]*region["time_base"][0]/region["time_base"][1]
                    fused=tracker.update(fid,seconds,candidates,masks)
                    row={"frame":frame,"status":"replayed","fused":fused,"native_unavailable":unavailable,
                        "native_status":str((directory/"STATUS.json").relative_to(ROOT)),"session_id":session,"generation":0,
                        "A0_connected":False,"full_box_accuracy_verified":False}
                audit.add(row);destination.write(json.dumps(row,allow_nan=False)+"\n")
            if source.readline():raise ValueError("Unexpected extra B0 pool frame")
        summary=audit.finish()
        if (native/"run/FAILED.json").exists() or read_json(native/"run/COMPLETE.json")!=complete:
            raise ValueError("Native completion changed during replay")
        partial.rename(out/"FUSED_BOXES.jsonl")
        summary.update(native_job=str(native.relative_to(ROOT)),native_source_identity=manifest["identity"],
                       b0_evidence=cfg["b0_evidence"],source_identity=config["source_identity"],geometry_mode=config.get("geometry_mode","legacy"),prompt_protocol=cfg.get("prompt_protocol","legacy_points"))
        atomic_json(out/"AUDIT.json",summary)
        atomic_json(out/"COMPLETE.json",{"status":"FULL2850_BOX_REPLAY_AND_AUDIT_COMPLETE_NOT_ACCURACY",
                    "frames":2850,"fused":"FUSED_BOXES.jsonl","audit":"AUDIT.json","A0_connected":False,
                    "source_identity":config["source_identity"],"native_source_identity":manifest["identity"]})
        return summary
    except BaseException as error:
        import traceback
        atomic_json(out/"FAILED.json",{"error":str(error),"traceback":traceback.format_exc(),"frames_seen":audit.next_frame,
                    "partial_preserved":True,"accuracy_accepted":False})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    print(json.dumps(run(read_json(args.config)),allow_nan=False))

if __name__=="__main__":main()
