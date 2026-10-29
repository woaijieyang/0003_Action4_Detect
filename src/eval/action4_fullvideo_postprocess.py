"""After native exit, audit and render all original video frames without inference."""
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def validate_native_complete(done,identity,track_ids,frame_count=2850):
    if (done.get("status")!="CONTINUOUS_NATIVE_SAM_RESPONSES_COMPLETE_NOT_ASSOCIATION_EFFECT_ACCEPTANCE"
        or done.get("source_identity")!=identity):
        raise ValueError("Native completion type/source mismatch")
    rows=done["tracks"]
    if len(rows)!=len(track_ids) or {r["track_id"] for r in rows}!=set(track_ids):
        raise ValueError("Native track domain differs")
    for row in rows:
        if type(row["frames"]) is not int or row["frames"]!=frame_count:
            raise ValueError("Incomplete native frame domain")
        if type(row["available"]) is not int or not 0<=row["available"]<=frame_count:
            raise ValueError("Invalid availability count")
    return {r["track_id"]:r["available"] for r in rows}

def main():
    from data.action4_paths import PROJECT_ROOT as R,contained_path
    from eval.action4_sam_b0_box_replay import replay,render,read_bound
    from eval.action4_video_track_audit import audit
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    cfg=json.loads(contained_path(R,args.config,exists=True).read_text())
    out=contained_path(R,cfg["control"],exists=True)
    parent=contained_path(R,cfg["native_dispatch"],exists=True)
    source=contained_path(R,cfg["native_run"],exists=True)
    try:
        for _ in range(720):
            if (parent/"FAILED.json").exists():raise RuntimeError("Native stage failed; no replay or rerun")
            if (parent/"COMPLETE.json").exists():break
            time.sleep(30)
        else:raise TimeoutError("Native completion not received; no success inferred")
        native_cfg=json.loads((source/"source/config.json").read_text())
        done=json.loads((source/"run/NATIVE_COMPLETE.json").read_text())
        seeds=json.loads((source/"run/SEEDS.json").read_text())["seeds"]
        availability=validate_native_complete(done,cfg["native_source_identity"],[s["track_id"] for s in seeds])
        region=read_bound(native_cfg["region"])
        if [f["global_frame_id"] for f in region["frames"]]!=list(range(2850)):
            raise ValueError("Expected every original frame 0..2849")
        summary=replay(cfg["native_run"],cfg["replay_output"],include_prefix=True)
        replay_path=contained_path(R,cfg["replay_output"],exists=True)
        rows=[json.loads(line) for line in (replay_path/"FUSED_BOXES.jsonl").read_text().splitlines()]
        audited=audit(rows,region["frames"])
        (replay_path/"TRACE_AUDIT.json").write_text(json.dumps(audited,indent=2)+"\n")
        video="outputs/videos/"+datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")+"-stage5-full2850-SAM-B0-box"
        render(cfg["replay_output"],cfg["native_run"],video,
               preview_frames=(0,4,114,545,900,1088,1178,1374,1500,1949,2072,2400,2849))
        result={"status":"FULL2850_REPLAY_AUDIT_VIDEO_COMPLETE_REVIEW_REQUIRED",
                "native_available":availability,"replay":summary,"audit_counts":audited["counts"],
                "video":video,"accuracy_accepted":False,"A0_new_inference":False}
        (out/"COMPLETE.json").write_text(json.dumps(result,indent=2)+"\n")
    except BaseException as exc:
        (out/"FAILED.json").write_text(json.dumps({"error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n")
        raise
if __name__=="__main__":main()
