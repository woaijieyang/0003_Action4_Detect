"""Manifested release housekeeping, confined to this repository.

No generic recursive delete: each removable file is identified by role, then
inode/size/mtime checked again before unlink. All old material is archived
unless it is a redundant checkpoint or a verified regenerable process output.
"""
import argparse,json,os,re,subprocess
from pathlib import Path
from collections import Counter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json

PREP=R/"outputs/action4_release_preparation_v001"

def files_under(root):
    for parent,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink()]
        for name in names:
            p=Path(parent)/name
            if not p.is_symlink():yield p

def stopped():
    jobs=json.loads((R/"outputs/action4_aux_eval_acceleration_v001/DISPATCH.json").read_text())
    pids=[x["pid"] for x in jobs["workers"]]+[x["pid"] for x in jobs["original_evaluators"]]
    pids += [2253388,2253390,2226978,2336333,2336334,2336338,2336339,2366728]
    result=subprocess.run(["ps","-p",",".join(map(str,pids)),"-o","pid=,stat=,args="],capture_output=True,text=True)
    live=[line for line in result.stdout.splitlines() if not line.split()[1].startswith("Z")]
    if live:raise RuntimeError("Known consumers still alive: "+"; ".join(live))
    for run in ["action4_a0_additive_v001/train_run_v002","action4_direction_normalization_v001/pooled_control","action4_direction_normalization_v001/per_roi"]:
        p=R/"outputs"/run
        complete=json.loads((p/"TRAIN_COMPLETE.json").read_text())
        if complete["epochs"]!=80:raise ValueError("Incomplete training")
        if not (p/"ALL_EVAL_SELECTION.json").exists():raise ValueError("Missing complete asynchronous selection")
    for name in ["outputs/action4_release_bundle_check_v001/REPORT.json","outputs/action4_final_video_v001/COMPLETE.json"]:
        if not (R/name).exists():raise ValueError("Release verification missing")
    if not json.loads((R/"outputs/action4_release_bundle_check_v001/REPORT.json").read_text())["passed"]:raise ValueError("Bundle check failed")

def prepare():
    stopped()
    plan=json.loads((PREP/"ARCHIVE_CANDIDATES.json").read_text())
    refs=set()
    for top in ("src","configs"):
        for p in files_under(R/top):
            if p.suffix in (".py",".json") and p.stat().st_size<4000000:
                refs.update(re.findall(r"scripts/[A-Za-z0-9_./-]+\.py",p.read_text(errors="replace")))
    archives=[n for n in plan["archive_paths"] if not n.startswith("scripts/") or n not in refs]
    # Canonical active models remain only A0/B0; old per-run bests stay in archive.
    archives.append("models/action4")
    model_keep={"pretrained.lock.json","action4_online_b0_v001.json","action4_gdr_pose_v002_resolved.json",
        "action4_gdr_surface_v001.json","action4_gdr_surface_train_v001.json",
        "action4_a0_continue_best_v001.json","action4_b0_continue_best_v001.json"}
    archives.extend(str(p.relative_to(R)) for p in (R/"configs/model").glob("*.json") if p.name not in model_keep)
    archives.extend(str(p.relative_to(R)) for p in (R/"configs/eval").glob("*smoke*.json"))
    archives.extend(str(p.relative_to(R)) for p in (R/"configs/eval").glob("action4_final_*_v001.json"))
    deletions=[];retained=[]
    for top in ("models/action4","outputs"):
        for p in files_under(R/top):
            name=p.name
            removable=(name in ("last.pt","joint_init.pt","component_final.pt","component_partial.pt","component_weights_only.pt","selected_parent.pt") or re.fullmatch(r"(?:eval_step|selected_step|epoch|step)_\d+(?:_step_\d+)?\.pt",name)
                or re.fullmatch(r"best_alias_\d+\.tmp",name) or re.fullmatch(r"[^/]+\.pt\.[0-9a-f]+\.tmp",name))
            if removable:
                st=p.stat();deletions.append({"path":str(p.relative_to(R)),"stat":[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],"blocks":st.st_blocks,"nlink":st.st_nlink})
            elif p.suffix in (".pt",".pth",".ckpt"):retained.append(str(p.relative_to(R)))
    # Exact regenerable cache and complete-video shard products.
    extra=[R/"outputs/action4_aux_eval_acceleration_v001/features.pt"]
    for p in (R/"outputs/action4_final_video_v001").glob("part_*"):
        if p.is_dir() and not p.is_symlink():extra.extend(files_under(p))
    for p in extra:
        if p.exists() and not p.is_symlink() and p.is_file():
            st=p.stat();deletions.append({"path":str(p.relative_to(R)),"stat":[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],"blocks":st.st_blocks,"nlink":st.st_nlink})
    unique={x["path"]:x for x in deletions}
    result={"status":"PREPARED","archive_paths":archives,"delete_files":list(unique.values()),"retained_weight_or_feature_paths":retained,
        "policy":"canonical stage best bundles retained; old per-run bests and dependencies archived; only intermediate snapshots/temp aliases/closed shared feature cache/merged video shards unlinked"}
    atomic_json(PREP/"CLEANUP_MANIFEST.json",result)
    print(json.dumps({"archive_paths":len(archives),"delete_files":len(unique),"logical_unlink_bytes":sum(x["stat"][2] for x in unique.values())}))

def apply():
    stopped();plan=json.loads((PREP/"CLEANUP_MANIFEST.json").read_text())
    if (PREP/"CLEANUP_APPLIED.json").exists():raise FileExistsError("Already applied")
    journal=PREP/"cleanup_journal.jsonl"
    if journal.exists():raise FileExistsError("Inspect interrupted cleanup; do not replay blindly")
    # Validate the COMPLETE manifest before the first mutation.
    for item in plan["delete_files"]:
        p=contained_path(R,item["path"],exists=True)
        if p.is_symlink() or not p.is_file():raise ValueError("Unexpected target type")
        st=p.stat()
        if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]!=item["stat"]:raise ValueError("Changed target "+item["path"])
    for name in plan["archive_paths"]:
        source=contained_path(R,name,exists=True);dest=contained_path(R,"archive/legacy/"+name,exists=False)
        if dest.exists():raise FileExistsError(str(dest))
    deleted=0
    with journal.open("x") as log:
        for item in plan["delete_files"]:
            p=contained_path(R,item["path"],exists=True);p.unlink();deleted+=1
            log.write(json.dumps({"delete":item["path"],"bytes":item["stat"][2]})+"\n")
        for name in plan["archive_paths"]:
            source=contained_path(R,name,exists=True);dest=contained_path(R,"archive/legacy/"+name,exists=False)
            dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest)
            log.write(json.dumps({"archive":name,"destination":str(dest.relative_to(R))})+"\n")
    atomic_json(PREP/"CLEANUP_APPLIED.json",{"status":"APPLIED","deleted_files":deleted,"archived_paths":len(plan["archive_paths"]),
        "logical_unlinked_bytes":sum(x["stat"][2] for x in plan["delete_files"]),
        "physical_space_note":"logical bytes include hard links and are NOT claimed as disk space reclaimed",
        "data_modified":False,"canonical_bundles":["models/A0/best.pt","models/B0/best.pt"]})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--apply",action="store_true");a=p.parse_args()
    apply() if a.apply else prepare()
