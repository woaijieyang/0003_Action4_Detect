"""Prepare bounded cleanup manifest; never mutate assets during discovery."""
import ast,json,os,re
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R

def inventory():
    # Retain all data/render implementations: frozen scene provenance references
    # source files by identity. Keep training/eval implementations for provenance.
    # Historical launch-only helpers are the narrow safe source archive scope.
    archive=[]
    for p in (R/"scripts").glob("*.py"):
        if any(token in p.name for token in ("_preview","_diagnostic","_probe_","_compare_amp","_collective_probe","_separated_","_retry_startup")):
            archive.append(str(p.relative_to(R)))
    # Metadata references from maintained code/configuration determine runtime
    # outputs to protect. Reports are archived, not interpreted as instructions.
    protected={"required","action4_final_selection_v001","action4_final_test_v001","action4_final_video_v001",
        "action4_final_proposal_audit_v001","action4_release_bundle_check_v001","action4_release_preparation_v001",
        "action4_aux_eval_acceleration_v001","action4_gdr_surface_candidate_v001",
        "action4_online_b0_v001","action4_gdr_pose_candidate_v002",
        "action4_direction_normalization_v001","action4_a0_additive_v001"}
    # Conservative literal-reference protection; dynamic references are protected
    # by preserving all published data provenance and configured runtime outputs.
    for top in ("src","configs"):
        for root,dirs,files in os.walk(R/top,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and d not in ("__pycache__","local")]
            for name in files:
                p=Path(root)/name
                if p.is_symlink() or p.suffix not in (".py",".json") or p.stat().st_size>4000000:continue
                text=p.read_text(errors="replace")
                protected.update(re.findall(r"outputs/([A-Za-z0-9_\-]+)",text))
    # Follow explicit immutable metadata refs reached by published datasets/assets.
    todo=[R/"data/action4_dataset_commit.json",R/"configs/data/action4_frozen_modules_v003.json"]
    seen=set()
    while todo:
        p=todo.pop()
        if p in seen or not p.exists() or p.is_symlink() or not p.is_file() or p.suffix not in (".json",".jsonl") or p.stat().st_size>20000000:continue
        if not p.resolve().is_relative_to(R):raise ValueError("Boundary")
        seen.add(p);text=p.read_text()
        protected.update(re.findall(r"outputs/([A-Za-z0-9_\-]+)",text))
        for token in re.findall(r'"((?:data|configs|outputs)/[^"\n]+\.jsonl?)"',text):
            target=R/token
            if target.exists() and not target.is_symlink():todo.append(target)
    for p in (R/"outputs").iterdir():
        if p.is_dir() and not p.is_symlink() and p.name not in protected:
            archive.append(str(p.relative_to(R)))
    return {"schema":"action4.cleanup_manifest.v1","status":"PREPARE_ONLY","archive_root":"archive/legacy",
        "archive_paths":sorted(archive),"protected_outputs":sorted(protected),
        "asset_policy":"original data/frozen Blender and all source data modules preserved",
        "delete_paths":[],"delete_policy":"exact manifest after best bundle and all consumers closed"}
if __name__=="__main__":
    out=R/"outputs/action4_release_preparation_v001/ARCHIVE_CANDIDATES.json"
    with out.open("x") as f:json.dump(inventory(),f,indent=2)
    data=json.loads(out.read_text());print(json.dumps({"archive_candidates":len(data["archive_paths"]),"protected_outputs":len(data["protected_outputs"])}))
