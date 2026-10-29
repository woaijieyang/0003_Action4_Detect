"""Profile post-prediction scoring only; no model execution or selection."""
import argparse,json,os,sys,cProfile,pstats,hashlib,time
from pathlib import Path
def main():
    root=Path("/data/zy/vln/a-test");assert Path.cwd().resolve()==root
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    out=(root/args.output).resolve(strict=True)
    if not out.is_relative_to(root):raise ValueError("Output outside project")
    artifact=root/"outputs/action4_corner_capacity32_v001/worker_00/PREDICTION_ARTIFACT.json"
    payload=json.loads(artifact.read_text());manifest=json.loads((root/payload["source_manifest"]["path"]).read_text())
    source=root/manifest["code_root"];os.environ["ACTION4_PROJECT_ROOT"]=str(root)
    sys.path[:0]=[str(source/"runtime/surface_deps"),str(source/"src")]
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from eval.action4_corner_scoring_inputs import build_scoring_inputs
    def ref(path):
        with path.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        return {"path":str(path.relative_to(root)),"sha256":sha}
    bindings={"predictions":ref(artifact),"annotations":ref(root/"data/phrase2/eval/annotations.jsonl"),"source":payload["source_manifest"]}
    profile=cProfile.Profile();start=time.monotonic();profile.enable()
    result=build_scoring_inputs(bindings["predictions"],bindings["annotations"],bindings["source"],out/"scoring",diagnostic=False)
    profile.disable();profile.dump_stats(str(out/"SCORING.prof"))
    with (out/"TOP_CUMULATIVE.txt").open("w") as stream:pstats.Stats(profile,stream=stream).sort_stats("cumulative").print_stats(45)
    receipt={"status":"FULL650_POST_PREDICTION_PROFILE_COMPLETE","model_forward_calls":0,"images":650,
       "GT":result["raw_report"]["pose_gt_count"],"All8_002":result["raw_report"]["pose_all8_at_002_success"],
       "wall_seconds_with_profiler":time.monotonic()-start,"source":bindings,"not_capacity_admission":True}
    (out/"COMPLETE.json").write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=="__main__":main()
