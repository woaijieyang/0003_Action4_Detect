"""Prepare face hypotheses from authenticated saved selected-model surfaces."""
import argparse,hashlib,json,os,sys
from pathlib import Path
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-manifest",required=True)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    root=Path("/data/zy/vln/a-test")
    if Path.cwd().resolve()!=root:raise ValueError("Enter project root")
    manifest=(root/args.source_manifest).resolve(strict=True)
    if not manifest.is_relative_to(root):raise ValueError("Snapshot outside project")
    source=manifest.parent
    for name,digest in json.loads(manifest.read_text())["files"].items():
        path=(source/name).resolve(strict=True)
        if not path.is_relative_to(source) or path.stat().st_mode&0o222 or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError("Immutable source changed")
    os.environ["ACTION4_PROJECT_ROOT"]=str(root)
    sys.path[:0]=[str(source/"src"),str(root/"runtime/surface_deps")]
    from eval.action4_face_pilot_inputs import prepare_saved
    result=prepare_saved(args.config)
    print(json.dumps({k:v for k,v in result.items() if k!="results"}))
if __name__=="__main__":main()
