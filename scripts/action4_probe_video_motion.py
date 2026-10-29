"""Thin entry for the frozen, paired video image-motion diagnostic."""
import argparse,hashlib,json,os,runpy
from pathlib import Path
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    args=parser.parse_args()
    root=Path("/data/zy/vln/a-test")
    if Path.cwd().resolve()!=root:raise ValueError("Enter the authorized project root")
    source=(root/args.source).resolve(strict=True)
    if not source.is_relative_to(root):raise ValueError("Snapshot escapes project")
    manifest=json.loads((source/"SOURCE_MANIFEST.json").read_text())
    for name,digest in manifest["files"].items():
        path=(source/name).resolve(strict=True)
        if not path.is_relative_to(source) or path.stat().st_mode&0o222 or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError("Immutable snapshot differs")
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    runpy.run_path(str(source/"run.py"),run_name="__main__")
if __name__=="__main__":main()
