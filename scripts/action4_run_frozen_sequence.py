"""Bootstrap the existing SAM sequence module from one verified frozen source."""
import argparse,hashlib,json,os,runpy,sys
from pathlib import Path
ROOT=Path("/data/zy/vln/a-test")

def verify(config_path):
    config=Path(config_path)
    if not config.is_absolute():config=ROOT/config
    if config.is_symlink() or not config.resolve().is_relative_to(ROOT):raise ValueError("Config path escaped project")
    cfg=json.loads(config.read_text())
    source=ROOT/cfg["source_snapshot"]
    if source.is_symlink() or not source.resolve().is_relative_to(ROOT):raise ValueError("Source path escaped project")
    source=source.resolve(strict=True)
    manifest=json.loads((source/"SNAPSHOT_MANIFEST.json").read_text())
    if manifest.get("schema")!="action4.sam.immutable_source.v1" or manifest.get("config_file")!="config.json" or config.resolve()!=source/"config.json":
        raise ValueError("Explicit immutable sequence source/config required")
    payload={k:v for k,v in manifest.items() if k!="identity"}
    if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()!=manifest["identity"]:
        raise ValueError("Source manifest identity changed")
    if manifest["config_file"] not in manifest["files_sha256"]:raise ValueError("Config is not in frozen source")
    for name,expected in manifest["files_sha256"].items():
        file=source/name
        if file.is_symlink() or not file.resolve(strict=True).is_relative_to(source) or file.stat().st_mode&0o222:
            raise ValueError("Writable/escaped source")
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError("Source content changed")
    for prefix in ("src","runtime/surface_deps","upstream/sam3"):
        base=source/prefix
        if not base.exists():continue
        for current,dirs,names in os.walk(base,followlinks=False):
            if any((Path(current)/n).is_symlink() for n in dirs+names):raise ValueError("Source contains symlink")
            for name in names:
                file=Path(current)/name
                if file.suffix in (".py",".pyc",".pyo",".so",".pyd",".dll") and str(file.relative_to(source)) not in manifest["files_sha256"]:
                    raise ValueError("Unlisted importable source")
    return cfg,source

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--mode",required=True,choices=["sam"])
    a=p.parse_args()
    if Path.cwd().resolve()!=ROOT:raise ValueError("Expected project working directory")
    cfg,source=verify(a.config)
    r=cfg["resources"]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=r["gpu_uuid"] or set(os.sched_getaffinity(0))!=set(r["cpu_affinity"]):
        raise ValueError("Assigned native resources differ")
    os.environ["ACTION4_PROJECT_ROOT"]=str(ROOT);os.environ["PYTHONDONTWRITEBYTECODE"]="1";sys.dont_write_bytecode=True
    sys.path[:]=[str(source/"runtime/surface_deps"),str(source/"src")]+[
        x for x in sys.path if x and not Path(x).resolve().is_relative_to(ROOT/"src")
        and Path(x).resolve() not in (ROOT,ROOT/"scripts")]
    sys.argv=["sequence","--config",str(source/"config.json"),"--mode",a.mode]
    runpy.run_module("eval.action4_sam_b0_sequence",run_name="__main__")
if __name__=="__main__":main()
