"""Thin immutable-source entry for saved video pose stabilization."""
import argparse,hashlib,json,os,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source-manifest",required=True)
 p.add_argument("--config",required=True)
 p.add_argument("--reuse",action="store_true")
 a=p.parse_args();root=Path("/data/zy/vln/a-test")
 if Path.cwd().resolve()!=root:raise ValueError("Enter authorized project root first")
 manifest=(root/a.source_manifest).resolve(strict=True)
 if not manifest.is_relative_to(root):raise ValueError("Source outside project")
 source=manifest.parent;doc=json.loads(manifest.read_text())
 for name,digest in doc["files"].items():
  f=(source/name).resolve(strict=True)
  if not f.is_relative_to(source) or f.stat().st_mode&0o222 or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
   raise ValueError("Immutable source changed")
 os.environ["ACTION4_PROJECT_ROOT"]=str(root);os.environ["PYTHONDONTWRITEBYTECODE"]="1"
 sys.path[:]=[str(source/"runtime/surface_deps"),str(source/"src")]+[
  x for x in sys.path if x and Path(x).resolve()!=root and not Path(x).resolve().is_relative_to(root/"src")]
 from eval.action4_video_stabilization import generate,bound,read,validate_render_receipt
 from data.action4_paths import contained_path
 cfg_ref=bound(a.config);cfg=read(cfg_ref);output=contained_path(root,cfg["output"],exists=False)
 receipt=output/"STABILIZATION_RECEIPT.json";source_ref=bound(manifest)
 if receipt.exists() and a.reuse:
  record=validate_render_receipt(cfg["output"],cfg["input_lock"],bound(receipt))
  if record["config"]["sha256"]!=cfg_ref["sha256"] or record["postprocess_source"]!=source_ref:
   raise ValueError("Existing result has another numeric/source identity")
  print(json.dumps(dict(status="REUSED_IDENTICAL_STABILIZED_PREDICTIONS",receipt=bound(receipt),new_model_calls=0)));return
 print(json.dumps(generate(a.config,source_ref=source_ref)))
if __name__=="__main__":main()
