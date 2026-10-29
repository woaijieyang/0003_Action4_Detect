"""One-shot immutable handoff from completed full-video bootstrap to SAM tracking."""
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path

ROOT=Path("/data/zy/vln/a-test")

def contained(path):
    p=(ROOT/path).resolve()
    if not p.is_relative_to(ROOT):raise ValueError("Path escaped project")
    return p

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {"path":str(path.relative_to(ROOT)),"sha256":sha(path)}

def prepare(bootstrap,output):
    old=bootstrap/"source";source=read(old/"SNAPSHOT_MANIFEST.json")
    if source.get("schema")!="action4.sam.immutable_source.v1":raise ValueError("Unknown source schema")
    payload={k:v for k,v in source.items() if k!="identity"}
    if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()!=source["identity"]:
        raise ValueError("Bootstrap manifest identity changed")
    cfg=read(old/"config.json");done=read(bootstrap/"run/NATIVE_COMPLETE.json")
    b0=read(bootstrap/"run/B0_COMPLETE.json");seeds=read(bootstrap/"run/SEEDS.json")
    if done["status"]!="DISCOVERY_SEEDS_ONLY_COMPLETE_NOT_TRACKING" or done["source_identity"]!=source["identity"]:
        raise ValueError("Bootstrap did not finish seed-only operation")
    if (b0["source_identity"]!=source["identity"] or b0["b0"]!=cfg["b0"]
        or b0["region"]!=cfg["region"]):
        raise ValueError("B0 source/model/frame identity differs")
    if b0["frames"]!=2850 or cfg["sequence"]["frame_count"]!=2850:
        raise ValueError("Full2850 domain required")
    if len(seeds["seeds"])!=2 or {x["track_id"] for x in done["tracks"]}!={x["track_id"] for x in seeds["seeds"]}:
        raise ValueError("Expected two verified initial seeds; do not invent missing object")
    load=read(bootstrap/"run/NATIVE_LOAD.json")
    if load["status"]!="LOADED_ZERO_FINAL_MISSING_UNEXPECTED":raise ValueError("Native load not verified")
    for seed in seeds["seeds"]:
        folder=bootstrap/"run"/seed["track_id"]
        if read(folder/"SEED_BINDING.json")["reason"]!="bound_unique_prediction_support":
            raise ValueError("Initial seed binding failed")
        if read(folder/"CLOSE.json")["released"] is not True:raise ValueError("Bootstrap session not released")
    if output.exists():raise FileExistsError("Tracking output exists; never start duplicate run")
    S=output/"source";S.mkdir(parents=True)
    for relative,expected in source["files_sha256"].items():
        if relative=="config.json":continue
        p=old/relative
        if p.is_symlink() or not p.resolve().is_relative_to(old) or p.stat().st_mode&0o222:
            raise ValueError("Bootstrap file is writable or escaped")
        # Bootstrap sealed runner already verified bytes; native runner verifies copied manifest before execution.
        q=S/relative;q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q)
    cfg["source_snapshot"]=str(S.relative_to(ROOT));cfg["output"]=str((output/"run").relative_to(ROOT))
    (output/"run").mkdir()
    cfg.update(seed_mode="instance_points",discovery_seed_only=False,discovery_decode_prefix=False,instance_output_bridge=True)
    cfg.pop("discovery_track_ids",None)
    cfg["b0_evidence"]=ref(bootstrap/"run/B0_COMPLETE.json")
    cfg["b0_source_manifest"]=ref(old/"SNAPSHOT_MANIFEST.json")
    cfg["instance_seed_sources"]={}
    for seed in seeds["seeds"]:
        tid=seed["track_id"]
        cfg["instance_seed_sources"][tid]={"config":ref(old/"config.json"),
            "seeds":ref(bootstrap/"run/SEEDS.json"),"native_load":ref(bootstrap/"run/NATIVE_LOAD.json"),
            "binding":ref(bootstrap/f"run/{tid}/SEED_BINDING.json"),
            "raw_receipt":ref(bootstrap/f"run/{tid}/RAW_RECEIPT.json")}
    for key in ("source_root","source_receipt","asset_receipt"):
        cfg["native"][key]=cfg["native"][key].replace(str(old.relative_to(ROOT)),str(S.relative_to(ROOT)))
    (S/"config.json").write_text(json.dumps(cfg,indent=2)+"\n");(S/"config.json").chmod(0o444)
    files=dict(source["files_sha256"]);files["config.json"]=sha(S/"config.json")
    manifest={k:v for k,v in source.items() if k not in ("identity","files_sha256")}
    manifest["files_sha256"]=files
    manifest["identity"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    (S/"SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n");(S/"SNAPSHOT_MANIFEST.json").chmod(0o444)
    launcher=output/"run_frozen.py";launcher.write_bytes((bootstrap/"run_frozen.py").read_bytes());launcher.chmod(0o444)
    return cfg,manifest,launcher

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--bootstrap",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--dispatch",required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise ValueError("Wrong working directory")
    bootstrap=contained(args.bootstrap);output=contained(args.output);dispatch=contained(args.dispatch)
    try:
        # Existing bootstrap remains sole owner. Do not restart it on absent or slow receipts.
        for _ in range(120):
            if (bootstrap/"CONTROLLER_FAILED.json").exists():raise RuntimeError("Bootstrap failed; preserve and inspect")
            if (bootstrap/"CONTROLLER_COMPLETE.json").exists():break
            time.sleep(15)
        else:raise TimeoutError("Bootstrap has no completion receipt; no tracking launched")
        cfg,manifest,launcher=prepare(bootstrap,output)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg["resources"]["gpu_uuid"],CUBLAS_WORKSPACE_CONFIG=":4096:8",
                 OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
        cmd=["nice","-n","10","taskset","-c","88-93",str(ROOT/".venv/sam31/bin/python"),"-I",str(launcher),"sam"]
        with (output/"sam.log").open("xb") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            receipt={"pid":child.pid,"command":cmd,"source_identity":manifest["identity"],"time_unix":time.time(),
                     "frames":[0,2850],"B0_reused":True,"seed_masks_reused":True}
            (output/"LAUNCH.json").write_text(json.dumps(receipt,indent=2)+"\n")
            (dispatch/"CHILD.json").write_text(json.dumps(receipt,indent=2)+"\n")
            code=child.wait()
        if code:raise RuntimeError("Full SAM tracking exited "+str(code)+"; no automatic restart")
        (dispatch/"COMPLETE.json").write_text(json.dumps({"status":"NATIVE_PROCESS_EXITED_REVIEW_REQUIRED","output":str(output.relative_to(ROOT))})+"\n")
    except BaseException as exc:
        (dispatch/"FAILED.json").write_text(json.dumps({"error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n")
        raise

if __name__=="__main__":main()
