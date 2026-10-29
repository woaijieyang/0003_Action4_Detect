"""Bounded read-only keyframe telemetry replay; no model training or publication."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path("/data/zy/vln/a-test")

def run():
 out=ROOT/"outputs/action4_m3t_keyframe_audit_v001"
 if not (out/"BUILD_COMPLETE.json").is_file():raise RuntimeError("Native build incomplete")
 if (out/"REPLAY_STARTED.json").exists():raise FileExistsError("Existing replay must not be restarted")
 parent=ROOT/"outputs/action4_m3t_sam80_continuous_v001"
 source=out/"source"
 for directory,dirs,files in os.walk(parent/"source",followlinks=False):
  if any((Path(directory)/n).is_symlink() for n in dirs+files):raise ValueError("Source symlink")
 shutil.copytree(parent/"source",source)
 copied_clip=source/"src/eval/action4_native_clip.py"
 copied_clip.chmod(0o644)
 shutil.copyfile(ROOT/"src/eval/action4_native_clip.py",copied_clip)
 copied_clip.chmod(0o444)
 identities={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
  for p in source.rglob("*") if p.is_file()}
 (out/"REPLAY_SOURCE.json").write_text(json.dumps(identities,indent=2))
 native=ROOT/"runtime/m3t_native";prefix=ROOT/"runtime/m3t_keyframe_audit_v001"
 cache=out/"cache"
 for name in ["home","tmp","config","mesa"]:(cache/name).mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(prefix)+":"+str(native/"lib/python3.11/site-packages"),
  PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
  LP_NUM_THREADS="2",LIBGL_ALWAYS_SOFTWARE="1",GALLIUM_DRIVER="llvmpipe",
  LD_LIBRARY_PATH=str(prefix/"lib")+":"+str(ROOT/"runtime/m3t_osmesa/lib")+":"+str(native/"lib"),
  HOME=str(cache/"home"),TMPDIR=str(cache/"tmp"),XDG_CACHE_HOME=str(cache),XDG_CONFIG_HOME=str(cache/"config"),
  MESA_SHADER_CACHE_DIR=str(cache/"mesa"))
 (out/"REPLAY_STARTED.json").write_text(json.dumps(dict(pid=os.getpid(),cpu=sorted(os.sched_getaffinity(0)),time=time.time(),changes="read-only native getters only",published=False)))
 def invoke(name,parent_name):
  target=out/name;target.mkdir()
  for filename in ["INPUT_MANIFEST.json","SUPPORT_INPUTS.json","MODEL_COMPLETE.json","track_00_body.yaml","track_01_body.yaml"]:
   shutil.copyfile(parent/parent_name/filename,target/filename)
  with (target/"RUN.log").open("x") as log:
   process=subprocess.Popen([str(native/"bin/python"),"-B",str(source/"track.py"),"--output",str(target.relative_to(ROOT))],
    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   (target/"PROCESS.json").write_text(json.dumps(dict(pid=process.pid,time=time.time())))
   rc=process.wait()
  if rc:raise RuntimeError(name+" native exit "+str(rc))
  rows=[json.loads(x) for x in (target/"POSES.jsonl").read_text().splitlines()]
  reference=[json.loads(x) for x in (parent/parent_name/"POSES.jsonl").read_text().splitlines()]
  if [(x["frame_id"],x["track_id"]) for x in rows]!=[(x["frame_id"],x["track_id"]) for x in reference]:
   raise RuntimeError("Replay identity mismatch")
  delta=max(float(np.max(np.abs(np.asarray(a["native_pose"])-np.asarray(b["native_pose"])))) for a,b in zip(rows,reference))
  events=json.loads((target/"FRAME_EVENTS.json").read_text())
  if not all(all(x["available"] for x in e["keyframe_state_after"]) for e in events):
   raise RuntimeError("Missing actual native telemetry")
  receipt=dict(records=len(rows),maximum_pose_element_difference=delta,numerical_parity=(delta==0.0),accuracy_accepted=False)
  (target/"PARITY.json").write_text(json.dumps(receipt,indent=2))
  return receipt
 try:
  short=invoke("parity3","parity_sparse_debug")
  if not short["numerical_parity"]:raise RuntimeError("Telemetry changed actual pose output; stop before continuous replay")
  continuous=invoke("continuous352","continuous_prior")
  (out/"REPLAY_COMPLETE.json").write_text(json.dumps(dict(short=short,continuous=continuous,status="DIAGNOSTIC_REPLAY_COMPLETE_NOT_TRACKING_ACCEPTANCE"),indent=2))
 except Exception as e:
  (out/"REPLAY_FAILED.json").write_text(json.dumps(dict(error=repr(e),time=time.time())))
  raise
