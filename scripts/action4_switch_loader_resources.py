"""Dry-run-default, explicit-approved A0 resource switch coordinator.

Only --execute with exact gate and human-approval receipt can signal/launch.
Original controller/train/eval/observer processes remain STOP-preserved until
the successor has >=S+10 and a real successful full650 suffix evaluation.
"""
import argparse, contextlib, fcntl, hashlib, json, math, os, signal, shutil, subprocess, sys, time, traceback
from pathlib import Path

R=Path("/data/zy/vln/a-test")
W=R/"outputs/action4_a0_resource_continuation_v001"
D=W/"control_preparation"
OLD_SPEC=R/"runtime/action4_runs/p2_a0_parallel80_eval10_v001/RUN_SPEC.json"
OLD=R/"outputs/action4_improvement/stage1_baseline/p2_a0_parallel80_eval10_v001"
OLD_OBS=R/"outputs/action4_training_candidate_selection_v001"
VARIANT="source_variant_v002"
RUN_ID="p2_a0_loader12_eval8_v001"
NEW_OBS=R/"outputs/action4_training_candidate_selection_loader12_eval8_v001"
COMPOSER=R/"scripts/action4_build_resource_continuation.py"
GATE=D/"RESOURCE_EXECUTION_PASS_V002.json"
OLD_PIDS={"controller":3357320,"torchrun":3357371,"trainer":3357408,
          "eval_0":3357372,"eval_1":3357373,"eval_2":3357374,"eval_3":3357375,"observer":3759830}

def path(value,exists=True):
 p=Path(value);p=p if p.is_absolute() else R/p
 if p.is_symlink() or ".." in p.parts or not p.resolve(strict=exists).is_relative_to(R):raise ValueError("Project path escape")
 return p.resolve(strict=exists)

def sha(p):
 with path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def ref(p):return {"path":str(path(p).relative_to(R)),"sha256":sha(p)}
def bound(ref_):
 if set(ref_)!={"path","sha256"} or sha(ref_["path"])!=ref_["sha256"]:raise ValueError("Bound reference changed")
 return path(ref_["path"])
def read(p):
 def reject(v):raise ValueError("Nonfinite JSON:"+v)
 return json.loads(path(p).read_text(),parse_constant=reject)
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def atomic(p,v):
 p=path(p,False);temp=p.with_name(p.name+".tmp")
 with temp.open("w") as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def finite(v):
 if isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,(list,tuple)):
  for x in v:finite(x)
 elif isinstance(v,float) and not math.isfinite(v):raise ValueError("Nonfinite runtime evidence")

def process(pid):
 result=subprocess.run(["ps","-p",str(pid),"-o","pid=,pgid=,lstart=,args="],capture_output=True,text=True)
 text=result.stdout.strip()
 if not text:return None
 pieces=text.split(None,7)
 if len(pieces)!=8:raise RuntimeError("Cannot obtain exact process identity")
 return {"pid":int(pieces[0]),"pgid":int(pieces[1]),"lstart":" ".join(pieces[2:7]),"args":pieces[7]}
def state(pid):
 return subprocess.run(["ps","-p",str(pid),"-o","stat="],capture_output=True,text=True).stdout.strip()
def alive(pid):return bool(state(pid)) and not state(pid).startswith("Z")
def members(pgid):
 p=subprocess.run(["pgrep","-g",str(pgid)],capture_output=True,text=True)
 if p.returncode not in (0,1):raise RuntimeError("Cannot inspect owned process group")
 return sorted(int(x) for x in p.stdout.split())
def children(pid):
 p=subprocess.run(["pgrep","-P",str(pid)],capture_output=True,text=True)
 if p.returncode not in (0,1):raise RuntimeError("Cannot inspect known children")
 return sorted(int(x) for x in p.stdout.split())
def descendants(pid):
 seen=set();todo=[pid]
 while todo:
  parent=todo.pop()
  for child in children(parent):
   if child not in seen:seen.add(child);todo.append(child)
 return sorted(seen)

class GuardedSignals:
 def __init__(self,recorder):self.record=recorder
 def verify(self,identity):
  actual=process(identity["pid"])
  if actual!=identity:raise RuntimeError("PID identity changed before signal:"+str(identity["pid"]))
 def single(self,identity,sig):
  self.verify(identity)
  self.record("signal",{"scope":"single_pid","identity":identity,"signal":signal.Signals(sig).name})
  os.kill(identity["pid"],sig)
 def group(self,identities,sig):
  if not identities:return
  pgid=identities[0]["pgid"]
  if any(x["pgid"]!=pgid for x in identities):raise ValueError("Mixed group identities")
  actual=members(pgid)
  if actual!=sorted(x["pid"] for x in identities):raise RuntimeError("Group membership changed before signal:"+str(pgid))
  for identity in identities:self.verify(identity)
  self.record("signal",{"scope":"owned_group","pgid":pgid,"members":identities,"signal":signal.Signals(sig).name})
  os.killpg(pgid,sig)
 def stopped(self,identities,seconds=10):
  end=time.monotonic()+seconds
  while True:
   for x in identities:self.verify(x)
   if all(state(x["pid"]).startswith("T") for x in identities):return
   if time.monotonic()>end:raise RuntimeError("Signal did not establish stopped state")
   time.sleep(.05)

def capture_old():
 identities={role:process(pid) for role,pid in OLD_PIDS.items()}
 if any(x is None for x in identities.values()):raise RuntimeError("Original process missing")
 if any(state(x["pid"]).startswith(("T","Z")) for x in identities.values()):raise RuntimeError("Original already stopped/exited; cannot start a new switch")
 for role,x in identities.items():
  if "p2_a0_parallel80_eval10_v001" not in x["args"]:raise RuntimeError("Old role command not bound to original run:"+role)
 groups={}
 for role in ("trainer","eval_0","eval_1","eval_2","eval_3"):
  lead=identities[role];items=[process(pid) for pid in members(lead["pgid"])]
  if any(x is None or "p2_a0_parallel80_eval10_v001" not in x["args"] for x in items):raise RuntimeError("Unowned old group member")
  groups[role]=items
 if {x["pid"] for x in groups["trainer"]}!={3357408,3357681,3357683,3357690,3357692}:
  raise RuntimeError("Original trainer/four-loader process set changed")
 return {"roles":identities,"groups":groups}

def resume_preserved(signals,old):
 # Return exact old processes; no DISPATCH reset or new old launcher.
 for role in ("eval_0","eval_1","eval_2","eval_3"):
  signals.group(old["groups"][role],signal.SIGCONT)
 signals.single(old["roles"]["observer"],signal.SIGCONT)
 signals.single(old["roles"]["torchrun"],signal.SIGCONT)
 signals.group(old["groups"]["trainer"],signal.SIGCONT)
 signals.single(old["roles"]["controller"],signal.SIGCONT)

def successful_suffix(row,cfg,step):
 finite(row)
 if (row.get("step"),row.get("role"),row.get("config_id"),row.get("training_mode"))!=(step,"A0",canonical(cfg),"full_training"):
  raise RuntimeError("First suffix evaluation identity invalid")
 report=row.get("report",{})
 images=report.get("bbox_only",{}).get("per_image",[])
 ids=[x.get("sample_id") for x in images]
 if (report.get("evaluation_images")!=650 or report.get("pose_gt_count")!=1300
     or report.get("score_schema")!=cfg.get("score_schema") or len(ids)!=650
     or len(set(ids))!=650 or any(not isinstance(x,str) or not x for x in ids)):
  raise RuntimeError("First suffix report is not unique full650/original1300GT")


def runtime_env():
 env=dict(os.environ,ACTION4_PROJECT_ROOT=str(R),PYTHONDONTWRITEBYTECODE="1",
          OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
          TMPDIR=str(R/"runtime/tmp/a0switch"))
 return env

def dry_plan(gate_ref):
 gate=read(bound(gate_ref));delta=read(W/VARIANT/"REVIEWED_DELTAS.json")
 if gate.get("status")!="STAGE1_EXECUTION_PASS" or gate["project_code_sha256"]!=delta["project_code_sha256"]:
  raise ValueError("Actual-math/metadata gate does not bind source variant2")
 for destination in (R/"runtime/action4_runs"/RUN_ID,R/"outputs/action4_improvement/stage1_baseline"/RUN_ID,NEW_OBS):
  if destination.exists():raise FileExistsError("Successor destination already exists:"+str(destination))
 old=capture_old()
 plan={"schema":"action4.loader_switch_plan.v1","old_run_spec":ref(OLD_SPEC),"old_processes":old,
       "new_run_id":RUN_ID,"new_observer_output":str(NEW_OBS.relative_to(R)),
       "source_variant":VARIANT,"source_spec":ref(W/VARIANT/"SOURCE_SPEC.json"),
       "reviewed_deltas":ref(W/VARIANT/"REVIEWED_DELTAS.json"),"gate":gate_ref,
       "composer":ref(COMPOSER),"frozen_actions":ref(D/"frozen_actions.py"),"switcher_sha256":sha(Path(__file__).resolve()),
       "strategy":"STOP preserve originals; validate successor S+10/full650; rollback samePIDs or controlled retirement",
       "observation_timeout_is_not_failure":True,"signals_sent":False,"GPU_jobs_started":False}
 return plan

class Switcher:
 def __init__(self,plan,approval_ref):
  self.plan=plan;self.old=plan["old_processes"];self.approval_ref=approval_ref
  self.data={"schema":"action4.loader_cutover.v1","phase":"APPROVAL_VERIFIED","events":[],
             "old_processes":self.old,"plan":plan,"approval":approval_ref,"possible_inflight_S_plus_1_discard":True}
  self.signals=GuardedSignals(self.event);self.old_paused=False;self.new_roots=[];self.new_handles=[]
  self.new_known={};self.commit_started=False;self.capture=None;self.S=None
 def save(self):atomic(D/"CUTOVER.json",self.data)
 def event(self,kind,value):
  self.data["events"].append({"unix":time.time(),"kind":kind,"details":value});self.save()
 def phase(self,name,**details):self.data.update(phase=name,**details);self.save()
 def verify_old(self):
  for x in self.old["roles"].values():self.signals.verify(x)
  for group in self.old["groups"].values():
   if members(group[0]["pgid"])!=sorted(x["pid"] for x in group):raise RuntimeError("Original group ownership changed")
   for x in group:self.signals.verify(x)
 def run_python(self,source,args,logname):
  command=[str(R/".venv/bin/python"),"-B",str(source)]+list(args)
  self.event("command",command)
  result=subprocess.run(command,cwd=R,env=runtime_env(),text=True,capture_output=True)
  (D/logname).write_text(result.stdout+"\nSTDERR\n"+result.stderr)
  if result.returncode:raise RuntimeError("Preparation command failed:"+logname)
  lines=[x for x in result.stdout.splitlines() if x.startswith("{")]
  return json.loads(lines[-1]) if lines else None
 def wait_boundary(self):
  import torch
  initial=read(OLD/"STATUS.json")["step"]
  self.old_paused=True
  self.signals.single(self.old["roles"]["controller"],signal.SIGSTOP)
  self.signals.stopped([self.old["roles"]["controller"]])
  self.phase("OLD_CONTROLLER_STOPPED_WAIT_FRESH_BOUNDARY",initial_step=initial)
  while True:
   self.verify_old()
   for failure in list(OLD.glob("*FAILURE*.json"))+list((OLD/"eval").glob("*.failed.json")):
    raise RuntimeError("Original runtime failure:"+str(failure))
   status=read(OLD/"STATUS.json");S=status["step"]
   if S>initial and S%10==0 and (OLD/"eval"/f"step_{S:08d}.json").exists():
    before=(OLD/"last.pt").stat()
    checkpoint=torch.load(OLD/"last.pt",map_location="cpu",weights_only=True,mmap=True)
    good=checkpoint.get("step")==S
    del checkpoint
    after=(OLD/"last.pt").stat()
    if good and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns):
     self.signals.group(self.old["groups"]["trainer"],signal.SIGSTOP)
     self.signals.stopped(self.old["groups"]["trainer"])
     fresh=read(OLD/"STATUS.json")
     checkpoint=torch.load(OLD/"last.pt",map_location="cpu",weights_only=True,mmap=True)
     stable=fresh["step"]==S and checkpoint.get("step")==S
     del checkpoint
     if not stable:
      self.event("boundary_race_retry",{"candidate_S":S,"actual_status":fresh["step"]})
      self.signals.group(self.old["groups"]["trainer"],signal.SIGCONT);initial=max(initial,S);continue
     self.signals.single(self.old["roles"]["torchrun"],signal.SIGSTOP)
     self.signals.stopped([self.old["roles"]["torchrun"]])
     capture_dir=D/f"capture_step_{S:08d}";capture_dir.mkdir(exist_ok=False)
     target=capture_dir/"PREDECESSOR_STATE.pt";shutil.copyfile(OLD/"last.pt",target);target.chmod(0o444)
     if target.stat().st_ino==(OLD/"last.pt").stat().st_ino or sha(target)!=sha(OLD/"last.pt"):raise RuntimeError("Boundary copy mismatch")
     self.capture=target;self.S=S
     self.phase("OLD_TRAIN_STOPPED_CAPTURED",boundary_step=S,captured_state=ref(target))
     return
   time.sleep(.2)
 def drain_and_pause(self):
  expected=set(range(0,self.S+1,10))
  self.phase("DRAINING_ORIGINAL_EVAL_AND_OBSERVER")
  while True:
   self.verify_old()
   failed=list((OLD/"eval").glob("*.failed.json"))
   if failed:raise RuntimeError("Original eval failure during drain:"+str(failed[0]))
   done={int(p.name[5:13]) for p in (OLD/"eval").glob("step_*.done.json")}
   records={int(p.name[5:13]) for p in (OLD_OBS/"evaluated").glob("step_*.json")}
   if expected<=done and expected<=records:
    lock=(OLD/"selection.lock").open("rb")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();time.sleep(.5);continue
    with lock:
     if read(OLD/"STATUS.json")["step"]!=self.S:raise RuntimeError("Paused original status changed")
     for role in ("eval_0","eval_1","eval_2","eval_3"):
      self.signals.group(self.old["groups"][role],signal.SIGSTOP);self.signals.stopped(self.old["groups"][role])
     self.signals.single(self.old["roles"]["observer"],signal.SIGSTOP)
     self.signals.stopped([self.old["roles"]["observer"]])
     # All selection writers are stopped while this lock is held.
    self.phase("ALL_ORIGINALS_STOPPED_PREFIX_STABLE")
    return
   time.sleep(1)
 def prepare_successor(self):
  for key in ("composer","frozen_actions","source_spec","reviewed_deltas","gate"):bound(self.plan[key])
  root=W/VARIANT/"source"
  script=D/"frozen_actions.py"
  prefix=self.run_python(script,["prefix",str(root),str(OLD_SPEC),sha(OLD_SPEC),str(self.capture),str(self.S),
                                str(OLD_OBS),str(D/f"prefix_step_{self.S:08d}")],"PREFIX_BUILD.log")
  self.data["prefix"]=prefix;self.save()
  self.run_python(COMPOSER,["--run-id",RUN_ID,"--purpose","training_continuation","--checkpoint",str(self.capture),
                   "--source-variant",VARIANT,"--formal-gate",self.plan["gate"]["path"]],"FORMAL_BUILD.log")
  out=R/"outputs/action4_improvement/stage1_baseline"/RUN_ID
  prepared=read(out/"PREPARATION.json");self.data["successor"]=prepared
  spec=read(prepared["run_spec"]);code=R/spec["code_root"]
  verified=self.run_python(script,["verify",str(code),prepared["receipt"]["path"],prepared["receipt"]["sha256"],prepared["config"]],
                           "FULL_STATE_VERIFY.log")
  if verified["step"]!=self.S:raise RuntimeError("Successor continuation boundary differs")
  self.phase("SUCCESSOR_FROZEN_AND_FULL_STATE_VERIFIED",verification=verified)
 def spawn(self,command,logname):
  self.event("launch_command",command)
  with (D/logname).open("x") as stream:
   p=subprocess.Popen(command,cwd=R,env=runtime_env(),stdout=stream,stderr=stream,start_new_session=True)
  self.new_handles.append(p)
  identity=None
  for _ in range(50):
   identity=process(p.pid)
   if identity:break
   time.sleep(.02)
  if identity is None:raise RuntimeError("New process identity unavailable")
  self.new_roots.append(identity);self.new_known[p.pid]=identity
  self.data["new_roots"]=self.new_roots;self.save()
  return identity
 def launch_successor(self):
  bound(self.plan["frozen_actions"]);bound(self.plan["gate"])
  prepared=self.data["successor"];prefix=self.data["prefix"]
  controller=["taskset","-c","87",str(R/".venv/bin/python"),"-B",str(R/prepared["launcher"]),
      "model.training.action4_stage2_dispatch","--continue-receipt",prepared["receipt"]["path"],
      "--continue-receipt-sha256",prepared["receipt"]["sha256"],"--prefix-ledger",prefix["path"],
      "--prefix-ledger-sha256",prefix["sha256"]]
  self.spawn(controller,"NEW_CONTROLLER.log")
  spec=read(prepared["run_spec"]);code=R/spec["code_root"]
  observer=["taskset","-c","94",str(R/".venv/bin/python"),"-B",str(D/"frozen_actions.py"),"observer",str(code),
     prepared["run_spec"],sha(prepared["run_spec"]),str(NEW_OBS.relative_to(R)),prefix["path"],prefix["sha256"]]
  self.spawn(observer,"NEW_OBSERVER.log")
  self.phase("SUCCESSOR_RUNNING_ORIGINALS_STOP_PRESERVED")
 def refresh_new(self):
  # Controller failure may orphan children before the next ancestry sample.
  # Authenticated successor-owned process receipts retain these launch PIDs.
  receipts=[]
  if "successor" in self.data:
   out=R/self.data["successor"]["output"]
   dispatch=out/"DISPATCH.json"
   if dispatch.exists():
    receipts.extend(w["pid"] for w in read(dispatch)["workers"])
   training=out/"TRAIN_PROCESS.json"
   if training.exists():receipts.extend(read(training)["pids"])
   for entry in out.glob("EVAL_PROCESS_*.json"):receipts.append(read(entry)["pid"])
  for pid in set(receipts):
   current=process(pid)
   if current is None or state(pid).startswith("Z"):continue
   if RUN_ID not in current["args"]:raise RuntimeError("Successor receipt PID no longer belongs to run")
   prior=self.new_known.get(pid)
   if prior and prior!=current:raise RuntimeError("Successor worker identity changed")
   self.new_known[pid]=current
   for child in descendants(pid):
    identity=process(child)
    if identity is not None:self.new_known[child]=identity
  for root in self.new_roots:
   current=process(root["pid"])
   if current is not None and not state(root["pid"]).startswith("Z"):
    # taskset/exec can change args once during launch; bind after execution.
    if current["lstart"]!=root["lstart"] or current["pgid"]!=root["pgid"]:raise RuntimeError("New root PID reused")
    self.new_known[root["pid"]]=current
    for pid in descendants(root["pid"]):
     identity=process(pid)
     if identity is not None:self.new_known[pid]=identity
  self.data["new_owned_processes"]=list(self.new_known.values());self.save()
 def monitor(self):
  prepared=self.data["successor"];out=R/prepared["output"];cfg=read(prepared["config"]);cid=canonical(cfg)
  last_notice=0
  while True:
   self.refresh_new();self.verify_old()
   if not all(state(x["pid"]).startswith("T") for x in self.old["roles"].values()):raise RuntimeError("Original unexpectedly resumed during successor validation")
   failures=list(out.glob("*FAILURE*.json"))+list((out/"eval").glob("*.failed.json"))
   if failures:raise RuntimeError("Actual successor failure:"+str(failures[0]))
   for handle in self.new_handles:
    if handle.poll() is not None:raise RuntimeError("Actual new owned process exited before cutover validation:"+str(handle.pid))
   accepted=out/"CONTINUATION_STATE_ACCEPTED.json";status_path=out/"STATUS.json"
   if accepted.exists() and status_path.exists():
    acceptance=read(accepted);status=read(status_path);finite(status)
    if acceptance["boundary"]["step"]!=self.S or acceptance["purpose"]!="training_continuation" or acceptance["successor"]["config_id"]!=cid:raise RuntimeError("New full-state acceptance identity differs")
    if status["step"]>=self.S+10:
     done=out/"eval"/f"step_{self.S+10:08d}.done.json"
     if done.exists():
      row=read(done);finite(row)
      successful_suffix(row,cfg,self.S+10)
      if not (NEW_OBS/"ONELINEAGE_TRAINING_BEST.json").exists() or not (NEW_OBS/"evaluated"/f"step_{self.S+10:08d}.json").exists():
       time.sleep(1);continue
      self.phase("SUCCESSOR_ACTUAL_S_PLUS_10_FULL650_ACCEPTED",accepted_step=status["step"],first_suffix_eval=ref(done))
      return
   if time.time()-last_notice>=30:
    self.event("observation",{"status":"WAITING_FOR_ACTUAL_SUCCESSOR_UPDATES_EVAL_NOT_FAILURE",
       "observed_step":read(status_path).get("step") if status_path.exists() else None});last_notice=time.time()
   time.sleep(1)
 def stop_new_only(self):
  if not self.new_roots and not self.new_handles:return
  self.refresh_new()
  # Freeze launch roots first, then known descendant groups; never select by GPU.
  for root in self.new_roots:
   identity=self.new_known.get(root["pid"])
   if identity and alive(identity["pid"]):self.signals.single(identity,signal.SIGSTOP)
  self.refresh_new()
  groups={}
  for identity in self.new_known.values():
   if alive(identity["pid"]):groups.setdefault(identity["pgid"],[]).append(process(identity["pid"]))
  for pgid,known in groups.items():
   current=[process(pid) for pid in members(pgid)]
   if any(x is None or x["pid"] not in self.new_known for x in current):raise RuntimeError("Unowned new group member during rollback")
   self.signals.group(current,signal.SIGSTOP)
  for pgid in list(groups):
   current=[process(pid) for pid in members(pgid) if alive(pid)]
   # Zombies remain in group membership; retain exact identity records for signal verification.
   all_current=[process(pid) for pid in members(pgid)]
   if all_current:
    if any(x is None or x["pid"] not in self.new_known for x in all_current):raise RuntimeError("New rollback ownership changed")
    self.signals.group(all_current,signal.SIGKILL)
  self.event("new_owned_stopped",{"preserved_failure_outputs":True})
 def retire_old(self):
  self.refresh_new();self.verify_old()
  out=R/self.data["successor"]["output"]
  if any(h.poll() is not None for h in self.new_handles) or list(out.glob("*FAILURE*.json")) or list((out/"eval").glob("*.failed.json")):
   raise RuntimeError("Successor failed at final retirement gate")
  self.commit_started=True;self.phase("COMMITTING_CONTROLLED_OLD_RETIREMENT")
  # SIGKILL the STOP-preserved controller prevents Python finally cleanup from
  # targeting process groups. No old TRAIN_COMPLETE is manufactured.
  self.signals.single(self.old["roles"]["controller"],signal.SIGKILL)
  self.signals.group(self.old["groups"]["trainer"],signal.SIGKILL)
  self.signals.single(self.old["roles"]["torchrun"],signal.SIGKILL)
  for role in ("eval_0","eval_1","eval_2","eval_3"):self.signals.group(self.old["groups"][role],signal.SIGKILL)
  self.signals.single(self.old["roles"]["observer"],signal.SIGKILL)
  atomic(D/"CONTROLLED_RETIREMENT.json",{"old_processes":self.old,"boundary_step":self.S,
     "captured_state":ref(self.capture),"prefix":self.data["prefix"],"successor":self.data["successor"],
     "reason":"validated resource continuation","TRAIN_COMPLETE_fabricated":False,
     "possible_inflight_S_plus_1_discard":True,"unix":time.time()})
  self.phase("CUTOVER_COMMITTED_SUCCESSOR_CONTINUES")
 def execute(self):
  self.save()
  try:
   self.verify_old();self.wait_boundary();self.drain_and_pause()
   self.prepare_successor();self.launch_successor();self.monitor();self.retire_old()
  except BaseException as error:
   self.event("error",{"type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
   if self.commit_started:
    self.phase("SUCCESSOR_RUNNING_RETIREMENT_NEEDS_ROOT_REVIEW")
   elif isinstance(error,(KeyboardInterrupt,SystemExit)):
    self.phase("OBSERVATION_INTERRUPTED_ROOT_REVIEW_REQUIRED_ORIGINALS_PRESERVED")
   else:
    try:
     self.stop_new_only()
     if self.old_paused:resume_preserved(self.signals,self.old)
     self.phase("ROLLED_BACK_SAME_ORIGINAL_PIDS_NO_DISPATCH_RESET")
    except BaseException as rollback_error:
     self.phase("ROLLBACK_REQUIRES_ROOT_ATTENTION",rollback_error=repr(rollback_error))
   raise

def main():
 if Path.cwd().resolve()!=R:raise ValueError("Expected authorized cwd")
 sys.dont_write_bytecode=True;D.mkdir(parents=True,exist_ok=True)
 parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
 parser.add_argument("--gate",default=str(GATE));parser.add_argument("--gate-sha256")
 parser.add_argument("--approval");parser.add_argument("--approval-sha256");a=parser.parse_args()
 gate_ref={"path":str(path(a.gate).relative_to(R)),"sha256":a.gate_sha256 or sha(a.gate)}
 plan=dry_plan(gate_ref)
 if not a.execute:
  atomic(D/"DRY_RUN.json",plan)
  atomic(D/"USER_APPROVAL_TEMPLATE.json",{"schema":"action4.loader_switch_approval.v1","approved":False,
         "user_instruction":"ROOT MUST record the actual latest explicit user authorization",
         "plan":plan})
  print(json.dumps(plan));return
 if not all((a.gate_sha256,a.approval,a.approval_sha256)):raise ValueError("Execute requires explicit gate SHA and approval path/SHA")
 approval_ref={"path":str(path(a.approval).relative_to(R)),"sha256":a.approval_sha256}
 approval=read(bound(approval_ref))
 if approval.get("schema")!="action4.loader_switch_approval.v1" or approval.get("approved") is not True or approval.get("plan")!=plan:
  raise ValueError("Explicit user approval is absent or no longer matches exact processes/source/gate")
 if not isinstance(approval.get("user_instruction"),str) or not approval["user_instruction"].strip():
  raise ValueError("Actual user authorization text required")
 if (D/"CUTOVER.json").exists():raise FileExistsError("Existing cutover history: root must review before another attempt")
 (R/"runtime/tmp/a0switch").mkdir(parents=True,exist_ok=True)
 with (D/"switcher.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  Switcher(plan,approval_ref).execute()

if __name__=="__main__":main()
