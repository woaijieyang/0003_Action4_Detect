"""Separate train-candidate selection from frozen worker deployment gates.

Standalone observer: never edits the running trainer, its BEST, or models/default.
"""
import argparse,fcntl,hashlib,json,math,os,time
from pathlib import Path
ROOT=Path("/data/zy/vln/a-test")
SCHEMA="action4.pose_projection_all8_002.v1"
POLICY="action4.training_candidate_all8_separate_deployment.v1"

def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def inside(root,value,exists=True):
    p=Path(value);p=p if p.is_absolute() else root/p;p=p.resolve(strict=exists)
    if not p.is_relative_to(root):raise ValueError("Path outside project")
    return p

def atomic(path,value):
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    with tmp.open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def rank(report,step):
    if report.get("score_schema")!=SCHEMA:raise ValueError("Wrong score schema")
    n=report["pose_gt_count"];tp=report["pose_all8_at_002_success"];published=report["pose_output_count"]
    if any(type(v) is not int or v<0 for v in (n,tp,published)) or tp>min(n,published):raise ValueError("Invalid pose counts")
    if (report["pose_fn"],report["pose_fp"])!=(n-tp,published-tp):raise ValueError("Failure denominators differ")
    value=report["pose_all8_at_002_rate"];p95=report["pose_corner_p95_valid_matched_only"]
    expected=tp/n if n else None
    if value!=expected or (p95 is not None and (not math.isfinite(p95) or p95<0)):raise ValueError("Invalid metric value")
    return (value if value is not None else -1.,-(report["pose_fn"]+report["pose_fp"]),-p95 if p95 is not None else -float("inf"),-step)

class CandidateObserver:
    def __init__(self,root,run,output,config):
        self.root=Path(root).resolve();self.run=inside(self.root,run);self.out=inside(self.root,output,False)
        self.cfg=config;self.config_id=canonical(config)
        if config["role"]!="A0":raise ValueError("A0 candidate observer only")
        self.out.mkdir(parents=True,exist_ok=True)
        for folder in ("pending","selected","evaluated"):(self.out/folder).mkdir(exist_ok=True)
        self.identity={"policy":POLICY,"run":str(self.run.relative_to(self.root)),"config_id":self.config_id,
            "score_schema":SCHEMA,"deployment_guards_changed":False,"default_weights_changed":False}
        ip=self.out/"OBSERVER_IDENTITY.json"
        if ip.exists() and json.loads(ip.read_text())!=self.identity:raise ValueError("Observer identity changed")
        if not ip.exists():atomic(ip,self.identity)
        self.records={}
        for p in (self.out/"evaluated").glob("step_*.json"):
            v=json.loads(p.read_text())
            if v["config_id"]!=self.config_id:raise ValueError("Mixed evaluation identity")
            self.records[v["step"]]=v
        bp=self.out/"TRAINING_BEST.json"
        self.best=json.loads(bp.read_text()) if bp.exists() else None
        if self.best is not None:
            if self.best["config_id"]!=self.config_id:raise ValueError("Mixed candidate pointer")
            checkpoint=inside(self.root,self.best["checkpoint"])
            with checkpoint.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest()!=self.best["checkpoint_sha256"]:
                    raise ValueError("Retained training candidate bytes changed")
    def sweep(self):
        requests=[]
        # Same lock as immutable eval worker: pin inode before it deletes its alias.
        with (self.run/"selection.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for q in sorted((self.run/"eval").glob("step_*.json")):
                if q.name.count(".")!=1:continue
                step=int(q.stem.split("_")[1])
                if step in self.records:continue
                req=json.loads(q.read_text())
                if req["role"]!="A0" or req["config_id"]!=self.config_id or req["step"]!=step:raise ValueError("Foreign request")
                src=inside(self.root,req["snapshot"],False)
                if src.parent!=self.run:raise ValueError("Unexpected snapshot directory")
                if not src.exists():
                    src=self.run/f"selected_step_{step:08d}.pt"
                if not src.exists() and self.best is not None and self.best["step"]==step:
                    src=inside(self.root,self.best["checkpoint"])
                target=self.out/"pending"/f"step_{step:08d}.pt"
                if src.exists() and not target.exists():os.link(src,target)
                requests.append((step,q,target))
        changed=False
        for step,q,target in requests:
            done=q.with_suffix(".done.json")
            if not done.exists():continue
            raw=done.read_bytes();result=json.loads(raw);report=result["report"]
            if (result["step"],result["role"],result["config_id"])!=(step,"A0",self.config_id):raise ValueError("Foreign completed evaluation")
            if report.get("evaluation_images")!=self.cfg["expected_eval_images"]:raise ValueError("Not full development evaluation")
            if type(result.get("deployment_eligible")) is not bool:raise ValueError("Explicit deployment gate required")
            rank(report,step)
            fields=("score_schema","pose_gt_count","pose_output_count","pose_all8_at_002_success",
                "pose_all8_at_002_rate","pose_fn","pose_fp","pose_corner_p95_valid_matched_only",
                "pose_all8_at_002_precision","pose_instance_precision","pose_groups")
            record={"step":step,"config_id":self.config_id,"metrics":{k:report[k] for k in fields},
                "source_done":{"path":str(done.relative_to(self.root)),"sha256":hashlib.sha256(raw).hexdigest()},
                "deployment_guard_passed":result["deployment_eligible"],"weight_available_at_processing":target.exists()}
            if target.exists() and (self.best is None or rank(record["metrics"],step)>rank(self.best["metrics"],self.best["step"])):
                selected=self.out/"selected"/f"step_{step:08d}.pt"
                if not selected.exists():os.link(target,selected)
                if not os.path.samefile(target,selected):raise ValueError("Candidate inode collision")
                with selected.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
                self.best={**record,"checkpoint":str(selected.relative_to(self.root)),"checkpoint_sha256":sha,
                    "policy":POLICY,"purpose":"training_candidate_not_deployment_release","sealed":False,
                    "publication_default_changed":False,"selected_unix":time.time()}
                atomic(self.out/"TRAINING_BEST.json",self.best)
            atomic(self.out/"evaluated"/f"step_{step:08d}.json",record)
            self.records[step]=record;changed=True
            if target.exists():target.unlink() # own staging hardlink only; winners remain selected
        if changed:
            global_best=max(self.records.values(),key=lambda z:rank(z["metrics"],z["step"]))
            missing=[v["step"] for v in self.records.values() if not v["weight_available_at_processing"]]
            atomic(self.out/"STATUS.json",{"status":"OBSERVING","policy":POLICY,"evaluated_reports":len(self.records),
                "training_best_step":None if self.best is None else self.best["step"],
                "training_best_all8":None if self.best is None else self.best["metrics"]["pose_all8_at_002_success"],
                "training_best_deployment_guard_passed":None if self.best is None else self.best["deployment_guard_passed"],
                "historical_metric_best_step":global_best["step"],"historical_metric_best_all8":global_best["metrics"]["pose_all8_at_002_success"],
                "retained_best_matches_historical_rank":self.best is not None and rank(self.best["metrics"],self.best["step"])==rank(global_best["metrics"],global_best["step"]),
                "historical_weights_unavailable":missing,"pending_weights":len(list((self.out/"pending").glob("*.pt"))),
                "selected_retention":"improving candidates held during active run; final reference audit keeps required best only",
                "existing_deployment_BEST_untouched":True,"time_unix":time.time()})
        return changed
    def complete_if_terminal(self):
        terminal=self.run/"CONTROLLER_COMPLETE.json"
        if not terminal.exists():return False
        controller=json.loads(terminal.read_text())
        if (controller.get("config_id"),controller.get("role"),controller.get("mode"))!=(self.config_id,"A0",self.cfg["mode"]):
            raise ValueError("Terminal controller identity mismatch")
        evaluation=json.loads((self.run/"EVAL_COMPLETE.json").read_text())
        training=json.loads((self.run/"TRAIN_COMPLETE.json").read_text())
        actual=training["step"]
        from model.training.action4_eval_schedule import expected_eval_steps,read_schedule_ledger
        ledger=read_schedule_ledger(self.cfg,self.run,training)
        expected=expected_eval_steps(self.cfg,training,ledger)
        if (actual<=0 or actual+training.get("no_supervision_update_count",0)!=self.cfg["epochs"]*math.ceil(self.cfg["expected_units"]/self.cfg["batch"])
                or evaluation.get("config_id")!=self.config_id
                or evaluation.get("attempt_id")!=controller.get("attempt_id") or training.get("attempt_id")!=controller.get("attempt_id")
                or set(evaluation["all_steps"])!=expected or evaluation["evaluations"]!=len(expected)
                or set(self.records)!=expected):
            raise ValueError("Terminal training/evaluation coverage mismatch")
        requests=[p for p in (self.run/"eval").glob("step_*.json") if p.name.count(".")==1]
        if any(not p.with_suffix(".done.json").exists() for p in requests):return False
        if len(self.records)!=len(requests):return False
        if list((self.run/"eval").glob("*.failed.json")):raise ValueError("Terminal run has failed eval")
        if self.best is None:raise ValueError("No loadable training candidate")
        self.best=dict(self.best,sealed=True);atomic(self.out/"TRAINING_BEST.json",self.best)
        atomic(self.out/"COMPLETE.json",{"status":"CANDIDATE_SELECTION_COMPLETE_PENDING_FINAL_SEAL_AND_RETENTION_AUDIT",
            "training_best":self.best,"evaluated_reports":len(self.records),"deployment_release":False})
        return True

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-spec",required=True);p.add_argument("--run-spec-sha256",required=True)
    p.add_argument("--output",required=True);p.add_argument("--source-sha256",required=True);args=p.parse_args()
    if Path.cwd().resolve()!=ROOT or os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Project CPU observer required")
    if len(os.sched_getaffinity(0))!=1 or not set(os.sched_getaffinity(0))<=set(range(24,96)):raise ValueError("One authorized CPU required")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=args.source_sha256:raise ValueError("Observer source changed")
    sp=inside(ROOT,args.run_spec);raw=sp.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.run_spec_sha256:raise ValueError("RunSpec changed")
    spec=json.loads(raw);cp=inside(ROOT,spec["config"])
    rel=str(cp.relative_to(inside(ROOT,spec["code_root"])))
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=spec["files"][rel]:raise ValueError("Run config changed")
    cfg=json.loads(cp.read_text());observer=CandidateObserver(ROOT,cfg["output"],args.output,cfg)
    with (observer.out/"observer.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(observer.out/"PROCESS.json",{"pid":os.getpid(),"source_sha256":args.source_sha256,"run_spec":args.run_spec,
            "CPU":sorted(os.sched_getaffinity(0)),"GPU":False})
        try:
            while True:
                observer.sweep()
                if observer.complete_if_terminal():break
                time.sleep(2)
        except BaseException as error:
            import traceback
            atomic(observer.out/"FAILURE.json",{"status":"CANDIDATE_OBSERVER_FAILED","error":str(error),"traceback":traceback.format_exc()})
            raise
if __name__=="__main__":main()
