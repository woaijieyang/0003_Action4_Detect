"""Scoped independent-role controller; dynamic RGB eval workers and explicit terminal checks."""
import json,os,uuid,time,fcntl,subprocess,signal,traceback,math
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json,canonical_hash
from eval.action4_stage2_eval import rank_result
from model.training.action4_stage2_core import validate_run_gate
from model.training.action4_best_only_retention import finalize_best_only

def finalize_evaluations(cfg,out,attempt,terminal):
    """Validate completed evaluation domain and publish truthful terminal selection."""
    from model.training.action4_async_recovery import CORNER_VARIANT,corner_result_signature,checked_binding
    from eval.action4_stage2_eval import _corner_identity
    total=cfg["epochs"]*math.ceil(cfg["expected_units"]/cfg["batch"])
    if terminal["step"]+terminal.get("no_supervision_update_count",0)!=total or terminal["attempt_id"]!=attempt:
        raise ValueError("Training completion mismatch")
    requests=[p for p in (out/"eval").glob("step_*.json") if p.name.count(".")==1]
    reports=[json.loads(p.with_suffix(".done.json").read_text()) for p in requests]
    from model.training.action4_eval_schedule import expected_eval_steps,read_schedule_ledger
    ledger=read_schedule_ledger(cfg,out,terminal)
    actual=terminal["step"];expected=expected_eval_steps(cfg,terminal,ledger)
    if {r["step"] for r in reports}!=expected or list((out/"eval").glob("*.failed.json")):
        raise ValueError("Incomplete optimizer-step evaluations")
    corner=cfg.get("a0_variant")==CORNER_VARIANT
    if corner:
        signature=_corner_identity(cfg)
        if len(reports)!=len(expected):raise ValueError("Duplicate corner evaluation completion")
        if not (out/"CORNER_EVAL_IDENTITY.json").exists() or json.loads((out/"CORNER_EVAL_IDENTITY.json").read_text())!=signature:
            raise ValueError("Corner evaluation ledger/config mismatch")
        for request,result in zip(requests,reports):
            req=json.loads(request.read_text())
            if (req.get("step"),req.get("role"),req.get("config_id"))!=(result["step"],"A0",canonical_hash(cfg)):
                raise ValueError("Mixed corner completion request/config")
            if corner_result_signature(result)!=signature:raise ValueError("Mixed corner completion source/config")
            contained_path(R,result["evaluated_snapshot"]["path"],exists=True)
        eligible_reports=[r for r in reports if r["deployment_eligible"]]
        if not eligible_reports:
            if (out/"BEST.json").exists() or (out/"best.pt").exists() or list(out.glob("selected_step_*.pt")):
                raise ValueError("Unadmitted corner run has selected checkpoint artifacts")
            fallback=signature["parent_stage_fallback"]
            checked_binding({"path":fallback["path"],"sha256":fallback["sha256"]})
            no_best={"status":"NO_DEPLOYABLE_CHECKPOINT","attempt_id":attempt,"config_id":canonical_hash(cfg),
                "best_step":None,"evaluations":len(reports),"all_steps":sorted(expected),
                "parent_stage_fallback":fallback,"accuracy_accepted":False,
                "rejected_evaluated_snapshots":[r["evaluated_snapshot"] for r in sorted(reports,key=lambda r:r["step"])],
                "reason":"No corner checkpoint passed BOTH whole-checkpoint quality and original pose protocol guard"}
            atomic_json(out/"NO_DEPLOYABLE_CHECKPOINT.json",no_best)
            best_step=None;status="NO_DEPLOYABLE_CHECKPOINT"
        else:
            winner=max(eligible_reports,key=rank_result)
            best=json.loads((out/"BEST.json").read_text())
            if (corner_result_signature(best)!=signature or best["deployment_eligible"] is not True
                or best["step"]!=winner["step"] or rank_result(best)!=rank_result(winner)):
                raise ValueError("Corner BEST differs from eligible original-protocol winner")
            selected=checked_binding({"path":best["checkpoint"],"sha256":best["evaluated_snapshot"]["sha256"]})
            if not os.path.samefile(selected,out/"best.pt"):raise ValueError("Corner best alias changed")
            best_step=best["step"];status="COMPLETE_WITH_DEPLOYABLE_CHECKPOINT"
        evaluation={"attempt_id":attempt,"evaluations":len(reports),"all_steps":sorted(expected),"best_step":best_step,
            "config_id":canonical_hash(cfg),"status":status,"quality_gate_passed_evaluations":sum(r["report"]["quality_gate_passed_for_selection"] for r in reports),
            "deployment_eligible_evaluations":len(eligible_reports),"rejected_evaluations":len(reports)-len(eligible_reports),
            "parent_stage_fallback":signature["parent_stage_fallback"]}
        research=None
        if cfg.get("roi_finetune") is not None:
            from .action4_corner_research_selection import verify_complete
            research=verify_complete(out,reports)
            evaluation["pnp_research_best"]=research
        atomic_json(out/"EVAL_COMPLETE.json",evaluation)
        # Keep snapshot/recovery dependencies in both integration and future formal runs.
        finalize_best_only(str(out),remove_recovery=False)
        controller={"attempt_id":attempt,"mode":cfg["mode"],"role":cfg["role"],"best_step":best_step,
            "accuracy_accepted":False,"config_id":canonical_hash(cfg),"status":status,
            "parent_stage_fallback":signature["parent_stage_fallback"]}
    else:
        best=json.loads((out/"BEST.json").read_text());winner=max(reports,key=rank_result)
        if best["step"]!=winner["step"]:raise ValueError("Best checkpoint does not match all completed evaluations")
        atomic_json(out/"EVAL_COMPLETE.json",{"attempt_id":attempt,"evaluations":len(reports),"all_steps":sorted(expected),"best_step":best["step"],"config_id":canonical_hash(cfg)})
        if cfg["mode"]=="full_training":finalize_best_only(str(out),remove_recovery=True)
        controller={"attempt_id":attempt,"mode":cfg["mode"],"role":cfg["role"],"best_step":best["step"],"accuracy_accepted":False,"config_id":canonical_hash(cfg)}
    if corner and research is not None:controller["pnp_research_best"]=research
    atomic_json(out/"CONTROLLER_COMPLETE.json",controller)
    return controller


def run(config):
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    out=contained_path(R,cfg["output"],exists=True)
    from model.training.action4_run_snapshot import verify_running_snapshot,child_pythonpath
    verify_running_snapshot(cfg)
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"DISPATCH.json").exists():raise FileExistsError("No duplicate stage launch")
    validate_run_gate(cfg,json.loads(contained_path(R,cfg["integration_or_admission_receipt"],exists=True).read_text()))
    from model.training.action4_resource_policy import validate_device_policy,process_cpu_affinity,evaluation_threads
    validate_device_policy(cfg,R)
    train_gpus=cfg["training_gpus"];eval_gpus=cfg["eval_gpus"]
    if len(train_gpus)!=cfg["world_size"] or len(set(train_gpus))!=len(train_gpus):raise ValueError("Train device topology mismatch")
    if any(g not in range(0,8) for g in train_gpus+eval_gpus) or 1 in train_gpus or eval_gpus.count(1)>8:
        raise ValueError("Authorized GPU scope violated")
    cpu=cfg["world_size"]*(cfg.get("threads",2)+cfg.get("loader_workers",0))+(evaluation_threads(cfg)+cfg.get("eval_loader_workers",4))*len(eval_gpus)+1
    if cpu>cfg["reserved_cpu_budget"] or cfg["project_total_reserved_cpu"]>72:raise ValueError("Project CPU budget exceeded")
    uuid_by_gpu={}
    for g in set(train_gpus+eval_gpus):
        fields=subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        need=cfg.get("min_free_mib",12000)
        if cfg.get("training_allocator_budget_mib") is not None:
            need=cfg.get("uncommitted_gpu_margin_mib",8192)+(cfg["training_allocator_budget_mib"] if g in train_gpus else 0)+eval_gpus.count(g)*cfg["evaluation_allocator_budget_mib"]
        if int(fields[1])<need:raise RuntimeError("Insufficient available GPU memory")
        uuid_by_gpu[g]=fields[0].strip()
    attempt=uuid.uuid4().hex
    env=dict(os.environ,PYTHONPATH=child_pythonpath(),TMPDIR=str(R/"runtime"),
        PYTHONDONTWRITEBYTECODE="1",ACTION4_ATTEMPT_ID=attempt,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",NCCL_P2P_DISABLE="1")
    specs=[("train",[str(R/".venv/bin/python"),"-m","torch.distributed.run","--standalone","--nnodes=1",
        "--nproc_per_node="+str(cfg["world_size"]),"--max_restarts=0","--module","model.training.action4_stage2_train","--config",config],train_gpus)]
    if cfg.get("scoring_repair") is not None:
        if cfg.get("integration_resume_step") is not None:raise ValueError("Repair and engineering stop cannot be combined")
        specs[0][1].extend(["--resume",cfg["scoring_repair"]["checkpoint"]["path"]])
    if cfg.get("integration_resume_step") is not None:
        if cfg["mode"]!="integration":raise ValueError("Automatic resume probe is engineering-only")
        specs[0][1].extend(["--stop-after",str(cfg["integration_resume_step"])])
    for i,g in enumerate(eval_gpus):
        specs.append((f"eval_{i}",[str(R/".venv/bin/python"),"-m","eval.action4_stage2_eval","--config",config,"--worker-id",str(i)],[g]))
    processes=[];workers=[]
    try:
        for name,cmd,gpus in specs:
            if name=="eval_0" and cfg.get("wait_for_training_ready_before_eval",False):
                from model.training.action4_startup_readiness import wait_training_ready
                ready=wait_training_ready(out,attempt,processes[0],cfg.get("training_startup_timeout_s",1800.))
                atomic_json(out/"EVALUATOR_STARTUP_READY.json",ready)
            cpus=process_cpu_affinity(cfg,name,R)
            if cpus is not None:cmd[:]=["taskset","-c",",".join(map(str,cpus))]+cmd
            with (out/(name+".log")).open("x") as log:
                p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(uuid_by_gpu[g] for g in gpus)),stdout=log,stderr=log,start_new_session=True)
            processes.append(p);workers.append({"role":name,"pid":p.pid,"gpus":gpus,"command":cmd})
        atomic_json(out/"DISPATCH.json",{"attempt_id":attempt,"workers":workers,"cpu_budget":cpu,"mode":cfg["mode"]})
        resumed=False
        while any(p.poll() is None for p in processes):
            if any(p.poll() is not None and p.returncode!=0 for p in processes):raise RuntimeError("Owned stage worker failed")
            if cfg.get("integration_resume_step") is not None and not resumed and processes[0].poll()==0:
                stop=json.loads((out/"CHECKPOINTED_STOP.json").read_text())
                if stop["step"]!=cfg["integration_resume_step"] or stop["attempt_id"]!=attempt:raise ValueError("Controlled stop identity mismatch")
                cmd=specs[0][1][:-2]+["--resume",str((out/"last.pt").relative_to(R))]
                with (out/"train_resume.log").open("x") as log:
                    p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(uuid_by_gpu[g] for g in train_gpus)),stdout=log,stderr=log,start_new_session=True)
                processes.append(p);resumed=True
                atomic_json(out/"RESUME_DISPATCH.json",{"pid":p.pid,"command":cmd,"attempt_id":attempt,"from_step":stop["step"],"scope":"explicit controlled integration recovery"})
            time.sleep(5)
        terminal=json.loads((out/"TRAIN_COMPLETE.json").read_text())
        finalize_evaluations(cfg,out,attempt,terminal)
    except BaseException:
        atomic_json(out/"CONTROLLER_FAILURE.json",{"attempt_id":attempt,"traceback":traceback.format_exc()})
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);run(p.parse_args().config)
