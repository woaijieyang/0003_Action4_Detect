"""Sequential v002 continuation from completed v001 best; no live mutation."""
import argparse,json,os,time,hashlib,fcntl,subprocess,uuid,shutil,traceback
from pathlib import Path
import torch
from model.training.action4_runtime import atomic_json,canonical_hash
R=Path("/data/zy/vln/a-test")

def run(declaration,resume=None,retry_failed=False):
    assert Path.cwd().resolve()==R
    cfg=json.loads((R/declaration).read_text());out=R/cfg["output"]
    lock=(out/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resolved=R/"configs/model/action4_gdr_pose_v002_resolved.json"
    if (out/"DISPATCH.json").exists() and not resume:raise FileExistsError("Already dispatched; no automatic repeat")
    if retry_failed and not resume:raise ValueError("retry-failed requires explicit same-run resume")
    if not resume:
        parent=R/cfg.pop("wait_for_parent")
        while not (parent/"CONTROLLER_COMPLETE.json").exists():
            if (parent/"CONTROLLER_FAILURE.json").exists() or (parent/"TRAIN_FAILURE.json").exists():
                raise RuntimeError("Parent failed; do not auto restart")
            atomic_json(out/"CONTROLLER_STATUS.json",{"state":"WAIT_PARENT_COMPLETE","parent":str(parent)})
            time.sleep(30)
        complete=json.loads((parent/"TRAIN_COMPLETE.json").read_text())
        if complete["epochs"]!=80:raise ValueError("Expected completed80epoch parent")
        if any(json.loads((parent/"CONTROLLER_COMPLETE.json").read_text())["codes"]):raise ValueError("Parent exit failure")
        selected=out/"selected_parent.pt"
        if selected.exists():raise FileExistsError("Parent already selected; inspect interrupted setup")
        os.link(parent/"best.pt",selected)
        state=torch.load(selected,map_location="cpu",weights_only=True)
        receipt=json.loads((parent/"eval"/f"step_{state['step']:08d}.done.json").read_text())
        if receipt["step"]!=state["step"]:raise ValueError("Parent receipt mismatch")
        parent_cfg=json.loads((R/"configs/model/action4_gdr_pose_v001.json").read_text())
        if state["config_id"]!=canonical_hash(parent_cfg):raise ValueError("Parent config mismatch")
        cfg.update(joint_best=str(selected.relative_to(R)),parent_step=state["step"],parent_config_id=state["config_id"],
                   parent_weight_sha256=hashlib.sha256(selected.read_bytes()).hexdigest())
        atomic_json(out/"PARENT_SELECTION.json",{"step":state["step"],"config_id":state["config_id"],"sha256":cfg["parent_weight_sha256"],"receipt":receipt})
        if resolved.exists():raise FileExistsError("Resolved config already exists")
        atomic_json(resolved,cfg)
    else:
        cfg=json.loads(resolved.read_text())
        state=torch.load(R/resume,map_location="cpu",weights_only=True)
        if state["config_id"]!=canonical_hash(cfg) or not state.get("resumable",False):raise ValueError("Not same-run recovery state")
    for f in ("loss_regression.log","recovery_regression.log"):
        assert json.loads((out/f).read_text().splitlines()[-1])["passed"]
    attempt=uuid.uuid4().hex;history=out/"attempts"/attempt;history.mkdir(parents=True)
    queue=out/"eval";queue.mkdir(exist_ok=True)
    failed=list(queue.glob("*.failed.json"))
    if failed and not retry_failed:raise RuntimeError("Recorded eval failures require explicit diagnosis and --retry-failed")
    for p in failed:shutil.move(str(p),str(history/p.name))
    for name in ("DISPATCH.json","TRAIN_FAILURE.json","TRAIN_COMPLETE.json","CONTROLLER_FAILURE.json","CONTROLLER_COMPLETE.json"):
        p=out/name
        if p.exists():shutil.copy2(p,history/name)
    for gpu in (1,4):
        assert int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True))>8000
    env=dict(os.environ,PYTHONPATH=str(R/"src"),TMPDIR=str(R/"runtime"),
             ACTION4_ATTEMPT_ID=attempt,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",
             MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    workers=[];processes=[]
    for role,module,gpu,device in [
        ("eval","eval.action4_gdr_pose_eval_v002",1,"GPU-a77a9d4c-7642-7112-698c-b02673d254cb"),
        ("train","model.training.action4_gdr_pose_train_v002",4,"GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f")]:
        command=[str(R/".venv/bin/python"),"-B","-m",module,"--config",str(resolved.relative_to(R))]
        if role=="train" and resume:command+=["--resume",resume]
        with (history/(role+".log")).open("x") as log:
            p=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=log,start_new_session=True)
        workers.append({"role":role,"pid":p.pid,"gpu":gpu,"cpu":1,"log":str((history/(role+".log")).relative_to(R))});processes.append(p)
    dispatch={"attempt_id":attempt,"workers":workers,"config":str(resolved.relative_to(R)),"resume":resume,"retry_failed_explicit":retry_failed}
    atomic_json(history/"DISPATCH_CURRENT.json",dispatch);atomic_json(out/"DISPATCH.json",dispatch)
    while any(p.poll() is None for p in processes):
        if any(p.poll() is not None and p.returncode!=0 for p in processes):
            for p in processes:
                if p.poll() is None:p.terminate()
            atomic_json(out/"CONTROLLER_FAILURE.json",{"attempt_id":attempt,"codes":[p.poll() for p in processes],"policy":"fail closed; no automatic retry"});return
        time.sleep(15)
    atomic_json(out/"CONTROLLER_COMPLETE.json",{"attempt_id":attempt,"codes":[p.returncode for p in processes],"accuracy_accepted":False})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");p.add_argument("--retry-failed",action="store_true");a=p.parse_args()
    try:run(a.config,a.resume,a.retry_failed)
    except BaseException:
        cfg=json.loads((R/a.config).read_text());atomic_json(R/cfg["output"]/"CONTROLLER_FAILURE.json",{"traceback":traceback.format_exc()});raise
