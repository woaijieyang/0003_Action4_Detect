"""Actual full-input data preflight followed by four-rank NCCL; no training."""
import os,json,time,datetime,importlib.util,sys
from pathlib import Path
R=Path("/data/zy/vln/a-test")
O=R/"outputs/action4_corner_startup_repair_v001"
spec=json.loads((R/"runtime/action4_runs/p2_corner_joint_capacity20_v004/RUN_SPEC.json").read_text())
sys.path[:0]=[str(R/spec["code_root"]/"src"),str(R/spec["code_root"]/"runtime/surface_deps")]
os.environ["ACTION4_PROJECT_ROOT"]=str(R)
loader=importlib.util.spec_from_file_location("startup_probe",O/"source/readiness.py")
bridge=importlib.util.module_from_spec(loader);loader.loader.exec_module(bridge)
import torch,torch.distributed as dist
from data.action4_stage2_data import Stage2Units
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
from model.training.action4_runtime import atomic_json
rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
cfg=json.loads((R/"runtime/action4_runs/p2_a0_corner_prediction15_formal80_v001/source/resolved_config.json").read_text())
start=time.time()
atomic_json(O/f"rank_{rank}_START.json",dict(rank=rank,pid=os.getpid(),time=start,phase="full10400_native_data_preflight"))
data=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
assert len(data)==10400
ready=bridge.wait_data_ready(O,"full10400",rank,world,len(data),1800)
torch.cuda.set_device(local)
dist.init_process_group("nccl",device_id=torch.device(f"cuda:{local}"),timeout=datetime.timedelta(seconds=180))
value=torch.tensor(float(rank+1),device=f"cuda:{local}");dist.all_reduce(value);assert value.item()==10
dist.barrier()
atomic_json(O/f"rank_{rank}_COMPLETE.json",dict(rank=rank,pid=os.getpid(),units=len(data),elapsed_seconds=time.time()-start,readiness=ready,actual_nccl_sum=value.item(),training_updates=0))
dist.destroy_process_group()
