"""Thin ROI sweep evaluator entrypoint; no training or video admission bypass."""
import argparse,os,subprocess,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-lock",required=True);p.add_argument("--ratio",type=float,required=True);p.add_argument("--output",required=True)
    p.add_argument("--physical-gpu",type=int,required=True);p.add_argument("--cpu-affinity",type=int,nargs="+",required=True)
    a=p.parse_args()
    from data.action4_paths import PROJECT_ROOT as R
    from model.training.action4_resource_policy import validate_device_policy
    if Path.cwd().resolve()!=R:raise ValueError("Authorized working directory required")
    validate_device_policy({"training_gpus":[],"eval_gpus":[a.physical_gpu]},R)
    if a.physical_gpu not in (4,5,6,7) or set(os.sched_getaffinity(0))!=set(a.cpu_affinity) or not set(a.cpu_affinity)<=set(range(24,96)):
        raise ValueError("Explicit allowed GPU/CPU allocation required")
    fields=subprocess.check_output(["nvidia-smi","-i",str(a.physical_gpu),"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=fields[0].strip() or int(fields[1])<16384:
        raise ValueError("Visible device or memory headroom differs")
    import torch
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from eval.action4_roi_sweep import evaluate_fixed_ratio
    report=evaluate_fixed_ratio(a.input_lock,a.ratio,a.output,device="cuda:0")
    print(json.dumps({k:report[k] for k in ["roi_expansion","pose_all8_at_002_success","pose_gt_count","B0_forward_calls"]}),flush=True)
if __name__=="__main__":main()
