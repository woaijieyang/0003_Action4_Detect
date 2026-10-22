"""Read device identity for generation GPUs4-7 or explicitly separated quality GPU3."""
import subprocess
from .action4_devices import gpu_identity as generation_identity
QUALITY_GPU_UUID='GPU-510dc520-5a9d-1317-019d-de5ed02c1b37'
def gpu_identity(uuid):
    if uuid!=QUALITY_GPU_UUID:return generation_identity(uuid)
    raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,pci.bus_id,memory.free,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    values=[v.strip() for v in raw.split(',')]
    if len(values)!=6 or values[1]!=uuid or int(values[0])!=3:raise RuntimeError('Quality GPU physical identity changed')
    return dict(index=3,uuid=uuid,pci_bus_id=values[2],free_mib=int(values[3]),used_mib=int(values[4]),total_mib=int(values[5]))
