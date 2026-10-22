"""Explicit allowed GPU identities; query only devices assigned to this project."""
import os
import subprocess

ALLOWED_GPU_UUIDS = (
    "GPU-8dfd78d1-6a27-4292-3338-9d33fffee9ef",
    "GPU-510dc520-5a9d-1317-019d-de5ed02c1b37",
    "GPU-1648afa2-3e25-b1c7-b149-38b07306eb8f",
    "GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8",
    "GPU-b1ae5be6-1dba-7343-e233-59df41402f33",
    "GPU-a68477c3-1c2a-d557-58f5-405d80e5383e",
)


def gpu_identity(uuid):
    if uuid not in ALLOWED_GPU_UUIDS:
        raise ValueError("GPU UUID is not authorized for this project")
    line = subprocess.check_output(["nvidia-smi", "-i", uuid,
        "--query-gpu=index,uuid,pci.bus_id,memory.free,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip()
    values = [v.strip() for v in line.split(",")]
    if len(values) != 6 or values[1] != uuid or int(values[0]) != 2 + ALLOWED_GPU_UUIDS.index(uuid):
        raise RuntimeError("GPU physical index/UUID mapping changed")
    return dict(index=int(values[0]),uuid=uuid,pci_bus_id=values[2],free_mib=int(values[3]),
                used_mib=int(values[4]),total_mib=int(values[5]))


def worker_environment(uuid=None):
    env = dict(os.environ)
    for key in ("RANK","WORLD_SIZE","LOCAL_RANK","LOCAL_WORLD_SIZE","GROUP_RANK",
                "ROLE_RANK","ROLE_WORLD_SIZE","MASTER_ADDR","MASTER_PORT"):
        env.pop(key,None)
    if uuid is not None:
        gpu_identity(uuid)
    env["CUDA_VISIBLE_DEVICES"] = uuid or ""
    return env
