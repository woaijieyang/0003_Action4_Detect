"""Live resource authorization is separate from immutable numerical run identity."""
import hashlib,json
from pathlib import Path

def validate_device_policy(cfg, root):
    root=Path(root).resolve(strict=True)
    def checked(relative):
        p=root/relative
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root):raise ValueError('Resource authority path escaped root')
        return p
    pointer=json.loads(checked('configs/execution/action4_active.json').read_text())
    ref=pointer['active_ref'];p=checked(ref['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Resource authority digest mismatch')
    policy=json.loads(p.read_text());train=cfg['training_gpus'];evaluate=cfg['eval_gpus']
    if any(type(g) is not int or g not in range(8) for g in train+evaluate):raise ValueError('Invalid physical GPU ID')
    forbidden=set(policy.get('forbidden_gpu_ids',[]))
    if policy.get('GPU0_forbidden',False):forbidden.add(0)
    if forbidden.intersection(train+evaluate):raise ValueError('Physical GPU permission revoked: '+str(sorted(forbidden.intersection(train+evaluate))))
    allowed=policy.get('authorized_training_gpu',[2,3,4,5,6,7])
    if any(g not in allowed for g in train) or 1 in train:raise ValueError('Unauthorized training GPU')
    allowed_eval=policy.get('authorized_evaluation_gpu',[g for g in range(8) if g not in forbidden])
    if any(g not in allowed_eval for g in evaluate):raise ValueError('Unauthorized evaluation GPU')
    if evaluate.count(1)>policy.get('gpu1_eval_only_max_workers',8):raise ValueError('GPU1 evaluation worker budget exceeded')
    return {'authority':ref,'train_gpus':train,'eval_gpus':evaluate,'GPU0_forbidden':policy.get('GPU0_forbidden',False),'forbidden_gpu_ids':sorted(forbidden),'cpu_policy':policy.get('cpu_policy')}


def process_cpu_affinity(cfg, name, root):
    """Explicit opt-in process allocation, inherited by its child loader threads."""
    groups=cfg.get('process_cpu_affinity')
    if groups is None:return None
    policy=validate_device_policy(cfg,root)['cpu_policy']
    required=['train']+[f'eval_{i}' for i in range(len(cfg['eval_gpus']))]
    if set(groups)!=set(required):raise ValueError('CPU process allocation incomplete')
    flat=[c for key in required for c in groups[key]]
    if len(flat)!=len(set(flat)) or any(type(c) is not int or c not in policy['allowed_logical_cpu_ids'] for c in flat):
        raise ValueError('CPU allocation overlap or outside authorized tail pool')
    for key in required:
        need=cfg['world_size']*(cfg.get('threads',2)+cfg.get('loader_workers',0)) if key=='train' else evaluation_threads(cfg)+cfg.get('eval_loader_workers',4)
        if len(groups[key])<need:raise ValueError('Insufficient allocated CPUs for '+key)
    if cfg['project_total_reserved_cpu']>policy['max_logical_cpus']:raise ValueError('Total CPU budget exceeded')
    return groups[name]


def apply_allocator_budget(cfg, kind, logical_device=0):
    budget=cfg.get(kind+'_allocator_budget_mib')
    if budget is None:return
    if type(budget) is not int or budget<=0:raise ValueError('Invalid CUDA allocator budget')
    import torch
    total=torch.cuda.get_device_properties(logical_device).total_memory
    fraction=budget*1024*1024/total
    if not 0<fraction<1:raise ValueError('Allocator budget exceeds physical capacity')
    torch.cuda.set_per_process_memory_fraction(fraction,logical_device)


def evaluation_threads(cfg):
    value=cfg.get("eval_threads",2)
    if type(value) is not int or value not in (1,2):
        raise ValueError("Explicit tested evaluation thread count must be1 or2")
    return value
