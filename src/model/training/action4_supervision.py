"""Update-level sufficient statistics and explicit AdamW no-supervision policy."""
import torch
from torch.nn import functional as F
from model.action4_sparse_surface_losses import sample_points
from model.action4_pose_maps import geometry_losses

A0_KEYS=('xyz','region','visible_mask','amodal_mask','rotation_point_matching','centroid',
         'relative_depth','visible_corners','hidden_corners','positive_depth')

def eligibility(item):
    active=bool(item['active']);meta=item.get('metadata',{});g=item['geometry']
    present=meta.get('target_visible_input_support_count',float(item['surface_masks'][0].sum()))>0
    identity=bool(meta.get('target_identity_valid',True));valid=bool(meta.get('gt_geometry_valid',True))
    pose=active and present and identity and valid
    surface=active and identity and len(item['surface_sparse']['xy_hm'])>0
    return {'active':active,'pose':pose,'surface':surface,'visible_mask':active and identity,
            'amodal_mask':pose,'visible_corners':int(g['visible'].sum()) if pose else 0,
            'hidden_corners':int((~g['visible']).sum()) if pose else 0}

def local_counts(items, *, sam_enabled=False, corner_enabled=False):
    # The caller passes the SAME configured variant on every rank. Never infer
    # collective vector shape from local availability, supervision or markers.
    from model.action4_sam_joint import sam_items_enabled, sam_quality_eligible, QUALITY_KEY
    if type(corner_enabled) is not bool or (corner_enabled and not sam_enabled):
        raise ValueError("Explicit configured corner flag requires SAM on every rank")
    if type(sam_enabled) is not bool:
        raise ValueError("Explicit boolean configured SAM variant required")
    if sam_enabled:
        if not items or any(item.get("sam_enabled") is not True for item in items):
            raise ValueError("Configured SAM rank requires marked items, including missing/dummy")
    elif sam_items_enabled(items):
        raise ValueError("SAM items supplied to configured baseline counts")
    counts={k:0 for k in A0_KEYS}
    if sam_enabled:counts[QUALITY_KEY]=0
    for item in items:
        e=eligibility(item)
        for k in A0_KEYS:
            counts[k]+=e[k] if k in ('visible_mask','amodal_mask','visible_corners','hidden_corners') else e['surface'] if k in ('xyz','region') else e['pose']
        if sam_enabled:counts[QUALITY_KEY]+=sam_quality_eligible(item)
    if corner_enabled:
        from model.training.action4_corner_targets import corner_targets_from_items,corner_target_counts
        counts.update(corner_target_counts(corner_targets_from_items(items,dtype=items[0]["roi"].dtype)))
    return counts


def a0_loss_sums(output,items,device):
    zero=sum(v.sum()*0 for v in output['surface'].values())
    zero=zero+output['rotation'].sum()*0+output['translation'].sum()*0
    sums={k:zero for k in A0_KEYS}
    for i,item in enumerate(items):
        e=eligibility(item)
        if e['surface']:
            s=item['surface_sparse'];xy=s['xy_hm'].to(device)
            sums['xyz']=sums['xyz']+(sample_points(output['surface']['xyz'][i],xy)-s['xyz'].to(device)).abs().sum(-1).mean()
            sums['region']=sums['region']+F.cross_entropy(sample_points(output['surface']['regions'][i],xy),s['regions'].to(device))
        for channel,key in enumerate(['visible_mask','amodal_mask']):
            if e[key]:sums[key]=sums[key]+F.binary_cross_entropy_with_logits(output['surface']['masks'][i,channel],item['surface_masks'][channel].to(device))
        if e['pose']:
            geometry={k:v[None].to(device) for k,v in item['geometry'].items()}
            if not all(torch.isfinite(v).all() for v in geometry.values()):raise ValueError('Illegal eligible GT before geometry arithmetic')
            one={k:v[i:i+1] for k,v in output.items() if torch.is_tensor(v)}
            loss=geometry_losses(one,geometry,{'visible':1.,'hidden':1.})
            for k,v in loss.items():sums[k]=sums[k]+v
    return sums

def weighted_objective(sums,counts,weights,world_size):
    unknown=set(weights)-set(sums)
    if unknown:raise ValueError('Unconsumed loss weights: '+str(sorted(unknown)))
    total=sum(v*0 for v in sums.values())
    for k,value in sums.items():
        if counts[k]>0:total=total+weights.get(k,1.)*world_size*value/counts[k]
    return total

def has_supervision(counts,weights):
    return any(weights.get(k,1.)>0 and n>0 for k,n in counts.items())

def clear_unsupervised_heads(model,counts,weights,dependencies):
    """Each entry owns parameters exclusively; shared modules must list all driving losses."""
    named=dict(model.named_parameters());seen=set()
    for entry in dependencies:
        names=[n for n in named if any(n==p or n.startswith(p+'.') for p in entry['parameter_prefixes'])]
        if not names:raise ValueError('Unknown independent-head parameter prefix')
        if seen.intersection(names):raise ValueError('Overlapping parameter-loss ownership')
        seen.update(names)
        if any(k not in counts for k in entry['losses']):raise ValueError('Unknown head supervision key')
        if not any(counts[k]>0 and weights.get(k,1.)>0 for k in entry['losses']):
            for name in names:named[name].grad=None

def step_if_supervised(model,optimizer,scheduler,counts,weights,dependencies=(),clip=1.):
    if not has_supervision(counts,weights):
        optimizer.zero_grad(set_to_none=True)
        return False,0.
    clear_unsupervised_heads(model,counts,weights,dependencies)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return True,float(norm)
