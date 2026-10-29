"""Conditional, independent-size proposals for the targeted extension.

Uses the admitted CAD, mounting and perspective solvers unchanged.  Quota
bands describe actual output measurements; this module never edits labels.
"""
from copy import deepcopy
import math
import random
import numpy as np
from data.action4_phrase2_plan import seed_for
from data.action4_phrase2_geometry import (rotation_for_arm, pose_for_box,
    projected, junction_outside_frame, upperarm_reenters)
from data.action4_parameter_sampling import project
from data.action4_coverage1000_contract import (SMALL_BANDS, RATIO_BANDS,
    GAP_BANDS, SIDES, geometry_accept, occlusion_ranges, inside, source_eligible)


def draw_widths(rng, slot):
    side = SIDES.index(slot['designated_side'])
    group = slot['group']
    if group in 'AC':
        small = rng.uniform(*SMALL_BANDS[slot['small_width_band']])
        other = rng.uniform(max(.04, small / 1.5), min(.08, small * 1.5))
    else:
        ratio = rng.uniform(*RATIO_BANDS[slot['ratio_band']])
        band = (.08, min(.12, .23 / ratio)) if group == 'D' else SMALL_BANDS[slot['small_width_band']]
        small = rng.uniform(*band)
        other = small * ratio
    widths = [None, None]
    widths[side], widths[1-side] = small, other
    return widths


def orientation_hint(runtime, catalog, side, interval, center, rng):
    cache = catalog.setdefault('_coverage_cache', {})
    key = (side, tuple(interval))
    if key not in cache:
        lo, hi, last = interval
        choices = []
        for anchor in catalog['rows'][side]['anchors']:
            values = [a for a in anchor['samples']
                if a['occlusion_estimate'] is not None
                and inside(a['occlusion_estimate'], (lo, hi), last)
                and source_eligible(a['occlusion_estimate'], a['primary_class'])]
            if values:
                choices.append((anchor['center_uv'], values))
        if not choices:
            raise ValueError('No admitted orientation hint for requested interval')
        cache[key] = choices
    choices = cache[key]
    weights = [math.exp(-sum((a-b)**2 for a,b in zip(c,center))/(2*.25**2))+.015 for c,_ in choices]
    _, values = rng.choices(choices, weights=weights, k=1)[0]
    sample = rng.choice(values)
    roll = sample['roll_deg'] + rng.uniform(-3., 3.)
    direction = [sample['direction_x']+rng.uniform(-.15,.15), 1., sample['direction_z']+rng.uniform(-.12,.12)]
    return rotation_for_arm(runtime, side, roll, direction), roll, direction


def paired_centers(rng, cfg, size, widths, heights, gap, mode, slide, sign):
    """Construct a physical 2-D edge gap; y uses W units for Euclidean metrics."""
    a, b = sum(widths)/2, sum(heights)/2
    if mode == 0:
        dx = sign[0]*(a+gap)
        maxdy = min(b, math.sqrt(max(0., .20**2-dx**2)))
        if abs(dx) >= .20: return None
        dy = slide*maxdy
    elif mode == 1:
        dy = sign[1]*(b+gap)
        maxdx = min(a, math.sqrt(max(0., .20**2-dy**2)))
        if abs(dy) >= .20: return None
        dx = slide*maxdx
    else:
        angle = (slide+1)*math.pi/4
        dx, dy = sign[0]*(a+gap*math.cos(angle)), sign[1]*(b+gap*math.sin(angle))
        if math.hypot(dx,dy) > .20: return None
    delta = np.array([dx, dy])
    ranges = []
    for i,side in enumerate(SIDES):
        domain=cfg['position_ranges'][side]
        low=np.maximum([domain['u'][0],domain['v'][0]*size[1]/size[0]], [.001+widths[i]/2,.001+heights[i]/2])
        high=np.minimum([domain['u'][1],domain['v'][1]*size[1]/size[0]], [.999-widths[i]/2,size[1]/size[0]-.001-heights[i]/2])
        # center_left = midpoint-delta/2; center_right = midpoint+delta/2.
        ranges.append((low+delta/2 if i==0 else low-delta/2, high+delta/2 if i==0 else high-delta/2))
    low=np.maximum(ranges[0][0],ranges[1][0]); high=np.minimum(ranges[0][1],ranges[1][1])
    if (high<=low).any():return None
    mid=np.array([rng.uniform(low[k],high[k]) for k in range(2)])
    return [(mid-delta/2)/[1.,size[1]/size[0]], (mid+delta/2)/[1.,size[1]/size[0]]]


def side_guard(runtime, side, B, C, corners, size):
    if not .24 <= C[2,3] <= 1.4:return 'optical_Z_range'
    _,box,cam=projected(corners,C,runtime.K)
    if (cam[:,2]<=runtime.metadata['near_plane_m']).any() or (box[:2]<.5).any() or (box[2:]>size-.5).any():return 'body_outside'
    # Preserve the admitted scene and arm continuity guards unchanged.
    if runtime.support_solid is not None:
        inverse,low,high=runtime.support_solid
        actor=runtime.local_actor[side]@B[:3,:3].T+B[:3,3]
        local=actor@inverse[:3,:3].T+inverse[:3,3]
        if ((local>low+1e-6)&(local<high-1e-6)).all(1).any():return 'inside_table_solid'
    end=runtime.arm_terminal_points_B[side]@B[:3,:3].T+B[:3,3]
    start=B[:3,3]
    if start[2]<=.02:return 'wrist_near_camera'
    behind=end[:,2]<.02
    if behind.any():
        fraction=(start[2]-.02)/(start[2]-end[behind,2])
        end=end.copy();end[behind]=start+(end[behind]-start)*fraction[:,None]
    eh=end@runtime.K.T;eu=eh[:,:2]/eh[:,2:]/size
    if not any([(eu[:,0]<-.01).all(),(eu[:,0]>1.01).all(),(eu[:,1]<-.01).all(),(eu[:,1]>1.01).all()]):return 'arm_terminal_in_frame'
    if not junction_outside_frame(runtime,side,B,size) and not getattr(runtime,'coverage_continuous_junction_geometry',False):return 'elbow_junction_in_frame'
    if upperarm_reenters(runtime,side,B,size):return 'upperarm_reentry'
    return None


def propose(runtime, cfg, slot, attempt, catalog, local_visibility=None, ray_resolution=28):
    from data.action4_coverage1000_domains import vertical_domains,intersect,draw_domain
    catalog = catalog.get('by_split', {}).get(slot['split'], catalog)
    rng=random.Random(seed_for(cfg['master_seed'],slot['split'],slot['slot_id'],attempt))
    size=np.array([runtime.scene.render.resolution_x,runtime.scene.render.resolution_y])
    corners=np.array(runtime.metadata['canonical_definition']['corners_3d_m'])
    widths=draw_widths(rng,slot)
    centers=[np.array([rng.uniform(*cfg['position_ranges'][s]['u']),rng.uniform(*cfg['position_ranges'][s]['v'])]) for s in SIDES]
    intervals=occlusion_ranges(slot)
    hints=[orientation_hint(runtime,catalog,s,intervals[s],centers[i],rng) for i,s in enumerate(SIDES)]
    from data.action4_coverage1000_pose import augment_hints
    hints,pose_notes,error=augment_hints(runtime,cfg,slot,centers,widths,hints,rng)
    if error:return None,error
    gap=rng.uniform(*GAP_BANDS[slot['gap_band']]) if slot['group'] in 'CDE' else None
    mode=rng.randrange(3);slide=rng.uniform(-.95,.95);sign=(1 if rng.random()<.9 else -1, rng.choice([-1,1]))
    if gap is not None:
        # Construct the close pair BEFORE intersecting arm translation domains.
        # Unrelated initial centers would otherwise create a spurious dy gate.
        heights=[]
        for i,side in enumerate(SIDES):
            A=np.array(runtime.anchor['sides'][side]['A_brace_from_CAD'])
            try:C=pose_for_box(hints[i][0],A,corners,runtime.K,size,centers[i],widths[i])[1]
            except ValueError:return None,'projection_depth'
            _,box,_=projected(corners,C,runtime.K);heights.append((box[3]-box[1])/size[0])
        placement=random.Random(seed_for(cfg['master_seed'],slot['split'],slot['slot_id'],attempt,'placement'))
        centers=paired_centers(placement,cfg,size,widths,heights,gap,mode,slide,sign)
        if centers is None:return None,'pair_conditional_domain_empty'
    poses={};sides={};notes={};metrics={};boxes=[]
    for iteration in range(3 if gap is not None else 1):
        domains=[]
        for i,side in enumerate(SIDES):
            allowed,why=vertical_domains(runtime,cfg,side,hints[i][0],corners,size,float(centers[i][0]),widths[i])
            if not allowed:return None,side+':'+why
            domains.append(allowed)
        placement_y=random.Random(seed_for(cfg['master_seed'],slot['split'],slot['slot_id'],attempt,'physical_y'))
        if gap is None:
            for i in range(2):centers[i][1]=draw_domain(placement_y,domains[i])
        else:
            dy=float(centers[1][1]-centers[0][1])
            joint=intersect(domains[0],[(a-dy,b-dy) for a,b in domains[1]])
            if not joint:return None,'pair_arm_table_translation_domain_empty'
            y=draw_domain(placement_y,joint);centers[0][1]=y;centers[1][1]=y+dy
        solved=[]
        for i,side in enumerate(SIDES):
            fixed=runtime.anchor['sides'][side];A=np.array(fixed['A_brace_from_CAD'])
            try:B,C=pose_for_box(hints[i][0],A,corners,runtime.K,size,centers[i],widths[i])
            except ValueError:return None,'projection_depth'
            _,box,_=projected(corners,C,runtime.K)
            solved.append((B,C,box))
        if gap is None or iteration==2:break
        heights=[(b[3]-b[1])/size[0] for _,_,b in solved]
        # Reuse placement random numbers on the cheap fixed-point correction.
        placement=random.Random(seed_for(cfg['master_seed'],slot['split'],slot['slot_id'],attempt,'placement'))
        next_centers=paired_centers(placement,cfg,size,widths,heights,gap,mode,slide,sign)
        if next_centers is None:return None,'pair_conditional_domain_empty'
        centers=next_centers
    for i,side in enumerate(SIDES):
        B,C,box=solved[i];error=side_guard(runtime,side,B,C,corners,size)
        if error:return None,side+':'+error
        fixed=runtime.anchor['sides'][side];source=runtime.anchor['actor_pose_capture_by_side'][side]
        Bbase=np.linalg.inv(np.array(source['whole_actor_delta_capture']))@np.array(source['brace_B_capture'])
        poses[side]={'brace_B_capture':B.tolist(),'core_C_capture':C.tolist(),'whole_actor_delta_capture':(B@np.linalg.inv(Bbase)).tolist()}
        sides[side]={k:deepcopy(fixed[k]) for k in ['A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace']};sides[side].update(present=True,**poses[side])
        metrics[side]=project(corners,C,runtime.K,size,runtime.metadata['near_plane_m'])
        notes[side]={'requested_center_uv':centers[i].tolist(),'requested_center_semantics':'projected full-body bbox center','bbox_center_uv':((box[:2]+box[2:])/(2*size)).tolist(),
            'requested_width_fraction':widths[i],'actual_width_fraction':float((box[2]-box[0])/size[0]),'forearm_roll_deg':hints[i][1],
            'forearm_direction_capture':hints[i][2],'optical_Z_m':float(C[2,3]),'euclidean_CAD_origin_distance_m':float(np.linalg.norm(C[:3,3]))}
        notes[side].update(pose_notes[side])
        boxes.append(box)
    ok,error,pair=geometry_accept(slot,boxes,size,[solved[i][1] for i in range(2)])
    if not ok:return None,error
    if local_visibility is not None:
        for i,side in enumerate(SIDES):
            B,C,_=solved[i];m=local_visibility[side].estimate(B,C,ray_resolution,return_sources=True)
            lo,hi,last=intervals[side]
            if m['occlusion_estimate'] is None or not inside(m['occlusion_estimate'],(lo,hi),last) or not source_eligible(m['occlusion_estimate'],m['primary_class']):return None,side+':estimated_occlusion_or_source'
            notes[side]['local_ray_estimate']=m
    entry={'sample_id':slot['sample_id'],'index':slot['slot_id'],'seed':seed_for(cfg['master_seed'],slot['split'],slot['slot_id'],attempt,'render')%(2**31-1),
        'scene_family':{'wall':'wall_rear','table_full_garment':'table_turn','cloth_lift10_full_garment':'cloth_lift'}[slot['scene_id']],
        'target_count':2,'group_id':slot['group_id'],'background_source_id':'phrase2_'+slot['scene_id'],'background_seed':cfg['master_seed'],
        'split':slot['split'],'target_presence':[True,True],'batch_id':cfg['batch_id']}
    return {'schema':'action4.phrase2_arm_proposal.v1','parameter_contract_revision':'coverage1000_conditional_v001','occlusion_mode':'arm_geometry','occluder_proposals':{},
        'entry':entry,'target_presence':[True,True],'W_from_capture':runtime.W.tolist(),'K_native':runtime.K.tolist(),'canonical8_m':corners.tolist(),'corner_ids':list(range(8)),
        'actor_pose_capture_by_side':poses,'sides':sides,'derived':{'by_side':metrics},'background_request':{'template':runtime.name},
        'environment_controls':{'area_light_gain':rng.uniform(.85,1.15),'exposure_offset_ev':0.},'provenance':{'anchor_ref':runtime.metadata['baseline_parameters_ref']},
        'phrase2':{'slot_id':slot['slot_id'],'attempt':attempt,'by_side':notes,'master_seed':cfg['master_seed'],'source_RNG':'master+split+slot+attempt',
            'batch_id':cfg['batch_id'],'coverage_group':slot['group'],'slot_contract':slot,'pair_metrics':pair}}, None
