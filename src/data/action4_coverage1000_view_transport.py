"""Transport admitted orientation priors to the current observation ray.

All actual geometry, collision and mask predicates remain unchanged. This is
an isolated sampler candidate until its bounded actual-scene probe succeeds.
"""
import math,random
import numpy as np
from data import action4_coverage1000_sampling as base
from data.action4_phrase2_geometry import align,rotation_for_arm,pose_for_box,projected
from data.action4_coverage1000_contract import SIDES,inside,source_eligible,occlusion_ranges,geometry_accept
from data.action4_parameter_sampling import project

_original_propose=base.propose
_chosen={}

def hint(runtime,catalog,side,interval,center,rng):
    cache=catalog.setdefault('_transport_cache',{});key=(side,tuple(interval))
    if key not in cache:
        lo,hi,last=interval;choices=[]
        for a in catalog['rows'][side]['anchors']:
            values=[x for x in a['samples'] if x['occlusion_estimate'] is not None and inside(x['occlusion_estimate'],(lo,hi),last) and source_eligible(x['occlusion_estimate'],x['primary_class'])]
            if values:choices.append((a['center_uv'],values))
        if not choices:raise ValueError('Missing orientation prior')
        cache[key]=choices
    choices=cache[key];weights=[math.exp(-sum((a-b)**2 for a,b in zip(c,center))/(2*.25**2))+.015 for c,_ in choices]
    c,values=rng.choices(choices,weights=weights,k=1)[0];sample=rng.choice(values)
    roll=sample['roll_deg']+rng.uniform(-3,3);direction=[sample['direction_x']+rng.uniform(-.15,.15),1.,sample['direction_z']+rng.uniform(-.12,.12)]
    _chosen[side]=sample.get('calibration_center_uv',c)
    return rotation_for_arm(runtime,side,roll,direction),roll,direction


def propose(runtime,cfg,slot,attempt,catalog,local_visibility=None,ray_resolution=28):
    _chosen.clear();previous=base.orientation_hint;base.orientation_hint=hint
    try:p,error=_original_propose(runtime,cfg,slot,attempt,catalog,local_visibility=None,ray_resolution=ray_resolution)
    finally:base.orientation_hint=previous
    if error:return None,error
    size=np.array([runtime.scene.render.resolution_x,runtime.scene.render.resolution_y]);corners=np.array(p['canonical8_m']);boxes=[];Ts=[]
    for side in SIDES:
        note=p['phrase2']['by_side'][side];oldB=np.array(p['actor_pose_capture_by_side'][side]['brace_B_capture']);oldC=np.array(p['actor_pose_capture_by_side'][side]['core_C_capture'])
        pixel=np.r_[np.array(_chosen[side])*size,1.];oldray=np.linalg.inv(runtime.K)@pixel
        newray=oldC[:3,3];RB=align(oldray,newray)@oldB[:3,:3]
        d=RB@runtime.arm_axis_B[side]
        if d[1]<=0:return None,side+':transport_outside_arm_motion_domain'
        direction=d/d[1]
        if abs(direction[0])>.8 or not -1.42<=direction[2]<=-.28:return None,side+':transport_outside_arm_motion_domain'
        A=np.array(p['sides'][side]['A_brace_from_CAD']);B,C=pose_for_box(RB,A,corners,runtime.K,size,note['bbox_center_uv'],note['actual_width_fraction'])
        why=base.side_guard(runtime,side,B,C,corners,size)
        if why:return None,side+':transport_'+why
        delta0=rotation_for_arm(runtime,side,0.,direction);rot=RB@delta0.T;axis=d/np.linalg.norm(d)
        sin=float(np.dot(axis,np.array([rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])/2));cos=float((np.trace(rot)-1)/2)
        roll=math.degrees(math.atan2(sin,cos))
        if not np.allclose(rotation_for_arm(runtime,side,roll,direction),RB,atol=1e-8,rtol=0):raise ValueError('Transported orientation cannot be represented in admitted arm coordinates')
        source=runtime.anchor['actor_pose_capture_by_side'][side];B0=np.linalg.inv(np.array(source['whole_actor_delta_capture']))@np.array(source['brace_B_capture'])
        pose={'brace_B_capture':B.tolist(),'core_C_capture':C.tolist(),'whole_actor_delta_capture':(B@np.linalg.inv(B0)).tolist()}
        p['actor_pose_capture_by_side'][side]=pose;p['sides'][side].update(pose)
        _,box,_=projected(corners,C,runtime.K);boxes.append(box);Ts.append(C)
        note.update(forearm_roll_deg=roll,forearm_direction_capture=direction.tolist(),optical_Z_m=float(C[2,3]),euclidean_CAD_origin_distance_m=float(np.linalg.norm(C[:3,3])),
            prior_sightline_center_uv=_chosen[side],view_transport_applied=True,bbox_center_uv=((box[:2]+box[2:])/(2*size)).tolist())
        p['derived']['by_side'][side]=project(corners,C,runtime.K,size,runtime.metadata['near_plane_m'])
        if local_visibility is not None:
            m=local_visibility[side].estimate(B,C,ray_resolution,return_sources=True);lo,hi,last=occlusion_ranges(slot)[side]
            if m['occlusion_estimate'] is None or not inside(m['occlusion_estimate'],(lo,hi),last) or not source_eligible(m['occlusion_estimate'],m['primary_class']):return None,side+':transport_local_occlusion_or_source'
            note['local_ray_estimate']=m
    ok,why,metrics=geometry_accept(slot,boxes,size,Ts)
    if not ok:return None,why
    p['phrase2']['pair_metrics']=metrics;p['parameter_contract_revision']='coverage1000_view_transport_candidate_v001'
    return p,None
