"""Analytic translation domains; no changes to arm, body, mounting or camera.

For a fixed orientation, horizontal center and projected width, only capture-Y
translation remains free.  Projected rings are affine in that translation.
The admitted terminal/junction rules and table vertex rule can therefore be
applied before sampling rather than discovered repeatedly by blind rejection.
"""
import numpy as np
from data.action4_phrase2_geometry import pose_for_box,projected


def intersect(left,right):
    return [(max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<min(b,d)]


def subtract(domains,forbidden):
    result=domains
    for c,d in forbidden:
        new=[]
        for a,b in result:
            if d<=a or c>=b:new.append((a,b));continue
            if a<c:new.append((a,c))
            if d<b:new.append((d,b))
        result=new
    return result


def outside_ring_domain(points,B,K,size,clip_near=False):
    p=points@B[:3,:3].T+B[:3,3]
    if clip_near:
        start=B[:3,3];behind=p[:,2]<.02
        if start[2]<=.02:return []
        if behind.any():
            f=(start[2]-.02)/(start[2]-p[behind,2]);p=p.copy();p[behind]=start+(p[behind]-start)*f[:,None]
    if (p[:,2]<=0).any():return []
    # B is solved at C_y=0. Translation in capture Y preserves X and Z.
    x=K[0,0]*p[:,0]/p[:,2]+K[0,2]
    if (x<-.01*size[0]).all() or (x>1.01*size[0]).all():return [(-float('inf'),float('inf'))]
    upper=np.min((-.01*size[1]-K[1,2])*p[:,2]/K[1,1]-p[:,1])
    lower=np.max((1.01*size[1]-K[1,2])*p[:,2]/K[1,1]-p[:,1])
    return [(-float('inf'),upper),(lower,float('inf'))]


def table_forbidden(runtime,side,B):
    if runtime.support_solid is None:return []
    inv,low,high=runtime.support_solid
    actor=runtime.local_actor[side]@B[:3,:3].T+B[:3,3]
    local=actor@inv[:3,:3].T+inv[:3,3]
    slope=inv[:3,1];lower=np.full(len(local),-np.inf);upper=np.full(len(local),np.inf)
    valid=np.ones(len(local),dtype=bool)
    for axis in range(3):
        if abs(slope[axis])<1e-10:
            valid&=(local[:,axis]>low[axis]+1e-6)&(local[:,axis]<high[axis]-1e-6)
        else:
            a=(low[axis]+1e-6-local[:,axis])/slope[axis]
            b=(high[axis]-1e-6-local[:,axis])/slope[axis]
            lower=np.maximum(lower,np.minimum(a,b));upper=np.minimum(upper,np.maximum(a,b))
    intervals=sorted(zip(lower[valid&(upper>lower)],upper[valid&(upper>lower)]))
    merged=[]
    for a,b in intervals:
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return merged


def vertical_domains(runtime,cfg,side,RB,corners,size,u,width):
    A=np.array(runtime.anchor['sides'][side]['A_brace_from_CAD'])
    B,C=pose_for_box(RB,A,corners,runtime.K,size,[u,.5],width)
    # Exact one-dimensional coordinate is physical camera translation C_y.
    C[1,3]=0.;B=C@np.linalg.inv(A)
    if not .24<=C[2,3]<=1.4:return [],'optical_Z_range'
    lo,hi=cfg['position_ranges'][side]['v']
    c1=pose_for_box(RB,A,corners,runtime.K,size,[u,lo],width)[1][1,3]
    c2=pose_for_box(RB,A,corners,runtime.K,size,[u,hi],width)[1][1,3]
    domains=[(c1,c2)]
    required_rings=[runtime.arm_terminal_points_B[side]]
    if not getattr(runtime,'coverage_continuous_junction_geometry',False):required_rings.append(runtime.arm_junction_points_B[side])
    for points in required_rings:
        domains=intersect(domains,outside_ring_domain(points,B,runtime.K,size,True))
    if not domains:return [],'no_continuous_arm_translation_domain'
    domains=subtract(domains,table_forbidden(runtime,side,B))
    if not domains:return [],'no_table_free_translation_domain'
    result=[]
    for a,b in domains:
        # Convert endpoints to actual full-bbox v, not CAD-origin v.
        vals=[]
        for t in [a,b]:
            T=C.copy();T[1,3]=t;_,box,_=projected(corners,T,runtime.K)
            vals.append(float((box[1]+box[3])/(2*size[1])))
        result.append(tuple(vals))
    return result,None


def draw_domain(rng,domains):
    lengths=[b-a for a,b in domains];v=rng.random()*sum(lengths)
    for (a,b),length in zip(domains,lengths):
        if v<length:return a+v
        v-=length
    return domains[-1][1]
