"""Measurements and exact acceptance predicates for the targeted coverage batch."""
import math
import numpy as np

SMALL_BANDS=((.04,.05),(.05,.06),(.06,.07),(.07,.08))
RATIO_BANDS=((1.5,1.8),(1.8,2.1),(2.1,2.4))
GAP_BANDS=((.003,.01),(.01,.02),(.02,.03))
HIGH_BANDS=((.40,.50),(.50,.60),(.60,.75))
SIDES=('left','right')

def inside(value,band,last=False):
    lo,hi=band
    return math.isfinite(value) and value>=lo and (value<=hi if last else value<hi)

def pair_metrics(boxes,width):
    b=np.asarray(boxes,dtype=float)
    if b.shape!=(2,4) or not np.isfinite(b).all() or (b[:,2:]<=b[:,:2]).any():raise ValueError('Invalid full-body boxes')
    sizes=(b[:,2]-b[:,0])/width;centers=(b[:,:2]+b[:,2:])/2
    gap=np.maximum(np.maximum(b[0,:2]-b[1,2:],b[1,:2]-b[0,2:]),0)
    intersects=bool((np.minimum(b[0,2:],b[1,2:])-np.maximum(b[0,:2],b[1,:2])>0).all())
    return {'widths':sizes.tolist(),'ratio':float(max(sizes)/min(sizes)),'d_edge':float(np.linalg.norm(gap)/width),
            'center_distance':float(np.linalg.norm(centers[1]-centers[0])/width),'center_delta_width_units':((centers[1]-centers[0])/width).tolist(),
            'gap_xy_pixels':gap.tolist(),'boxes_intersect':intersects}

def geometry_accept(slot,boxes,size,transforms=None):
    m=pair_metrics(boxes,size[0]);w=m['widths'];group=slot['group'];designated=SIDES.index(slot['designated_side']);other=1-designated
    if m['boxes_intersect']:return False,'full_body_boxes_overlap',m
    b=np.asarray(boxes);canvas=np.tile(np.asarray(size),2)
    if (b[:,:2]<.5).any() or (b[:,2:]>np.asarray(size)-.5).any():return False,'body_outside',m
    if group in 'ABCE':
        k=slot['small_width_band'];
        if not inside(w[designated],SMALL_BANDS[k],k==3):return False,'actual_small_width_stratum',m
    if group in 'AC':
        if not all(.04<=x<=.08 for x in w) or not 1<=m['ratio']<=1.5:return False,'double_small_or_ratio',m
    else:
        k=slot['ratio_band']
        if not inside(m['ratio'],RATIO_BANDS[k],k==2):return False,'actual_ratio_stratum',m
        if w[designated]>=w[other]:return False,'designated_is_not_smaller',m
        if max(w)>.23:return False,'larger_width_over_023',m
        if group=='D' and not .08<=w[designated]<=.12:return False,'D_smaller_width',m
    if group in 'CDE':
        k=slot['gap_band']
        if not inside(m['d_edge'],GAP_BANDS[k],k==2):return False,'actual_edge_gap_stratum',m
        if m['center_distance']>.20:return False,'actual_2d_center_distance',m
    if transforms is not None:
        for side,T in zip(SIDES,transforms):
            t=np.asarray(T,dtype=float)[:3,3];z=float(t[2]);distance=float(np.linalg.norm(t))
            if not np.isfinite(t).all() or not .24<=z<=1.4:return False,'optical_Z_range',m
            m.setdefault('distance_by_side',{})[side]={'optical_Z_m':z,'euclidean_CAD_origin_distance_m':distance}
            band=slot.get('depth_bands_by_side',{}).get(side)
            if band and not inside(z,band,band[1]==1.4):return False,'required_optical_Z_stratum',m
    return True,None,m

def occlusion_ranges(slot):
    if slot['group']=='A':return {s:(0.,.35,False) for s in SIDES}
    if not slot['high_required']:return {s:(0.,.40,False) for s in SIDES}
    result={s:(0.,.75,True) for s in SIDES};k=slot['high_band'];result[slot['high_side']]=(*HIGH_BANDS[k],k==2)
    return result

def actual_accept(slot,record):
    if record.get('planned_target_count',2)!=2:raise ValueError('Wrong physical target plan')
    if record['excluded_instances'] or record['target_count']!=2:return False,'insufficient_actual_visible_body_instances',{}
    if [o['instance_id'] for o in record['instances']]!=[1,2]:raise ValueError('Physical camera identities corrupted')
    objects=record['instances'];Ts=[o['canonical8_geometry']['object_to_capture_cv_m'] for o in objects]
    ok,why,metrics=geometry_accept(slot,[o['bbox_full_xyxy'] for o in objects],(record['width'],record['height']),Ts)
    if not ok:return ok,why,metrics
    for o,side in zip(objects,SIDES):
        lo,hi,last=occlusion_ranges(slot)[side];value=float(o['occlusion_inframe'])
        if not inside(value,(lo,hi),last):return False,'actual_occlusion_stratum_'+side,metrics
        if o['truncation']>1e-5:return False,'actual_truncation',metrics
        if o.get('ignore',False) or o['visible_area_at_fullinput_px']<16:return False,'existing_visible_support_rule',metrics
    metrics['actual_occlusion_by_side']={s:float(o['occlusion_inframe']) for s,o in zip(SIDES,objects)}
    return True,None,metrics

def source_eligible(value,primary):
    return primary=='own_wrist_forearm' or (value<=.35 and primary in ['own_mount','none_sampled'])
