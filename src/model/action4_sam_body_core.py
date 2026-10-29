"""Geometry-only thick-core candidate. Morphology is NOT semantic body truth."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy.ndimage import distance_transform_edt,label

@dataclass(frozen=True)
class BodyCorePolicy:
    radius_fraction:float=.4
    minimum_radius_px:float=1.
    maximum_radius_px:float=128.
    minimum_retained_fraction:float=.5
    minimum_dominant_eroded_fraction:float=.65
    maximum_competitor_ratio:float=.5
    maximum_aspect:float=4.
    minimum_eroded_pixels:int=9
    prior_distance_in_max_radii:float=1.
    def __post_init__(self):
        values=(self.radius_fraction,self.minimum_radius_px,self.maximum_radius_px,self.minimum_retained_fraction,
                self.minimum_dominant_eroded_fraction,self.maximum_competitor_ratio,self.maximum_aspect,self.prior_distance_in_max_radii)
        if any(isinstance(x,bool) or not math.isfinite(x) for x in values):raise ValueError('Finite explicit candidate policy required')
        if not (0<self.radius_fraction<1 and 0<self.minimum_radius_px<=self.maximum_radius_px and .5<=self.minimum_retained_fraction<=1 and .5<self.minimum_dominant_eroded_fraction<=1 and 0<=self.maximum_competitor_ratio<1 and self.maximum_aspect>=1 and self.prior_distance_in_max_radii>0):raise ValueError('Invalid candidate policy')
        if type(self.minimum_eroded_pixels) is not int or self.minimum_eroded_pixels<1:raise ValueError('Positive eroded-area floor required')

def geometry(mask):
    y,x=np.nonzero(mask)
    if not len(x):return None,None
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],[float(x.mean()),float(y.mean())]

def body_core_geometry(mask,*,prior_body_center=None,policy=BodyCorePolicy()):
    """Return a candidate mask plus explicit qualification; never mutates mask.

    Prior center is an already bound prediction, not a pixel chosen by frame ID.
    Qualified geometry does not certify camera identity or exclude a thick mount.
    """
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=2 or min(mask.shape)<=0:raise ValueError('Native bool HxW mask required')
    if not isinstance(policy,BodyCorePolicy):raise ValueError('Typed fixed policy required')
    if prior_body_center is not None:
        prior_body_center=np.asarray(prior_body_center,float)
        if prior_body_center.shape!=(2,) or not np.isfinite(prior_body_center).all():raise ValueError('Finite native XY prior required')
    raw_box,raw_center=geometry(mask);area=int(mask.sum());output=np.zeros_like(mask)
    receipt={'protocol':'action4.sam.body_core_geometry.v1','source':'candidate_NOT_semantic_GT','policy':asdict(policy),
        'raw_mask_unchanged':True,'raw_pixels':area,'raw_box_native':raw_box,'raw_centroid':raw_center,
        'geometry_qualified':False,'full_body_certified':False,'GT_used':False,'prior_used':False}
    if not area:return {'mask':output,'candidate_box':None,'qualified_box':None,'receipt':dict(receipt,reason='empty_support')}
    x0,y0,x1,y1=raw_box;crop=np.pad(mask[y0:y1,x0:x1],1,constant_values=False)
    edt=distance_transform_edt(crop);maximum=float(edt.max());radius=float(np.clip(policy.radius_fraction*maximum,policy.minimum_radius_px,policy.maximum_radius_px))
    eroded=edt>radius;labels,n=label(eroded,np.ones((3,3),bool));sizes=np.bincount(labels.ravel());sizes[0]=0
    choices=sorted(range(1,n+1),key=lambda k:(-int(sizes[k]),k))
    receipt.update(crop_native=[x0,y0,x1,y1],max_interior_radius_px=maximum,opening_radius_px=radius,eroded_components=n,eroded_pixels=int(eroded.sum()))
    if not choices:return {'mask':output,'candidate_box':None,'qualified_box':None,'receipt':dict(receipt,reason='no_thick_core')}
    selected=choices[0];prior_reason='not_supplied'
    if prior_body_center is not None:
        distances=[]
        for k in choices:
            yy,xx=np.nonzero(labels==k)
            distances.append((float(np.hypot(xx+x0-1-prior_body_center[0],yy+y0-1-prior_body_center[1]).min()),k))
        distance,k=min(distances)
        if distance<=policy.prior_distance_in_max_radii*maximum:
            selected=k;receipt['prior_used']=True;prior_reason='nearest_eroded_component'
        else:prior_reason='prior_too_far_for_selection'
    selected_mask=labels==selected
    # Exact Euclidean dilation with the SAME radius, clipped to original support.
    restored=(distance_transform_edt(~selected_mask)<=radius)&crop
    output[y0:y1,x0:x1]=restored[1:-1,1:-1]
    box,center=geometry(output);retained=int(output.sum());fraction=retained/area
    competing=max((int(sizes[k]) for k in choices if k!=selected),default=0)
    dominance=int(sizes[selected])/max(1,int(eroded.sum()));competitor=competing/max(1,int(sizes[selected]))
    aspect=None if box is None else max((box[2]-box[0])/(box[3]-box[1]),(box[3]-box[1])/(box[2]-box[0]))
    reasons=[]
    if int(sizes[selected])<policy.minimum_eroded_pixels:reasons.append('insufficient_thick_pixels')
    if dominance<policy.minimum_dominant_eroded_fraction or competitor>policy.maximum_competitor_ratio:reasons.append('ambiguous_thick_components')
    if fraction<policy.minimum_retained_fraction:reasons.append('retained_area_below_half')
    if aspect is None or aspect>policy.maximum_aspect:reasons.append('implausible_core_aspect')
    receipt.update(selected_component=selected,selected_eroded_pixels=int(sizes[selected]),dominant_eroded_fraction=dominance,
        largest_competitor_ratio=competitor,retained_pixels=retained,retained_fraction=fraction,removed_pixels=area-retained,
        candidate_centroid=center,candidate_aspect=aspect,prior_reason=prior_reason,
        reason='qualified_geometry_candidate' if not reasons else '|'.join(reasons),geometry_qualified=not reasons,
        ambiguity_reasons=reasons,semantic_limitation='A connected thick mount or device-shaped false support can pass morphology; this is not a semantic test')
    return {'mask':output,'candidate_box':box,'qualified_box':box if not reasons else None,'receipt':receipt}
