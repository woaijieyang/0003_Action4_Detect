"""Finite 1D RGB-gradient profiles, not foreground/background region likelihood."""
import numpy as np
from scipy.signal import find_peaks,peak_widths

def normal_profile_matches(gx,gy,allowed,uv,tangent2,*,radius,step_px,tangent_half_width_px,
                           min_side_support_px,min_gradient,min_prominence,max_peak_width_px,
                           ambiguity_ratio,orientation_max_degrees=None,sampling_valid=None):
    """Unknown remains unmatched; no nearest-point fallback, no mask-derived edge."""
    if gx.shape!=gy.shape or gx.shape!=allowed.shape or allowed.dtype!=np.bool_:
        raise ValueError("Matching gradient/support arrays required")
    independent=sampling_valid is not None
    if independent and (sampling_valid.shape!=allowed.shape or sampling_valid.dtype!=np.bool_):
        raise ValueError("RGB sampling validity must share native pixel domain")
    sample_domain=allowed if sampling_valid is None else sampling_valid
    h,w=allowed.shape
    offsets=np.linspace(-radius,radius,int(np.ceil(2*radius/step_px))+1)
    actual_step=float(offsets[1]-offsets[0])
    cross=np.linspace(-tangent_half_width_px,tangent_half_width_px,
                      max(1,int(np.ceil(2*tangent_half_width_px))+1))
    if tangent_half_width_px==0:cross=np.array([0.])
    output_xy=np.zeros((len(uv),2));output_normal=np.zeros((len(uv),2));matched=np.zeros(len(uv),bool)
    diagnostics=[];side=int(np.ceil(min_side_support_px/actual_step))
    for row,(center,tangent) in enumerate(zip(uv,tangent2)):
        diagnostic={"status":"unknown","reason":None,"offsets_px":offsets.tolist(),"profile":[],
                    "support_valid":[],"peaks":[],"second_to_first_peak_ratio":None,"selected_peak":None}
        length=np.linalg.norm(tangent)
        if not np.isfinite(center).all() or not np.isfinite(length) or length<=1e-12:
            diagnostic["reason"]="undefined_projected_curve_normal";diagnostics.append(diagnostic);continue
        t=tangent/length;n=np.array([-t[1],t[0]])
        coords=center[None,None,:]+offsets[:,None,None]*n+cross[None,:,None]*t
        xy=np.rint(coords).astype(np.int64)
        inside=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)
        valid=np.zeros(inside.shape,bool);valid[inside]=sample_domain[xy[inside,1],xy[inside,0]]
        grad=np.zeros((*inside.shape,2))
        grad[inside]=np.c_[gx[xy[inside,1],xy[inside,0]],gy[xy[inside,1],xy[inside,0]]]
        strength=np.abs(grad@n)
        line_valid=valid.all(axis=1)
        profile=strength.mean(axis=1)
        diagnostic["raw_RGB_profile"]=profile.tolist()
        diagnostic["sampling_domain"]=("valid_RGB_separate_from_candidate_ownership" if independent else "legacy_candidate_support")
        profile[~line_valid]=0.
        diagnostic["profile"]=profile.tolist();diagnostic["support_valid"]=line_valid.tolist()
        if not line_valid.any():
            diagnostic["reason"]="insufficient_profile_support";diagnostics.append(diagnostic);continue
        peaks,properties=find_peaks(profile,height=min_gradient,prominence=min_prominence)
        widths=peak_widths(profile,peaks,rel_height=.5)[0]*actual_step if len(peaks) else []
        candidates=[]
        for pi,peak in enumerate(peaks):
            reasons=[]
            if peak-side<0 or peak+side>=len(offsets) or not line_valid[peak-side:peak+side+1].all():
                reasons.append("insufficient_two_side_support")
            if independent:
                margin=side+int(np.ceil(widths[pi]/(2*actual_step)))
                if peak-margin<0 or peak+margin>=len(offsets) or not line_valid[peak-margin:peak+margin+1].all():
                    reasons.append("truncated_RGB_profile")
            if widths[pi]>max_peak_width_px:reasons.append("broad_peak")
            ci=int(np.argmax(strength[peak]));point=xy[peak,ci].astype(float);g=grad[peak,ci]
            if not allowed[int(point[1]),int(point[0])]:reasons.append("peak_outside_candidate_ownership")
            norm=np.linalg.norm(g);angle=90. if norm<=1e-12 else float(np.degrees(np.arcsin(np.clip(abs(t@(g/norm)),0,1))))
            if np.linalg.norm(point-center)>radius:reasons.append("outside_original_radius")
            if norm<min_gradient:reasons.append("weak_actual_RGB_gradient")
            if orientation_max_degrees is not None and angle>orientation_max_degrees:reasons.append("orientation_conflict")
            item={"index":int(peak),"offset_px":float(offsets[peak]),"height":float(profile[peak]),
                  "prominence":float(properties["prominences"][pi]),"width_px":float(widths[pi]),
                  "pixel_xy":point.tolist(),"normal":(g/max(norm,1e-12)).tolist(),"angle_degrees":angle,
                  "reasons":reasons}
            diagnostic["peaks"].append(item)
            if not reasons:candidates.append(item)
        candidates.sort(key=lambda p:p["height"],reverse=True)
        if not candidates:
            diagnostic["reason"]="flat_or_weak_profile" if not len(peaks) else "all_peaks_unsupported_or_invalid"
        else:
            ratio=candidates[1]["height"]/candidates[0]["height"] if len(candidates)>1 else 0.
            diagnostic["second_to_first_peak_ratio"]=float(ratio)
            if len(candidates)>1 and ratio>=ambiguity_ratio:
                diagnostic["reason"]="ambiguous_multiple_RGB_peaks"
            else:
                chosen=candidates[0];matched[row]=True;output_xy[row]=chosen["pixel_xy"];output_normal[row]=chosen["normal"]
                diagnostic.update(status="matched_RGB_profile_peak",reason=None,selected_peak=chosen)
        diagnostics.append(diagnostic)
    return output_xy,output_normal,matched,diagnostics
