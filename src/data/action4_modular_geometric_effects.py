"""Pure geometric effect proposals and explicit checks; no scene mutation/render on import.

These requests are NOT action4.modular_parameters or accepted layout proposals.
A caller needs a separately bound config/consumer to apply whole-actor transforms.
"""
from copy import deepcopy
import numpy as np
SCHEMA='action4.modular_geometric_effect_request.v1'

def rigid(value):
 m=np.asarray(value,dtype=float)
 if m.shape!=(4,4) or not np.isfinite(m).all() or not np.array_equal(m[3],[0,0,0,1]) or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),rtol=0,atol=1e-6) or abs(np.linalg.det(m[:3,:3])-1)>1e-6:raise ValueError('Proper finite rigid transform only; no scale/mirror')
 return m

def transform(matrix,points):
 m=rigid(matrix);v=np.asarray(points,float)
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('Finite Nx3 points')
 return v@m[:3,:3].T+m[:3,3]

def project8(C,corners,K,wh=(3248,2464),near=.001):
 q=transform(C,corners);k=np.asarray(K,float)
 if q.shape!=(8,3) or k.shape!=(3,3) or not np.isfinite(k).all() or not (k[0,0]>0 and k[1,1]>0) or near<=0:raise ValueError('Original8 and finite pinhole K required')
 if np.any(q[:,2]<=near):raise ValueError('Near-plane rejection, no clamp')
 p=q@k.T;uv=p[:,:2]/p[:,2,None];lo=uv.min(0);hi=uv.max(0);w,h=wh
 return {'points_native_px':uv.tolist(),'depth_m':q[:,2].tolist(),'bbox_xyxy':[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],'bbox_area_fraction':float(np.prod(hi-lo)/(w*h)),'center_norm':((lo+hi)/2/[w,h]).tolist(),'corner_inframe':((uv[:,0]>=-.5)&(uv[:,0]<=w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<=h-.5)).tolist()}

def near_double_metrics(exports,wh=(3248,2464)):
 """f0-like large paired view; does not assert learned difficulty or occlusion."""
 if set(exports)!={'left','right'}:raise ValueError('Both real target exports required')
 rows=[exports[s] for s in ('left','right')];w,h=wh
 boxes=np.array([r['bbox_xyxy'] for r in rows]);widths=(boxes[:,2]-boxes[:,0])/w;areas=np.array([r['bbox_area_fraction'] for r in rows]);dx=abs(rows[1]['center_norm'][0]-rows[0]['center_norm'][0])
 passed=bool(np.all(widths>=.14)&np.all(areas>=.02)&(dx>=.20)&(dx<=.36)&all(all(r['corner_inframe']) for r in rows))
 return {'effect':'near_double','projection_condition_pass':passed,'bbox_width_fraction':widths.tolist(),'bbox_area_fraction':areas.tolist(),'center_horizontal_separation_fraction':float(dx),'definition':'Both large targets, f0-supported paired layout cluster; no difficulty claim','engineering_thresholds':{'min_width_fraction':.14,'min_area_fraction':.02,'horizontal_center_separation':[.20,.36]},'actual_collision_and_pixels_required':True,'training_accepted':False}

def _request(layout,family):
 if layout.get('schema')!='action4.modular_layout_proposal.v1' or layout.get('mode')!='layout_conditioned_preview':raise ValueError('Explicit upstream layout-conditioned proposal required')
 p=deepcopy(layout)
 return {'schema':SCHEMA,'status':'EFFECT_CANDIDATE_PENDING_EXPLICIT_CONSUMER','family':family,'input_layout_proposal':p,'effect_controls':{},'effect_evidence':{},'clears_family_pending':False,'whole_actor_only':True,'dataset_generation_allowed':False,'training_accepted':False}

def propose_near_double(layout,corners,K,wh=(3248,2464)):
 p=_request(layout,'near_double')
 if layout['target_presence']!=[True,True] or layout['layout_conditioning']['source_cluster_frame']!=0:raise ValueError('Use real f0 paired high-occupancy cluster')
 exports={s:project8(layout['sides'][s]['core_C_capture'],corners,K,wh) for s in ('left','right')};p['effect_evidence']=near_double_metrics(exports,wh)
 if not p['effect_evidence']['projection_condition_pass']:raise ValueError('Layout does not meet preregistered large paired effect')
 p['effect_controls']={'whole_actor_delta_capture_by_side':{s:np.eye(4).tolist() for s in ('left','right')},'reuse_upstream_paired_layout':True};return p

def propose_truncation(layout,corners,K,side='right',edge='right',outside_bbox_fraction=.15,wh=(3248,2464)):
 """Exact pinhole translation of entire selected actor; no clipping of geometry/GT.
 outside_bbox_fraction is pretranslation bbox fraction beyond selected edge,
 not visible-mask truncation. Actual final masks determine the latter.
 """
 p=_request(layout,'wall_transition_truncation')
 if side not in ('left','right') or not layout['sides'][side]['present'] or edge not in ('left','right','top','bottom') or not .05<=outside_bbox_fraction<=.25:raise ValueError('Explicit present side/edge and preregistered5-25percent control')
 C=rigid(layout['sides'][side]['core_C_capture']);B=rigid(layout['sides'][side]['brace_B_capture']);k=np.asarray(K,float);before=project8(C,corners,k,wh);v=transform(C,corners);uv=np.asarray(before['points_native_px']);axis=0 if edge in ('left','right') else 1;positive=edge in ('right','bottom');size=wh[axis];span=float(np.ptp(uv[:,axis]));boundary=size-.5 if positive else -.5;desired=boundary+(1 if positive else -1)*outside_bbox_fraction*span
 distances=(desired-uv[:,axis])*v[:,2]/k[axis,axis];shift=float(distances.min() if positive else distances.max());D=np.eye(4);D[axis,3]=shift;after=project8(D@C,corners,k,wh)
 if all(after['corner_inframe']) or not any(after['corner_inframe']):raise ValueError('Requires genuinely partial original8 in/out, not all visible/all out')
 p['effect_controls']={'side':side,'edge':edge,'outside_fraction_of_pretranslation_bbox':outside_bbox_fraction,'whole_actor_delta_capture':D.tolist(),'core_C_capture':(D@C).tolist(),'brace_B_capture':(D@B).tolist(),'translation_capture_m':D[:3,3].tolist(),'preserve_fixed_A_S_dimensions_K':True}
 p['effect_evidence']={'before':before,'after_projection_only':after,'declared_edge_extreme_px':desired,'actual_mask_truncation_unknown':True,'required_actual_checks':['full actor nearplane','internal/crossactor/environment collisions','positive visible target pixels','amodal_total>amodal_inframe>0','0<truncation_fraction<1 from actual masks'],'engineering_translation_not_measured_motion':True};return p

def orientation_effect_request(layout,side,C,B,orientation_provenance):
 p=_request(layout,'self_forearm_occlusion');C=rigid(C);B=rigid(B);oldC=rigid(layout['sides'][side]['core_C_capture']);oldB=rigid(layout['sides'][side]['brace_B_capture'])
 if not np.allclose(np.linalg.inv(B)@C,np.linalg.inv(oldB)@oldC,rtol=0,atol=1e-9):raise ValueError('Fixed A changed')
 p['effect_controls']={'side':side,'core_C_capture':C.tolist(),'brace_B_capture':B.tolist(),'whole_actor_delta_capture':(B@np.linalg.inv(oldB)).tolist(),'orientation_provenance':deepcopy(orientation_provenance),'requires_explicit_engineering_template_permission':True};return p

def nearest_ray_hit(direction,vertices,faces,near=.001):
 """Double-sided Moller-Trumbore; camera at zero; cache diagnostic, not a mask."""
 d=np.asarray(direction,float);d=d/np.linalg.norm(d);v=np.asarray(vertices,float);f=np.asarray(faces,int);tri=v[f];a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;p=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.sum(e1*p,axis=1);valid=np.abs(det)>1e-12;inv=np.zeros_like(det);inv[valid]=1/det[valid];tvec=-a;u=np.sum(tvec*p,axis=1)*inv;q=np.cross(tvec,e1);vv=q@d*inv;t=np.sum(e2*q,axis=1)*inv;valid&=(u>=-1e-10)&(vv>=-1e-10)&(u+vv<=1+1e-10)&(t>near)
 if not valid.any():return float('inf'),None
 t=np.where(valid,t,np.inf);i=int(np.argmin(t));return float(t[i]),i

def mask_effect_check(effect,record,instance_id,forearm_attribution=None):
 rows=[r for r in record['instances'] if r['instance_id']==instance_id]
 if len(rows)!=1:raise ValueError('Actual retained target instance required')
 r=rows[0]
 if effect=='truncation':
  # These are the existing mask_measurements names; do not infer from outside corners.
  total=r['area_full_amodal_px'];inside=r['area_inframe_amodal_px'];visible=r['area_visible_px'];passed=0<inside<total and visible>0
 elif effect=='self_forearm':
  if not forearm_attribution or forearm_attribution.get('actual_scene_rays') is not True:raise ValueError('Actual nearest-hit source attribution required, cache proof insufficient')
  passed=forearm_attribution.get('same_actor_forearm_first_hits',0)>0 and r['occlusion_inframe']>0 and r['area_visible_px']>0
 else:raise ValueError('Unknown actual effect check')
 return {'effect_condition_pass':bool(passed),'training_accepted':False,'not_production_acceptance':True}
