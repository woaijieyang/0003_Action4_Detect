"""Version2: relative near-double semantics and explicit engineering shaft views.
No scene/pose/render/search on import. Old15 failures and v1 module stay intact.
"""
import numpy as np
from data import action4_modular_geometric_effects as base
rigid,transform,project8=base.rigid,base.transform,base.project8
nearest_ray_hit=base.nearest_ray_hit
propose_truncation=base.propose_truncation
orientation_effect_request=base.orientation_effect_request
SHAFT_SECTIONS_M=(-.06,-.12,-.18)
ENGINEERING_CORE_DISTANCE_M=.55

def high_area_pair_metrics(exports,wh=(3248,2464)):
 result=base.near_double_metrics(exports,wh);result['effect']='high_area_pair_scale_diagnostic';result['counts_as_near_double']=False;return result

def near_double_metrics(exports,wh=(3248,2464)):
 if set(exports)!={'left','right'}:raise ValueError('Two projected targets required')
 left,right=(exports[x] for x in ('left','right'));a,b=np.array(left['bbox_xyxy']),np.array(right['bbox_xyxy']);w,h=wh
 dx=abs(left['center_norm'][0]-right['center_norm'][0]);edgegap=float(max(a[0],b[0])-min(a[2],b[2]))/w;verticalgap=float(max(a[1],b[1])-min(a[3],b[3]))/h
 passed=dx<=.15 and edgegap<=.06 and verticalgap<=.02 and all(left['corner_inframe']) and all(right['corner_inframe'])
 return {'effect':'near_double','projection_condition_pass':bool(passed),'center_horizontal_distance_fraction':dx,'horizontal_signed_bbox_edge_gap_fraction':edgegap,'vertical_signed_bbox_edge_gap_fraction':verticalgap,'negative_gap_means_2D_overlap_not_3D_collision':True,'reference_frames':[540,1200,1590],'engineering_bounds':{'max_center_horizontal_distance':.15,'max_horizontal_edge_gap':.06,'max_vertical_edge_gap':.02},'not_large_area_definition':True,'actual_nonintersection_and_two_distinguishable_visible_masks_required':True,'training_accepted':False}

def propose_near_double(layout,corners,K,wh=(3248,2464)):
 request=base._request(layout,'near_double');frame=layout['layout_conditioning']['source_cluster_frame']
 if layout['target_presence']!=[True,True] or frame not in [540,1200,1590]:raise ValueError('Near-double uses audited close paired clusters; f0 higharea is not near-double')
 evidence=near_double_metrics({s:project8(layout['sides'][s]['core_C_capture'],corners,K,wh) for s in ('left','right')},wh)
 if not evidence['projection_condition_pass']:raise ValueError('Pair fails relative-neighbor projection condition')
 request['effect_controls']={'reuse_upstream_paired_layout':True,'whole_actor_delta_capture_by_side':{s:np.eye(4).tolist() for s in ('left','right')}};request['effect_evidence']=evidence;return request

def shaft_sections(vertices,faces,source_regions):
 V=np.asarray(vertices,float);F=np.asarray(faces,int);regions=np.asarray(source_regions)
 if regions.shape!=(len(F),) or not np.isfinite(V).all():raise ValueError('Exact mesh/region identity required')
 triangles=V[F[regions==102]];out=[]
 for y in SHAFT_SECTIONS_M:
  points=[]
  for tri in triangles[(triangles[:,:,1].min(1)<=y)&(triangles[:,:,1].max(1)>=y)]:
   for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
    if (a[1]-y)*(b[1]-y)<0:points.append(a+(b-a)*(y-a[1])/(b[1]-a[1]))
    elif a[1]==y:points.append(a.copy())
  if len(points)<6:raise ValueError('Missing real forearm section; no synthesized fallback')
  p=np.unique(np.round(points,12),axis=0);lo=p.min(0);hi=p.max(0);center=(lo+hi)/2;center[1]=y;half=(hi-lo)/2
  if half[0]<=0 or half[2]<=0:raise ValueError('Degenerate shaft cross section')
  out.append({'y_m':y,'source_region':102,'intersection_points_m':p.tolist(),'center_m':center.tolist(),'half_width_x_m':float(half[0]),'half_width_z_m':float(half[2]),'center_is_section_bbox_proxy_not_anatomical_measurement':True})
 return out

def engineering_shaft_views(vertices,faces,source_regions,A):
 """Exactly3section x3directions. Eye distance is fixed engineering. No scoring/search."""
 A=rigid(A);q=A[:3,3];out=[]
 for section in shaft_sections(vertices,faces,source_regions):
  for edge in [-1,0,1]:
   aim=np.array(section['center_m']);aim[0]+=edge*section['half_width_x_m'];direction=aim-q;direction/=np.linalg.norm(direction);eye=q+ENGINEERING_CORE_DISTANCE_M*direction;z=-direction;x=np.array([1.,0,0]);x-=z*(x@z)
   if np.linalg.norm(x)<1e-8:raise ValueError('Undefined source-right basis; no orientation fallback')
   x/=np.linalg.norm(x);y=np.cross(z,x);B=np.eye(4);B[:3,:3]=np.array([x,y,z]);B[:3,3]=-B[:3,:3]@eye;C=B@A;rigid(B)
   out.append({'id':f"shaft_y{section['y_m']:.2f}_edge{edge:+d}",'section':section,'edge':edge,'aim_point_brace_m':aim.tolist(),'eye_brace_m':eye.tolist(),'B_capture':B.tolist(),'C_capture':C.tolist(),'core_distance_m':ENGINEERING_CORE_DISTANCE_M,'source_right_projected_into_imageX':True,'engineering_only':True,'measured_orientation':False,'counts_toward_reference':False})
 if len(out)!=9:raise ValueError('Exact9only')
 return out

def actual_near_double_check(record,collision_report):
 rows=record['instances']
 if len(rows)!=2 or {r['instance_id'] for r in rows}!={1,2}:raise ValueError('Both original real target records required')
 if collision_report.get('outside_or_unresolved_surface_pairs')!=0:raise ValueError('No3Dintersection bypass for 2Dnear/overlap')
 w,h=record['width'],record['height'];exports={}
 for row in rows:
  pts=np.array(row['canonical8_geometry']['points_native_px']);lo=pts.min(0);hi=pts.max(0);exports['left' if row['instance_id']==1 else 'right']={'bbox_xyxy':[*lo,*hi],'center_norm':((lo+hi)/2/[w,h]).tolist(),'corner_inframe':row['canonical8_geometry']['point_inframe']}
 condition=near_double_metrics(exports,(w,h));condition['actual_two_mask_support_pass']=all(r['area_visible_px']>0 and r.get('ignore') is False for r in rows);condition['effect_condition_pass']=condition['projection_condition_pass'] and condition['actual_two_mask_support_pass'];return condition
