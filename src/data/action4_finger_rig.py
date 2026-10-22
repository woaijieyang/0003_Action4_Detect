"""Source-preserving G1 finger controls. Initialization only, no calibrated pose ranges."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test')
DIGITS={'index':(50,49,78,79),'middle':(45,47,53,77),'ring':(48,44,88,89),'little':(52,46,90,91),'thumb':(51,92)}
SIDES={'left':1,'right':-1}

def bounded(value,exists=False):
 import os
 p=Path(os.path.abspath(str(ROOT/value)))
 if not p.is_relative_to(ROOT):raise ValueError('Outside project')
 p=p.resolve(strict=exists)
 if not p.is_relative_to(ROOT):raise ValueError('Outside project via symlink')
 return p

def sha(path):return hashlib.sha256(bounded(path,True).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_new(path,value):
 p=bounded(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def validate_pose(pose,limit=25):
 keys={s+'_'+d for s in SIDES for d in DIGITS}
 if not isinstance(pose,dict) or set(pose)-keys:raise ValueError('Known finger angles only; no scale, wrist, mount or source overrides')
 for key,v in pose.items():
  if not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>limit for x in v):raise ValueError('Finite bounded engineering angles required')
 return pose

def validate_config(c,check_files=True):
 keys={'schema_version','status','dataset_generation_allowed','source_asset','source_contract','digits','palm_region','thumb_split_closed_ring','joint_blend_halfwidth_m','root_transition_length_m','diagnostic_input_limit_degrees','calibrated_sampling_ranges','proxy_claim','diagnostic_poses'}
 if set(c)!=keys or c['schema_version']!='action4.finger_rig_init.v1':raise ValueError('Exact finger initialization schema required')
 if c['status']!='initialization_pending_fit' or c['dataset_generation_allowed'] is not False or c['calibrated_sampling_ranges'] is not None:raise ValueError('Initialization only; no calibrated ranges or generation approval')
 if c['digits']!={k:list(v) for k,v in DIGITS.items()} or c['palm_region']!=93 or c['thumb_split_closed_ring']!=6:raise ValueError('Source topology mapping requires independent review to change')
 if c['joint_blend_halfwidth_m']!=.004 or c['root_transition_length_m']!=.008 or c['diagnostic_input_limit_degrees']!=25:raise ValueError('Versioned engineering skinning/control priors changed')
 if c['proxy_claim']!='Source face-set boundary/ring centroids are initial control proxies, not measured anatomical joints':raise ValueError('Proxy scope changed')
 if not isinstance(c['diagnostic_poses'],list):raise ValueError('Explicit diagnostic poses required')
 for p in c['diagnostic_poses']:
  if set(p)!={'name','pose'}:raise ValueError('Unexpected diagnostic parameters')
  validate_pose(p['pose'])
 if check_files:
  for k in ['source_asset','source_contract']:
   ref=c[k]
   if set(ref)!={'path','sha256'} or sha(ref['path'])!=ref['sha256']:raise ValueError('Source identity changed')
  contract=json.loads(bounded(c['source_contract']['path'],True).read_text())
  if contract['asset']!=c['source_asset'] or contract['dataset_generation_allowed'] is not False:raise ValueError('G1 contract mismatch')
 return c

def source_mapping(vertices,faces,face_sets):
 """Build controls from observed surface boundaries; never claim anatomical joints."""
 import numpy as np
 v=np.asarray(vertices,float);mapping={}
 for side,sign in SIDES.items():
  sets={}
  for face,region in zip(faces,face_sets):
   if all(v[i,0]*sign>0 for i in face):sets.setdefault(region,set()).update(face)
  for digit,regions in DIGITS.items():
   root=sorted(sets[93]&sets[regions[0]])
   if len(root)!=12:raise ValueError('Expected source palm-digit 12-vertex ring')
   boundaries=[root];joints=[v[root].mean(0)]
   if digit!='thumb':
    for a,b in zip(regions[:2],regions[1:3]):
     ring=sorted(sets[a]&sets[b])
     if len(ring)!=12:raise ValueError('Expected phalange proxy 12-vertex ring')
     boundaries.append(ring);joints.append(v[ring].mean(0))
   else:
    allowed=sets[51];adj={i:set() for i in allowed}
    for face,region in zip(faces,face_sets):
     if region!=51 or not all(i in allowed for i in face):continue
     for a,b in zip(face,face[1:]+face[:1]):adj[a].add(b);adj[b].add(a)
    seen=set(root);ring=set(root)
    for _ in range(6):ring=set().union(*(adj[i] for i in ring))-seen;seen|=ring
    if len(ring)!=12 or any(len(adj[i]&ring)!=2 for i in ring):raise ValueError('Thumb ring6 is not a verified closed12 loop')
    boundaries.append(sorted(ring));joints.append(v[list(ring)].mean(0))
   # Nail patch is a surface feature; tip proxy is distal surface extremity, not another phalanx.
   nail=sorted(sets[regions[-1]]);direction=v[nail].mean(0)-joints[-1];direction/=np.linalg.norm(direction)
   distal=sorted(sets[regions[-2]]|sets[regions[-1]]);projections=(v[distal]-joints[-1])@direction
   tip_ids=[distal[i] for i in np.argsort(projections)[-max(3,len(distal)//20):]]
   joints.append(v[tip_ids].mean(0));joints=np.array(joints)
   lengths=np.linalg.norm(np.diff(joints,axis=0),axis=1)
   if np.any(lengths<.008) or np.any(lengths>.06):raise ValueError('Unexpected proxy segment dimensions')
   ids=sorted(set().union(*(sets[r] for r in regions)))
   inward=np.array([-sign,0.,0.]);axes=[]
   for a,b in zip(joints[:-1],joints[1:]):
    y=(b-a)/np.linalg.norm(b-a);z=inward-y*np.dot(y,inward);z/=np.linalg.norm(z);x=np.cross(y,z);axes.append({'local_X_flexion_world':x.tolist(),'local_Y_longitudinal_world':y.tolist(),'local_Z_inward_world':z.tolist()})
   mapping[side+'_'+digit]={'side':side,'digit':digit,'region_ids':list(regions),'vertex_ids':ids,'joint_boundary_vertex_ids':boundaries,'tip_proxy_vertex_ids':tip_ids,'joints_m':joints.tolist(),'lengths_m':lengths.tolist(),'rest_axes':axes,'joint_definition':'Surface topology proxy; thumb ring6 location ambiguous anatomically; not measured CMC/MCP/IP coordinates'}
 # Same semantic opposition sign on both hands, derived from source geometry.
 for side in SIDES:
  m=mapping[side+'_thumb'];root=np.array(m['joints_m'][0]);tip=np.array(m['joints_m'][-1]);target=np.array(mapping[side+'_index']['joints_m'][0])-root;axis=np.array(m['rest_axes'][0]['local_Z_inward_world'])
  toward=float(np.dot(np.cross(axis,tip-root),target))
  if abs(toward)<1e-8:raise ValueError('Thumb opposition direction ambiguous')
  m['root_opposition_sign_toward_index']=1 if toward>0 else -1
  m['root_opposition_axis_world']=(axis*m['root_opposition_sign_toward_index']).tolist()
 return mapping

def skin_fractions(point,joints,halfwidth=.004,root_length=.008):
 import numpy as np
 q=np.asarray(point);j=np.asarray(joints);lens=np.linalg.norm(np.diff(j,axis=0),axis=1);cumulative=np.r_[0,np.cumsum(lens)];candidates=[]
 for i,(a,b,L) in enumerate(zip(j[:-1],j[1:],lens)):
  t=float(np.dot(q-a,b-a)/(L*L));clamped=max(0.,min(1.,t));dist=float(np.linalg.norm(q-a-clamped*(b-a)));candidates.append((dist,cumulative[i]+clamped*L))
 arc=min(candidates)[1]
 def sm(x):x=max(0.,min(1.,x));return x*x*(3-2*x)
 root=sm(arc/root_length);trans=[sm((arc-x+halfwidth)/(2*halfwidth)) for x in cumulative[1:-1]]
 weights=[];remain=root
 for t in trans:weights.append(remain*(1-t));remain*=t
 weights.append(remain)
 return [1-root,*weights]

def finger_names(mapping):return [key+'_'+str(i) for key,m in mapping.items() for i in range(len(m['lengths_m']))]

def apply_pose(rig,pose,mapping):
 """Non-thumb [MCPproxy,PIPproxy,DIPproxy]; thumb [rootflex,tipflex,oppositionTowardIndex]; all in degrees."""
 import bpy
 validate_pose(pose)
 for key,m in mapping.items():
  angles=pose.get(key,[0,0,0]);n=len(m['lengths_m'])
  for i in range(n):
   b=rig.pose.bones[key+'_'+str(i)]
   if tuple(b.scale)!=(1.,1.,1.) or any(abs(x)>1e-9 for x in b.location):raise ValueError('Finger translation/scaling forbidden')
 # Validate all controls before mutating.
 for key,m in mapping.items():
  angles=pose.get(key,[0,0,0])
  for i in range(len(m['lengths_m'])):
   b=rig.pose.bones[key+'_'+str(i)];b.rotation_mode='XYZ'
   b.rotation_euler=(math.radians(angles[i]),0,math.radians(angles[2]*m['root_opposition_sign_toward_index']) if m['digit']=='thumb' and i==0 else 0)
 bpy.context.view_layer.update()

def evaluated(body):
 import bpy,numpy as np
 obj=body.evaluated_get(bpy.context.evaluated_depsgraph_get());me=obj.to_mesh()
 try:
  me.calc_loop_triangles();return np.array([list(v.co) for v in me.vertices]),np.array([list(t.vertices) for t in me.loop_triangles],int)
 finally:obj.to_mesh_clear()

def nonadjacent_overlaps(vertices,triangles,face_ids):
 from mathutils.bvhtree import BVHTree
 subset=[triangles[i].tolist() for i in face_ids];tree=BVHTree.FromPolygons(vertices.tolist(),subset,all_triangles=True,epsilon=0)
 return {tuple(sorted((face_ids[i],face_ids[j]))) for i,j in tree.overlap(tree) if i<j and not set(subset[i])&set(subset[j])}

def check_pose(body,rig,mapping,rest,triangles,baseline,camera_world,old_bones,pose):
 import numpy as np
 apply_pose(rig,pose,mapping);current,tris=evaluated(body)
 if not np.array_equal(tris,triangles):raise ValueError('Evaluated topology changed')
 delta=np.linalg.norm(current-rest,axis=1);allowed=set().union(*(set(mapping[k]['evaluated_vertex_ids']) for k in pose)) if pose else set()
 outside=[i for i in range(len(rest)) if i not in allowed]
 if np.max(delta[outside],initial=0)>1e-6:raise ValueError('Non-target evaluated skin moved max_m='+str(float(np.max(delta[outside]))))
 for name,row in old_bones.items():
  b=rig.data.bones[name]
  if float(b.length)!=row['length'] or not np.array_equal(np.array(b.matrix_local),np.array(row['matrix'])):raise ValueError('Original G1 bone changed')
 for key,m in mapping.items():
  for i,L in enumerate(m['lengths_m']):
   b=rig.pose.bones[key+'_'+str(i)]
   if abs((b.tail-b.head).length-L)>1e-6:raise ValueError('Finger length changed')
 import bpy
 for name,M in camera_world.items():
  if not np.array_equal(np.array(bpy.data.objects[name].matrix_world),M):raise ValueError('Finger moved fixed camera or brace')
 active=np.flatnonzero(np.any(np.isin(triangles,list(allowed)),axis=1)) if allowed else np.array([],int)
 ratios=[];dots=[]
 if len(active):
  t=triangles[active];a=np.cross(rest[t[:,1]]-rest[t[:,0]],rest[t[:,2]]-rest[t[:,0]]);b=np.cross(current[t[:,1]]-current[t[:,0]],current[t[:,2]]-current[t[:,0]])
  A=np.linalg.norm(a,axis=1);B=np.linalg.norm(b,axis=1);valid=A>1e-12
  ratios=B[valid]/A[valid];dots=np.einsum('ij,ij->i',a[valid],b[valid])/(A[valid]*B[valid]+1e-30)
  # Conservative SMALL-angle diagnostic only. Rest-vs-posed normal reversal
  # is NOT a general deformation inversion test for large joint rotations.
  if min(ratios)<.2 or max(ratios)>5 or min(dots)<0:
   diagnostics={'area_ratio_min':float(min(ratios)),'area_ratio_max':float(max(ratios)),'min_rest_posed_normal_dot':float(min(dots)),'min_area_triangle_id':int(active[valid][np.argmin(ratios)]),'min_normal_triangle_id':int(active[valid][np.argmin(dots)]),'area_rejected':bool(min(ratios)<.2 or max(ratios)>5),'normal_small_angle_screen_rejected':bool(min(dots)<0),'pose':pose,'warning':'Normal dot screens these small-angle diagnostics only, not general anatomical pose legality or orientation-preserving deformation'}
   raise ValueError('CONSERVATIVE_SMALL_ANGLE_DIAGNOSTIC_REJECTION '+json.dumps(diagnostics))
 overlaps=nonadjacent_overlaps(current,triangles,baseline['hand_triangles']);new=overlaps-baseline['overlaps']
 if new:raise ValueError('New non-adjacent hand triangle intersections: '+str(len(new)))
 tips={key:np.array(rig.pose.bones[key+'_'+str(len(m['lengths_m'])-1)].tail) for key,m in mapping.items()}
 return {'skin_max_displacement_m':float(delta.max()),'outside_selected_digit_max_displacement_m':float(np.max(delta[outside],initial=0)),'active_triangle_min_area_ratio':float(min(ratios)) if len(ratios) else None,'active_triangle_max_area_ratio':float(max(ratios)) if len(ratios) else None,'active_triangle_min_normal_dot':float(min(dots)) if len(dots) else None,'new_nonadjacent_hand_intersections':len(new),'baseline_nonadjacent_hand_intersections':len(baseline['overlaps']),'thumb_index_tip_distance_m':{s:float(np.linalg.norm(tips[s+'_thumb']-tips[s+'_index'])) for s in SIDES}}

def build(config_path,output_dir,report_dir):
 import bpy,numpy as np
 from mathutils import Vector
 if Path.cwd().resolve()!=ROOT:raise ValueError('Exact project root required')
 cfg=validate_config(json.loads(bounded(config_path,True).read_text()));out=bounded(output_dir);reports=bounded(report_dir)
 if not out.is_relative_to(ROOT/'data/action4_scene_assets/finger_rig_init_v001') or not reports.is_relative_to(ROOT/'outputs/action4_realism_reassessment_v001/finger_rig_init_v001'):raise ValueError('Assigned directories only')
 if out.exists() or (reports/'report.json').exists():raise FileExistsError('New asset/evidence version required')
 reports.mkdir(parents=True,exist_ok=True)
 bpy.ops.wm.open_mainfile(filepath=str(bounded(cfg['source_asset']['path'],True)),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;scene.render.threads_mode='FIXED';scene.render.threads=2
 body=bpy.data.objects['global_source_anatomy'];rig=bpy.data.objects['global_metric_armature'];source_geometry={'vertices':[list(v.co) for v in body.data.vertices],'faces':[list(p.vertices) for p in body.data.polygons]};source_geometry_hash=digest(source_geometry)
 face_sets=[x.value for x in body.data.attributes['.sculpt_face_set'].data];mapping=source_mapping(source_geometry['vertices'],source_geometry['faces'],face_sets)
 rest,triangles=evaluated(body);old_bones={b.name:{'length':float(b.length),'matrix':[list(r) for r in b.matrix_local]} for b in rig.data.bones}
 if len(old_bones)!=7 or any(np.max(abs(np.array(b.matrix_basis)-np.eye(4)))>1e-7 for b in rig.pose.bones):raise ValueError('Expected neutral original7boneG1')
 camera_world={n:np.array(bpy.data.objects[n].matrix_world) for n in ['left_Action4_core','right_Action4_core','left_brace_frame','right_brace_frame']}
 original_weights={v.index:{body.vertex_groups[g.group].name:float(g.weight) for g in v.groups} for v in body.data.vertices}
 bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
 for key,m in mapping.items():
  parent=rig.data.edit_bones[m['side']+'_hand'];j=m['joints_m']
  for i in range(len(j)-1):
   b=rig.data.edit_bones.new(key+'_'+str(i));b.head=j[i];b.tail=j[i+1];b.parent=parent;b.use_connect=i>0;b.align_roll(Vector(m['rest_axes'][i]['local_Z_inward_world']));parent=b
 bpy.ops.object.mode_set(mode='OBJECT')
 for name in finger_names(mapping):rig.data.bones[name].inherit_scale='NONE';body.vertex_groups.new(name=name)
 modified=[]
 for key,m in mapping.items():
  hand=m['side']+'_hand';names=[hand]+[key+'_'+str(i) for i in range(len(m['lengths_m']))]
  for idx in m['vertex_ids']:
   old=original_weights[idx];amount=old.get(hand,0)
   if amount<=0:raise ValueError('Digit missing parent hand influence')
   fractions=skin_fractions(source_geometry['vertices'][idx],m['joints_m'])
   for name,w in zip(names,fractions):
    group=body.vertex_groups[name]
    if w>0:group.add([idx],float(amount*w),'REPLACE')
    else:group.remove([idx])
   modified.append(idx)
 bpy.context.view_layer.update();current,tris=evaluated(body)
 if np.max(abs(current-rest))>1e-6:raise ValueError('Neutral evaluated surface changed; max_m='+str(float(np.max(abs(current-rest)))))
 for v in body.data.vertices:
  if abs(sum(x.weight for x in v.groups)-1)>1e-6:raise ValueError('Unnormalized skin')
 # Source has Multires before Armature: evaluated IDs differ from control IDs.
 # Read actual interpolated deform weights to test isolation on the real surface.
 ev=body.evaluated_get(bpy.context.evaluated_depsgraph_get());em=ev.to_mesh()
 try:
  for key,m in mapping.items():
   group_ids={body.vertex_groups[key+'_'+str(i)].index for i in range(len(m['lengths_m']))}
   m['evaluated_vertex_ids']=[v.index for v in em.vertices if any(g.group in group_ids and g.weight>0 for g in v.groups)]
  hand_ids={body.vertex_groups[s+'_hand'].index for s in SIDES}|{body.vertex_groups[n].index for n in finger_names(mapping)}
  all_hand={v.index for v in em.vertices if any(g.group in hand_ids and g.weight>0 for g in v.groups)}
 finally:ev.to_mesh_clear()
 hand_triangles=[i for i,t in enumerate(triangles) if any(int(j) in all_hand for j in t)]
 baseline={'hand_triangles':hand_triangles,'overlaps':nonadjacent_overlaps(rest,triangles,hand_triangles)}
 results=[];neutral=check_pose(body,rig,mapping,rest,triangles,baseline,camera_world,old_bones,{})
 for case in cfg['diagnostic_poses']:
  result=check_pose(body,rig,mapping,rest,triangles,baseline,camera_world,old_bones,case['pose']);results.append({'name':case['name'],'pose':case['pose'],'check':result})
  points,_=evaluated(body);write_new(reports/(case['name']+'_geometry.json'),{'vertices':points.tolist(),'faces':triangles.tolist(),'mapping_evaluated_support':{k:m['evaluated_vertex_ids'] for k,m in mapping.items()}})
 apply_pose(rig,{},mapping);again,_=evaluated(body)
 if np.max(abs(again-rest))>1e-6 or digest({'vertices':[list(v.co) for v in body.data.vertices],'faces':[list(p.vertices) for p in body.data.polygons]})!=source_geometry_hash:raise ValueError('Neutral/source geometry not preserved')
 validate_config(cfg)
 result={'status':'INITIALIZATION_ENGINEERING_ONLY_PENDING_REVIEW','dataset_generation_allowed':False,'calibrated_sampling_ranges':None,'config':cfg,'mapping':mapping,'original_G1_bones':old_bones,'source_control_geometry_sha256':source_geometry_hash,'neutral_evaluated_max_error_m':float(np.max(abs(again-rest))),'neutral_evaluated_error_distribution':{'max_vertex_L2_m':float(np.linalg.norm(again-rest,axis=1).max()),'mean_vertex_L2_m':float(np.linalg.norm(again-rest,axis=1).mean()),'p95_vertex_L2_m':float(np.quantile(np.linalg.norm(again-rest,axis=1),.95)),'count_nonzero':int(np.count_nonzero(np.linalg.norm(again-rest,axis=1))),'evaluated_vertex_count':len(rest),'signed_mean_xyz_m':np.mean(again-rest,axis=0).tolist(),'numerical_tolerance_m':1e-6},'original7bone_matrices_lengths_exact':True,'fixed_brace_camera_world_exact':True,'bone_count':len(rig.data.bones),'modified_vertex_count':len(set(modified)),'neutral_check':neutral,'diagnostic_checks':results,'fixed_camera_and_brace_world':{k:v.tolist() for k,v in camera_world.items()},'limitations':['Proxy joints from source topology, not anatomical measurement or video fit','Thumb ring6 is an explicitly uncertain intermediate control location; two-segment thumb proxy, no independently identified CMC','Limited diagnostic poses only; calibrated ranges null; rest-normal dot is only conservative small-angle screening, not general pose legality','Source Multires baseline has336 nonadjacent hand triangle intersections, largely distal/nail region; zero NEW pairs does not mean self-intersection-free','Approach/grasp candidates do not establish actual contact; real pinch/cloth handling remains required downstream','No real contact, grasp success, photoreal rendering, dataset or training acceptance']}
 scene['finger_rig_status']='initialization_pending_fit';scene['finger_dataset_generation_allowed']=False
 bpy.data.texts.new('ACTION4_FINGER_RIG_INITIALIZATION.json').write(json.dumps(result,indent=2))
 out.mkdir(parents=True);asset=out/'finger_rig_initial.blend';bpy.ops.wm.save_as_mainfile(filepath=str(asset),check_existing=False)
 result['saved_asset']={'path':str(asset.relative_to(ROOT)),'sha256':sha(asset)}
 write_new(out/'finger_contract.json',result);write_new(reports/'report.json',result)
 return result

def main():
 import sys,traceback
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True)
 args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
 try:
  r=build(args.config,args.output,args.report);print(json.dumps({'status':r['status'],'asset':r['saved_asset']}))
 except Exception:
  f=bounded(args.report)/'failure.json'
  if not f.exists():write_new(f,{'status':'REJECTED_INITIALIZATION','traceback':traceback.format_exc()})
  raise
