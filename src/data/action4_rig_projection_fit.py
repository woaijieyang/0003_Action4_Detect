"""Bounded G2a projection candidates. Not PnP, not calibrated/fitted production scene."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path("/data/zy/vln/a-test")
OBS="outputs/action4_realism_reassessment_v001/rig_relation/calibration_observations/observations.json"
CONTRACT="data/action4_scene_assets/global_rig_init_v001/validated_build_v002/rig_contract.json"
PRIORS="outputs/action4_realism_reassessment_v001/anatomy_scale/source_bundle_inventory"
G1_CONFIG="configs/data/action4_global_rig_init_v001.json"
TRACKS=("L","R")
SIDES={"L":"left","R":"right"}
SUPPORTED={"body_edge","brace_visible_edge","forearm_axis"}


def path(p):
 p=(ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError("Outside project")
 return p


def sha(p):
 h=hashlib.sha256()
 with path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()


def write(p,value):
 p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)


def transform(v):
 T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(v[:3]).as_matrix();T[:3,3]=v[3:6];return T


def apply(T,p):return np.asarray(p)@T[:3,:3].T+T[:3,3]


def project(p,f,wh):
 p=np.asarray(p);z=p[:,2]
 return p[:,:2]/np.maximum(z[:,None],.005)*f+(np.array(wh)-1)/2


def segment_deltas(points,segments):
 a,b=segments[:,0],segments[:,1];d=b-a
 t=np.clip(((points[:,None,:]-a)*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1)
 return points[:,None,:]-(a+t[...,None]*d)


def obb_gap(Ta,ha,Tb,hb):
 Ra,Rb=Ta[:3,:3],Tb[:3,:3];delta=Tb[:3,3]-Ta[:3,3]
 cross=np.cross(Ra.T[:,None,:],Rb.T[None,:,:]).reshape(-1,3)
 axes=np.concatenate([Ra.T,Rb.T,cross]);norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-8]/norm[norm>1e-8,None]
 gaps=np.abs(axes@delta)-np.abs(axes@Ra)@ha-np.abs(axes@Rb)@hb
 return float(gaps.max())


def prepare():
 obs=json.loads(path(OBS).read_text());contract=json.loads(path(CONTRACT).read_text())
 if sha(OBS)!="2bddfa1c9a462ec0f448a28440a7a81cd42a3318c26e7013391b1364d1746240":raise ValueError("Observation identity changed")
 g1config=json.loads(path(G1_CONFIG).read_text())
 if g1config["body"]!=contract["source_body"] or g1config["camera"]!=contract["source_camera"]:raise ValueError("G1 source prior config differs")
 if contract["dataset_generation_allowed"] is not False:raise ValueError("Expected unaccepted G1")
 raw=dict(contract);digest=raw.pop("invariants_sha256")
 if hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()!=digest:raise ValueError("Rig contract hash changed")
 if sha(contract["asset"]["path"])!=contract["asset"]["sha256"]:raise ValueError("Rig asset changed")
 verts=np.load(path(PRIORS+"/male_body_control_geometry.npz"),allow_pickle=False)["local"]
 faces=json.loads(path(PRIORS+"/male_body_faces.json").read_text())
 if hashlib.sha256(verts.tobytes()+json.dumps(faces).encode()).hexdigest()!=contract["source_geometry_sha256"]:raise ValueError("Source anatomy geometry differs from G1")
 fs=np.load(path(PRIORS+"/male_body_face_sets.npz"),allow_pickle=False)["face_set"]
 dims=np.array(contract["fixed_camera_mesh_xyz_m"]);half=dims/2
 corners=np.array([[x,y,z] for x in [-half[0],half[0]] for y in [-half[1],half[1]] for z in [-half[2],half[2]]])
 edges=[];adj=[]
 for i,a in enumerate(corners):
  for j,b in enumerate(corners):
   if j>i and np.sum(a!=b)==1:
    edges.append([i,j]);adj.append([(k,1 if a[k]>0 else -1) for k in range(3) if a[k]==b[k]])
 model={"obs":obs,"contract":contract,"wh":[3248,2464],"half":half,"corners":corners,"edges":np.array(edges),"edge_faces":adj,"sides":{}}
 for track,side in SIDES.items():
  b=contract["fixed_brace_and_camera_attachment"][side];T=np.array(b["rest_world_frame"])
  local=apply(np.linalg.inv(T),verts)
  sign=contract["source_body"]["side_sign"][side]
  ids={i for face,region in zip(faces,fs) if region==102 for i in face if verts[i,0]*sign>0}
  selected=local[list(ids)];bounds=np.linspace(selected[:,1].min(),selected[:,1].max(),5);boxes=[]
  for lo,hi in zip(bounds[:-1],bounds[1:]):
   band=selected[(selected[:,1]>=lo-.01)&(selected[:,1]<=hi+.01)]
   lower,upper=band.min(0),band.max(0);lower[1]=lo;upper[1]=hi
   box=np.eye(4);box[:3,3]=(lower+upper)/2;boxes.append((box,(upper-lower)/2))
  rx,rz=b["brace_inner_radii_m"];thick=g1config["brace"]["thickness_m"];width=g1config["brace"]["width_m"]
  boxes.append((np.eye(4),np.array([rx+thick,width/2,rz+thick])))
  theta=np.linspace(0,2*np.pi,73);rings={}
  for kind,yy in [("proximal",-width/2),("distal",width/2)]:
   rings[kind]=np.column_stack([(rx+thick)*np.cos(theta),np.full(len(theta),yy),(rz+thick)*np.sin(theta)])
  model["sides"][track]={"A0":np.array(b["A_brace_camera"]),"rings":rings,"boxes":boxes,"radii":[rx+thick,rz+thick]}
 for frame in obs["frames"]:
  if frame["source_size_wh"]!=model["wh"] or sha(frame["source"])!=frame["source_sha256"]:raise ValueError("Source frame identity changed")
  for o in frame["observations"]:
   if o.get("model_points_3d_mm") is not None:raise ValueError("Unexpected fixed3D correspondences")
 return model


def observations(model,frames,brace_only=False):
 return [(f["frame_id"],o) for f in model["obs"]["frames"] if f["frame_id"] in frames for o in f["observations"]
         if o.get("points_px") is not None and o["kind"] in ({"brace_visible_edge","forearm_axis"} if brace_only else SUPPORTED)]


def init_pose(rows,track,f,wh):
 local=[o for _,o in rows if o["track"]==track];curves=[o for o in local if o["kind"]=="brace_visible_edge"]
 allpoints=np.concatenate([o["points_px"] for o in curves]);uv=allpoints.mean(0)
 means={}
 for o in curves:means.setdefault("proximal" if o["semantic_boundary"].startswith("proximal") else "distal",[]).extend(o["points_px"])
 axes=[np.diff(np.array(o["points_px"]),axis=0)[0] for o in local if o["kind"]=="forearm_axis"]
 direction=axes[0] if axes else np.array([0.,-1.]);direction=direction/np.linalg.norm(direction)
 y=np.r_[direction*.94,.34];y/=np.linalg.norm(y);z=np.array([0.,0.,-1.]);z-=y*(z@y);z/=np.linalg.norm(z);x=np.cross(y,z)
 R=np.column_stack([x,y,z])
 if set(means)=={"proximal","distal"}:
  sep=np.linalg.norm(np.mean(means["proximal"],0)-np.mean(means["distal"],0));depth=np.clip(f*.028*.94/max(sep,30),.16,1.8)
 else:depth=np.clip(f*.065/max(np.ptp(allpoints,axis=0).max(),80),.16,1.8)
 translation=np.r_[(uv-(np.array(wh)-1)/2)*depth/f,depth]
 return np.r_[Rotation.from_matrix(R).as_rotvec(),translation]


def unpack(q,frames):
 f=np.exp(q[0]);A={t:transform(q[1+6*i:7+6*i]) for i,t in enumerate(TRACKS)}
 B={(frame,t):transform(q[13+6*j:19+6*j]) for j,(frame,t) in enumerate((fr,t) for fr in frames for t in TRACKS)}
 return f,A,B


def body_segments(model,C,f,face):
 vertices=apply(C,model["corners"]);uv=project(vertices,f,model["wh"])
 scores=[];select=[]
 for edge,faces in zip(model["edges"],model["edge_faces"]):
  visible=False
  for k,sign in faces:
   normal=C[:3,k]*sign;center=C[:3,3]+normal*model["half"][k]
   visible|=float(normal@center)<0
  if face=="rear_screen_face_+Y":
   eligible=(1,1) in faces
  elif face=="front_lens_face_-Y":
   eligible=(1,-1) in faces
  else:eligible=visible
  select.append(bool(eligible))
 segments=uv[model["edges"]]
 return segments,np.array(select),vertices


def residual_rows(model,rows,f,A,B):
 result=[];metadata=[]
 for frame,o in rows:
  track=o["track"];T=B[(frame,track)];kind=o["kind"];points=np.array(o["points_px"],float);sigma=o["manual_uncertainty_radius_px"];match=None
  if kind=="body_edge":
   segments,eligible,_=body_segments(model,T@A[track],f,o.get("body_local_face"))
   deltas=segment_deltas(points,segments);cost=(deltas*deltas).sum((0,2));cost[~eligible]+=1e12
   match=int(np.argmin(cost));delta=deltas[:,match,:]
  elif kind=="brace_visible_edge":
   boundary="proximal" if o["semantic_boundary"].startswith("proximal") else "distal"
   ring=model["sides"][track]["rings"][boundary];p=apply(T,ring);uv=project(p,f,model["wh"])
   mid=(ring[:-1]+ring[1:])/2;normal=np.column_stack([mid[:,0]/model["sides"][track]["radii"][0]**2,np.zeros(len(mid)),mid[:,2]/model["sides"][track]["radii"][1]**2])@T[:3,:3].T
   visible=(normal*apply(T,mid)).sum(1)<0
   deltas=segment_deltas(points,np.stack([uv[:-1],uv[1:]],1));cost=(deltas*deltas).sum(-1);cost[:,~visible]+=1e12
   idx=cost.argmin(1);delta=deltas[np.arange(len(points)),idx];match=idx.tolist()
  else:
   uv=project(apply(T,[[0,-.12,0],[0,0,0]]),f,model["wh"]);pred=uv[1]-uv[0];pred/=max(np.linalg.norm(pred),1e-9)
   observed=points[1]-points[0];length=np.linalg.norm(observed);observed/=length
   delta=((pred-observed)*length)[None,:] # equivalent pixel direction error only, never joint endpoint correspondence
  values=(delta/sigma).ravel();face_cos=None
  face=o.get("body_local_face")
  if kind=="body_edge" and face in ("rear_screen_face_+Y","front_lens_face_-Y"):
   C=T@A[track];sign=1 if face=="rear_screen_face_+Y" else -1
   normal=C[:3,1]*sign;center=C[:3,3]+normal*model["half"][1]
   face_cos=float(normal@center/max(np.linalg.norm(center),1e-9))
   values=np.r_[values,10*max(0,face_cos)]
  result.extend(values)
  metadata.append({"frame_id":frame,"id":o["id"],"track":track,"kind":kind,"pixel_equivalent_errors":np.linalg.norm(delta,axis=1).tolist(),"manual_tolerance_px":sigma,"latent_edge_or_ring_segment":match,"known_face_facing_cos":face_cos,"normalized_squared_sum":float(values@values)})
 return np.array(result),metadata


def physical(model,A,B):
 rows=[];labels=[]
 for t in TRACKS:
  for i,(box,half) in enumerate(model["sides"][t]["boxes"]):
   gap=obb_gap(A[t],model["half"],box,half);rows.append(20*max(0,.002-gap)/.005);labels.append({"constraint":"camera_vs_own_forearm_or_brace_proxy","track":t,"box":i,"separation_m":gap})
  distance=np.linalg.norm(A[t][:3,3]);rows.extend([20*max(0,.04-distance)/.01,20*max(0,distance-.15)/.01])
 for fr in sorted({k[0] for k in B}):
  if (fr,"L") in B and (fr,"R") in B:
   gap=obb_gap(B[(fr,"L")]@A["L"],model["half"],B[(fr,"R")]@A["R"],model["half"])
   rows.append(20*max(0,.002-gap)/.005);labels.append({"constraint":"two_camera_OBBs","frame_id":fr,"separation_m":gap})
  for t in TRACKS:
   if (fr,t) not in B:continue
   minz=apply(B[(fr,t)]@A[t],model["corners"])[:,2].min()
   rows.append(20*max(0,.03-minz)/.01);labels.append({"constraint":"positive_depth","frame_id":fr,"track":t,"minimum_depth_m":float(minz)})
 return np.array(rows),labels


def prior(q,model):
 r=[.05*(q[0]-math.log(3248*.8125))]
 for i,t in enumerate(TRACKS):
  v=q[1+6*i:7+6*i];A0=model["sides"][t]["A0"]
  r.extend(.05*Rotation.from_matrix(Rotation.from_rotvec(v[:3]).as_matrix()@A0[:3,:3].T).as_rotvec()/np.pi)
  r.extend(.1*(v[3:]-A0[:3,3])/.1)
 return np.array(r)


def summary(meta):
 groups={}
 for row in meta:
  key=f"{row['frame_id']}/{row['track']}/{row['kind']}";groups.setdefault(key,[]).extend(row["pixel_equivalent_errors"])
 return {k:{"count":len(v),"mean_px":float(np.mean(v)),"rms_px":float(np.sqrt(np.mean(np.square(v)))),"max_px":float(max(v))} for k,v in groups.items()}


def data_rank(fun,q):
 base=fun(q);J=np.empty((len(base),len(q)));steps=np.r_[1e-5,np.tile([1e-5]*3+[1e-6]*3,(len(q)-1)//6)]
 scales=np.r_[1.,np.tile([1.]*3+[.1]*3,(len(q)-1)//6)]
 for j,step in enumerate(steps):
  x=q.copy();x[j]+=step;J[:,j]=(fun(x)-base)/step
 values=np.linalg.svd(J*scales,compute_uv=False)
 return {"scope":"DATA residual Jacobian only, excluding all priors/barriers; finite-difference local latent matches",
         "shape":list(J.shape),"singular_values":values.tolist(),"column_scale":scales.tolist(),
         "rank_by_relative_threshold":{str(t):int(np.sum(values>values[0]*t)) for t in [1e-4,1e-6,1e-8]},
         "condition_number":float(values[0]/max(values[-1],1e-30)),"caution":"Partial-curve latent correspondence and local numerical rank do not prove global identifiability or statistical confidence."}


def sparsity(model,rows,frames):
 from scipy.sparse import lil_matrix
 size=61;patterns=[];global_ids=list(range(13))
 def block(frame,track):return list(range(13+6*(2*frames.index(frame)+TRACKS.index(track)),19+6*(2*frames.index(frame)+TRACKS.index(track))))
 for frame,o in rows:
  count=2 if o["kind"]=="forearm_axis" else 2*len(o["points_px"])
  if o["kind"]=="body_edge" and o.get("body_local_face") in ("rear_screen_face_+Y","front_lens_face_-Y"):count+=1
  patterns.extend([global_ids+block(frame,o["track"])]*count)
 for i,t in enumerate(TRACKS):patterns.extend([list(range(1+6*i,7+6*i))]*(len(model["sides"][t]["boxes"])+2))
 for frame in frames:
  patterns.append(global_ids+block(frame,"L")+block(frame,"R"))
  for t in TRACKS:patterns.append(global_ids+block(frame,t))
 patterns.extend([global_ids]*13)
 result=lil_matrix((len(patterns),size),dtype=int)
 for i,indices in enumerate(patterns):result[i,indices]=1
 return result.tocsr()


def run(output):
 out=path(output)
 if out.exists():raise FileExistsError("Use a fresh fit attempt directory")
 out.mkdir(parents=True);model=prepare();frames=model["obs"]["fit_frames"];rows=observations(model,frames)
 inputs={OBS:sha(OBS),CONTRACT:sha(CONTRACT),G1_CONFIG:sha(G1_CONFIG),**{PRIORS+"/"+p:sha(PRIORS+"/"+p) for p in ["male_body_control_geometry.npz","male_body_faces.json","male_body_face_sets.npz"]}}
 inputs["src/data/action4_rig_projection_fit.py"]=sha("src/data/action4_rig_projection_fit.py")
 config={"scope":"G2a bounded projection initialization candidates only","parameter_count":61,"fit_frames":frames,"held_out_frames":model["obs"]["held_out_frames"],"max_global_starts":4,"max_nfev_per_start":200,"f_over_width_engineering_search":[.4,1.6],"principal_point":((np.array(model["wh"])-1)/2).tolist(),"square_pixels":True,"zero_distortion_assumed":True,"camera_scale_fixed":True,"brace_dimensions_fixed_G1_prior":True,"seed":20261020,"inputs":inputs,"omitted":[{"frame":f["frame_id"],"id":o["id"],"kind":o["kind"],"reason":"null/unobservable" if o.get("points_px") is None else "unsupported pixel geometry correspondence in G2a"} for f in model["obs"]["frames"] for o in f["observations"] if o.get("points_px") is None or o["kind"] not in SUPPORTED]}
 write(out/"config.json",config)
 results=[]
 for start,(ratio,rx) in enumerate(zip([.5,.8,1.2,1.5],[90,0,180,90])):
  f=model["wh"][0]*ratio;q=[math.log(f)]
  for t in TRACKS:
   rotation=Rotation.from_euler("x",rx,degrees=True)
   if start==3:rotation=rotation*Rotation.from_euler("z",180,degrees=True)
   q.extend(np.r_[rotation.as_rotvec(),model["sides"][t]["A0"][:3,3]])
  for frame in frames:
   for t in TRACKS:q.extend(init_pose(observations(model,[frame]),t,f,model["wh"]))
  q=np.array(q);lo=np.r_[math.log(3248*.4),np.tile([-np.pi]*3+[-.15]*3,2),np.tile([-np.pi]*3+[-2,-2,.12],8)]
  hi=np.r_[math.log(3248*1.6),np.tile([np.pi]*3+[.15]*3,2),np.tile([np.pi]*3+[2,2,3.],8)]
  q=np.clip(q,lo+1e-8,hi-1e-8);calls=[0];started=time.perf_counter()
  def data(v):
   ff,aa,bb=unpack(v,frames);return residual_rows(model,rows,ff,aa,bb)[0]
  def cost(v):
   calls[0]+=1;ff,aa,bb=unpack(v,frames)
   return np.r_[residual_rows(model,rows,ff,aa,bb)[0],physical(model,aa,bb)[0],prior(v,model)]
  initial=unpack(q,frames);initial_values,initial_meta=residual_rows(model,rows,*initial)
  try:
   fit=least_squares(cost,q,bounds=(lo,hi),max_nfev=200,jac_sparsity=sparsity(model,rows,frames),loss="soft_l1",f_scale=1.,x_scale="jac",ftol=1e-7,xtol=1e-7,gtol=1e-7)
   ff,aa,bb=unpack(fit.x,frames);final_values,meta=residual_rows(model,rows,ff,aa,bb);_,checks=physical(model,aa,bb)
   physical_ok=all(row.get("known_face_facing_cos") is None or row["known_face_facing_cos"]<=.001 for row in meta) and all(c.get("separation_m",1)>0 and c.get("minimum_depth_m",1)>.03 for c in checks) and all(.04<=np.linalg.norm(a[:3,3])<=.15 for a in aa.values())
   result={"start":start,"status":"candidate_only" if physical_ok else "rejected_physical_constraints","optimizer_success":bool(fit.success),"message":fit.message,"nfev":fit.nfev,"actual_cost_calls_including_finite_differences":calls[0],"seconds":time.perf_counter()-started,"initial_f_px":f,"f_px":ff,"A_brace_camera":{t:a.tolist() for t,a in aa.items()},"capture_from_brace":{f"{fr}/{t}":b.tolist() for (fr,t),b in bb.items()},"parameters":fit.x.tolist(),"data_normalized_rms":float(np.sqrt(np.mean(final_values**2))),"initial_per_type_frame":summary(initial_meta),"final_per_type_frame":summary(meta),"final_observation_residuals":meta,"initial_observation_residuals":initial_meta,"physical_checks":checks,"physical_ok":physical_ok,"data_jacobian":data_rank(data,fit.x),"prior_and_barrier_used":True,"parameter_near_bounds":[int(i) for i in np.where(np.minimum(fit.x-lo,hi-fit.x)<1e-4)[0]]}
  except Exception as exc:
   import traceback
   result={"start":start,"status":"failed_retained","error":str(exc),"traceback":traceback.format_exc(),"actual_cost_calls":calls[0],"seconds":time.perf_counter()-started}
  write(out/f"start_{start}.json",result);results.append(result);print("FIT_START_COMPLETE",start,result["status"],flush=True)
 valid=[r for r in results if r.get("physical_ok")];best=min(valid,key=lambda r:r["data_normalized_rms"]) if valid else None
 held=[]
 if best:
  ff,aa,_=unpack(np.array(best["parameters"]),frames)
  for frame in model["obs"]["held_out_frames"]:
   for t in TRACKS:
    rr=[(fr,o) for fr,o in observations(model,[frame],True) if o["track"]==t]
    rings={"proximal" if o["semantic_boundary"].startswith("proximal") else "distal" for _,o in rr if o["kind"]=="brace_visible_edge"}
    if len(rings)<2 or not any(o["kind"]=="forearm_axis" for _,o in rr):
     held.append({"frame_id":frame,"track":t,"status":"insufficient_brace_axis_observability","body_points_used_for_optimization":False});continue
    init=init_pose(rr,t,ff,model["wh"])
    def localdata(v):return residual_rows(model,rr,ff,aa,{(frame,t):transform(v)})[0]
    fit=least_squares(localdata,init,bounds=([-np.pi]*3+[-2,-2,.12],[np.pi]*3+[2,2,3]),max_nfev=100,loss="soft_l1",x_scale="jac")
    # Six-column local DATA Jacobian, no body observations or global optimization.
    J=np.empty((len(localdata(fit.x)),6));base=localdata(fit.x)
    for j in range(6):
     v=fit.x.copy();v[j]+=1e-6;J[:,j]=(localdata(v)-base)/1e-6
    sv=np.linalg.svd(J*np.array([1,1,1,.1,.1,.1]),compute_uv=False);rank=int(np.sum(sv>sv[0]*1e-6))
    body=[(fr,o) for fr,o in observations(model,[frame]) if o["track"]==t and o["kind"]=="body_edge"]
    _,scores=residual_rows(model,body,ff,aa,{(frame,t):transform(fit.x)})
    min_depth=float(apply(transform(fit.x)@aa[t],model["corners"])[:,2].min())
    held.append({"frame_id":frame,"track":t,"minimum_predicted_body_depth_m":min_depth,"two_camera_physical_check":"unavailable unless both heldoutbrace poses supported","status":"heldout_candidate_only" if rank==6 and min_depth>.03 else "insufficient_local_rank_or_depth","body_points_used_for_optimization":False,"global_f_A_frozen":True,"nfev":fit.nfev,"brace_axis_data_rank":rank,"singular_values":sv.tolist(),"body_prediction_per_type_frame":summary(scores),"capture_from_brace":transform(fit.x).tolist(),"body_residuals":scores})
 for name,h in inputs.items():
  if sha(name)!=h:raise ValueError("Inputs changed during fitting")
 summary_result={"status":"projection_initialization_only_not_calibrated","dataset_generation_allowed":False,"selected_start":best["start"] if best else None,"all_start_status":[{"start":r["start"],"status":r["status"],"data_normalized_rms":r.get("data_normalized_rms")} for r in results],"heldout":held,"remaining_required":["Actual global FK/reach and skin/finger/contact feasibility","Native visual review of candidate projections and occlusions","Lens/mount feature geometry and observer distortion analysis","Joint environment and photometric fit"],"ambiguity_note":"Fixed brace dimensions remain G1 priors. Shared focal/depth/pose and latent edge choices can be weakly identifiable; low residual or prior-assisted rank is not calibration proof."}
 write(out/"summary.json",summary_result);return summary_result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);a=p.parse_args()
 print(json.dumps(run(a.output)))
if __name__=="__main__":main()
